# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from ...pysot.core.config import cfg
from ...pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from ...pysot.models.backbone import get_backbone
from ...pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from ...pysot.models.neck import get_neck

from .head.rpn import DepthwiseXCorr
from ..core.xcorr import xcorr_depthwise

import onnx
import torch
import os
import os.path as osp
from loguru import logger


class ModelBuilder(nn.Module):

    export = False

    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)


    def template(self, z):

        print("template input shape:", z.shape)

        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

        try:
            print("template output shape:", zf.shape)
        except:
            print("template output shape:", [f_.shape for f_ in zf])

        if self.export:
            return zf

    def export_template(self):
        self.export = True
        self.forward = self.template

    def export_track(self):
        self.export = True
        self.forward = self._track

    def _track(self, zf, x):

        print("track input shape:", x.shape)
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc = self.rpn_head(zf, xf)
        return cls, loc

    def track(self, x):

        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs


class ExportModelBuilder(ModelBuilder):

    def __init__(self, state_dict=None):
        super(ExportModelBuilder, self).__init__()
        if state_dict is not None:
            self.load_state_dict(state_dict)
        self.eval()

        self.cls: DepthwiseXCorr = self.rpn_head.cls
        self.loc: DepthwiseXCorr = self.rpn_head.loc

        self.cls_kernel = None
        self.loc_kernel = None

        self.fp16 = True
        self.workspace = 8
        self.template = self.template_forward

    def template_forward(self, z):
        out = self.backbone(z)
        self.cls_kernel = self.cls.conv_kernel(out)
        self.loc_kernel = self.loc.conv_kernel(out)
        return self.cls_kernel, self.loc_kernel

    def track1(self, x):
        out = self.backbone(x)
        cls_search = self.cls.conv_search(out)
        loc_search = self.loc.conv_search(out)
        return cls_search, loc_search

    def track_reserve(self, zf, x):
        feature = xcorr_depthwise(x, zf)
        return feature

    def track2(self, cls_feature, loc_feature):
        cls = self.cls.head(cls_feature)
        loc = self.loc.head(loc_feature)
        return cls, loc

    def track(self, x):
        cls_search, loc_search = self.track1(x)
        cls_feature = self.track_reserve(self.cls_kernel, cls_search)
        loc_feature = self.track_reserve(self.loc_kernel, loc_search)
        return self.track2(cls_feature, loc_feature)

    @staticmethod
    def onnx_simplify(filename):
        onnx_model = onnx.load(filename)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
            onnx.save(onnx_model, filename)
        except Exception as e:
            print(f'Simplifier failure: {e}')

    def export_one_part(self, forward_func, net_inputs, filename, io_names):
        filename = filename.replace("\\", "/")
        self.forward = forward_func

        # export
        torch.onnx.export(
            self,
            tuple(net_inputs),
            filename,
            verbose=False,
            opset_version=11,
            **io_names,
            dynamic_axes=None
        )

        self.onnx_simplify(filename)

        if osp.isfile(filename):
            logger.info(f"export file {filename} is saved.")


    def export_onnx(self, template, x, dirname):
        import json

        os.makedirs(dirname, exist_ok=True)

        logger.info("start export to onnx")

        template_input_names = ["z"]
        template_output_names = ["cls_k", "loc_k"]
        track1_input_names = ["x"]
        track1_output_names = ["cls_s", "loc_s"]
        track2_input_names = ["cls_f", "loc_f"]
        track2_output_names = ["cls", "loc"]
        names = {
            "template": {"input_names": template_input_names, "output_names": template_output_names},
            "track1": {"input_names": track1_input_names, "output_names": track1_output_names},
            "track2": {"input_names": track2_input_names, "output_names": track2_output_names}
        }

        logger.info("1. export template")

        filename1 = osp.join(dirname, "template.onnx")
        self.export_one_part(self.template_forward, [template], filename1, names["template"])


        logger.info("2. export track part 1.")

        cls_k, loc_k = self(template)

        filename2 = osp.join(dirname, "track1.onnx")
        self.export_one_part(self.track1, [x], filename2, names["track1"])


        logger.info("3. export track part 2.")

        cls_s, loc_s = self(x)
        cls_f = self.track_reserve(cls_k, cls_s)
        loc_f = self.track_reserve(loc_k, loc_s)

        filename3 = osp.join(dirname, "track2.onnx")
        self.export_one_part(self.track2, [cls_f, loc_f], filename3, names["track2"])


        json.dump(names, open(osp.join(dirname, "names.json"), "w"))

        return [filename1, filename2, filename3], [names["template"], names["track1"], names["track2"]]

    def export_trt(self, filenames, names):


        def onnx2trt(onnxfile, save_data):
            save_engine = onnxfile[:-5] + ".engine"
            save_trt = onnxfile[:-5] + ".trt"
            command1 = f"trtexec --onnx={onnxfile}{' --fp16' if self.fp16 else ''} " \
                       f"--saveEngine={save_engine} --workspace={int(self.workspace * 1024)} --buildOnly"

            print("#" * 100 + "\n")
            os.system(command1)
            if not osp.isfile(save_engine):
                logger.error(f"failed to build tensorRT engine file {save_engine}.")
                return

            save_data["engine"] = bytearray(open(save_engine, "rb").read())
            torch.save(save_data, save_trt)
            if osp.isfile(save_trt):
                logger.info(f"engine file saved to {save_trt}")

        for filename, name in zip(filenames, names):
            onnx2trt(filename, name)

    def start_export(self, template, x, dirname, trt=True):

        if trt:
            self.export_trt(*self.export_onnx(template, x, dirname))
        else:
            self.export_onnx(template, x, dirname)


class ExportModelBuilder2(ModelBuilder):

    def __init__(self, state_dict=None):
        super(ExportModelBuilder2, self).__init__()
        if state_dict is not None:
            self.load_state_dict(state_dict)
        self.eval()

        self.rpns = [self.rpn_head.rpn2, self.rpn_head.rpn3, self.rpn_head.rpn4]

        # self.cls: DepthwiseXCorr = self.rpn_head.cls
        # self.loc: DepthwiseXCorr = self.rpn_head.loc

        self.cls_kernels = []
        self.loc_kernels = []

        self.fp16 = True
        self.workspace = 8
        self.template = self.template_forward

    def template_forward(self, z):
        out = self.neck(self.backbone(z))
        self.cls_kernels = []
        self.loc_kernels = []
        for idx, rpn in enumerate(self.rpns):
            self.cls_kernels.append(rpn.cls.conv_kernel(out[idx]))
            self.loc_kernels.append(rpn.loc.conv_kernel(out[idx]))
        # print(len(self.cls_kernels), len(self.loc_kernels))
        return *self.cls_kernels, *self.loc_kernels

    def track1(self, x):
        out = self.neck(self.backbone(x))
        cls_searchs = []
        loc_searchs = []
        for idx, rpn in enumerate(self.rpns):
            cls_searchs.append(rpn.cls.conv_search(out[idx]))
            loc_searchs.append(rpn.loc.conv_search(out[idx]))
        return *cls_searchs, *loc_searchs

    def track_reserve(self, zf, x):
        feature = xcorr_depthwise(x, zf)
        return feature

    def track2(self, cls1_feature, cls2_feature, cls3_feature, loc1_feature, loc2_feature, loc3_feature):
        clss = []
        locs = []
        cls_features = [cls1_feature, cls2_feature, cls3_feature]
        loc_features = [loc1_feature, loc2_feature, loc3_feature]
        for idx, rpn in enumerate(self.rpns):
            clss.append(rpn.cls.head(cls_features[idx]))
            locs.append(rpn.loc.head(loc_features[idx]))

        def avg(lst):
            add = lst[0]
            for i in range(1, len(lst)):
                add += lst[i]

            return add / len(lst)

        return avg(clss), avg(locs)


    @staticmethod
    def onnx_simplify(filename):
        onnx_model = onnx.load(filename)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
            onnx.save(onnx_model, filename)
        except Exception as e:
            print(f'Simplifier failure: {e}')

    def export_one_part(self, forward_func, net_inputs, filename, io_names):
        filename = filename.replace("\\", "/")
        self.forward = forward_func

        # export
        torch.onnx.export(
            self,
            tuple(net_inputs),
            filename,
            verbose=False,
            opset_version=11,
            **io_names,
            dynamic_axes=None
        )

        self.onnx_simplify(filename)

        if osp.isfile(filename):
            logger.info(f"export file {filename} is saved.")


    def export_onnx(self, template, x, dirname):
        import json

        os.makedirs(dirname, exist_ok=True)

        logger.info("start export to onnx")

        template_input_names = ["z"]
        template_output_names = ["cls1_k", "cls2_k", "cls3_k", "loc1_k", "loc2_k", "loc3_k"]
        track1_input_names = ["x"]
        track1_output_names = ["cls1_s", "cls2_s", "cls3_s", "loc1_s", "loc2_s", "loc3_s"]
        track2_input_names = ["cls1_f", "cls2_f", "cls3_f", "loc1_f", "loc2_f", "loc3_f"]
        track2_output_names = ["cls", "loc"]
        names = {
            "template": {"input_names": template_input_names, "output_names": template_output_names},
            "track1": {"input_names": track1_input_names, "output_names": track1_output_names},
            "track2": {"input_names": track2_input_names, "output_names": track2_output_names}
        }

        logger.info("1. export template")

        filename1 = osp.join(dirname, "template.onnx")
        self.export_one_part(self.template_forward, [template], filename1, names["template"])


        logger.info("2. export track part 1.")

        cls1_k, cls2_k, cls3_k, loc1_k, loc2_k, loc3_k = self(template)

        filename2 = osp.join(dirname, "track1.onnx")
        self.export_one_part(self.track1, [x], filename2, names["track1"])


        logger.info("3. export track part 2.")

        cls1_s, cls2_s, cls3_s, loc1_s, loc2_s, loc3_s = self(x)
        clss_k = [cls1_k, cls2_k, cls3_k]
        locs_k = [loc1_k, loc2_k, loc3_k]
        clss_s = [cls1_s, cls2_s, cls3_s]
        locs_s = [loc1_s, loc2_s, loc3_s]
        clss_f = []
        locs_f = []
        for idx in range(3):
            clss_f.append(self.track_reserve(clss_k[idx], clss_s[idx]))
            locs_f.append(self.track_reserve(locs_k[idx], locs_s[idx]))

        filename3 = osp.join(dirname, "track2.onnx")
        self.export_one_part(self.track2, [*clss_f, *locs_f], filename3, names["track2"])


        json.dump(names, open(osp.join(dirname, "names.json"), "w"))

        return [filename1, filename2, filename3], [names["template"], names["track1"], names["track2"]]

    def export_trt(self, filenames, names):


        def onnx2trt(onnxfile, save_data):
            save_engine = onnxfile[:-5] + ".engine"
            save_trt = onnxfile[:-5] + ".trt"
            command1 = f"trtexec --onnx={onnxfile}{' --fp16' if self.fp16 else ''} " \
                       f"--saveEngine={save_engine} --workspace={int(self.workspace * 1024)} --buildOnly"

            print("#" * 100 + "\n")
            os.system(command1)
            if not osp.isfile(save_engine):
                logger.error(f"failed to build tensorRT engine file {save_engine}.")
                return

            save_data["engine"] = bytearray(open(save_engine, "rb").read())
            torch.save(save_data, save_trt)
            if osp.isfile(save_trt):
                logger.info(f"engine file saved to {save_trt}")

        for filename, name in zip(filenames, names):
            onnx2trt(filename, name)

    def start_export(self, template, x, dirname, trt=True):

        if trt:
            self.export_trt(*self.export_onnx(template, x, dirname))
        else:
            self.export_onnx(template, x, dirname)