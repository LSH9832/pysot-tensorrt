from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import cv2
import torch
import numpy as np
from .pysot.core.config import cfg
from .pysot.models.model_builder import ModelBuilder
from .pysot.tracker.tracker_builder import build_tracker, TensorRTTracker, TensorRTTracker2
torch.set_num_threads(1)

this_file_dir = str(__file__).replace('\\', '/').replace('__init__.py', '')


class Tracker(object):

    def __init__(self, model_name='siamrpn_alex_dwxcorr_otb', trt=False):
        # load config
        config = this_file_dir + 'experiments/%s/config.yaml' % model_name

        assert os.path.isfile(config), "config file does not exist!"

        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        self.trt = trt

        if self.trt:
            from torch2trt import TRTModule

            def load(name, map_location="cpu"):
                ckpt = torch.load(name, map_location=map_location)
                if "model" in ckpt:
                    ckpt["engine"] = ckpt.pop("model")
                    torch.save(ckpt, name)
                return ckpt

            template_weights = this_file_dir + 'experiments/%s/export/template.trt' % model_name
            track1_weights = this_file_dir + 'experiments/%s/export/track1.trt' % model_name
            track2_weights = this_file_dir + 'experiments/%s/export/track2.trt' % model_name

            template_model = TRTModule()
            template_model.load_state_dict(load(template_weights, map_location="cpu"))

            track1_model = TRTModule()
            track1_model.load_state_dict(load(track1_weights, map_location="cpu"))

            track2_model = TRTModule()
            track2_model.load_state_dict(load(track2_weights, map_location="cpu"))

            tracker = TensorRTTracker2 if cfg.RPN.TYPE == 'MultiRPN' else TensorRTTracker

            self.tracker = tracker(template_model, track1_model, track2_model)

        else:
            # create model
            self.model = ModelBuilder()
            # load model
            snapshot = this_file_dir + 'experiments/%s/model.pth' % model_name
            assert os.path.isfile(snapshot), "weights does not exist!"
            self.model.load_state_dict(torch.load(snapshot, map_location="cpu"))
            self.model.eval().to(device)

            # build tracker
            self.tracker = build_tracker(self.model)
            # self.model.half()

        # warmup
        self.set_boundingbox(np.zeros([3, 3, 3]), [0, 0, 1, 1])

    def set_boundingbox(self, image, boundingbox):
        """
        input:
            frame: bgr image
            bb: bounding box (x_left_top, y_left_top, w, h)

        """
        self.tracker.init(image, boundingbox)

    @staticmethod
    def draw_boundingbox(frame, result, color=(0, 255, 0), thickness=3):
        frame = frame.copy()
        if 'polygon' in result:
            polygon = np.array(result['polygon']).astype(np.int32)
            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            mask = ((result['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = np.round(mask).astype(np.uint8)
            # print('mask', np.max(mask))
            mask = np.stack([0 * mask, mask, 0 * mask]).transpose([1, 2, 0])
            # cv2.imshow('mask',mask)
            mask_weight = 0.8

            background = frame * (1 - (mask == 255)).astype(np.uint8)
            this_object = frame * (mask == 255).astype(np.uint8)
            frame = background + cv2.addWeighted(this_object, 1. - mask_weight, mask, mask_weight, -1)
        else:
            bbox = list(map(int, result['bbox']))
            cv2.putText(frame, '%.3f' % result['best_score'], (bbox[0], bbox[1]+20), 0, 0.7, color, thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(
                img=frame,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color=color,
                thickness=thickness
            )
        return frame

    def update(self, frame):
        outputs = self.tracker.track(frame)
        return outputs




