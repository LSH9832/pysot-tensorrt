import os

from pysot.pysot.models.model_builder import ModelBuilder, ExportModelBuilder, ExportModelBuilder2
from pysot.pysot.core.config import cfg

import argparse
import torch
import onnx
import os.path as osp


def get_args():
    parser = argparse.ArgumentParser("SiamRPN export parser")

    parser.add_argument("--name", type=str, default="siamrpn_alex_dwxcorr_otb")
    parser.add_argument("--weights", type=str, default="pysot/experiments/siamrpn_alex_dwxcorr_otb/model.pth", help="model weights")

    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--opset", type=int, default=11)

    parser.add_argument("--no-trt", action="store_true", help="do not export tensorrt model")
    parser.add_argument("-w", "--workspace", type=float, default=8.0)
    parser.add_argument("--no-fp16", action="store_true")

    return parser.parse_args()


def main_export(args):

    args.weights = f"pysot/experiments/{args.name}/model.pth"

    dirname = osp.join(osp.dirname(args.weights), "export")
    os.makedirs(dirname, exist_ok=True)

    config = osp.join(osp.dirname(args.weights), "config.yaml")
    assert osp.isfile(config), "config file does not exist!"

    cfg.merge_from_file(config)

    exporter = ExportModelBuilder2 if cfg.RPN.TYPE == 'MultiRPN' else ExportModelBuilder

    model = exporter(state_dict=torch.load(args.weights, map_location="cpu"))

    model.fp16 = not args.no_fp16
    model.workspace = args.workspace

    template = torch.ones([1, 3, cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE])
    x = torch.ones([1, 3, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

    model.start_export(template, x, dirname, not args.no_trt)


def onnx_simplify(filename):
    onnx_model = onnx.load(filename)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    try:
        import onnxsim
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'
        onnx.save(onnx_model, filename)
    except Exception as e:
        print(f'Simplifier failure: {e}')\


def export_onnx(args):
    os.makedirs(osp.join(osp.dirname(args.weights), "export"), exist_ok=True)
    config = osp.join(osp.dirname(args.weights), "config.yaml")
    assert osp.isfile(config), "config file does not exist!"

    template_input_names = ["z"]
    template_output_names = ["zf"]
    track_input_names = ["zf", "x"]
    track_output_names = ["cls", "loc"]
    data = {
        "template": {"input_names": template_input_names, "output_names": template_output_names},
        "track": {"input_names": track_input_names, "output_names": track_output_names}
    }
    template_file = osp.join(osp.dirname(args.weights), "export", "template.onnx")
    track_file = osp.join(osp.dirname(args.weights), "export", "track.onnx")

    if args.onnx or (args.trt and not (osp.isfile(template_file) and (osp.isfile(track_file)))):
        cfg.merge_from_file(config)
        model = ModelBuilder()
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.eval()

        z = torch.ones([1, 3, cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE])
        x = torch.ones([1, 3, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])

        model.export_template()
        zf = model(z)
        print(zf.shape)

        torch.onnx.export(model,
                          z,
                          template_file,
                          verbose=False,
                          opset_version=args.opset,
                          input_names=template_input_names,
                          output_names=template_output_names,
                          dynamic_axes=None)

        if not args.no_sim:
            onnx_simplify(template_file)

        model.export_track()
        cls, loc = model(zf, x)
        print(cls.shape, loc.shape)

        torch.onnx.export(model,
                          (zf, x),
                          track_file,
                          verbose=False,
                          opset_version=args.opset,
                          input_names=track_input_names,
                          output_names=track_output_names,
                          dynamic_axes=None)

        if not args.no_sim:
            onnx_simplify(track_file)

    import json
    json.dump(data, open(osp.join(osp.dirname(args.weights), "export", "io_names.json"), "w"))

    return data, template_file, track_file


def export_tensorrt(args, data, template_file, track_file):
    if not args.trt:
        return

    def onnx2trt(onnxfile, save_data):
        save_engine = onnxfile[:-5] + ".engine"
        save_trt = onnxfile[:-5] + ".trt"
        command1 = f"trtexec --onnx={onnxfile}{' --fp16' if not args.no_fp16 else ''} " \
                   f"--saveEngine={save_engine} --workspace={int(args.workspace*1024)} --buildOnly"

        print("#" * 100 + "\n")
        os.system(command1)
        if not osp.isfile(save_engine):
            print(f"failed to build tensorRT engine file {save_engine}.")
            return

        save_data["model"] = bytearray(open(save_engine, "rb").read())
        torch.save(save_data, save_trt)
        if osp.isfile(save_trt):
            print(f"engine file saved to {save_trt}")

    onnx2trt(template_file, data["template"])
    onnx2trt(track_file, data["track"])


def main():
    args = get_args()
    # export_tensorrt(args, *export_onnx(args))
    main_export(args)


if __name__ == '__main__':
    main()




