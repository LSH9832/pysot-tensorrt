# pysot-tensorrt

## Intro
- we convert most of pysot network layers to tensorrt fp16 model to get speed up.
- **mask** model is not supported now
- here's the results on UAV123-car6 (4864 frames)

|             Name              | FPS<sup>RTX3060<br>trt fp16 | FPS<sup>RTX3060<br>pytorch |
|:------------------------------|:---------------------------:|:--------------------------:|
| siamrpn_alex_dwxcorr(_otb)    | 128.49                      |  78.54                     |
| siamrpn_mobilev2_l234_dwxcorr | 80.59                       |  72.91                     |
| siamrpn_r50_l234_dwxcorr      | bad precision(I don't know why) |  -                     |

## Setup

### warning: make sure torch2trt and TensorRT Development Toolkit(version>7.1.3.0) are all installed.

```shell
# install torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install
```
then
```
# setup
git clone https://github.com/LSH9832/pysot-tensorrt.git
cd pysot-tensorrt
pip install -r requirements
```

## Export to tensorrt

- download weights in origin repository [STVIR/pysot](https://github.com/STVIR/pysot) and move weights to **pysot/experiments/${RELATIVE DIR}**
- then
```
python export.py --name siamrpn_alex_dwxcorr --workspace 4

# full options
python export.py --name siamrpn_alex_dwxcorr   # (str)   experience name
                 --no-sim                      # (bool)  do not simplify model
                 --opset                       # (int)   onnx opset version
                 --workspace 8                 # (float) max workspace(GB), default is 8
                 --no-fp16                     # (bool)  use fp32 precision
                 --no-trt                      # (bool)  export onnx only (for debug)
```
all generated tensorrt engine files will be saved in **pysot/experiments/${RELATIVE DIR}/export**

## Test FPS
```
python test.py --trt --name siamrpn_alex_dwxcorr --source ${PATH TO YOUR VIDEO}

# full options
python test.py --trt 
               --name siamrpn_alex_dwxcorr 
               --source ${PATH TO YOUR VIDEO}
               --bbox 508 180 139 110    # (this is UAV123-car6 init bbox) input bbox instead of selecting roi bbox by hand
               --max-fps                 # limit of display fps
```


