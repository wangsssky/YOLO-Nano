# Introduction
- YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection. [paper](https://arxiv.org/abs/1910.01271).
- This repo is based on [liux0614]( https://github.com/liux0614/yolo_nano).

# Project Structure
<pre>
root/
  datasets/
    coco/
      images/
        train/
        val/
      annotation/
        instances_train2017.json
</pre>

# Installation
```bash
git clone https://github.com/wangsssky/YOLO-Nano.git
pip3 install -r requirements.txt
```
# COCO
To use COCO dataset loader, _pycocotools_ should be installed via the following command.
```bash 
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

To train on COCO dataset:
```bash
python3 main.py --dataset_path datasets/coco/images --annotation_path datasets/coco/annotations 
                --dataset coco --conf_thresh=0.8 --gpu
```

# Convert to onnx
- cd deploy, run convert2onnx.py
- run ```python -m onnxsim yolo-nano.onnx simplified.onnx```, you may install onnx-simplifier first.
- try run_onnx.py, test it by onnx runtime
