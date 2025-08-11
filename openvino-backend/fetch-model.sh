#!/usr/bin/env bash

set -xeuo pipefail
mkdir -p "openvino/1"
curl -fSLo "./openvino/1/model.onnx" "https://github.com/onnx/models/raw/b1eeaa1ac722dcc1cd1a8284bde34393dab61c3d/validated/vision/classification/resnet/model/resnet50-caffe2-v1-9.onnx"
set +x
echo "Model successfully fetched to openvino/1/model.onnx"
