#!/usr/bin/env bash

set -xeuo pipefail
mkdir -p "onnxruntime/1"
curl -fSLo "./onnxruntime/1/model.onnx" "https://github.com/triton-inference-server/onnxruntime_backend/raw/604ee7ae2d75d0204ec756aaf7d7edf5317e7dcc/test/initializer_as_input/models/add_with_initializer/1/model.onnx"
set +x
echo "Model successfully fetched to onnxruntime/1/model.onnx"
