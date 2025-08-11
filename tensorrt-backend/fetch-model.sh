#!/usr/bin/env bash

set -xeuo pipefail
curl -fSLo "./model.onnx" "https://raw.githubusercontent.com/triton-inference-server/onnxruntime_backend/refs/heads/main/test/initializer_as_input/models/add_with_initializer/1/model.onnx"
set +x
echo "Model successfully fetched to the current working directory"
