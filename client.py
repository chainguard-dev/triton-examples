import argparse
import json
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

def log_result(input: list[str], expected: str, output: str, successful: bool):
    result = [{
        "input": input,
        "expected": expected,
        "output": output,
        "successful": successful
    }]
    print(json.dumps(result, indent=4))

def main() -> int:
    _parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Testing for Tritonserver", prog="Tritonserver Client Tests")

    _parser.add_argument('model',
                         type=str,
                         help="Model that will be used with the client")

    _parser.add_argument('-s', '--server',
                         type=str,
                         default="localhost:8001",
                         help="Host that will be used for the GRPC client (e.g.: localhost:8001)")

    args = _parser.parse_args()


    match args.model:
        case "python" | "openvino" | "onnxruntime" | "tensorrt":
            ...
        case _:
            print("Failed finding supported model")
            return 1

    with grpcclient.InferenceServerClient(args.server) as client:
        match args.model:
            case "python":
                return handle_python(client, model_name=args.model)
            case "openvino":
                return handle_openvino(client, model_name=args.model)
            case "onnxruntime" | "onnxruntime_gpu":
                return handle_onnxruntime(client, model_name=args.model)
            case "tensorrt":
                return handle_tensorrt(client, model_name=args.model)

def handle_openvino(client, model_name: str) -> int:
    # https://github.com/triton-inference-server/openvino_backend/blob/64651dcd5a7e465c2a9d37d9c3a701b75f923df2/tests/functional/model_config.py#L28
    shape = [1, 3, 224, 224]
    input0_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        grpcclient.InferInput(
            "gpu_0/data_0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]
    outputs = [
        grpcclient.InferRequestedOutput("gpu_0/softmax_1"),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    response = client.infer(
        model_name, inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("gpu_0/softmax_1")

    success_bool = output0_data.shape == (1, 1000)

    log_result(input=[str(input0_data)],
               expected="(1,1000)",
               output=output0_data.shape,
               successful=success_bool)

    return 0 if success_bool else 1

def handle_onnxruntime(client, model_name: str) -> int:
    input0_data = np.random.rand(5, 5).astype(np.float32)
    inputs = [
        grpcclient.InferInput(
            "INPUT", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        grpcclient.InferInput(
            "INITIALIZER", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]
    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT"),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input0_data)

    response = client.infer(
        model_name, inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("OUTPUT")

    expected_bool = np.allclose(input0_data*2, output0_data)

    log_result(input=[str(input0_data)],
                expected=str(input0_data*2),
                output=str(output0_data),
                successful=expected_bool)

    return 0 if expected_bool else 1

def handle_tensorrt(client, model_name: str) -> int:
    input0_data = np.random.rand(5, 5).astype(np.float32)
    inputs = [
        grpcclient.InferInput(
            "INPUT", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]
    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT"),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    response = client.infer(
        model_name, inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("OUTPUT")

    expected_bool = np.allclose(input0_data+1, output0_data)

    log_result(input=[str(input0_data)],
                expected=str(input0_data+1),
                output=str(output0_data),
                successful=expected_bool)

    return 0 if expected_bool else 1

def handle_python(client, model_name: str) -> int:
    shape = [4]
    input0_data = np.random.rand(*shape).astype(np.float32)
    input1_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        grpcclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        grpcclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]
    
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)
    
    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT0"),
        grpcclient.InferRequestedOutput("OUTPUT1"),
    ]
    
    response = client.infer(
        model_name, inputs, request_id=str(1), outputs=outputs
    )

    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")

    addition_success = np.allclose(input0_data + input1_data, output0_data)
    subtraction_success = np.allclose(input0_data - input1_data, output1_data)

    result = [
        {
            "input": [
                str(input0_data),
                str(input1_data)
            ],
            "expected": str(input0_data + input1_data),
            "output": str(output0_data),
            "successful": addition_success
        },
        {
            "input": [
                str(input0_data),
                str(input1_data)
            ],
            "expected": str(input0_data - input1_data),
            "output": str(output0_data),
            "successful": subtraction_success
        }
    ]
    print(json.dumps(result, indent=4))

    return (0 if addition_success else 1) + (0 if subtraction_success else 1)

if __name__ == "__main__":
    raise SystemExit(main())

