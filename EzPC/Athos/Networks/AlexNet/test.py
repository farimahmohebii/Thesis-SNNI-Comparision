# Load the ONNX model
# model_path = 'alexnet_Opset16.onnx'


import onnx

# Load the ONNX model
model_path = "alexnet_Opset16.onnx"
model = onnx.load(model_path)

# Initialize counters for operations
conv_ops = 0
relu_ops = 0
pool_ops = 0
fc_ops = 0
flatten_ops = 0
total_ops = 0

# Inspect the model's nodes
for node in model.graph.node:
    total_ops += 1  # Count every node as an operation and layer

    if node.op_type == "Conv":
        conv_ops += 1

    elif node.op_type == "Gemm":
        fc_ops += 1

    elif node.op_type in ["MaxPool", "AveragePool"]:
        pool_ops += 1

    elif node.op_type == "Relu":
        relu_ops += 1

    elif node.op_type == "Flatten":
        flatten_ops += 1

# Display the layer counts
print(f"Number of Layers (Operations considered as Layers) in the ONNX model:")
print(f"Conv (Convolutional layers): {conv_ops}")
print(f"ReLU (Activation layers): {relu_ops}")
print(f"Pooling layers: {pool_ops}")
print(f"Flatten layers: {flatten_ops}")
print(f"Fully Connected layers (Gemm): {fc_ops}")
print(f"Total Number of Layers: {total_ops}\n")

# To include input image size
input_tensor = model.graph.input[0]
input_image_size = None
for dim in input_tensor.type.tensor_type.shape.dim:
    if input_image_size is None:
        input_image_size = []
    input_image_size.append(dim.dim_value if dim.dim_value > 0 else None)

print(f"Input Image Size: {input_image_size}")
