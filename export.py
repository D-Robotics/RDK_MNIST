from models import Net2
import torch

from onnxsim import simplify
import onnx


model = Net2()

# 设置模型为评估模式
model.eval()

# 定义一个示例输入张量（batch_size为1）
dummy_input = torch.randn(1, 1, 28, 28)
y = model(dummy_input)
# 导出模型
output_file = "net2_28x28.onnx"
torch.onnx.export(model, dummy_input, output_file,
                  verbose=False,  # 输出详细信息
                  input_names=['input'],  # 输入节点名称
                  output_names=['output'],  # 输出节点名称
                  opset_version=11)  # ONNX操作集版本

print(f"Model has been exported to {output_file}")



# 加载原始的ONNX模型
original_model = onnx.load(output_file)

# 使用onnxsimplify简化模型
simplified_model, check = simplify(original_model)

# 检查简化后的模型是否仍然与原始模型相同
if check:
    print("Simplified model is valid and equivalent to the original model.")
else:
    print("There might be issues with the simplified model.")

# 保存简化的模型
onnx.save(simplified_model, output_file)
print("Simplified model has been saved as 'simplified_model.onnx'")



