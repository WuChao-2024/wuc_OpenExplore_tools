import torch
import copy
import torch.nn as nn
import torch.onnx
import onnx
from onnxsim import simplify


class BPU_Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        #   (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
        #   (up_proj): Linear(in_features=896, out_features=4864, bias=False)
        #   (down_proj): Linear(in_features=4864, out_features=896, bias=False)
        self.hidden_size=896
        self.intermediate_size=4864
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


hidden_size = 896
model = BPU_Qwen2MLP(hidden_size)

# 创建一个随机输入张量
input_data = torch.randn(1, 896, 1, 128)  # 假设批量大小为1

# 运行模型
output = model(input_data)
print(output.shape)  # 应该输出 [1, 3840]

# 设置模型为评估模式
model.eval()
onnx_save_path = "Qwen2MLP_1x896x1x128.onnx"
# 导出模型
torch.onnx.export(
    model,  # 要转换的模型
    input_data,  # 模型的输入
    onnx_save_path,  # 输出文件名
    export_params=True,  # 存储训练后的参数
    opset_version=11,  # ONNX版本
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=['input'],  # 输入节点名称
    output_names=['output'],  # 输出节点名称
    dynamic_axes=None
)

# Checks
model_onnx = onnx.load(onnx_save_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

model_onnx, check = simplify(
    model_onnx,
    dynamic_input_shape=False,
    input_shapes=None)
assert check, 'assert check failed'
onnx.save(model_onnx, onnx_save_path)

print('Onnx model save as {}'.format(onnx_save_path))
