import torch.onnx
from model import Model


model = Model(58)

# 加载模型状态字典
model_path = "C:/Users/H2250/Desktop/Traffic annotation classification/Traffic annotation classification/save_model/best_model.pt"
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)
model.eval()  # 切换到评估模式

# 创建一个符合模型输入形状的虚拟输入张量
x = torch.randn(1, 3, 224, 224)

# 确保模型和输入都在相同的设备上，这里假设使用CPU
model.to('cpu')
x = x.to('cpu')

# 导出模型
torch.onnx.export(model,               # 运行的模型
                  x,                   # 模型输入（或一个元组，对于多个输入）
                  "model.onnx",        # 输出的ONNX文件名
                  export_params=True,  # 导出模型时包括训练参数
                  opset_version=10,    # ONNX版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],   # 输入名
                  output_names=['output'], # 输出名
                  dynamic_axes={'input': {0: 'batch_size'},    # 动态轴的定义
                                'output': {0: 'batch_size'}})

print('finish')
