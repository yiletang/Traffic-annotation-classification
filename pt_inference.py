import torch
from torchvision import models, transforms
from PIL import Image
from model import Model

model_path = r'C:\Users\H2250\Desktop\Traffic annotation classification\Traffic annotation classification\save_model\best_model.pt'
model = Model(58)
model.load_state_dict(torch.load(model_path))
model.eval()  # 将模型设置为评估模式


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = r'C:\Users\H2250\Desktop\Traffic annotation classification\Traffic annotation classification\003_xs40.png'  # 新图片的文件路径
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # 添加batch维度

# 在不需要计算梯度的情况下执行前向传播
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

class_names = {
    0: "限速5km", 1: "限速15km", 2: "限速30km", 3: "限速40km", 5: "限速60km",
    6: "限速70km", 7: "限速80km", 8: "禁止左转和直行", 9: "禁止直行和右转",
    10: "禁止直行", 11: "禁止左转", 12: "禁止左右转弯", 14: "禁止超车",
    15: "禁止掉头", 16: "禁止机动车驶入", 17: "禁止鸣笛", 18: "解除40km限制",
    19: "解除50km限制", 20: "直行和右转", 21: "单直行", 22: "向左转弯",
    23: "向左向右转弯", 24: "向右转弯", 25: "靠左侧通道行驶", 26: "靠右侧道路行驶",
    27: "环岛行驶", 28: "机动车行驶", 29: "鸣喇叭", 30: "非机动车行驶",
    31: "允许掉头", 32: "左右绕行", 33: "注意红绿灯", 34: "注意危险",
    35: "注意行人", 36: "注意非机动车", 37: "注意儿童", 38: "向右急转弯",
    39: "向左急转弯", 40: "下陡坡", 41: "上陡坡", 42: "慢行", 43: "T形交叉",
    44: "T形交叉", 45: "村庄", 46: "反向弯路", 47: "无人看守铁路道口",
    48: "施工", 49: "连续弯路", 50: "有人看守铁路道口", 51: "事故易发生路段",
    52: "停车让行", 53: "禁止通行", 54: "禁止车辆临时或长时间停放", 55: "禁止输入",
    56: "减速让车", 57: "停车检查"
}

# 获取预测类别的标签，如果标签不存在则返回"类别未知"
predicted_class = class_names.get(predicted.item(), "类别未知")
print(f'Predicted class: {predicted_class}')
