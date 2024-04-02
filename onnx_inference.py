import onnxruntime as ort
from PIL import Image
import numpy as np
from torchvision import transforms

# ONNX模型的路径
onnx_model_path = r'C:\Users\H2250\Desktop\Traffic annotation classification\Traffic annotation classification\save_model\model.onnx'

# ONNX模型
ort_session = ort.InferenceSession(onnx_model_path)

# 定义图像的转换操作
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载图像并应用转换
image_path = r'C:\Users\H2250\Desktop\Traffic annotation classification\Traffic annotation classification\003_xs40.png'
image = Image.open(image_path)
image = transform(image)
image_np = np.expand_dims(image.numpy(), 0)  # 将图像转换为模型所需的numpy数组格式

# 进行推理
inputs = {ort_session.get_inputs()[0].name: image_np}
outputs = ort_session.run(None, inputs)

# 获取预测结果
output = outputs[0]
predicted_idx = np.argmax(output)

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

# 获取预测类别的标签
predicted_class = class_names.get(predicted_idx, "类别未知")
print(f'Predicted class: {predicted_class}')
