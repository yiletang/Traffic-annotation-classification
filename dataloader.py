from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#数据增强操作
transform = transforms.Compose([
        transforms.Resize((224, 224)), #图像裁剪为 224
        transforms.ToTensor(), #将图像转化为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.show()

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]

        return transform(image), int(label)





if __name__ == "__main__":
    train_dataset = ImageDataset(
        annotations_file=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\train_data.csv',
        img_dir=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\images'
    )

    # 使用 DataLoader 来加载数据集
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 获取一个批次的数据
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    # 显示图像和标签
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        ax = axes[i]
        imshow(images[i])
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')

    for data, target in train_dataloader:
        print(data.shape)
        print(target.shape)
        print(data)
        print(target)
        break

