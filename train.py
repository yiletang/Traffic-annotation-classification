import torch
from dataloader import ImageDataset
from model import Model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_metrics(model, dataloader, device, criterion):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, filename='training_metrics.png'):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model(58)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = ImageDataset(
        annotations_file=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\train_data.csv',
        img_dir=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\images'
    )

    val_data = ImageDataset(
        annotations_file=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\val_data.csv',
        img_dir=r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\data\images'
    )

    train_dataloader = DataLoader(train_data, batch_size=160, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=160, shuffle=True)

    num_epochs = 25
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for imgs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(imgs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        # 计算训练集和验证集的准确率
        train_loss, train_accuracy = calculate_metrics(model, train_dataloader, device, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = calculate_metrics(model, val_dataloader, device, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # 保存模型
            torch.save(model.state_dict(), r'C:\Users\JAN\Desktop\Traffic annotation classification\Traffic annotation classification\save_model\best_model.pt')
            print(f"New best model saved with val_accuracy: {best_val_accuracy}%")

        print(f"Loss: {running_loss / len(train_dataloader)}, "
              f"Train Accuracy: {train_accuracy}%, Val Accuracy: {val_accuracy}%")

    plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 'metrics.png')
    print('Finished Training')
