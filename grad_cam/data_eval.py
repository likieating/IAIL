import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from convs.xception_dmp import xception

# 定义模型结构（需要与训练时的模型结构一致）
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 112 * 112, 2)  # 假设输入尺寸为 224x224

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_model(model_path, device):
    # 初始化模型
    model = xception(num_classes=2)
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(image_path):
    # 定义预处理 pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    # 应用预处理
    image = transform(image)
    # 添加 batch 维度
    image = image.unsqueeze(0)
    return image


def predict(image_path, model, device):
    # 预处理图片
    image = preprocess_image(image_path)
    image = image.to(device)

    # 推理
    with torch.no_grad():
        outputs = model(image)[0]
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # 获取预测结果
    label = predicted.item()
    prob = probabilities[0][label].item()
    return label, prob


def main():
    # 设置设备
    device = torch.device("cuda:2")

    # 模型和图片路径
    model_path = "/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/dmp/0111-17-50-45-019_OSMA_split1_resnet32_1993_B0_Inc1/task2/val_acc87.24_model.pth"  # 替换为你的 .pth 文件路径
    image_path = "/data/ssd2/tangshuai/DFIL299_new/DFD/train/fake/01_02__exit_phone_room__YVGY8LOK_frame_21_face_0.png"  # 替换为你的测试图片路径

    # 加载模型
    model = load_model(model_path, device)

    # 进行预测
    label, prob = predict(image_path, model, device)

    # 输出结果
    result = "真" if label == 0 else "假"
    print(f"预测结果: {result}, 置信度: {prob:.4f}")


if __name__ == "__main__":
    main()