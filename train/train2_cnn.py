import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Định nghĩa mạng nơ-ron tích chập (CNN 5-Block) tối ưu cho ảnh 224x224
class FruitCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(FruitCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: 14x14x256 -> 7x7x256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    print("=" * 60, flush=True)
    print("   HUAN LUYEN MANG CNN 224x224 PHAN LOAI HOA QUA (GPU)   ", flush=True)
    print("=" * 60, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] Thiet bi su dung: CUDA (GPU: {gpu_name})", flush=True)
    else:
        print(f"[*] Thiet bi su dung: CPU", flush=True)

    # Đường dẫn thư mục dữ liệu
    base_dir = "D:/nam4/vinh/xla/bai_tap_lon"
    train_dir = os.path.join(base_dir, "dataset/train")
    test_dir = os.path.join(base_dir, "dataset/test")
    model_save_dir = os.path.join(base_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    onnx_path = os.path.join(model_save_dir, "fruit_cnn.onnx")

    # Pipeline tiền xử lý và tăng cường dữ liệu độ phân giải cao 224x224
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(), # Chuẩn hóa về [0, 1]
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("[*] Dang tai tap du lieu Train & Test (224x224)...", flush=True)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    classes = train_dataset.classes
    print(f"[*] Danh sach lop ({len(classes)}): {classes}", flush=True)
    print(f"[*] So luong anh Train: {len(train_dataset)} | Test: {len(test_dataset)}", flush=True)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Khởi tạo mô hình
    model = FruitCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    epochs = 5
    print(f"\n[*] Bat dau huan luyen {epochs} epochs tren {device.type.upper()}...", flush=True)
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Đánh giá nhanh trên test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = (test_correct / test_total) * 100.0

        print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%", flush=True)

    total_time = time.time() - start_time
    print(f"\n[+] Huan luyen hoan tat trong: {total_time:.1f} giay.", flush=True)

    # Chuyển model về CPU trước khi xuất ONNX cho OpenCV DNN
    print(f"[*] Dang xuat mo hinh sang ONNX: {onnx_path} ...", flush=True)
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamo=False
        )
    except Exception as e:
        print(f"[*] Fallback export: {e}", flush=True)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[+] Xuat ONNX thanh cong! Kich thuoc file: {file_size_mb:.2f} MB", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
