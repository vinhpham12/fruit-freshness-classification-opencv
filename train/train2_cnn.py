import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Định nghĩa mạng nơ-ron tích chập (CNN) tối ưu cho OpenCV DNN
class FruitCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(FruitCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128x3 -> 64x64x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 8 * 8, 256),
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
    print("=" * 60)
    print("   HUAN LUYEN MANG CNN PHAN LOAI HOA QUA (PIPELINE 2)   ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiet bi su dung: {device}")

    # Đường dẫn thư mục dữ liệu
    base_dir = "D:/nam4/vinh/xla/bai_tap_lon"
    train_dir = os.path.join(base_dir, "dataset/train")
    test_dir = os.path.join(base_dir, "dataset/test")
    model_save_dir = os.path.join(base_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    onnx_path = os.path.join(model_save_dir, "fruit_cnn.onnx")

    # Pipeline tiền xử lý nhanh và hiệu quả trên CPU
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), # Chuẩn hóa về [0, 1]
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("[*] Dang tai tap du lieu Train & Test...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    classes = train_dataset.classes
    print(f"[*] Danh sach lop ({len(classes)}): {classes}")
    print(f"[*] So luong anh Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Khởi tạo mô hình
    model = FruitCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    print(f"\n[*] Bat dau huan luyen {epochs} epochs...")
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

        print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"\n[+] Huan luyen hoan tat trong: {total_time:.1f} giay.")

    # Xuất mô hình sang định dạng chuẩn ONNX cho C++ OpenCV
    print(f"[*] Dang xuat mo hinh sang ONNX: {onnx_path} ...", flush=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    try:
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
    except Exception as e:
        print(f"[*] Chuyen sang cu phap fallback legacy: {e}", flush=True)
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
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[+] Xuat ONNX thanh cong! Kich thuoc file: {file_size_mb:.2f} MB", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
