import os
import sys
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Định nghĩa mạng nơ-ron tích chập (FruitCNN - Cách 1)
# Áp dụng Global Average Pooling (GAP) để tối ưu trích xuất vân bề mặt (texture)
# và triệt tiêu bùng nổ tham số ở tầng fully-connected.
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

        # Global Average Pooling: Biến đặc trưng 7x7x256 -> 1x1x256
        # Giúp mạng học tính bất biến vị trí, tập trung vào cấu trúc vỏ quả
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier gọn nhẹ (~33,000 tham số thay vì 3.2 triệu tham số cũ)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    print("=" * 65, flush=True)
    print("   HUAN LUYEN MANG CNN 224x224 (CACH 1: GAP + COLOR JITTER)     ", flush=True)
    print("=" * 65, flush=True)

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

    # Tăng cường dữ liệu: thêm ColorJitter để phân biệt rõ vân vỏ và tránh phụ thuộc ánh sáng
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )

    # Khởi tạo mô hình
    model = FruitCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    # Giảm dần learning rate qua các epoch để mạng tinh chỉnh chi tiết
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[*] Bat dau huan luyen {epochs} epochs tren {device.type.upper()}...", flush=True)
    start_time = time.time()

    best_test_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

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

        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        # Đánh giá trên test set
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

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch [{epoch+1:2d}/{epochs}] (lr: {current_lr:.6f}) - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}%", flush=True)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print(f"\n[+] Huan luyen hoan tat trong: {total_time:.1f} giay. Test Acc tot nhat: {best_test_acc:.2f}%", flush=True)

    # Nạp lại checkpoint tốt nhất trước khi xuất ONNX
    model.load_state_dict(best_model_weights)
    model.eval()
    model.to('cpu')

    print(f"[*] Dang xuat mo hinh sang ONNX: {onnx_path} ...", flush=True)
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

    # Dọn dẹp file .data thừa nếu có từ lần trước
    old_data_file = onnx_path + ".data"
    if os.path.exists(old_data_file):
        try:
            os.remove(old_data_file)
            print(f"[*] Da don dep file thua: {old_data_file}", flush=True)
        except Exception:
            pass

    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
