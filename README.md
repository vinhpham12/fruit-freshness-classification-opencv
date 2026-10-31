# Fruit Freshness Classification System using C++ & OpenCV (Dual Pipeline: SVM & CNN 98.30%)

Hệ thống phân loại hoa quả tươi và hỏng sử dụng thị giác máy tính và học sâu (Deep Learning CNN / SVM), hỗ trợ 2 pipeline song song:
1. **Pipeline 1 (Truyền thống)**: SVM + HOG + 3D HSV Color Histogram (Độ chính xác: 93.40%).
2. **Pipeline 2 (Hiện đại - Khuyên dùng)**: Mạng tích chập 5-Block CNN 224x224 với Global Average Pooling qua OpenCV DNN (Độ chính xác: **98.30%**, tốc độ: ~10ms).

> 📖 **Xem báo cáo kỹ thuật chi tiết:**
> * [**Báo cáo kỹ thuật Mô hình CNN 224x224 (98.30%)**](file:///d:/nam4/vinh/xla/bai_tap_lon/BAO_CAO_MO_HINH_CNN_224x224.md)
> * [**So sánh chi tiết 2 Pipeline (SVM vs CNN)**](file:///d:/nam4/vinh/xla/bai_tap_lon/SO_SANH_HAI_PIPELINE.md)

---

## 📌 1. Giới thiệu tổng quan

Dự án thực hiện nhận diện và đánh giá độ tươi / hỏng của 3 loại quả phổ biến qua 6 lớp (classes):
* `freshapples` (Táo tươi) vs `rottenapples` (Táo hỏng)
* `freshbanana` (Chuối tươi) vs `rottenbanana` (Chuối hỏng)
* `freshoranges` (Cam tươi) vs `rottenoranges` (Cam hỏng)

---

## 🚀 2. Kiến trúc & Pipeline xử lý

Quy trình nhận diện trải qua 4 bước đồng bộ từ huấn luyện đến suy luận thời gian thực:

```
[Ảnh đầu vào / Webcam]
        │
        ▼
[1. Auto-Crop Fruit ROI] ──► Tự động phát hiện viền quả, loại bỏ nền trắng dư thừa
        │
        ▼
[2. Resize Aspect-Ratio] ──► Chuẩn hóa kích thước 128x128 giữ nguyên tỷ lệ quả
        │
        ▼
[3. Feature Extraction]
  ├── 3D HSV Color Histogram (512 bins)  ──► Đặc trưng phân bố màu sắc (Chuẩn hóa L2)
  └── HOG (Histogram of Oriented Gradients) ──► Đặc trưng cạnh biên hình học (Chuẩn hóa L2)
        │
        ▼ (Ghép nối vector đặc trưng 8,612 chiều)
[4. Linear SVM Classifier] ──► Dự đoán nhãn hoa quả & trạng thái Tươi / Hỏng
```

---

## 📊 3. Kết quả đánh giá mô hình

Được huấn luyện trên **10,901 ảnh** tập train và kiểm thử trên **2,698 ảnh** tập test độc lập:

* **Độ chính xác tổng thể (Accuracy)**: **93.40%**
* **Ma trận nhầm lẫn (Confusion Matrix)**:

| Nhãn thực tế | freshapples | freshbanana | freshoranges | rottenapples | rottenbanana | rottenoranges | Độ chính xác |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **freshapples** | **346** | 2 | 2 | 44 | 0 | 1 | 87.6% |
| **freshbanana** | 1 | **373** | 0 | 0 | 7 | 0 | **97.9%** |
| **freshoranges** | 1 | 2 | **371** | 5 | 1 | 8 | **95.6%** |
| **rottenapples** | 10 | 0 | 3 | **577** | 2 | 9 | **96.0%** |
| **rottenbanana** | 0 | 7 | 0 | 2 | **521** | 0 | **98.3%** |
| **rottenoranges** | 5 | 0 | 24 | 34 | 8 | **332** | 82.4% |

---

## 🛠️ 4. Cấu trúc thư mục

```text
bai_tap_lon/
├── train/
│   └── main.cpp          # Mã nguồn huấn luyện mô hình SVM trên 10,901 ảnh
├── eval/
│   └── evaluate.cpp      # Mã nguồn đánh giá Accuracy và Confusion Matrix trên 2,698 ảnh
├── demo/
│   └── demo.cpp          # Ứng dụng GUI Demo tương tác (File Dialog + Live Webcam)
├── pre_set_up/
│   └── rename.py         # Script chuẩn hóa tiền xử lý tên file ảnh
├── model/
│   └── (svm_model.xml)   # File trọng số SVM (~547MB) sau khi train
├── HUONG_DAN_DU_AN.md    # Tài liệu hướng dẫn chi tiết toàn bộ dự án
└── README.md
```

---

## 💻 5. Yêu cầu cài đặt & Biên dịch

### Môi trường khuyến nghị:
* **Hệ điều hành**: Windows 10 / 11
* **Trình biên dịch**: MSYS2 MinGW UCRT64 (GCC/G++ hỗ trợ C++17)
* **Thư viện**: OpenCV (hỗ trợ các module `core`, `imgproc`, `imgcodecs`, `highgui`, `videoio`, `xobjdetect`, `ml`)

### Lệnh biên dịch mã nguồn (g++):

1. **Biên dịch bộ huấn luyện (Train)**:
   ```bash
   g++ -std=c++17 -O2 train/main.cpp -o train/main.exe -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_xobjdetect -lopencv_ml
   ```

2. **Biên dịch bộ đánh giá (Evaluate)**:
   ```bash
   g++ -std=c++17 -O2 eval/evaluate.cpp -o eval/evaluate.exe -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_xobjdetect -lopencv_ml
   ```

3. **Biên dịch ứng dụng Demo tương tác**:
   ```bash
   g++ -std=c++17 -O2 demo/demo.cpp -o demo/demo.exe -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_xobjdetect -lopencv_ml -lcomdlg32
   ```

---

## 🎮 6. Hướng dẫn sử dụng Demo

Chạy file `demo/demo.exe`. Các phím tắt tương tác:
* **`[O]` (Open)**: Mở hộp thoại Windows để chọn ảnh từ máy tính bằng chuột.
* **`[C]` (Camera)**: Bật / Tắt chế độ nhận diện theo thời gian thực từ Webcam máy tính.
* **`[T]` (Text Path)**: Nhập đường dẫn ảnh từ console.
* **`[ESC]` / `[Q]`**: Thoát ứng dụng.

---

## 📜 Giấy phép & Tác giả
* **Tác giả**: vinhpham12
* **Môn học**: Xử lý ảnh (Digital Image Processing)
