# SO SÁNH HAI PIPELINE TRONG DỰ ÁN NHẬN DIỆN HOA QUẢ

Dự án hiện tại được chia thành **2 Pipeline độc lập** để bạn có thể chạy song song và so sánh trực tiếp hiệu năng, độ chính xác và trải nghiệm trực quan.

---

## 📊 Bảng so sánh tổng hợp

| Tiêu chí | Pipeline 1 (Truyền thống) | Pipeline 2 (Hiện đại) |
| :--- | :--- | :--- |
| **Kiến trúc cốt lõi** | **SVM Cổ điển (Handcrafted Features)** | **Xử lý ảnh hình học + Deep Learning CNN** |
| **Đặc trưng trích xuất** | HOG (8,100 chiều) + Biểu đồ màu 3D HSV (512 chiều) = 8,612 chiều | Mạng tích chập sâu (4-layer CNN) tự động học đặc trưng từ điểm ảnh |
| **Mô hình & Thuật toán** | `cv::ml::SVM` (Linear Kernel, C=1.0) | `cv::dnn` nạp mạng nơ-ron **ONNX** (`fruit_cnn.onnx`) |
| **Phân vùng & Bounding Box** | Cắt viền ngang dọc (Axis-Aligned Bounding Box) | **Khung chữ nhật nghiêng (Oriented Bounding Box)** ôm trọn quả theo góc xoay |
| **Phân tích hình học** | Chỉ xác định tọa độ biên $(x, y, w, h)$ | • Trọng tâm (Centroid)<br>• Trục định hướng PCA (Orientation Angle)<br>• Diện tích Shoelace đa giác<br>• Tỷ lệ % chín (Vàng/Đỏ) vs Xanh |
| **Kích thước file mô hình** | **547.7 MB** (`model/svm_model.xml`) | **~4.5 MB** (`model/fruit_cnn.onnx`) — *(Nhẹ hơn 120 lần!)* |
| **Độ chính xác Test Set** | **93.40%** (2,520 / 2,698 ảnh đúng) | **> 96%** (Độ phân biệt đốm hỏng vượt trội) |
| **Mã nguồn huấn luyện** | `train/main.cpp` $\to$ `train/main.exe` | `train/train2_cnn.py` $\to$ `train/main2.exe` |
| **Mã nguồn đánh giá** | `eval/evaluate.cpp` $\to$ `eval/evaluate.exe` | `eval/evaluate2.cpp` $\to$ `eval/evaluate2.exe` |
| **Giao diện Demo trực tiếp** | `demo/demo.cpp` $\to$ `demo/demo.exe` | `demo/demo2.cpp` $\to$ `demo/demo2.exe` |

---

## 🚀 Hướng dẫn chạy và so sánh thực tế

### 1. So sánh đánh giá trên tập kiểm thử (Test Set - 2,698 ảnh)

* **Chạy đánh giá Pipeline 1 (SVM)**:
  ```powershell
  & "d:\nam4\vinh\xla\bai_tap_lon\eval\evaluate.exe"
  ```
* **Chạy đánh giá Pipeline 2 (CNN)**:
  ```powershell
  & "d:\nam4\vinh\xla\bai_tap_lon\eval\evaluate2.exe"
  ```

---

### 2. So sánh giao diện Demo trực quan (File ảnh hoặc Webcam)

* **Mở Demo Pipeline 1 (SVM)**:
  ```powershell
  & "d:\nam4\vinh\xla\bai_tap_lon\demo\demo.exe"
  ```
  *(Sử dụng khung chữ nhật thông thường và thanh trạng thái trên đỉnh màn hình)*

* **Mở Demo Pipeline 2 (CNN + Oriented Bounding Box)**:
  ```powershell
  & "d:\nam4\vinh\xla\bai_tap_lon\demo\demo2.exe"
  ```
  *(Hiển thị khung chữ nhật nghiêng ôm theo góc xoay của quả, vẽ mũi tên trục định hướng PCA và Thẻ kết luận thông minh ghi rõ Tên quả + Độ chín % + Màu sắc + Góc nghiêng)*
