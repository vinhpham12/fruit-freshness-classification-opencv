# HƯỚNG DẪN TỪNG BƯỚC HOÀN THÀNH BÀI TẬP LỚN: PHÂN LOẠI QUẢ TƯƠI VÀ HỎNG (C++ & OPENCV)

> **Mục tiêu bài toán**: Huấn luyện mô hình Học máy (Machine Learning) bằng C++ và OpenCV để nhận diện 6 lớp quả:
> `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.

---

## MỤC LỤC
1. [Tổng quan kiến trúc hệ thống](#1-tổng-quan-kiến-trúc-hệ-thống)
2. [Giai đoạn 1: Đọc và duyệt dữ liệu (Data Loading)](#giai-đoạn-1-đọc-và-duyệt-dữ-liệu-data-loading)
3. [Giai đoạn 2: Tiền xử lý & Trích xuất đặc trưng (Feature Extraction)](#giai-đoạn-2-tiền-xử-lý--trích-xuất-đặc-trưng-feature-extraction)
4. [Giai đoạn 3: Chuẩn bị ma trận huấn luyện cho OpenCV ML](#giai-đoạn-3-chuẩn-bị-ma-trận-huấn-luyện-cho-opencv-ml)
5. [Giai đoạn 4: Huấn luyện và lưu mô hình (Model Training & Saving)](#giai-đoạn-4-huấn-luyện-và-lưu-mô-hình-model-training--saving)
6. [Giai đoạn 5: Đánh giá mô hình trên tập Test (Evaluation)](#giai-đoạn-5-đánh-giá-mô-hình-trên-tập-test-evaluation)
7. [Giai đoạn 6: Xây dựng chương trình Demo / Dự đoán ảnh thực tế (Inference)](#giai-đoạn-6-xây-dựng-chương-trình-demo--dự-đoán-ảnh-thực-tế-inference)
8. [Hướng dẫn biên dịch và chạy (Build & Run)](#hướng-dẫn-biên-dịch-và-chạy-build--run)

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG

Luồng hoạt động chuẩn của bài toán Thị giác máy tính truyền thống (Traditional CV):

```
[Ảnh đầu vào (dataset/train)]
          │
          ▼
[1. Tiền xử lý (Resize, Lọc nhiễu, Không gian màu HSV/RGB)]
          │
          ▼
[2. Trích xuất đặc trưng (Color Histogram + HOG)]
          │
          ▼
[3. Gom thành Feature Vector 1D (CV_32F)]
          │
          ▼
[4. Nạp vào Thuật toán ML (SVM / Random Forest)]
          │
          ▼
[5. Lưu file Model (.xml / .yml)]
          │
          ▼
[6. Đánh giá tập Test & Dự đoán ảnh mới]
```

---

## GIAI ĐOẠN 1: ĐỌC VÀ DUYỆT DỮ LIỆU (DATA LOADING)

### Mục tiêu:
Duyệt qua toàn bộ thư mục `dataset/train` để nạp từng ảnh và gán nhãn tương ứng.

### Tư duy thực hiện:
1. **Thiết lập bảng ánh xạ nhãn (`std::map`)**:
   - Gán mỗi tên thư mục sang một số nguyên cố định:
     - `freshapples` $\rightarrow 0$
     - `freshbanana` $\rightarrow 1$
     - `freshoranges` $\rightarrow 2$
     - `rottenapples` $\rightarrow 3$
     - `rottenbanana` $\rightarrow 4$
     - `rottenoranges` $\rightarrow 5$
2. **Duyệt 2 tầng bằng `std::filesystem`**:
   - **Tầng 1**: Duyệt thư mục cha `dataset/train` để tìm 6 thư mục con.
   - **Tầng 2**: Duyệt từng file ảnh bên trong thư mục con đó.
3. **Đọc ảnh an toàn**:
   - Dùng `cv::imread(path)`.
   - Kiểm tra `img.empty()` để bỏ qua ảnh hỏng hoặc file rác.
4. **Kiểm tra cân bằng dữ liệu**:
   - In ra console số lượng ảnh đọc được của từng nhãn để đảm bảo không folder nào bị bỏ sót.

---

## GIAI ĐOẠN 2: TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)

> **Lưu ý quan trọng**: Không nên dùng trực tiếp điểm ảnh thô (raw pixels) vì số chiều quá lớn, dễ gây quá tải bộ nhớ và mô hình rất dễ bị phụ thuộc vào góc chụp/ánh sáng.

Đối với bài toán **Quả tươi vs Quả hỏng**, có 2 loại đặc trưng cốt lõi:

### 1. Đặc trưng Màu sắc (Color Histogram - HSV)
* **Ý nghĩa**: Quả hỏng sẽ đổi màu (thâm đen, mốc xanh/trắng, mất sắc thái tươi sáng). Không gian màu **HSV** phân tách rõ thông tin màu (Hue) và độ bão hòa (Saturation) khỏi cường độ sáng (Value).
* **Cách làm**:
  - Đổi màu ảnh từ BGR sang HSV: `cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV)`.
  - Tính biểu đồ tần số màu bằng hàm `cv::calcHist` (chia thành các bin, ví dụ 8 bin cho H, 8 bin cho S).
  - Chuẩn hóa histogram về `[0, 1]` bằng `cv::normalize` với chuẩn `cv::NORM_L1` hoặc `cv::NORM_L2`.

### 2. Đặc trưng Kết cấu / Cạnh (HOG - Histogram of Oriented Gradients)
* **Ý nghĩa**: Quả hỏng thường bị nhăn da, thối lõm, nứt nẻ hoặc biến dạng kết cấu bề mặt. HOG là công cụ bắt kết cấu góc nghiêng và viền cạnh cực kỳ chính xác.
* **Cách làm**:
  - Đưa ảnh về kích thước cố định (ví dụ: `64x64` hoặc `128x128`).
  - Chuyển sang ảnh xám (`cv::COLOR_BGR2GRAY`).
  - Khởi tạo `cv::HOGDescriptor hog(...)`.
  - Gọi `hog.compute(grayImg, descriptors)` để lấy vector đặc trưng.

### 3. Ghép đặc trưng (Feature Concatenation)
Nối vector Color Histogram và vector HOG lại thành **một vector đặc trưng 1 chiều** duy nhất cho mỗi bức ảnh.

---

## GIAI ĐOẠN 3: CHUẨN BỊ MA TRẬN HUẤN LUYỆN CHO OPENCV ML

OpenCV yêu cầu dữ liệu đầu vào cho các thuật toán học máy tuân theo định dạng chuẩn:

1. **Ma trận dữ liệu (`trainData`)**:
   - Kiểu dữ liệu: **`CV_32FC1`** (Float 32-bit).
   - Kích thước: `N x D`
     - $N$: Tổng số ảnh trong tập train.
     - $D$: Số lượng đặc trưng trong 1 vector (chiều dài vector trích xuất ở Giai đoạn 2).
2. **Ma trận nhãn (`trainLabels`)**:
   - Kiểu dữ liệu: **`CV_32SC1`** (Integer 32-bit).
   - Kích thước: `N x 1` (1 cột chứa mã nhãn từ `0` đến `5`).

> **Kỹ thuật gom dữ liệu**:
> Mỗi khi trích xuất xong 1 vector đặc trưng, biến đổi nó thành ma trận 1 hàng (`1 x D`), sau đó dùng `trainData.push_back(featureRow)` để ghép dần vào ma trận lớn.

---

## GIAI ĐOẠN 4: HUẤN LUYỆN VÀ LƯU MÔ HÌNH (MODEL TRAINING & SAVING)

OpenCV cung cấp sẵn các thuật toán Machine Learning cổ điển trong module `cv::ml`:

### Lựa chọn thuật toán:
1. **SVM (`cv::ml::SVM`)** *(Khuyên dùng)*:
   - Loại SVM: `cv::ml::SVM::C_SVC` (phân loại đa lớp tự động theo cơ chế One-vs-One).
   - Loại Kernel:
     - `cv::ml::SVM::RBF`: Phù hợp khi vector đặc trưng có mối quan hệ phi tuyến tính.
     - `cv::ml::SVM::LINEAR`: Huấn luyện nhanh hơn, ít bị quá khớp (overfit) khi số lượng đặc trưng lớn.
   - Siêu tham số: $C$ (mức độ phạt lỗi), $\gamma$ (gamma cho RBF kernel). Bạn có thể dùng `cv::ml::SVM::trainAuto` để OpenCV tự động tìm bộ tham số tối ưu thông qua K-Fold Cross Validation.
2. **Random Forest (`cv::ml::RTrees`)** *(Lựa chọn thay thế)*:
   - Huấn luyện nhiều cây quyết định, chống nhiễu cực tốt và không yêu cầu dữ liệu phải chuẩn hóa quá ngặt nghèo.

### Lưu mô hình:
Sau khi huấn luyện xong, lưu mô hình ra file XML/YML để sử dụng lại mà không cần train lại:
```cpp
model->save("fruit_classifier_model.xml");
```

---

## GIAI ĐOẠN 5: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (EVALUATION)

Để kiểm chứng xem mô hình có học vẹt (overfitting) hay không:

1. **Duyệt qua tập `dataset/test`**:
   - Áp dụng **chính xác các bước tiền xử lý & trích xuất đặc trưng** giống hệt tập train.
2. **Dự đoán nhãn**:
   - Dùng `float predictedLabel = model->predict(testFeatureVector);`.
3. **Thống kê độ chính xác (Accuracy)**:
   $$\text{Accuracy} = \frac{\text{Số lượng mẫu dự đoán đúng}}{\text{Tổng số mẫu test}} \times 100\%$$
4. **Vẽ Ma trận nhầm lẫn (Confusion Matrix $6 \times 6$)**:
   - Hàng biểu thị nhãn thực tế (Ground Truth), cột biểu thị nhãn dự đoán.
   - Ma trận này sẽ chỉ ra loại quả nào thường bị máy đoán nhầm sang loại quả nào (ví dụ: `freshapples` hay bị nhầm sang `freshoranges` do màu sắc tương đồng).

---

## GIAI ĐOẠN 6: XÂY DỰNG CHƯƠNG TRÌNH DEMO / DỰ ĐOÁN ẢNH THỰC TẾ

Tạo một hàm hoặc một file chạy độc lập (`predict.cpp`):
1. **Load mô hình đã lưu**:
   ```cpp
   cv::Ptr<cv::ml::SVM> model = cv::ml::StatModel::load<cv::ml::SVM>("fruit_classifier_model.xml");
   ```
2. **Đọc ảnh bất kỳ** do người dùng truyền vào (hoặc từ camera/webcam).
3. **Xử lý ảnh & dự đoán**:
   - Trích xuất đặc trưng $\rightarrow$ gọi `predict`.
   - Lấy mã số nhãn dự đoán tra ngược vào mảng `classNames` để lấy chuỗi chữ (ví dụ: `"freshapples"`).
4. **Vẽ kết quả lên ảnh**:
   - Dùng `cv::putText` để in tên quả và trạng thái tươi/hỏng lên góc ảnh (màu xanh nếu tươi, màu đỏ nếu hỏng).
   - Dùng `cv::imshow` để hiển thị trực quan cửa sổ cho người xem.

---

## HƯỚNG DẪN BIÊN DỊCH VÀ CHẠY (BUILD & RUN)

Vì dùng `std::filesystem`, bạn bắt buộc phải bật chuẩn **C++17** khi biên dịch.

### Nếu dùng g++ (MinGW trên Windows):
```bash
g++ -std=c++17 -O3 main.cpp -o train_app `pkg-config --cflags --libs opencv4`
```
*(Hoặc link thủ công các file thư viện `.lib`/`.dll` của OpenCV tương ứng nếu không dùng pkg-config)*.

### Cấu trúc project khuyến nghị:
```text
bai_tap_lon/
├── dataset/
│   ├── train/ (6 thư mục)
│   └── test/  (6 thư mục)
├── pre_set_up/
│   └── rename.py
├── train/
│   ├── main.cpp                 <-- Chương trình chính nạp dữ liệu & train
│   └── fruit_classifier.xml     <-- File model sau khi train
├── HUONG_DAN_DU_AN.md           <-- File hướng dẫn này
```

---
*Chúc bạn hoàn thành xuất sắc Bài tập lớn Xử lý ảnh!*
