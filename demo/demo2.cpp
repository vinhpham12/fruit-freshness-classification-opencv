#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <windows.h>
#include <commdlg.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// 1. Danh sách tên nhãn mô hình CNN
const vector<string> classNames = {
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"};

const vector<string> friendlyNames = {
    "Tao tuoi (Fresh Apple)",
    "Chuoi tuoi (Fresh Banana)",
    "Cam tuoi (Fresh Orange)",
    "Tao hong (Rotten Apple)",
    "Chuoi hong (Rotten Banana)",
    "Cam hong (Rotten Orange)"};

// Hàm làm sạch đường dẫn file khi kéo thả vào Terminal
string cleanFilePath(string path) {
    while (!path.empty() && (path.front() == ' ' || path.front() == '\t')) {
        path.erase(path.begin());
    }
    while (!path.empty() && (path.back() == ' ' || path.back() == '\t' || path.back() == '\r' || path.back() == '\n')) {
        path.pop_back();
    }
    if (path.size() >= 2 && ((path.front() == '"' && path.back() == '"') || 
                             (path.front() == '\'' && path.back() == '\''))) {
        path = path.substr(1, path.size() - 2);
    }
    return path;
}

// Hàm mở hộp thoại chọn file của Windows bằng chuột
string openFileDialog() {
    char filename[MAX_PATH] = {0};
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrTitle = "Chon anh hoa qua can kiem tra (Pipeline 2: CNN)";

    if (GetOpenFileNameA(&ofn)) {
        return string(filename);
    }
    return "";
}

// 2. Nhận diện tên màu chủ đạo theo HSV
string detectColorName(double h, double s, double v, Scalar& tagColor) {
    if (v < 45) {
        tagColor = Scalar(80, 80, 80);
        return "DEN (Black)";
    }
    if (s < 35 && v > 180) {
        tagColor = Scalar(255, 255, 255);
        return "TRANG (White)";
    }
    if (s < 35) {
        tagColor = Scalar(160, 160, 160);
        return "XAM (Gray)";
    }

    if ((h >= 0 && h <= 10) || (h >= 165 && h <= 179)) {
        tagColor = Scalar(0, 0, 255);      // Đỏ (BGR)
        return "DO (Red)";
    } else if (h > 10 && h <= 25) {
        tagColor = Scalar(0, 140, 255);    // Cam
        return "CAM (Orange)";
    } else if (h > 25 && h <= 35) {
        tagColor = Scalar(0, 255, 255);    // Vàng
        return "VANG (Yellow)";
    } else if (h > 35 && h <= 85) {
        tagColor = Scalar(0, 255, 0);      // Xanh lá
        return "XANH LA (Green)";
    } else if (h > 85 && h <= 100) {
        tagColor = Scalar(255, 255, 0);    // Xanh ngọc
        return "XANH NGOC (Cyan)";
    } else if (h > 100 && h <= 135) {
        tagColor = Scalar(255, 80, 0);     // Xanh dương
        return "XANH DUONG (Blue)";
    } else if (h > 135 && h <= 155) {
        tagColor = Scalar(255, 0, 200);    // Tím
        return "TIM (Purple)";
    } else if (h > 155 && h < 165) {
        tagColor = Scalar(180, 105, 255);  // Hồng
        return "HONG (Pink)";
    }
    tagColor = Scalar(0, 255, 255);
    return "KHONG RO";
}

// 3. Tính diện tích đa giác contour bằng công thức Shoelace
double calculateContourArea(const vector<Point>& pts) {
    int n = static_cast<int>(pts.size());
    if (n < 3) return 0.0;
    double area = 0.0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += static_cast<double>(pts[i].x) * pts[j].y - static_cast<double>(pts[j].x) * pts[i].y;
    }
    return std::abs(area) * 0.5;
}

// 4. Tính trọng tâm (Centroid)
Point2f calculateCentroid(const vector<Point>& pts) {
    if (pts.empty()) return Point2f(0.0f, 0.0f);
    double sumX = 0.0, sumY = 0.0;
    for (const auto& p : pts) {
        sumX += p.x;
        sumY += p.y;
    }
    return Point2f(static_cast<float>(sumX / pts.size()), static_cast<float>(sumY / pts.size()));
}

// 5. Tính góc nghiêng trục chính của quả bằng PCA
double calculateOrientation(const vector<Point>& pts, Point2f center) {
    if (pts.size() < 3) return 0.0;
    double cov_xx = 0.0, cov_yy = 0.0, cov_xy = 0.0;
    for (const auto& p : pts) {
        double dx = p.x - center.x;
        double dy = p.y - center.y;
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_xy += dx * dy;
    }
    double angleRad = 0.5 * atan2(2.0 * cov_xy, cov_xx - cov_yy);
    double angleDeg = angleRad * (180.0 / CV_PI);
    return angleDeg;
}

// 6. Tính toán khung chữ nhật nghiêng ôm trọn toàn bộ quả (Oriented BBox)
void getOrientedBoundingBox(const vector<Point>& pts, Point2f center, double angleDeg, Point2f rectPts[4]) {
    double angleRad = angleDeg * (CV_PI / 180.0);
    double cosA = cos(angleRad);
    double sinA = sin(angleRad);

    double minU = 1e9, maxU = -1e9;
    double minV = 1e9, maxV = -1e9;

    for (const auto& p : pts) {
        double dx = p.x - center.x;
        double dy = p.y - center.y;
        double u = dx * cosA + dy * sinA;
        double v = -dx * sinA + dy * cosA;
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
    }

    rectPts[0] = Point2f(static_cast<float>(center.x + minU * cosA - minV * sinA),
                         static_cast<float>(center.y + minU * sinA + minV * cosA));
    rectPts[1] = Point2f(static_cast<float>(center.x + maxU * cosA - minV * sinA),
                         static_cast<float>(center.y + maxU * sinA + minV * cosA));
    rectPts[2] = Point2f(static_cast<float>(center.x + maxU * cosA - maxV * sinA),
                         static_cast<float>(center.y + maxU * sinA + maxV * cosA));
    rectPts[3] = Point2f(static_cast<float>(center.x + minU * cosA - maxV * sinA),
                         static_cast<float>(center.y + minU * sinA + maxV * cosA));
}

// 7. Vẽ trục nghiêng và tâm của quả
void drawOrientationAxis(Mat& img, Point2f center, double angleDeg, int length = 55) {
    double angleRad = angleDeg * (CV_PI / 180.0);
    Point2f p1(static_cast<float>(center.x + length * cos(angleRad)),
               static_cast<float>(center.y + length * sin(angleRad)));
    Point2f p2(static_cast<float>(center.x - length * cos(angleRad)),
               static_cast<float>(center.y - length * sin(angleRad)));

    line(img, p2, p1, Scalar(0, 0, 255), 2, LINE_AA);
    arrowedLine(img, center, p1, Scalar(0, 255, 255), 2, LINE_AA, 0, 0.2);
    circle(img, center, 5, Scalar(255, 0, 0), -1);
}

// 8. Đánh giá tỷ lệ màu sắc bên trong quả (Vàng/Đỏ chín vs Xanh)
void analyzeFruitRipeness(const Mat& hsv, const Mat& fruitMask, 
                          double& ripePercent, double& greenPercent, 
                          string& mainColorName, Scalar& tagColor) {
    Mat greenMask;
    inRange(hsv, Scalar(35, 40, 40), Scalar(85, 255, 255), greenMask);
    if (!fruitMask.empty()) greenMask &= fruitMask;

    Mat mYellow, mRed1, mRed2;
    inRange(hsv, Scalar(11, 40, 40), Scalar(34, 255, 255), mYellow);
    inRange(hsv, Scalar(0, 40, 40), Scalar(10, 255, 255), mRed1);
    inRange(hsv, Scalar(165, 40, 40), Scalar(179, 255, 255), mRed2);
    Mat ripeMask = (mYellow | mRed1 | mRed2);
    if (!fruitMask.empty()) ripeMask &= fruitMask;

    int greenCount = countNonZero(greenMask);
    int ripeCount  = countNonZero(ripeMask);
    int totalValid = greenCount + ripeCount;

    Scalar meanHSV = (!fruitMask.empty()) ? mean(hsv, fruitMask) : mean(hsv);
    mainColorName = detectColorName(meanHSV[0], meanHSV[1], meanHSV[2], tagColor);

    if (totalValid < 50) {
        ripePercent = 0.0;
        greenPercent = 0.0;
        return;
    }

    ripePercent = (static_cast<double>(ripeCount) / totalValid) * 100.0;
    greenPercent = 100.0 - ripePercent;
}

// 9. Bảng thẻ thông tin kết hợp giữa AI CNN và Xử lý ảnh hình học
void drawFruitConclusionCard(Mat& img, Point origin, const string& cnnLabel, int classId,
                             float confidence, double ripePercent, double greenPercent,
                             const string& mainColor, double angleDeg) {
    int cardW = 340;
    int cardH = 125;

    int x = max(10, min(origin.x, img.cols - cardW - 10));
    int y = max(10, min(origin.y, img.rows - cardH - 10));

    bool isFresh = (classId >= 0 && classId <= 2);
    Scalar statusColor = (classId < 0) ? Scalar(180, 180, 180) : (isFresh ? Scalar(0, 255, 0) : Scalar(0, 0, 255));
    string statusText = (classId < 0) ? "CHUA XAC DINH" : (isFresh ? "STATUS: FRESH (TUOI)" : "STATUS: ROTTEN (HONG)");

    // Vẽ nền thẻ đen mờ viền màu trạng thái
    rectangle(img, Rect(x, y, cardW, cardH), Scalar(20, 20, 20), -1);
    rectangle(img, Rect(x, y, cardW, cardH), statusColor, 2);

    // Dòng 1: Tên quả từ CNN + Confidence
    char textCNN[100];
    snprintf(textCNN, sizeof(textCNN), "%s (%.0f%%)", cnnLabel.c_str(), confidence * 100.0f);
    putText(img, textCNN, Point(x + 12, y + 28), FONT_HERSHEY_SIMPLEX, 0.62, Scalar(255, 255, 255), 2, LINE_AA);

    // Dòng 2: Trạng thái Tươi / Hỏng
    putText(img, statusText, Point(x + 12, y + 58), FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2, LINE_AA);

    // Dòng 3: Tỷ lệ chín Vàng/Đỏ vs Xanh
    char textRatio[100];
    snprintf(textRatio, sizeof(textRatio), "Do chin: %.0f%% (Vang/Do) | Xanh: %.0f%%", ripePercent, greenPercent);
    putText(img, textRatio, Point(x + 12, y + 85), FONT_HERSHEY_SIMPLEX, 0.48, Scalar(220, 220, 220), 1, LINE_AA);

    // Dòng 4: Màu chủ đạo & Góc nghiêng PCA
    char textDetail[100];
    snprintf(textDetail, sizeof(textDetail), "Mau: %s | Goc: %.1f deg", mainColor.c_str(), angleDeg);
    putText(img, textDetail, Point(x + 12, y + 112), FONT_HERSHEY_SIMPLEX, 0.48, Scalar(0, 255, 255), 1, LINE_AA);
}

// 10. Hàm dự đoán quả bằng mô hình CNN qua OpenCV DNN
void predictFruitCNN(dnn::Net& net, const Mat& crop, int& outClassId, float& outConfidence, string& outLabel) {
    if (crop.empty() || crop.cols < 10 || crop.rows < 10) {
        outClassId = -1;
        outConfidence = 0.0f;
        outLabel = "Khong xac dinh";
        return;
    }

    Mat blob;
    dnn::blobFromImage(crop, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat prob = net.forward();

    Point classIdPoint;
    double maxConfidence;
    minMaxLoc(prob, 0, &maxConfidence, 0, &classIdPoint);

    outClassId = classIdPoint.x;
    outConfidence = static_cast<float>(maxConfidence);
    if (outClassId >= 0 && outClassId < (int)friendlyNames.size()) {
        outLabel = friendlyNames[outClassId];
    } else {
        outLabel = "Khong xac dinh";
    }
}

int main(int argc, char** argv) {
    cout << "========================================================\n";
    cout << "   DEMO PIPELINE 2: HE THONG NHAN DIEN HOA QUA VOI CNN  \n";
    cout << "   (KET HOP XU LY ANH HINH HOC + DEEP LEARNING CNN)     \n";
    cout << "========================================================\n";

    string onnxModelPath = "D:/nam4/vinh/xla/bai_tap_lon/model/fruit_cnn.onnx";
    cout << "Dang nap mo hinh CNN tu: " << onnxModelPath << " ...\n";

    if (!fs::exists(onnxModelPath)) {
        cerr << "[LOI] Chua co file mo hinh: " << onnxModelPath << "\n";
        cerr << "Vui long chay chuong trinh train2 de huan luyen va tao file ONNX!\n";
        return -1;
    }

    dnn::Net net = dnn::readNetFromONNX(onnxModelPath);
    if (net.empty()) {
        cerr << "[LOI] Khong the doc file mo hinh ONNX vao OpenCV DNN!\n";
        return -1;
    }
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    cout << "==> Nap mo hinh CNN vao C++ thanh cong!\n\n";

    cout << "HUONG DAN PHIM BAM TREN CUA SO DEMO:\n";
    cout << "  - [Phim O] : Mo hop thoai chon file anh bang chuot (File Dialog)\n";
    cout << "  - [Phim C] : Bat / Tat che do Camera Webcam truc tiep\n";
    cout << "  - [ESC / Q]: Thoat chuong trinh\n";
    cout << "========================================================\n\n";

    VideoCapture cap;
    bool useCamera = false;
    Mat staticImg;
    string currentImagePath = "";

    // Đọc ảnh mẫu nếu có
    if (argc > 1) {
        string argPath = cleanFilePath(argv[1]);
        staticImg = imread(argPath);
        if (!staticImg.empty()) currentImagePath = argPath;
    }
    if (staticImg.empty()) {
        currentImagePath = "D:/nam4/vinh/xla/dataset/archive/MY_data/predict/img_171.jpeg";
        staticImg = imread(currentImagePath);
        if (staticImg.empty()) {
            staticImg = Mat(500, 750, CV_8UC3, Scalar(45, 45, 45));
            putText(staticImg, "PIPELINE 2: CNN DEMO", Point(200, 220), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
            putText(staticImg, "Nhan [O] de chon anh | Nhan [C] de mo Camera", Point(120, 270), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        }
    }

    string windowName = "Demo Pipeline 2: Classical CV + Deep Learning CNN";
    namedWindow(windowName, WINDOW_NORMAL);
    setWindowProperty(windowName, WND_PROP_TOPMOST, 1);

    Mat rawFrame, frame, hsv, blurred;
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat kernelOpen  = getStructuringElement(MORPH_RECT, Size(5, 5));

    while (true) {
        if (useCamera) {
            cap >> rawFrame;
            if (rawFrame.empty()) {
                cout << "[CANH BAO] Khong doc duoc frame tu Camera!\n";
                useCamera = false;
                cap.release();
                continue;
            }
            flip(rawFrame, rawFrame, 1); // Lật gương cho webcam
        } else {
            rawFrame = staticImg.clone();
        }

        // Tự động scale kích thước cho dễ quan sát
        int origW = rawFrame.cols;
        int origH = rawFrame.rows;
        if (origW < 550 || origH < 550) {
            double scale = max(550.0 / origW, 550.0 / origH);
            resize(rawFrame, frame, Size(), scale, scale, INTER_LINEAR);
        } else if (origW > 1000 || origH > 750) {
            double scale = min(1000.0 / origW, 750.0 / origH);
            resize(rawFrame, frame, Size(), scale, scale, INTER_AREA);
        } else {
            frame = rawFrame.clone();
        }

        Mat result = frame.clone();
        int imgW = frame.cols;
        int imgH = frame.rows;

        GaussianBlur(frame, blurred, Size(5, 5), 0);
        cvtColor(blurred, hsv, COLOR_BGR2HSV);

        // PHÂN VÙNG QUẢ BẰNG XỬ LÝ ẢNH TRUYỀN THỐNG (Contour + Morphology)
        Mat maskGreen, maskYellow, maskRed1, maskRed2;
        inRange(hsv, Scalar(35, 30, 25), Scalar(88, 255, 255), maskGreen);
        inRange(hsv, Scalar(10, 30, 25), Scalar(34, 255, 255), maskYellow);
        inRange(hsv, Scalar(0, 30, 25), Scalar(9, 255, 255), maskRed1);
        inRange(hsv, Scalar(165, 30, 25), Scalar(179, 255, 255), maskRed2);

        Mat combinedFruit = (maskGreen | maskYellow | maskRed1 | maskRed2);
        Mat fruitMask;
        morphologyEx(combinedFruit, fruitMask, MORPH_CLOSE, kernelClose);
        morphologyEx(fruitMask, fruitMask, MORPH_OPEN, kernelOpen);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(fruitMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int detectedFruitCount = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = calculateContourArea(contours[i]);
            if (area < 1500.0 || area > (imgW * imgH * 0.96)) continue;
            detectedFruitCount++;

            Point2f center = calculateCentroid(contours[i]);
            double angle = calculateOrientation(contours[i], center);

            // 1. Tính toán & Vẽ khung chữ nhật xoay (Oriented Bounding Box)
            Point2f rectPts[4];
            getOrientedBoundingBox(contours[i], center, angle, rectPts);

            // 2. Cắt vùng ảnh quả đưa vào mạng CNN
            int minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
            for (const auto& pt : contours[i]) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
            }
            Rect bBox(minX, minY, max(1, maxX - minX), max(1, maxY - minY));
            bBox = bBox & Rect(0, 0, imgW, imgH);
            Mat crop = frame(bBox);

            int predClassId = -1;
            float predConf = 0.0f;
            string predLabel = "";
            predictFruitCNN(net, crop, predClassId, predConf, predLabel);

            // Màu khung: Xanh lá nếu tươi, Đỏ nếu hỏng
            bool isFresh = (predClassId >= 0 && predClassId <= 2);
            Scalar boxColor = isFresh ? Scalar(0, 255, 0) : Scalar(0, 0, 255);

            for (int j = 0; j < 4; j++) {
                line(result, rectPts[j], rectPts[(j + 1) % 4], boxColor, 2, LINE_AA);
            }

            // 3. Vẽ trục định hướng PCA của quả
            drawOrientationAxis(result, center, angle, 55);

            // 4. Phân tích tỷ lệ chín (Vàng/Đỏ vs Xanh)
            Mat singleMask = Mat::zeros(frame.size(), CV_8UC1);
            drawContours(singleMask, contours, static_cast<int>(i), Scalar(255), -1);
            double ripePercent = 0.0, greenPercent = 0.0;
            string mainColor;
            Scalar tagColor;
            analyzeFruitRipeness(hsv, singleMask, ripePercent, greenPercent, mainColor, tagColor);

            // 5. Vẽ Thẻ kết luận thông minh
            Point cardPos(static_cast<int>(center.x - 170), static_cast<int>(center.y - 70));
            drawFruitConclusionCard(result, cardPos, predLabel, predClassId, predConf, ripePercent, greenPercent, mainColor, angle);
        }

        // Trường hợp nền phức tạp không tách được contour riêng lẻ, dự đoán toàn bộ khung hình
        if (detectedFruitCount == 0) {
            int predClassId = -1;
            float predConf = 0.0f;
            string predLabel = "";
            predictFruitCNN(net, frame, predClassId, predConf, predLabel);

            double ripePercent = 0.0, greenPercent = 0.0;
            string mainColor;
            Scalar tagColor;
            analyzeFruitRipeness(hsv, Mat(), ripePercent, greenPercent, mainColor, tagColor);

            Point cardPos(20, 20);
            drawFruitConclusionCard(result, cardPos, predLabel, predClassId, predConf, ripePercent, greenPercent, mainColor, 0.0);
        }

        // Banner dưới đáy
        rectangle(result, Point(0, result.rows - 35), Point(result.cols, result.rows), Scalar(20, 20, 20), FILLED);
        putText(result, "[PIPELINE 2: CNN] [O] Chon anh  |  [C] Camera  |  [ESC/Q] Thoat",
                Point(15, result.rows - 10), FONT_HERSHEY_SIMPLEX, 0.52, Scalar(0, 220, 255), 1, LINE_AA);

        imshow(windowName, result);

        int waitTime = useCamera ? 30 : 100;
        char key = static_cast<char>(waitKey(waitTime));
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        } else if (key == 'o' || key == 'O') {
            if (useCamera) {
                useCamera = false;
                cap.release();
            }
            string selected = openFileDialog();
            if (!selected.empty()) {
                Mat newImg = imread(selected);
                if (!newImg.empty()) {
                    staticImg = newImg;
                    currentImagePath = selected;
                    cout << "==> Da mo anh: " << currentImagePath << "\n";
                }
            }
        } else if (key == 'c' || key == 'C') {
            useCamera = !useCamera;
            if (useCamera) {
                cap.open(0);
                if (!cap.isOpened()) {
                    cerr << "[LOI] Khong the mo Camera Webcam!\n";
                    useCamera = false;
                } else {
                    cout << "==> Da mo Camera Webcam cho Pipeline 2!\n";
                }
            } else {
                cap.release();
                cout << "==> Da tat Camera.\n";
            }
        }
    }

    if (cap.isOpened()) cap.release();
    destroyAllWindows();
    return 0;
}
