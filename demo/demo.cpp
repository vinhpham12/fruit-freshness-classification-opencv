#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <windows.h>
#include <commdlg.h>

using namespace cv;
using namespace std;

// 1. Danh sách tên nhãn
std::vector<std::string> classNames = {
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"};

std::vector<std::string> friendlyNames = {
    "Tao tuoi (Fresh Apple)",
    "Chuoi tuoi (Fresh Banana)",
    "Cam tuoi (Fresh Orange)",
    "Tao hong (Rotten Apple)",
    "Chuoi hong (Rotten Banana)",
    "Cam hong (Rotten Orange)"};



// Hàm tự động phát hiện quả và cắt sát viền (Auto-Crop Fruit ROI)
Rect detectFruitROI(const Mat& img) {
    int minX = img.cols, maxX = 0;
    int minY = img.rows, maxY = 0;
    int count = 0;

    for (int y = 0; y < img.rows; y++) {
        const Vec3b* row = img.ptr<Vec3b>(y);
        for (int x = 0; x < img.cols; x++) {
            // Điểm ảnh không phải màu trắng tinh (nền trắng thường có B,G,R > 235)
            if (row[x][0] < 235 || row[x][1] < 235 || row[x][2] < 235) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                count++;
            }
        }
    }

    // Nếu không tìm thấy hoặc đối tượng quá nhỏ, trả về toàn bộ ảnh
    if (count < 100 || minX >= maxX || minY >= maxY) {
        return Rect(0, 0, img.cols, img.rows);
    }

    // Đệm thêm 5% viền an toàn để không cắt lẹm vào vỏ quả
    int padX = (maxX - minX) * 0.05;
    int padY = (maxY - minY) * 0.05;
    int x = max(0, minX - padX);
    int y = max(0, minY - padY);
    int w = min(img.cols - x, (maxX - minX) + 2 * padX);
    int h = min(img.rows - y, (maxY - minY) + 2 * padY);

    return Rect(x, y, w, h);
}

Mat resizeKeepAspectRatio(const Mat& input, int targetSize =128){
    int w= input.cols;
    int h= input.rows;
    double scale =(double)targetSize/max(w,h);
    int newW= (int)(w*scale);
    int newH= (int)(h*scale);

    Mat resized;
    resize(input,resized, Size(newW,newH));

    Mat output(targetSize,targetSize, input.type(),Scalar(255,255,255));
    int dx = (targetSize - newW)/2;
    int dy = (targetSize - newH)/2;
    resized.copyTo(output(Rect(dx,dy,newW,newH)));

    return output;
}

vector<float> extractFeatures(const Mat &img)
{
    // 1. Tự động tìm quả và cắt sát viền (Auto-Crop ROI)
    Rect roi = detectFruitROI(img);
    Mat croppedImg = img(roi);

    // 2. Resize giữ nguyên tỷ lệ trên phần quả đã cắt
    Mat resizedImg = resizeKeepAspectRatio(croppedImg, 128);

    Mat hsv;
    cvtColor(resizedImg, hsv, COLOR_BGR2HSV);

    int hBins = 8;
    int sBins = 8;
    int vBins = 8;
    int histSize[] = {hBins, sBins , vBins};

    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges, vRanges};

    int channels[] = {0, 1 ,2};

    Mat hist;
    calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    
    vector<float> colorFeat;
    colorFeat.reserve(512);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < 8; k++)
                colorFeat.push_back(hist.at<float>(i, j, k));
    Mat colorMat(1, colorFeat.size(), CV_32F, colorFeat.data());
    normalize(colorMat, colorMat, 1.0, 0.0, NORM_L2); // Chuẩn hóa Màu sắc
    // 3. Trích xuất HOG trên ảnh xám
    Mat gray;
    cvtColor(resizedImg, gray, COLOR_BGR2GRAY);
    HOGDescriptor hog(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    vector<float> hogFeat;
    hog.compute(gray, hogFeat);
    Mat hogMat(1, hogFeat.size(), CV_32F, hogFeat.data());
    normalize(hogMat, hogMat, 1.0, 0.0, NORM_L2); // Chuẩn hóa HOG
    // 4. Ghép Màu sắc và HOG với trọng số ngang nhau (50% - 50%)
    vector<float> finalFeat;
    finalFeat.reserve(colorFeat.size() + hogFeat.size());
    for (int i = 0; i < colorMat.cols; i++) finalFeat.push_back(colorMat.at<float>(0, i));
    for (int i = 0; i < hogMat.cols; i++) finalFeat.push_back(hogMat.at<float>(0, i));
    // Chuẩn hóa toàn bộ vector tổng hợp
    Mat finalMat(1, finalFeat.size(), CV_32F, finalFeat.data());
    normalize(finalMat, finalMat, 1.0, 0.0, NORM_L2);
    vector<float> result(finalMat.cols);
    for (int i = 0; i < finalMat.cols; i++) result[i] = finalMat.at<float>(0, i);
    return result;
}



// Hàm mở hộp thoại chọn file của Windows bằng chuột
string openFileDialog()
{
    char filename[MAX_PATH] = {0};
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrTitle = "Chon anh hoa qua can kiem tra";

    if (GetOpenFileNameA(&ofn))
    {
        return string(filename);
    }
    return "";
}

// Hàm vẽ giao diện kết quả dự đoán lên ảnh
void drawResult(Mat &canvas, int predictedId, const string &modeInfo)
{
    bool isFresh = (predictedId >= 0 && predictedId <= 2);
    string labelName = (predictedId >= 0) ? classNames[predictedId] : "Khong xac dinh";
    string detailName = (predictedId >= 0) ? friendlyNames[predictedId] : "";

    // Vẽ thanh banner đen trên cùng
    rectangle(canvas, Point(0, 0), Point(canvas.cols, 85), Scalar(30, 30, 30), FILLED);

    // Màu trạng thái: Xanh lá cho TƯƠI, Đỏ cho HỎNG
    Scalar statusColor = (predictedId < 0) ? Scalar(200, 200, 200) : (isFresh ? Scalar(0, 255, 0) : Scalar(0, 0, 255));
    string statusText = (predictedId < 0) ? "STATUS: WAITING..." : (isFresh ? "STATUS: FRESH (TUOI)" : "STATUS: ROTTEN (BI THOI/HONG)");

    // Vẽ thông tin chữ
    putText(canvas, "QUA: " + labelName + " (" + detailName + ")", Point(15, 32),
            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
    putText(canvas, statusText, Point(15, 68),
            FONT_HERSHEY_SIMPLEX, 0.85, statusColor, 2);

    // Vẽ thanh hướng dẫn phím bấm dưới đáy
    rectangle(canvas, Point(0, canvas.rows - 35), Point(canvas.cols, canvas.rows), Scalar(20, 20, 20), FILLED);
    putText(canvas, "[O] Chon anh  |  [C] Bat/Tat Camera  |  [T] Nhap duong dan  |  [ESC/Q] Thoat",
            Point(10, canvas.rows - 10), FONT_HERSHEY_SIMPLEX, 0.52, Scalar(0, 220, 255), 1);
}

int main()
{
    // BƯỚC 1: LOAD MÔ HÌNH SVM
    string modelPath = "D:/nam4/vinh/xla/bai_tap_lon/model/svm_model.xml";
    cout << "========================================================\n";
    cout << "   HE THONG DEMO NHAN DIEN HOA QUA TUOI / HONG (SVM)   \n";
    cout << "========================================================\n";
    cout << "Dang tai mo hinh tu: " << modelPath << " ...\n";

    Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>(modelPath);
    if (svm.empty())
    {
        cerr << "[LOI] Khong the tai file mo hinh: " << modelPath << "\n";
        return -1;
    }
    cout << "==> Tai mo hinh thanh cong!\n\n";

    cout << "HUONG DAN PHIM BAM TREN CUA SO DEMO:\n";
    cout << "  - [Phim O] : Mo hop thoai chon file anh bang chuot (File Dialog)\n";
    cout << "  - [Phim C] : Bat / Tat che do Camera Webcam truc tiep\n";
    cout << "  - [Phim T] : Nhap duong dan anh tu ban phim Console\n";
    cout << "  - [ESC / Q]: Thoat chuong trinh\n";
    cout << "========================================================\n\n";

    string windowName = "Demo Phan Loai Hoa Qua - SVM";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    setWindowProperty(windowName, WND_PROP_TOPMOST, 1); // Đưa cửa sổ lên trên cùng màn hình

    // Tạo một khung hình khởi tạo ban đầu
    Mat currentDisplay(500, 750, CV_8UC3, Scalar(45, 45, 45));
    putText(currentDisplay, "HE THONG PHAN LOAI HOA QUA", Point(130, 200),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(currentDisplay, "Nhan [O] de chon anh | Nhan [C] de mo Camera", Point(110, 260),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
    drawResult(currentDisplay, -1, "");
    imshow(windowName, currentDisplay);
    waitKey(1); // Cập nhật hiển thị cửa sổ ngay lập tức

    bool isCameraMode = false;
    VideoCapture cap;
    int lastPredictedId = -1;
    int frameCount = 0;

    while (true)
    {
        if (isCameraMode)
        {
            Mat frame;
            cap >> frame;
            if (frame.empty())
            {
                cout << "[CANH BAO] Khong doc duoc du lieu tu Camera!\n";
                isCameraMode = false;
                cap.release();
                continue;
            }

            // Lật gương ảnh webcam cho tự nhiên
            flip(frame, frame, 1);

            // Xác định vùng ngắm quả ở giữa khung hình (ROI: 240x240)
            int roiSize = 240;
            int x = (frame.cols - roiSize) / 2;
            int y = (frame.rows - roiSize) / 2;
            Rect fruitRoi(x, y, roiSize, roiSize);

            // Dự đoán mỗi 3 khung hình để tối ưu mượt mà FPS
            if (frameCount % 3 == 0)
            {
                Mat crop = frame(fruitRoi);
                vector<float> features = extractFeatures(crop);
                Mat featRow(1, features.size(), CV_32F, features.data());
                lastPredictedId = (int)svm->predict(featRow);
            }
            frameCount++;

            // Vẽ khung chữ nhật hướng dẫn đặt quả
            Scalar boxColor = (lastPredictedId <= 2) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
            rectangle(frame, fruitRoi, boxColor, 2);
            putText(frame, "Dat qua vao khung nay", Point(x + 10, y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, boxColor, 2);

            drawResult(frame, lastPredictedId, "CAMERA");
            imshow(windowName, frame);
        }

        // Bắt phím bấm (delay 30ms nếu đang chạy camera, 100ms nếu ảnh tĩnh)
        int waitTime = isCameraMode ? 30 : 100;
        int key = waitKey(waitTime);

        // Phím ESC (27) hoặc 'q' / 'Q' để thoát
        if (key == 27 || key == 'q' || key == 'Q')
        {
            cout << "Dang thoat chuong trinh...\n";
            break;
        }

        // Phím 'c' hoặc 'C': Bật / Tắt Webcam
        if (key == 'c' || key == 'C')
        {
            isCameraMode = !isCameraMode;
            if (isCameraMode)
            {
                cout << "Dang mo Camera (Webcam)...\n";
                cap.open(0);
                if (!cap.isOpened())
                {
                    cerr << "[LOI] Khong the ket noi den Camera/Webcam!\n";
                    isCameraMode = false;
                }
                else
                {
                    cout << "==> Camera da san sang! Dat qua vao khung vuong o giua.\n";
                }
            }
            else
            {
                cout << "Da dong Camera.\n";
                cap.release();
            }
        }

        // Phím 'o' hoặc 'O': Mở hộp thoại chọn ảnh của Windows
        if (key == 'o' || key == 'O')
        {
            if (isCameraMode)
            {
                isCameraMode = false;
                cap.release();
            }

            string selectedFile = openFileDialog();
            if (!selectedFile.empty())
            {
                cout << "Dang xu ly anh: " << selectedFile << " ...\n";
                Mat img = imread(selectedFile);
                if (!img.empty())
                {
                    // 1. Tự động nhận diện vùng quả (Auto ROI Detection)
                    Rect roi = detectFruitROI(img);

                    // 2. Trích xuất đặc trưng trên vùng quả vừa cắt & Dự đoán
                    vector<float> features = extractFeatures(img);
                    Mat featRow(1, features.size(), CV_32F, features.data());
                    int pred = (int)svm->predict(featRow);

                    // Resize ảnh hiển thị cho vừa màn hình
                    Mat display = img.clone();
                    double scale = 1.0;
                    if (display.cols > 850 || display.rows > 650)
                    {
                        scale = min(850.0 / display.cols, 650.0 / display.rows);
                        resize(display, display, Size(), scale, scale);
                    }

                    // Vẽ khung Bounding Box ôm sát quả trên ảnh hiển thị
                    Rect displayRoi(roi.x * scale, roi.y * scale, roi.width * scale, roi.height * scale);
                    Scalar boxColor = (pred <= 2) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                    rectangle(display, displayRoi, boxColor, 2);
                    putText(display, classNames[pred], Point(displayRoi.x, max(displayRoi.y - 8, 20)),
                            FONT_HERSHEY_SIMPLEX, 0.65, boxColor, 2);

                    drawResult(display, pred, "IMAGE");
                    imshow(windowName, display);

                    cout << "==> Phat hien qua tai toa do: [" << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << "]\n";
                    cout << "==> Du doan: " << classNames[pred] << " - " << friendlyNames[pred] << "\n\n";
                }
                else
                {
                    cerr << "[LOI] Khong the doc file anh: " << selectedFile << "\n";
                }
            }
        }

        // Phím 't' hoặc 'T': Nhập đường dẫn từ Console
        if (key == 't' || key == 'T')
        {
            if (isCameraMode)
            {
                isCameraMode = false;
                cap.release();
            }

            cout << "\nNhap duong dan file anh (hoac Copy as path roi paste vao day):\n> ";
            string inputPath;
            getline(cin, inputPath);

            if (!inputPath.empty())
            {
                if (inputPath.front() == '"' && inputPath.back() == '"')
                {
                    inputPath = inputPath.substr(1, inputPath.length() - 2);
                }

                Mat img = imread(inputPath);
                if (!img.empty())
                {
                    // 1. Tự động nhận diện vùng quả (Auto ROI Detection)
                    Rect roi = detectFruitROI(img);

                    // 2. Trích xuất đặc trưng trên vùng quả vừa cắt & Dự đoán
                    vector<float> features = extractFeatures(img);
                    Mat featRow(1, features.size(), CV_32F, features.data());
                    int pred = (int)svm->predict(featRow);

                    Mat display = img.clone();
                    double scale = 1.0;
                    if (display.cols > 850 || display.rows > 650)
                    {
                        scale = min(850.0 / display.cols, 650.0 / display.rows);
                        resize(display, display, Size(), scale, scale);
                    }

                    // Vẽ khung Bounding Box ôm sát quả trên ảnh hiển thị
                    Rect displayRoi(roi.x * scale, roi.y * scale, roi.width * scale, roi.height * scale);
                    Scalar boxColor = (pred <= 2) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                    rectangle(display, displayRoi, boxColor, 2);
                    putText(display, classNames[pred], Point(displayRoi.x, max(displayRoi.y - 8, 20)),
                            FONT_HERSHEY_SIMPLEX, 0.65, boxColor, 2);

                    drawResult(display, pred, "IMAGE");
                    imshow(windowName, display);

                    cout << "==> Phat hien qua tai toa do: [" << roi.x << ", " << roi.y << ", " << roi.width << ", " << roi.height << "]\n";
                    cout << "==> Du doan: " << classNames[pred] << " - " << friendlyNames[pred] << "\n\n";
                }
                else
                {
                    cerr << "[LOI] Khong doc duoc anh tu duong dan vua nhap!\n";
                }
            }
        }
    }

    if (cap.isOpened())
    {
        cap.release();
    }
    destroyAllWindows();
    return 0;
}
