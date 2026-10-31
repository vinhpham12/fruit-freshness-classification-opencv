#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <iomanip>
#include <chrono>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Bản đồ nhãn lớp
std::map<std::string, int> classMap = {
    {"freshapples", 0},
    {"freshbanana", 1},
    {"freshoranges", 2},
    {"rottenapples", 3},
    {"rottenbanana", 4},
    {"rottenoranges", 5}};

std::vector<std::string> classNames = {
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"};

int main()
{
    cout << "=======================================================\n";
    cout << "   DANH GIA MO HINH PIPELINE 2: CNN (OPENCV DNN)       \n";
    cout << "=======================================================\n";

    string modelPath = "D:/nam4/vinh/xla/bai_tap_lon/model/fruit_cnn.onnx";
    cout << "Dang tai mo hinh ONNX tu: " << modelPath << " ...\n";

    if (!fs::exists(modelPath)) {
        cerr << "[LOI] Chua co file mo hinh: " << modelPath << "\n";
        cerr << "Vui long chay chuong trinh train2 truoc de tao mo hinh!\n";
        return -1;
    }

    // 1. Nạp mạng ONNX bằng OpenCV DNN
    dnn::Net net = dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "[LOI] Khong the nap mo hinh ONNX vao OpenCV DNN!\n";
        return -1;
    }

    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);
    cout << "==> Mo hinh CNN da duoc nap thanh cong vao bo suy luan cv::dnn!\n\n";

    // 2. Thu thập danh sách ảnh test
    string testPath = "D:/nam4/vinh/xla/bai_tap_lon/dataset/test";
    vector<string> testImages;
    vector<int> groundTruth;

    for (const auto &entry : fs::directory_iterator(testPath)) {
        if (entry.is_directory()) {
            string folderName = entry.path().filename().string();
            if (classMap.find(folderName) != classMap.end()) {
                int label = classMap[folderName];
                for (const auto &imgFile : fs::directory_iterator(entry.path())) {
                    if (imgFile.is_regular_file()) {
                        string ext = imgFile.path().extension().string();
                        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                            testImages.push_back(imgFile.path().string());
                            groundTruth.push_back(label);
                        }
                    }
                }
            }
        }
    }

    int totalImages = (int)testImages.size();
    cout << "So luong anh test tim thay: " << totalImages << "\n";
    cout << "Dang tien hanh du doan bang mang CNN...\n";

    // 3. Khởi tạo ma trận nhầm lẫn (6x6)
    int confusionMatrix[6][6] = {0};
    int correctCount = 0;
    double totalInferenceTimeMs = 0.0;

    for (int i = 0; i < totalImages; i++) {
        Mat img = imread(testImages[i]);
        if (img.empty()) continue;

        // Chuẩn hóa và tạo Blob đầu vào [1, 3, 128, 128]
        auto tStart = chrono::high_resolution_clock::now();
        Mat blob;
        dnn::blobFromImage(img, blob, 1.0 / 255.0, Size(128, 128), Scalar(), true, false);
        net.setInput(blob);
        Mat prob = net.forward();
        auto tEnd = chrono::high_resolution_clock::now();

        double elapsedMs = chrono::duration<double, milli>(tEnd - tStart).count();
        totalInferenceTimeMs += elapsedMs;

        // Tìm lớp có xác suất cao nhất
        Point classIdPoint;
        double confidence;
        minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
        int predictedLabel = classIdPoint.x;

        int trueLabel = groundTruth[i];
        confusionMatrix[trueLabel][predictedLabel]++;

        if (predictedLabel == trueLabel) {
            correctCount++;
        }

        if ((i + 1) % 400 == 0 || (i + 1) == totalImages) {
            cout << "Da danh gia: " << (i + 1) << "/" << totalImages << " anh.\n";
        }
    }

    // 4. In bảng kết quả
    double accuracy = (double)correctCount / totalImages * 100.0;
    double avgLatency = totalInferenceTimeMs / totalImages;

    cout << "\n=======================================================\n";
    cout << "       KET QUA DANH GIA MO HINH PIPELINE 2 (CNN)       \n";
    cout << "=======================================================\n";
    cout << "Tong so anh test      : " << totalImages << "\n";
    cout << "So anh doan dung      : " << correctCount << "\n";
    cout << "DO CHINH XAC (ACC)    : " << fixed << setprecision(2) << accuracy << " %\n";
    cout << "Toc do suy luan TB    : " << fixed << setprecision(2) << avgLatency << " ms / anh\n";
    cout << "=======================================================\n\n";

    cout << "MA TRAN NHAM LAN (CONFUSION MATRIX 6x6):\n";
    cout << "(Hang = Nhan Thuc Te, Cot = Nhan Du Doan)\n\n";
    cout << "                 [0]       [1]       [2]       [3]       [4]       [5]   | Do chinh xac tung lop\n";
    cout << "------------------------------------------------------------------------------------------\n";

    for (int r = 0; r < 6; r++) {
        cout << setw(14) << right << classNames[r] << " |";
        int rowTotal = 0;
        for (int c = 0; c < 6; c++) {
            cout << setw(10) << right << confusionMatrix[r][c];
            rowTotal += confusionMatrix[r][c];
        }
        double classAcc = (rowTotal > 0) ? ((double)confusionMatrix[r][r] / rowTotal * 100.0) : 0.0;
        cout << "   | " << fixed << setprecision(1) << classAcc << " %\n";
    }
    cout << "------------------------------------------------------------------------------------------\n";
    cout << "\nGhi chu cac lop:\n";
    for (int k = 0; k < 6; k++) {
        cout << "  [" << k << "]: " << classNames[k] << "\n";
    }

    return 0;
}
