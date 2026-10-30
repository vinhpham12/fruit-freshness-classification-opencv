#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/ml.hpp>           // Thêm sẵn cho bước huấn luyện SVM tiếp theo
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <iomanip>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

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
    
// Hàm tự động phát hiện quả và cắt sát viền (Auto-Crop Fruit ROI)
Rect detectFruitROI(const Mat& img) {
    int minX = img.cols, maxX = 0;
    int minY = img.rows, maxY = 0;
    int count = 0;

    for (int y = 0; y < img.rows; y++) {
        const Vec3b* row = img.ptr<Vec3b>(y);
        for (int x = 0; x < img.cols; x++) {
            if (row[x][0] < 235 || row[x][1] < 235 || row[x][2] < 235) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                count++;
            }
        }
    }

    if (count < 100 || minX >= maxX || minY >= maxY) {
        return Rect(0, 0, img.cols, img.rows);
    }

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

int main(){
    string modelPath = "D:/nam4/vinh/xla/bai_tap_lon/model/svm_model.xml";
    cout << "Dang tai mo hinh SVM tu file: " << modelPath << endl;
    Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>(modelPath);
    if(svm.empty()){
        cerr << "Khong the tai mo hinh SVM tu file: " << modelPath << endl;
        return -1;
    }
    cout << "Mo hinh SVM da duoc tai thanh cong." << endl;
    String datasetPath = "D:/nam4/vinh/xla/bai_tap_lon/dataset/test";
    vector<String> testImagePaths;
    vector<int> testLabels;

    for (const auto &entry : fs::directory_iterator(datasetPath))
    {
        if(entry.is_directory()){
            string classFolder = entry.path().filename().string();
            if(classMap.find(classFolder)!=classMap.end()){
                int label = classMap[classFolder];
                for(const auto &imgEntry : fs::directory_iterator(entry.path()))
                {
                    if(imgEntry.is_regular_file()){
                        string ext=imgEntry.path().extension().string();
                        if(ext==".jpg" || ext==".png" || ext==".jpeg"){
                            testImagePaths.push_back(imgEntry.path().string());
                            testLabels.push_back(label);
                        }
                    }
                }
            }
        }
    }
    cout << "So luong anh test: " << testImagePaths.size() << endl;
    
    int totalCorrect = 0;
    int confusionMatrix[6][6] = {0};
    cout << "Dang tien hanh du doan tren tap test..." << endl;
    for (size_t i = 0; i < testImagePaths.size(); ++i)
    {
        Mat img = imread(testImagePaths[i]);
        if(img.empty()){
            cerr << "Khong the doc anh: " << testImagePaths[i] << endl;
            continue;
        }
        vector<float> features = extractFeatures(img);
        Mat featRow(1,features.size(),CV_32F,features.data());

        int predictedLabel = (int)svm->predict(featRow);
        int trueLabel = testLabels[i];
        confusionMatrix[trueLabel][predictedLabel]++;
        if(predictedLabel==trueLabel){
            totalCorrect++;
        }
        if ((i + 1) % 200 == 0 || i + 1 == testImagePaths.size())
        {
            cout << "Da danh gia: " << (i + 1) << "/" << testImagePaths.size() << " anh.\n";
        }
    }
    double accuracy = (double)totalCorrect / testImagePaths.size() * 100.0;
    cout << "\n=======================================================\n";
    cout << "                KET QUA DANH GIA MO HINH               \n";
    cout << "=======================================================\n";
    cout << "Tong so anh test   : " << testImagePaths.size() << "\n";
    cout << "So anh doan dung   : " << totalCorrect << "\n";
    cout << "DO CHINH XAC (ACC) : " << fixed << setprecision(2) << accuracy << " %\n";
    cout << "=======================================================\n\n";
    cout << "MA TRAN NHAM LAN (CONFUSION MATRIX 6x6):\n";
    cout << "(Hang = Nhan Thuc Te, Cot = Nhan Du Doan)\n\n";
    // In tieu de cot
    cout << setw(16) << " ";
    for (int c = 0; c < 6; c++)
    {
        cout << setw(10) << ("[" + to_string(c) + "]");
    }
    cout << "   | Do chinh xac tung lop\n";
    cout << string(90, '-') << "\n";
    // In tung hang
    for (int r = 0; r < 6; r++)
    {
        cout << setw(14) << classNames[r] << " |";
        int classTotal = 0;
        for (int c = 0; c < 6; c++)
        {
            cout << setw(10) << confusionMatrix[r][c];
            classTotal += confusionMatrix[r][c];
        }
        double classAcc = (classTotal > 0) ? ((double)confusionMatrix[r][r] / classTotal * 100.0) : 0.0;
        cout << "   | " << fixed << setprecision(1) << classAcc << " %\n";
    }
    cout << string(90, '-') << "\n";
    return 0;
}