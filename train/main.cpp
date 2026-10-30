#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/ml.hpp>           // Thêm sẵn cho bước huấn luyện SVM tiếp theo
#include <vector>
#include <string>
#include <map>
#include <filesystem>
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

int main()
{
    string datasetPath = "D:/nam4/vinh/xla/bai_tap_lon/dataset/train";

    vector<string> imagePaths;
    vector<int> imageLabels;

    // 1. Quét qua thư mục cha
    for (const auto &entry : fs::directory_iterator(datasetPath))
    {

        // 2. Chỉ xử lý nếu nó là thư mục (bỏ qua nếu có file rác)
        if (entry.is_directory())
        {

            // 3. Lấy tên thư mục con (freshapples, freshbanana, ...)
            string folderName = entry.path().filename().string();
            // 4. ĐÂY LÀ ĐOẠN IF...ELSE CỦA BẠN:
            if (classMap.find(folderName) != classMap.end())
            {
                int label = classMap[folderName];
                cout << "Folder hop le: " << folderName << " -> Nhan: " << label << "\n";
                // TODO: Tiếp tục duyệt từng ảnh bên trong entry.path() tại đây!
                int count = 0;
                for (const auto &imgEntry : fs::directory_iterator(entry.path()))
                {
                    if (imgEntry.is_regular_file())
                    {
                        string ext = imgEntry.path().extension().string();
                        if (ext == ".jpg" || ext == ".png" || ext == ".jpeg")
                        {
                            string imgPath = imgEntry.path().string();
                            imagePaths.push_back(imgPath);
                            imageLabels.push_back(label);
                            count++;
                        }
                    }
                }
                cout << "So luong anh trong thu muc " << folderName << ": " << count << "\n";
            }
            else
            {
                cout << "Bo qua thu muc la: " << folderName << "\n";
            }
        }
    }
    cout << "Tong so anh duoc tim thay: " << imagePaths.size() << "\n";

    Mat trainData;
    Mat trainLabels;
    cout << "\n Dang trich xuat dac trung cho " << imagePaths.size() << " anh...\n";
    for (int i = 0; i < imagePaths.size(); i++)
    {
        Mat img = imread(imagePaths[i]);
        if (img.empty())
        {
            cout << "Khong the doc anh: " << imagePaths[i] << "\n";
            continue;
        }
        vector<float> features = extractFeatures(img);
        // TODO: Thêm features vào trainData và imageLabels
        Mat featRow(1, features.size(), CV_32F, features.data());
        trainData.push_back(featRow);
        trainLabels.push_back(imageLabels[i]);

        if ((i + 1) % 500 == 0 || i + 1 == imagePaths.size())
        {
            cout << "Da xu ly: " << (i + 1) << "/" << imagePaths.size() << " anh.\n";
        }
    }
    cout << "==> Hoan tat! Kich thuoc trainData: " << trainData.rows << " x " << trainData.cols << "\n";
    cout << "==> Kich thuoc trainLabels: " << trainLabels.rows << " x " << trainLabels.cols << "\n";

    trainLabels.convertTo(trainLabels, CV_32S); // Chuyển đổi nhãn sang kiểu int
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setC(1.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    cout <<"\n ------------------------------------------\n";
    cout << "Dang huan luyen SVM \n";
    int64 start = getTickCount();
    svm->train(trainData, ml::ROW_SAMPLE, trainLabels);
    int64 end = getTickCount();
    double duration = (end - start) / getTickFrequency();
    cout << "==> Hoan tat! Thoi gian huan luyen: " << duration << " giay.\n";

    string modelPath = "D:/nam4/vinh/xla/bai_tap_lon/model/svm_model.xml";
    fs::create_directories(fs::path(modelPath).parent_path());
    svm->save(modelPath);
    cout << "==> Da luu mo hinh SVM vao: " << modelPath << "\n";
    return 0;
}
