#include <iostream>
#include <cstdlib>
#include <string>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

int main() {
    cout << "========================================================\n";
    cout << "        CHUONG TRINH HUAN LUYEN PIPELINE 2 (CNN)       \n";
    cout << "========================================================\n";
    cout << "-> Mo hinh su dung: Mang no-ron tich chap 4-layer CNN\n";
    cout << "-> Dinh dang xuat : ONNX (Chuan cong nghiep cho OpenCV)\n";
    cout << "-> Dang khoi chay tien trinh huan luyen Python...\n\n";

    string scriptPath = "D:/nam4/vinh/xla/bai_tap_lon/train/train2_cnn.py";
    if (!fs::exists(scriptPath)) {
        cerr << "[LOI] Khong tim thay script: " << scriptPath << "\n";
        return -1;
    }

    string command = "python \"" + scriptPath + "\"";
    int ret = system(command.c_str());

    if (ret == 0) {
        cout << "\n==> Pipeline 2 huan luyen va xuat ONNX thanh cong!\n";
        cout << "==> File mo hinh duoc luu tai: D:/nam4/vinh/xla/bai_tap_lon/model/fruit_cnn.onnx\n";
    } else {
        cerr << "\n[CANH BAO] Tien trinh huan luyen gap loi hoac bi huy (Exit code: " << ret << ").\n";
    }
    return ret;
}
