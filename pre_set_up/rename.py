import os
import re

# Đường dẫn tới thư mục cha chứa các thư mục con
root_dir = r"D:/nam4/vinh/xla/bai_tap_lon/dataset/test"
root_dir = r"D:/nam4/vinh/xla/bai_tap_lon/dataset/train"
# Các định dạng ảnh cần xử lý
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def format_prefix(folder_name):
    # Thêm dấu gạch dưới giữa các từ (ví dụ: freshapples -> fresh_apples, rottenbanana -> rotten_banana)
    name = re.sub(r'^(fresh|rotten)(.*)$', r'\1_\2', folder_name, flags=re.IGNORECASE)
    return f"{name}_"

# Duyệt qua từng thư mục con trong thư mục test
for item in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, item)
    
    if os.path.isdir(folder_path):
        prefix = format_prefix(item)
        
        # Lấy danh sách file ảnh hợp lệ
        files = [
            f for f in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, f)) 
            and os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        
        print(f"Đang xử lý thư mục: {item} ({len(files)} ảnh) -> Tiền tố: '{prefix}'")
        
        # BƯỚC 1: Đổi sang tên tạm thời để tránh xung đột tên file cũ
        temp_files = []
        for idx, filename in enumerate(files, start=1):
            old_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            temp_name = f"__temp_{idx}{ext}"
            temp_path = os.path.join(folder_path, temp_name)
            os.rename(old_path, temp_path)
            temp_files.append((temp_path, ext, idx))
            
        # BƯỚC 2: Đổi sang định dạng tên chính thức
        for temp_path, ext, idx in temp_files:
            new_name = f"{prefix}{idx}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(temp_path, new_path)

print("\nHoàn tất đổi tên toàn bộ ảnh trong các thư mục!")