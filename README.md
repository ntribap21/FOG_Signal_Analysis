# FOG_FIXED — một bộ đồng bộ, fog-unified-1.1

## Cài đặt trên Windows

1. Giải nén toàn bộ ZIP vào cùng một thư mục. File Python phải có đúng tên:
   main.py, signal_processing.py, data_prepare.py, batch_processing.py, train_fog.py.
   Không để các tên main(9).py / signal_processing(1).py thay cho tên module chuẩn.
2. Giữ file fog_config.json đang dùng và raw TXT/model/output của bạn.
   ZIP không chứa fog_config.json nên không ghi đè cấu hình cá nhân.
3. Trong PowerShell ở thư mục chương trình, với .venv đang kích hoạt:

```powershell
pip install -r requirements.txt
python check_install.py
python main.py
```

Nếu chưa có fog_config.json và muốn dùng cấu hình bạn đã cung cấp trước đây:

```powershell
Copy-Item fog_config_example.json fog_config.json
```

Chỉ thực hiện lệnh copy khi chưa có cấu hình cá nhân cần giữ.
Ví dụ giữ: Fs 64, resample 64→64, Low-pass 16 Hz order 4, detrend Linear,
window 2 s, overlap 50%, Majority, Ankle ba trục, loại label gốc 0.

## Những lỗi được xử lý

- API mismatch: signal_processing cung cấp Config, ProcessingEngine và các hàm GUI/batch cần dùng.
- Một pipeline numerical duy nhất trong signal_processing; data_prepare chỉ điều phối/export.
- Một lớp FOGSignalAnalyzer kế thừa QMainWindow; không có CoreFOGAnalyzer/_PreviousFOGAnalyzer.
- Config được chuẩn hóa tại mọi ranh giới xử lý; không trộn dict/dataclass gây .validate() crash.
- Version check khi mở main và check_install phát hiện module khác phiên bản trước khi chạy GUI.
- batch_processing chỉ chuyển tiếp import tới signal_processing, không chứa công thức riêng.

## Các đặc tính giữ nguyên

- Phương pháp numerical của main/batch bạn gửi: Butterworth b/a, filtfilt,
  lfilter fallback cho đoạn ngắn, resample và detrend như trước.
- 14 loại đặc trưng gốc, công thức FI và cách gán nhãn gốc.
- SVM StandardScaler + SVC, cách chia theo bệnh nhân và các tham số train hiện có.
- JSON GUI original-core-gui-v1 được chuyển về Config chung mà giữ nguyên controls/selection.
- Pan/zoom, Original/Binary labels, Data Selection và Selected Time Domain.
- Loại trừ theo Recording_ID, bật/tắt/xóa từng đoạn; không windowing qua đoạn bị loại.
- Batch config độc lập, 1–3 vị trí và 1–3 trục; input folder hoặc nhiều TXT.

JSON flat của nhánh SOS/modular cũ không được âm thầm sử dụng cho thuật toán gốc.
Nếu nạp nhầm schema, chương trình báo rõ; không tự thay thuật toán/đặc trưng.
File signal_processing(1).py bạn gửi dùng nhánh SOS khác với main/batch; bản này
chọn công thức main/batch làm chuẩn để giữ kết quả của bộ 42 đặc trưng đã kiểm tra.

## Xuất dữ liệu

NPZ/CSV có một hàng/cửa sổ, metadata bệnh nhân/bản ghi/segment/time/config.
Preview GUI có thể hiển thị hàng riêng cho từng trục; khi export được ghép về một vector/cửa sổ.
Ankle × ba trục: 42 cột; hai vị trí × ba trục: 84; ba vị trí × ba trục: 126.
Xuất kèm config JSON và report từng file. Không đưa đồng thời CSV và NPZ của cùng dataset vào train.

```powershell
python data_prepare.py --input raw_folder --config fog_config.json --output output/fog_batch --format both
python train_fog.py --input output/fog_batch.npz --output svm_output
```

## Kiểm tra

- Biên dịch Python và check_install trong thư mục giải nén độc lập.
- GUI offscreen, cấu hình tự nạp, Data Selection/exclusions/preprocessing/windowing.
- Export Current Windows và prepare_batch được gọi thực tế.
- Bộ Ankle trên đoạn S09R01 khớp numerical main/batch gốc.
- GUI và batch cùng input/config có cùng đặc trưng; xuất 42/84/126 cột; bộ train đọc được.
- Các thông số controls/selection trong cấu hình gốc được giữ nguyên.

Lần sửa này không thêm Jerk/CWT, không thay mục tiêu tuning hoặc ngưỡng quyết định SVM.
