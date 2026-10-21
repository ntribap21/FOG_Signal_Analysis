# FOG Analyzer — bổ sung đặc trưng, không DWT

Đây là bộ file đồng bộ dựa trên sáu file người dùng cung cấp. Giữ 14 công thức cũ/trục và pipeline selection → split → filter → detrend → windowing. Không có DWT, PyWavelets, gravity compensation hoặc nhánh LPMean trong bản này.

## Cài và chạy trên Windows

Giải nén toàn bộ vào một thư mục mới. Tên chạy phải là `main.py`, `signal_processing.py`, `data_prepare.py`, `train_fog.py`, `batch_processing.py`, `check_install.py`, `feature_registry.py`. Không trộn module bản trước.

```powershell
python -m pip install -r requirements.txt
python check_install.py
python main.py
```

Không cần tạo lại venv nếu đang có môi trường phù hợp. Lệnh pip trên không yêu cầu nâng cấp đồng loạt các thư viện đã cài.

## Thao tác GUI

- Mục EXPORT FEATURES có panel ADDITIONAL FEATURES cho phân tích một cảm biến. Tắt toàn bộ checkbox mới để dùng bộ gốc.
- Trong Batch Configuration có panel riêng. Chọn sensor/trục, preprocessing và windowing như trước, rồi chọn feature mới.
- Chọn cross-correlation: cần ít nhất hai vị trí, đủ ba trục. Chọn ATER: cần Ankle và Trunk, đủ ba trục. Hai nhóm này chỉ có trong Batch GUI vì cửa sổ tương tác hiện chỉ giữ một sensor.
- Các feature từng trục mới hiển thị trong bảng Additional feature của trang time analysis sau ANALYZE WINDOW. Feature liên kênh hiển thị qua batch CSV/NPZ.
- Khi thay feature tương tác, phải APPLY WINDOWING lại trước preview/export. Thay Batch feature không thay lựa chọn phân tích một cảm biến.
- Lưu/đọc config JSON lưu cả nhóm đang bật và tham số. Config cũ không có feature_options sẽ dùng 14 đặc trưng cũ/trục.
- Cross-correlation/ATER thiếu sensor gây lỗi cấu hình, không tạo số 0 giả. Report giữ thống kê bản ghi và thêm Feature_Count, Undefined_Feature_Values khi xuất batch.

## Các đặc trưng mới

Giữ 126 đặc trưng gốc khi đủ 9 trục. Mặc định tất cả nhóm mới tắt.

| Nhóm | Định nghĩa | Số cột thêm đủ 3 sensor/3 trục |
|---|---|---:|
| Jerk | RMS(diff(a) × fs), gia tốc processed, đơn vị g/s | 9 |
| SampEn | m=2, r=0.2×SD mặc định; cùng tập template độ dài m và m+1; không self-match; natural log | 9 |
| ApEn | Có self-match; phi(m) − phi(m+1); natural log | 9 |
| Autocorrelation | Trừ mean; R(k)/R(0); maximum trong lag, không phải tìm local peak; peak và lag giây | 18 |
| Kurtosis | Fisher excess, scipy bias=False | 9 |
| Crest Factor | max(abs(a−mean))/RMS(a−mean), không đơn vị | 9 |
| Correlation | Pearson lag 0: F–V, F–L, V–L tại mỗi sensor | 9 |
| Cross-correlation | Chuẩn hóa Pearson trên phần chồng lấp, magnitude của XYZ sau trừ mean từng trục; lấy hệ số có abs lớn nhất và lag | 6 |
| LogATER | log((sum XYZ band power ankle + eps)/(sum XYZ band power trunk + eps)); loco 0.5–3 và freeze 3–8 Hz | 2 |

Bật tất cả: **206 cột** đầu vào. Đây là số feature xuất, chưa bao gồm cột indicator do imputer tạo nội bộ khi có giá trị thiếu.

Các thông số có thể đổi ở panel và config:

- entropy_m=2, entropy_r_ratio=0.2; delay cố định 1 mẫu.
- autocorr_min_s=0.35, autocorr_max_s=1.0; ceil/floor theo fs; giới hạn lag ≤ n//2 để còn ít nhất nửa cửa sổ chồng lấp.
- crosscorr_max_s=0.25; k>0 nghĩa ghép sensor A[n] với B[n+k]. Đây là quy ước index, không khẳng định hướng truyền chuyển động. Nếu nhiều lag hòa, chọn lag đầu tiên theo thứ tự tăng dần.
- ater_epsilon_g2=1e-8; là tham số ổn định, không phải ngưỡng năng lượng đã được xác thực. Giữ các band power gốc để phát hiện cả tử/mẫu nhỏ.

Correlation giữa vị trí sử dụng norm của vector đã trừ mean theo trục, không dùng norm raw chứa trọng lực và không tương đương gravity compensation. Dữ liệu đầu vào các hàm này đã là g; tên khóa Signals vẫn giữ hậu tố `_mg` từ hệ thống cũ.

## Trường hợp không tính được

Tín hiệu đầu vào có NaN/Inf hoặc sai chiều bị từ chối; không xóa mẫu riêng từng trục. Các thống kê bổ sung không xác định (SampEn không có cặp m/m+1 khớp; variance gần 0 cho entropy/correlation/kurtosis/crest; lag không đủ) xuất NaN. Giá trị thiếu không bị thay bằng 0 khi trích xuất.

Trainer mới dùng `SimpleImputer(median, add_indicator=True, keep_empty_features=True) → StandardScaler → SVC`. Imputer/scaler được fit trong từng fold train, kể cả inner folds. Feature hoàn toàn thiếu trong tập train được impute 0 theo keep_empty_features; model lưu cả pipeline và thứ tự đầu vào. Xem tỷ lệ thiếu trước khi diễn giải kết quả: batch có quá nhiều thiếu cần xem lại cấu hình hoặc feature.

Không dùng NaN để che lỗi raw. Raw lỗi và thống kê không ước lượng được là hai tình huống khác nhau. Các feature cũ phải tiếp tục hữu hạn.

## Export và training

NPZ giữ feature_names theo thứ tự. CSV mở rộng phải có file `<prefix>_config.json` đi kèm. Long CSV mở rộng không được trainer chấp nhận; xuất wide qua GUI mới hoặc batch. Không sửa JSON sau khi xuất để đổi nội dung NPZ.

```powershell
python train_fog.py --input fog_batch.npz --output results_no_dwt_B1 --evaluation loso --C 1 --gamma scale --decision-threshold 0
python train_fog.py --input fog_batch.npz --output results_no_dwt_B2 --evaluation loso --threshold-tuning --threshold-method exact --cv-folds 3
python train_fog.py --input fog_batch.npz --output results_no_dwt_B3 --evaluation loso --tune --threshold-tuning --threshold-method exact --cv-folds 3
```

Mỗi lần dùng một NPZ hoặc một CSV, không nhập cả hai bản sao cùng dataset. Model cũ không dùng cho feature mới. Xuất batch và train lại khi thay feature. So sánh A0 tất cả nhóm mới tắt với A1 chỉ Jerk, A2 chỉ SampEn, A3 chỉ autocorrelation, A4 chỉ correlation... trước khi kết hợp. Giữ dữ liệu, khoảng loại, filter, window, label và ngưỡng cố định khi so nhóm.

## Kiểm tra đã thực hiện / giới hạn

- So sánh trực tiếp 14 công thức và config chế độ gốc với signal_processing(4).py: không đổi trong kiểm tra đối chứng.
- Kiểm tra cú pháp các module.
- Dữ liệu tổng hợp: batch → CSV/NPZ → loader trainer giữ đúng 206 tên và giá trị.
- Export GUI qua hàm xử lý → wide cho cùng dữ liệu khớp batch, kể cả feature liên kênh.
- Kiểm tra tín hiệu hằng, sin, correlation, autocorrelation, missing-value fit/inference.
- Chạy LOSO 4 bệnh nhân tổng hợp thành công, không phải kết quả hiệu năng DAPHNET.
- Chưa chạy mở GUI thực tế vì môi trường kiểm tra không có PySide6/pyqtgraph. Cần chạy check_install và mở thử trên máy Windows; kích thước panel mới có thể cần điều chỉnh theo màn hình thực tế.

Chạy lại smoke test số học/export/train không cần GUI:

```powershell
python smoke_test.py
```

SampEn/ApEn là tính toán bậc hai theo số mẫu; bật cả hai trên toàn bộ DAPHNET sẽ chậm hơn đáng kể. Thử một bản ghi trước. Không có khẳng định feature mới sẽ tăng đồng thời sensitivity/specificity/F1.
