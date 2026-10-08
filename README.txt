# FOG Signal Analyzer & SVM Training

Ứng dụng desktop phân tích tín hiệu DAPHNET và chương trình huấn luyện SVM để phân loại **Non-FOG / FOG theo window**.

Tài liệu này dành cho **bản `main.py` và bản `train_fog.py` .
## 1. Các file và thư mục

| File / thư mục | Công dụng |
|---|---|
| `main.py` | GUI: đọc raw, chọn dữ liệu, preprocessing, windowing, phân tích và xuất feature |
| `train_fog.py` | Đọc feature CSV/NPZ và huấn luyện SVM |
| `requirements.txt` | Thư viện Python cho cả GUI và training |
| `raw_data/` | Thư mục tự tạo để chứa DAPHNET raw TXT/CSV |
| `prepared_data/` | Thư mục tự tạo để chứa feature và config JSON xuất từ GUI |
| `svm_output/` | Thư mục kết quả do chương trình train tạo |

Đặt `main.py`, `train_fog.py`, `README.md` và `requirements.txt` trong cùng thư mục dự án. Nếu file tải về có tên `main(4).py`, đổi thành `main.py` trước khi chạy các lệnh bên dưới.

Bản GUI hiện tại chạy độc lập, không cần `signal_processing.py` hoặc `data_prepare.py`. Phần chuẩn bị batch được tích hợp trong `main.py`. Script train cũng chạy độc lập với GUI.

## 2. Cài đặt trên Windows

Dùng Python 3.12, phiên bản đã được dùng để kiểm tra chương trình. Mở PowerShell trong thư mục dự án:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh dùng trực tiếp Python trong môi trường ảo, nên không cần chạy `Activate.ps1`.

Kiểm tra thư viện:

```powershell
.\.venv\Scripts\python.exe -c "import numpy, pandas, scipy, PySide6, pyqtgraph, sklearn, joblib; print('Dependencies OK')"
```

Chạy app:

```powershell
.\.venv\Scripts\python.exe main.py
```

Nếu muốn dùng môi trường đang có, cài bằng `python -m pip install -r requirements.txt` và chạy `python main.py`.

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

Các bước GUI đã được kiểm tra trong môi trường Linux offscreen; cài đặt và hiển thị trên máy Windows của bạn cần được kiểm tra tại máy đó.

## 3. Dữ liệu và định nghĩa nhãn

Dataset: [DAPHNET Freezing of Gait — UCI](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait).

GUI nhận DAPHNET TXT có 11 cột, hoặc raw CSV có đúng tên cột:

| Nhóm | Các cột |
|---|---|
| Timestamp | `Time_ms` |
| Ankle | `Ankle_Forward_mg`, `Ankle_Vertical_mg`, `Ankle_Lateral_mg` |
| Thigh | `Thigh_Forward_mg`, `Thigh_Vertical_mg`, `Thigh_Lateral_mg` |
| Trunk | `Trunk_Forward_mg`, `Trunk_Vertical_mg`, `Trunk_Lateral_mg` |
| Nhãn | `Label` |

Bản đọc raw hiện tại kiểm tra bước thời gian xấp xỉ **64 Hz**, timestamp tăng nghiêm ngặt, dữ liệu hữu hạn và nhãn 0/1/2. Tín hiệu raw được lưu theo mg; khi vẽ và tạo feature, đổi sang g bằng cách chia 1000.

| Original Label | Ý nghĩa | Binary Label để train |
|---|---|---|
| `0` | Ngoài thí nghiệm | Không hợp lệ / để trống; loại khỏi window ML |
| `1` | Non-FOG | `0` |
| `2` | FOG | `1` |

**Original Label 0 khác Binary Label 0.** Original 0 không được coi là Non-FOG. Hai cột Original/Binary được tạo ngay khi mở dataset. Chọn Original/Binary trên đồ thị chỉ thay đổi cách hiển thị nhãn.

## 4. Workflow GUI

**Raw → Data Selection → Selected Time Domain → Preprocessing → Windowing → Analysis / Export → SVM Training.**

| Trang | Thao tác |
|---|---|
| Dataset | Mở bản ghi, xem thông tin và preview toàn bộ hàng bằng model/view |
| Raw Time Domain | Xem tín hiệu gốc và nhãn trên cùng một đồ thị |
| Data Selection | Chọn đoạn trong thí nghiệm, khoảng thời gian hoặc toàn bộ bản ghi; tùy chọn loại Original 0 |
| Selected Time Domain | Xem dữ liệu Selected hoặc Processed sau preprocessing |
| Preprocessing | Bật/tắt resample, Butterworth filter và detrend |
| Windowing | Chọn sensor, axes, độ dài window, overlap và cách gán nhãn |
| Time Analysis | So sánh tín hiệu raw với đầu vào window; xem time features |
| Frequency Analysis | Xem FFT, PSD và frequency features |
| Export | Xuất feature CSV/NPZ và lưu config JSON |
| Batch Preparation | Chuẩn bị feature cho nhiều bản ghi bằng cùng thông số |

### PAN, zoom và chọn region

1. Zoom vào một khoảng bằng lăn chuột hoặc nhập Start/End rồi bấm **VIEW RANGE**.
2. Bấm **PAN**, giữ chuột trái trong vùng đồ thị và kéo ngang.
3. Bấm **SELECT REGION** để tạo vùng chọn có thể di chuyển và chỉnh hai biên.
4. Bấm **USE REGION** để chuyển Start/End sang Data Selection, sau đó bấm **APPLY SELECTION**.
5. Bấm **RESET VIEW** để xem toàn bộ nguồn dữ liệu.

Raw và Selected Time Domain có PAN riêng, cùng trục thời gian cho tín hiệu và nhãn. PAN khóa khả năng kéo vùng chọn nhưng giữ lại khoảng đã chọn.

### Data Selection và preprocessing

- Mặc định giữ các đoạn trong thí nghiệm và loại Original Label 0.
- Timestamp gốc và khoảng gián đoạn được giữ. Các đoạn được xử lý riêng; không nối các đoạn thành tín hiệu liên tục giả.
- Bấm **APPLY SELECTION** trước preprocessing hoặc windowing.
- Thứ tự xử lý: **Selection → Resample → Filter → Detrend**.
- Bản giữ core gốc dùng Butterworth hệ số `b, a`: `filtfilt` khi đoạn đủ dài, `lfilter` khi đoạn ngắn. Bản này không có lựa chọn SOS zero-phase/causal như bản module trước.
- **VIEW PROCESSED** mở Selected Time Domain với nguồn Processed.
- Thay đổi thông số upstream sẽ xóa kết quả phụ thuộc cũ; cần Apply lại.
- Raw vẫn được giữ nguyên.

### Windowing

Mặc định: **Ankle X/Y/Z**, window **2 s**, overlap **50%**. Ở 64 Hz, mỗi window có 128 mẫu và step là 64 mẫu.

| Cách gán nhãn | Quy tắc |
|---|---|
| Majority | FOG nếu tỷ lệ mẫu FOG >50%; hòa là Non-FOG |
| FOG Threshold | FOG nếu tỷ lệ mẫu FOG >= ngưỡng đã chọn |
| Strict FOG | FOG nếu 100% mẫu trong window là FOG |

Window ML không chứa Original 0 và không vượt qua khoảng gián đoạn. Đoạn ngắn hơn một window và phần đuôi chưa đủ window không tạo mẫu train.

## 5. Xuất dữ liệu và batch

### Một bản ghi

1. Mở raw có tên gốc như `S01R01.txt`.
2. Apply Data Selection và preprocessing theo cấu hình mong muốn.
3. Apply Windowing.
4. Vào Export, chọn **CSV + NPZ**, **NPZ** hoặc **CSV**.
5. Đặt tên như `prepared_data/S01R01_features`.

Nếu export một bản ghi thiếu cột Subject/Recording, `train_fog.py` suy ra chúng từ tên `SxxRxx_features.csv/npz`. Không dùng tên chung như `features.npz` trong trường hợp đó.

### Nhiều bản ghi

1. Vào Batch Preparation, bấm **ADD FILES** hoặc **ADD FOLDER**.
2. Kiểm tra Data Selection, preprocessing và windowing; các thông số áp dụng chung cho toàn bộ batch.
3. Có thể **LOAD CONFIG JSON** của đúng phiên bản GUI hiện tại.
4. Chọn format, bấm **PREPARE BATCH** và lưu vào `prepared_data/`.
5. Kiểm tra report: bản ghi lỗi bị loại khỏi dataset. Nếu tất cả file đều lỗi, chỉ có report, không có dataset train.

Batch thêm metadata Subject/Recording. Giữ nguyên tên file gốc `SxxRxx` để xác định người tham gia. Time Selection áp dụng cùng khoảng Start/End tuyệt đối cho mỗi file.

### Config JSON

- **SAVE CONFIG JSON** ở Export lưu giá trị controls hiện tại để dùng lại; không chứa raw hay model.
- Khi xuất windows hiện tại, sidecar JSON ghi cấu hình đã áp dụng cho các windows đó.
- GUI giữ core gốc dùng schema `original-core-gui-v1`; không nhập JSON của bản GUI module khác.
- Giữ `*_config.json` cùng CSV/NPZ. Với CSV không có JSON, chương trình vẫn đọc được nhưng ghi `config_verified=false`.

## 6. Feature đầu vào SVM

Bản core gốc xuất **một hàng/window/sensor/axis**, gồm 14 features:

| Nhóm | Features |
|---|---|
| Time | Mean, RMS, Std, Variance, Min, Max, SMA |
| Frequency | Dominant Frequency, Total Power, Loco Power, Freeze Power, Freeze Index, Low Frequency Ratio, PSE |

Trainer ghép các hàng axis thành **một vector/window**. Ankle X/Y/Z tạo 42 features. Chọn một axis tạo 14 features.

Trainer cũng đọc export wide của bản `data_prepare` cũ: một hàng/window, với tên như `Ankle_Forward_RMS`. Không trộn schema long của core gốc với schema wide của bản module trong cùng lần train vì công thức feature khác nhau. Mọi dataset ghép chung cần cùng bộ channels và cấu hình tương thích.

Subject, Recording, Window ID, timestamp, label và FOG Ratio không được dùng làm features. `train_fog.py` không tự lọc tín hiệu hay tính feature từ raw TXT.

## 7. Pipeline train SVM

1. Đọc CSV/NPZ và metadata; kiểm tra window trùng, thiếu axis, NaN/Inf và cấu hình.
2. Tạo `X` kích thước `(N_windows, N_features)` và `y` kích thước `(N_windows,)`.
3. Chia train/test theo **Subject_ID**; một subject chỉ thuộc một tập.
4. Fit StandardScaler trên train, sau đó fit SVM trên features đã chuẩn hóa.
5. Nếu bật tuning, tìm C/gamma bằng grouped CV chỉ trong train; scaler được fit riêng trong mỗi fold.
6. Dự đoán test bằng scaler và SVM đã fit.
7. Tính metrics theo window và lưu pipeline.

Mặc định dùng `SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=False)`. Class weight tăng mức phạt cho lớp ít mẫu; không tạo thêm mẫu FOG. Test-size 0.25 là tỷ lệ subject, không phải tỷ lệ windows.

Cần ít nhất 2 subjects, và train/test đều có cả FOG và Non-FOG. S10R01 đơn lẻ không đủ cho thiết kế đánh giá này. Chương trình không tự thử nhiều seed rồi chọn kết quả test tốt nhất.

## 8. Lệnh train trên Windows

Các lệnh dùng feature trong `prepared_data/`; output đặt ở thư mục riêng để tránh lẫn file kết quả vào input.

### SVM RBF mặc định

```powershell
.\.venv\Scripts\python.exe train_fog.py --input prepared_data --output svm_output
```

### Chỉ định test subjects

```powershell
.\.venv\Scripts\python.exe train_fog.py --input prepared_data --output svm_output --test-subjects S02 S08
```

S02/S08 là ví dụ; thay bằng subjects thực sự có trong dataset và theo thiết kế thí nghiệm của bạn.

### Tuning C/gamma trong train

```powershell
.\.venv\Scripts\python.exe train_fog.py --input prepared_data --output svm_tuned --kernel rbf --tune --cv-folds 3 --test-subjects S02 S08
```

Grid C: `[0.1, 1, 10, 100]`; gamma: `['scale', 0.01, 0.1]`. Chọn theo **F1 FOG** trung bình trong CV. CV 3 folds cần ít nhất 3 train subjects, và mỗi phần fit/validation phải có cả hai lớp. Nếu điều kiện không đạt, trainer báo lỗi; không tự thay đổi thiết kế CV.

### SVM kernel linear

```powershell
.\.venv\Scripts\python.exe train_fog.py --input prepared_data --output svm_linear --kernel linear --C 1
```

Đây là `SVC(kernel='linear')`, không phải `LinearSVC`. Nếu bật tuning cho linear, chỉ tìm C.

### Nhiều input hoặc một file batch

```powershell
.\.venv\Scripts\python.exe train_fog.py --input prepared_data/S01R01_features.npz prepared_data/S02R01_features.npz prepared_data/S03R01_features.npz --output svm_output
.\.venv\Scripts\python.exe train_fog.py --input prepared_data/fog_batch.npz --output svm_output
```

Khi đọc một thư mục, chỉ xét CSV/NPZ trực tiếp trong thư mục đó, không duyệt thư mục con. CSV/NPZ cùng stem chỉ lấy NPZ; không đưa cùng window trong nhiều export vào train.

### Các tùy chọn

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--input` | Bắt buộc | Một/nhiều feature files hoặc thư mục |
| `--output` | `svm_output` | Thư mục kết quả |
| `--kernel` | `rbf` | `rbf` hoặc `linear` |
| `--C` / `--c` | `1` | Mức phạt của SVM |
| `--gamma` | `scale` | `scale`, `auto` hoặc số >0; dùng với RBF |
| `--test-size` | `0.25` | Tỷ lệ subject test |
| `--test-subjects` | Không đặt | Danh sách test subjects, thay thế test-size |
| `--seed` | `42` | Seed của split/CV |
| `--tune` | Tắt | Bật grid search trên train |
| `--cv-folds` | `3` | Số folds khi tune |
| `--n-jobs` | `1` | Số fits CV đồng thời |
| `--cache-size` | `512` | Kernel cache MiB mỗi SVC worker |

RBF và grid search có thể chạy lâu khi có nhiều windows. Tăng n-jobs làm tăng số fits đồng thời và lượng RAM sử dụng. Khi bật `--tune`, grid nêu trên quyết định C/gamma thay cho giá trị cố định truyền bằng `--C`/`--gamma`.

## 9. Evaluation metrics

FOG là positive class `1`; Non-FOG là negative class `0`.

| Thực tế / Dự đoán | Non-FOG | FOG |
|---|---:|---:|
| Non-FOG | TN | FP |
| FOG | FN | TP |

| Khóa JSON | Công thức / ý nghĩa | Tên tương đương |
|---|---|---|
| `sensitivity` | TP / (TP + FN) | Recall FOG |
| `specificity` | TN / (TN + FP) | Recall Non-FOG |
| `ppv` | TP / (TP + FP) | Precision FOG |
| `npv` | TN / (TN + FN) | Precision Non-FOG |
| `accuracy` | (TP + TN) / tổng mẫu | Accuracy |
| `youden_index` | Sensitivity + Specificity − 1 | Youden J |
| `fog_f1` | 2TP / (2TP + FP + FN) | F1 FOG |
| `balanced_accuracy` | (Sensitivity + Specificity) / 2 | Balanced Accuracy |
| `roc_auc` | ROC-AUC từ decision score | Đánh giá qua nhiều ngưỡng |
| `average_precision` | Average Precision từ decision score | Tổng hợp Precision–Recall |

Metrics được lưu dưới dạng số thập phân: 0.85 tương ứng 85%. Youden là chỉ số không đơn vị, từ −1 đến 1; `Youden = 2 × Balanced Accuracy − 1`.

Các khóa sensitivity/specificity/PPV/NPV dùng `null` nếu mẫu số bằng 0; giá trị tương ứng trong CSV để trống. `classification_report` giữ quy ước cũ `zero_division=0`, vì vậy trường hợp không xác định có thể hiển thị 0 trong report đó.

PPV/NPV phụ thuộc tỷ lệ FOG trong tập đánh giá. Không dùng Accuracy riêng để kết luận mô hình tốt khi dữ liệu lệch lớp. Tất cả metrics hiện tại là **window-level**, chưa đo số sự kiện FOG bỏ sót, độ trễ phát hiện hay báo động nhầm mỗi giờ.

Không chọn ngưỡng hoặc tham số theo kết quả test. Bản hiện tại dùng predict mặc định của SVM, không tối ưu ngưỡng bằng Youden. Đánh giá là một subject-held-out split, chưa phải LOSO toàn dataset.

## 10. Các file kết quả

| File | Nội dung |
|---|---|
| `fog_svm.joblib` | Pipeline StandardScaler + SVM, feature order, nhãn, config và metrics |
| `metrics.json` | 10 metrics, confusion counts, classification report, tham số, split subjects và versions |
| `evaluation_metrics.csv` | Bảng 10 metrics cùng định nghĩa |
| `test_predictions.csv` | Nhãn đúng, nhãn dự đoán, FOG decision score cho từng test window |
| `confusion_matrix.csv` | Hàng = thực tế; cột = dự đoán; Non-FOG trước FOG |
| `split_manifest.csv` | Metadata mỗi window và tập train/test |
| `feature_names.json` | Thứ tự features cần dùng khi dự đoán |
| `dataset_info.json` | Input files, schema, config và trạng thái xác minh |
| `cv_results.csv` | Grid search results; chỉ tạo khi tune |
| `cv_subjects.csv` | Subject fit/validation mỗi fold; chỉ tạo khi tune |

`FOG_Decision_Score` là điểm SVM, không phải xác suất. Model đã lưu được fit trên train; không refit trên test.

## 11. Dùng model để dự đoán

```python
import joblib
import pandas as pd

bundle = joblib.load('svm_output/fog_svm.joblib')
features = pd.read_csv('new_windows_wide.csv')

# features phải là bảng một hàng/window, với tên cột đúng như model đã lưu.
X_new = features[bundle['feature_names']].to_numpy(float)
labels = bundle['pipeline'].predict(X_new)
scores = bundle['pipeline'].decision_function(X_new)
```

Nếu dữ liệu mới là bảng long X/Y/Z, phải ghép axis thành vector/window và đặt tên cột như trainer trước khi dùng đoạn ví dụ này. Preprocessing, công thức features và bộ channels phải giống lúc train.

Không chuẩn hóa X_new thêm lần nữa: pipeline tự dùng scaler đã fit. Chỉ load model joblib từ nguồn tin cậy và sử dụng môi trường thư viện tương thích với môi trường train.

## 12. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError` | Cài requirements bằng đúng Python được dùng để chạy app/train |
| Không nhận Subject_ID | Xuất Batch có metadata hoặc giữ tên SxxRxx_features cho single-recording export |
| Dataset chỉ có một lớp | Chuẩn bị nhiều bản ghi có cả FOG và Non-FOG |
| Train/test thiếu một lớp | Thiết kế lại danh sách test subjects; không chọn theo điểm test |
| Config khác nhau | Chuẩn bị các file bằng cùng cấu hình; không trộn core gốc với bản module |
| Thiếu một axis trong window | Xuất lại với bộ channels nhất quán |
| Window trùng | Bỏ export trùng; không truyền cả CSV và NPZ cùng dataset qua hai input riêng |
| Không có window | Kiểm tra Data Selection và độ dài đoạn hợp lệ so với window length |
| CSV bị báo không có feature | Không đưa raw CSV hoặc report vào trainer |
| CV fold thiếu lớp | Điều chỉnh thiết kế folds/subjects hoặc chạy không tune |

## 13. Phạm vi kiểm tra và phiên bản

Đã kiểm tra đọc CSV/NPZ, ghép axis, huấn luyện RBF/linear, grouped tuning, scaler chỉ fit train, metrics và xuất kết quả bằng dữ liệu kiểm thử. GUI PAN đã được kiểm tra bằng thao tác drag mô phỏng trên Raw và Selected Time Domain. S10R01 đã được dùng để kiểm tra luồng chuẩn bị dữ liệu; chưa phải kết quả đánh giá SVM trên toàn DAPHNET.

`requirements.txt` dùng khoảng phiên bản cho dependencies, không phải lockfile tái lập chính xác. Khi train chính thức, lưu phiên bản môi trường cùng model:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment_train.txt
```

## 14. Tài liệu tham khảo

- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
