# FOG Signal Analyzer — Daphnet, SVM, Random Forest và SHAP

## 1. Các file cần có

| File | Chức năng |
|---|---|
| `main.py` | Mở app FOG: dữ liệu, tiền xử lý, đặc trưng và Batch |
| `signal_processing.py` | Xử lý tín hiệu, chia cửa sổ, tính đặc trưng |
| `feature_registry.py` | Khai báo và lựa chọn các nhóm đặc trưng |
| `data_prepare.py` | Chuẩn bị dữ liệu và hỗ trợ xuất Batch |
| `batch_processing.py` | File tương thích nếu bộ mã GUI hiện tại có sử dụng |
| `train_fog.py` | Train SVM/RF, đánh giá và tùy chọn chạy SHAP sau train |
| `shap_fog.py` | Tính SHAP từ mô hình đã train |
| `app_compare.py` | So sánh số liệu, confusion matrix, baseline và SHAP |
| `requirements.txt` | Các thư viện Python cần cài |

Đặt các file chương trình trong cùng thư mục dự án. Với bản gộp v3, không cần `train_fog_svm.py`, `train_rf.py`, `train_fog_shap.py` hoặc chạy lại `install_rf.py`.

## 2. Cài môi trường

### Cài mới

Dùng Python 3.12 64-bit. Kiểm tra Python trên máy:

```powershell
py -0p
```

Chỉ tạo môi trường mới khi dự án chưa có `.venv` đang sử dụng:

```powershell
cd E:\TEEP\FOG_Research
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

### Dự án đã có `.venv`

Giữ môi trường đang dùng, kích hoạt và cài các thư viện còn thiếu theo file:

```powershell
cd E:\TEEP\FOG_Research
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

Nếu PowerShell chặn kích hoạt, có thể gọi trực tiếp Python trong môi trường mà không thay chính sách PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Các lệnh `python ...` phía dưới giả định `.venv` đã được kích hoạt.

### Phiên bản mô hình đã lưu

`requirements.txt` cố định **scikit-learn 1.9.1** và **SHAP 0.54.1**, theo các run SVM/RF đã cung cấp. Các thư viện còn lại dùng khoảng phiên bản; đây không phải bản khóa đầy đủ của một môi trường.

Kiểm tra phiên bản hiện tại:

```powershell
python -c "import sklearn, shap; print('sklearn:', sklearn.__version__); print('shap:', shap.__version__)"
```

Model `.joblib` cần môi trường tương thích với lúc train, đặc biệt cùng phiên bản scikit-learn. Nếu cần mở model được train bằng phiên bản khác 1.9.1, giữ môi trường tương ứng cho model đó; không cài đè requirements này một cách máy móc.

Sau khi môi trường chạy ổn, có thể ghi lại đầy đủ phiên bản để tái lập:

```powershell
python -m pip freeze > requirements-lock.txt
```

## 3. Mở app FOG và chuẩn bị dữ liệu

```powershell
python main.py
```

Quy trình:

1. Nạp các file Daphnet `.txt`; kiểm tra bệnh nhân, cảm biến và nhãn.
2. Kiểm tra ánh xạ nhãn: nhãn gốc `1` thành Non-FOG `0`, nhãn gốc `2` thành FOG `1`; loại nhãn gốc `0` ngoài thí nghiệm.
3. Chọn đoạn dữ liệu và các vùng cần loại trừ.
4. Thiết lập preprocessing: lọc, detrend, resample.
5. Thiết lập cửa sổ, overlap và quy tắc gán nhãn.
6. Chọn cảm biến, trục và đặc trưng ở phần Feature Selection/Batch tương ứng.
7. Xuất Batch thành NPZ, ví dụ `fog_batch.npz`.

Để đối chiếu với các run hiện tại: dùng cửa sổ **2 giây**, overlap **50%**, cùng nhóm đặc trưng và cùng vùng loại trừ. Nếu cấu hình là 443 đặc trưng thì kiểm tra số cột trước khi train.

```powershell
python -c "import numpy as np; z=np.load('fog_batch.npz', allow_pickle=False); print('X:', z['X'].shape); print('y:', z['y'].shape); print('features:', len(z['feature_names'])); z.close()"
```

Trainer nhận đặc trưng đã chuẩn bị, không nhận trực tiếp raw `.txt`. NPZ chứa mỗi cửa sổ ở một dòng, nhãn và metadata bệnh nhân/bản ghi/thời gian. Không chuẩn hóa toàn dataset trước LOSO; imputer/scaler được fit riêng trong training fold.

Lưu riêng NPZ cho mỗi cấu hình, ví dụ `fog_2s_all.npz`, `fog_3s_all.npz`. Không ghi đè NPZ của một run nếu còn cần tính SHAP hoặc tái lập run đó.

## 4. Train baseline SVM

```powershell
python train_fog.py --model svm --input fog_batch.npz --output svm_2s_baseline --evaluation loso --C 1 --gamma scale --decision-threshold 0
```

- SVM RBF, `C=1`, `gamma=scale`, class weight balanced.
- Pipeline: imputer median có missing indicators → StandardScaler → SVM.
- FOG khi decision score **lớn hơn 0**.
- Với 10 bệnh nhân, LOSO lặp 10 lần: 9 người train, 1 người test.
- Không có `--tune`/`--threshold-tuning` thì đây là baseline tham số cố định.

SVM là model mặc định, nên các lệnh cũ không có `--model svm` vẫn dùng được.

## 5. Train baseline Random Forest

```powershell
python train_fog.py --model rf --input fog_batch.npz --output rf_2s_baseline --evaluation loso --rf-n-estimators 200 --rf-max-depth none --rf-min-samples-leaf 1 --decision-threshold 0.5
```

- Pipeline: imputer median giữ số chiều → Random Forest; không scaler và không thêm missing indicators.
- FOG khi xác suất lớp FOG **lớn hơn 0,5**; bằng ngưỡng được tính Non-FOG.
- Xác suất RF không mặc nhiên là xác suất đã hiệu chuẩn.
- Nhánh RF hiện hỗ trợ NPZ và LOSO. Nhánh SVM có thêm CSV và holdout theo bệnh nhân.

| Tùy chọn RF | Mặc định | Ý nghĩa |
|---|---:|---|
| `--rf-n-estimators` | 200 | Số cây |
| `--rf-max-depth` | `none` | Độ sâu tối đa |
| `--rf-min-samples-leaf` | 1 | Số mẫu tối thiểu ở lá |
| `--rf-max-features` | `sqrt` | Số/tỉ lệ đặc trưng xét ở mỗi lần chia; hỗ trợ `sqrt`, `log2`, `none` hoặc tỉ lệ `(0,1]` |
| `--class-weight` | `balanced` | Trọng số lớp; hỗ trợ `balanced_subsample` và `none` |
| `--n-jobs` | -1 | Số tác vụ RF; `-1` sử dụng các CPU có sẵn |
| `--seed` | 42 | Seed để tái lập |

Mỗi lần train dùng một tên output mới. Không trộn các lần chạy vào cùng thư mục kết quả.

## 6. Tuning model và ngưỡng bằng inner CV

SVM:

```powershell
python train_fog.py --model svm --input fog_batch.npz --output svm_2s_tuned --evaluation loso --tune --threshold-tuning --threshold-method exact --cv-folds 3 --min-sensitivity 0.85 --min-specificity 0.85
```

RF:

```powershell
python train_fog.py --model rf --input fog_batch.npz --output rf_2s_tuned --evaluation loso --tune --threshold-tuning --cv-folds 3 --min-sensitivity 0.85 --min-specificity 0.85
```

Trong mỗi outer LOSO, chỉ dữ liệu của nhóm train được dùng cho inner SGKF. Tham số/ngưỡng được chọn trong inner CV, sau đó model được fit lại trên outer train và đánh giá ở người test.

- `--tune`: chọn tham số model.
- `--threshold-tuning`: chọn ngưỡng từ các dự đoán inner OOF.
- `--cv-folds 3`: số fold inner.
- Các mục tiêu 0,85 không bảo đảm đạt trên người test. Xem thông tin `target_met`/fallback trong kết quả.
- RF grid mặc định: số cây `[200]`, độ sâu `[none,12]`, min leaf `[1,5]`.
- Đổi RF grid bằng `--rf-n-estimators-grid`, `--rf-max-depth-grid`, `--rf-min-samples-leaf-grid`.

Để so SVM/RF, dùng cùng NPZ và cùng outer folds; xem cả sensitivity, specificity, FP và FN. Không chọn ngưỡng trực tiếp trên outer test rồi báo cùng test như đánh giá độc lập.

## 7. Tính SHAP cho model đã train

SHAP giải thích model đã train, không train lại model. `--results` cần thư mục kết quả đã giải nén, chứa các thư mục `fold_*` và model tương ứng.

### Chạy thử SVM ở một fold

```powershell
python shap_fog.py --input fog_batch.npz --results svm_2s_baseline --folds S01 --background 24 --explain 20 --nsamples 256 --sampling errors-balanced
```

### SVM: background 50, giải thích 150 cửa sổ mỗi fold

```powershell
python shap_fog.py --input fog_batch.npz --results svm_2s_baseline --background 50 --explain 150 --nsamples 256 --sampling errors-balanced --overwrite
```

### RF: background 50, giải thích 150 cửa sổ mỗi fold

```powershell
python shap_fog.py --input fog_batch.npz --results rf_2s_baseline --background 50 --explain 150 --sampling errors-balanced --overwrite
```

Thay `--results` đúng tên thư mục của bạn, ví dụ `window_2s_all_features` hoặc `rf_2s`.

| Tham số | Ý nghĩa |
|---|---|
| `--background 50` | Tối đa 50 cửa sổ thuộc train của mỗi fold; chọn theo bệnh nhân × nhãn |
| `--explain 150` | Tối đa 150 cửa sổ thuộc test của mỗi fold |
| `--sampling errors-balanced` | Phân bổ giữa TP/FP/FN/TN có tồn tại; bù sang nhóm còn mẫu nếu một nhóm thiếu |
| `--sampling uniform` | Chọn ngẫu nhiên đều các cửa sổ test |
| `--nsamples 256` | Ngân sách xấp xỉ KernelExplainer; chỉ dùng với SVM |
| `--folds S01 S06` | Chỉ tính các fold được chọn |
| `--overwrite` | Tính lại SHAP cũ; giữ model và kết quả train |

Đây là số cửa sổ, không phải phần trăm. Toàn bộ cửa sổ test vẫn được dùng để đánh giá model; chỉ một tập con được SHAP giải thích. Dùng đúng NPZ và đúng thứ tự `--input` của lần train. Run v3 kiểm tra SHA256 NPZ.

SVM tự dùng **KernelExplainer**, giải thích `pipeline.decision_function`. RF tự dùng **TreeExplainer interventional**, giải thích xác suất FOG lớp 1. Không cộng hoặc so trực tiếp độ lớn SHAP giữa hai đơn vị này.

RF không dùng `nsamples` hay l1 sparse selection. SVM mặc định của công cụ dùng `num_features(20)`; tăng background/instances không tự bỏ giới hạn này. SHAP bằng 0 không chứng minh feature vô ích.

Với errors-balanced, importance dùng trọng số để bù tỉ lệ lấy mẫu. Mật độ điểm beeswarm vẫn phản ánh mẫu đã chọn. Background được cân bằng theo người/nhãn nên baseline không nhất thiết phản ánh tần suất FOG tự nhiên.

### Vì sao KernelExplainer có thể chạy lâu?

Tăng từ background 24 / explain 20 lên 50 / 150 ở cùng `nsamples=256` làm khối lượng đánh giá model tăng xấp xỉ **15,6 lần**. Tùy CPU và số support vectors, một fold có thể mất nhiều giờ.

Bản hiện tại ẩn thanh tiến độ nội bộ (`silent=True`), chỉ báo khi hoàn tất fold. File SHAP mới chỉ được lưu sau khi tính xong fold; các file cũ còn hiện diện trong lúc tính không đồng nghĩa lượt mới đã xong.

Có thể kiểm tra CPU/RAM của `python.exe` trong Task Manager. Nếu dừng bằng Ctrl+C, model và kết quả train vẫn còn; phần SHAP đang tính trong fold chưa được lưu sẽ cần tính lại. Không chạy hai tiến trình SHAP cùng ghi vào một thư mục kết quả.

## 8. Train rồi tự chạy SHAP bằng một lệnh

SVM, cấu hình chạy thử:

```powershell
python train_fog.py --model svm --input fog_batch.npz --output svm_2s_auto_shap --evaluation loso --shap --shap-background 24 --shap-explain 20 --shap-nsamples 256 --shap-sampling errors-balanced
```

RF, cấu hình 50/150:

```powershell
python train_fog.py --model rf --input fog_batch.npz --output rf_2s_auto_shap --evaluation loso --rf-n-estimators 200 --shap --shap-background 50 --shap-explain 150 --shap-sampling errors-balanced
```

Có thể thêm `--shap-folds S01` để chỉ tính SHAP cho S01 sau khi train hết LOSO. Auto SHAP yêu cầu lưu model từng fold. Nếu train đã xong nhưng SHAP lỗi, sửa nguyên nhân và chạy `shap_fog.py`; không cần train lại.

## 9. Xem kết quả

```powershell
python app_compare.py
```

Nạp một hoặc nhiều thư mục/ZIP kết quả, chọn baseline rồi xem các tab: Bảng số liệu, Chỉ số, Confusion matrix, So với baseline, SHAP.

Trong tab SHAP, chọn run, fold, TP/FP/FN/TN và biểu đồ Importance, Beeswarm, Waterfall hoặc độ ổn định giữa các fold. Không cần tính lại SHAP mỗi lần mở app.

Các file SHAP lưu trong `fold_XX_Sxx/shap/`:

| File | Nội dung |
|---|---|
| `manifest.json` | Trạng thái hoàn tất, cấu hình và nguồn mô hình |
| `values.npz` | SHAP values, raw features, base value và đầu ra model |
| `importance.csv` | Thứ hạng mean absolute SHAP |
| `samples.csv` | Những cửa sổ được giải thích và nhóm TP/FP/FN/TN |
| `background_rows.csv` | Những cửa sổ train được chọn làm background |

Để gửi kết quả sau khi SHAP chạy xong, nén lại thư mục hiện tại, ví dụ:

```powershell
Compress-Archive -Path .\svm_2s_baseline -DestinationPath .\svm_2s_baseline_SHAP.zip -Force
```

## 10. Lỗi thường gặp

| Thông báo / hiện tượng | Cách xử lý |
|---|---|
| `No module named 'shap'` | Kích hoạt đúng `.venv`, chạy `python -m pip install -r requirements.txt` |
| `No module named 'pywt'` | Thư viện pip tên `PyWavelets`; cài requirements |
| `--results cần thư mục đã giải nén` | Giải nén ZIP; trỏ tới thư mục trực tiếp chứa `fold_*` |
| `Đã có SHAP, bỏ qua` | Kết quả fold đã có; muốn thay đổi cấu hình thì thêm `--overwrite` |
| NPZ/manifest/score/SHA256 không khớp | Lấy lại đúng NPZ của lần train và đúng thứ tự input; không dùng NPZ của cấu hình khác |
| Cảnh báo khác phiên bản sklearn | Dùng đúng môi trường/phiên bản đã train model |
| Output đã có dữ liệu | Đặt tên output mới cho lần train mới |
| SVM vẫn báo NaN theo schema cũ | Kiểm tra đã chép đúng `train_fog.py` bản gộp v3; không dùng lại trainer cũ |
| SHAP lâu mà chưa in thêm | Kiểm tra CPU/RAM, xem mục thời gian KernelExplainer; không kết luận treo chỉ từ console |

## 11. Cập nhật từ GitHub

Trong thư mục đã clone và liên kết repository:

```powershell
git status
git pull --ff-only
```

Nếu có sửa đổi local, commit hoặc giữ chúng riêng trước khi cập nhật. Nếu Git báo nhánh phân kỳ hoặc xung đột, xử lý trước khi tiếp tục. Sau khi pull, nếu `requirements.txt` thay đổi thì cài lại trong đúng môi trường và kiểm tra tương thích với model đang lưu.

Không đưa `.venv/`, `__pycache__/` và các thư mục backup lên repository. Giữ NPZ/model/kết quả theo nhu cầu tái lập của dự án.

## 12. Phạm vi hiện tại

- Wavelet trong app hiện tại là **DWT**, không phải CWT.
- Chưa bổ sung learning curve hay tự động bỏ đặc trưng.
- SHAP trên outer test dùng để phân tích; muốn chọn đặc trưng và đánh giá khách quan phải thực hiện lựa chọn trong training/inner CV.

Tài liệu tham khảo: [pip requirements](https://pip.pypa.io/en/stable/reference/requirements-file-format/), [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
