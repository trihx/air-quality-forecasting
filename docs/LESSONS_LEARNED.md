# LESSONS LEARNED

> Ghi lại **MỌI** lỗi gặp phải và cách khắc phục. Đọc file này TRƯỚC KHI implement để tránh lặp lại lỗi cũ.

| Date | Cat | Pattern → Fix | Source |
|------|-----|---------------|--------|
| 2026-04-05 | DEPLOY | Lỗi port 8501 bị chiếm → `lsof -ti :8501 \| xargs kill -9` | Streamlit |
| 2026-04-11 | DEPLOY | `local_sources_watcher` + `transformers` làm app chậm 50% → `fileWatcherType = "none"` | Streamlit |
| 2026-04-12 | BUILD | Stacking kém hiệu quả với models tương đồng → Dùng Weighted Ensemble | Ensemble |
| 2026-04-12 | TEST | Cần phân biệt rõ evaluation policy (test-on-real vs incl-imputed) khi so sánh MAE | Eval |
| 2026-04-12 | ML | 117 features gây hại cho DL ở h=1 do curse of dimensionality → Dùng PCA/Select | Feature |
| 2026-04-12 | ML | Log transform effect tùy model/horizon → Test riêng rẽ, không áp dụng mù quáng | Transform |
| 2026-04-12 | ML | CV feature (std/mean) phát nổ khi mean≈0 → Cần safeguard clamp mean ≥ 1 | Feature |
| 2026-04-12 | ML | TFT tăng feature 22x cần tăng hidden_dim tương ứng → Nếu không sẽ bị bottleneck | TFT |
| 2026-04-12 | ML | LightGBM + PyTorch + M1 = crash OMP → `n_jobs=1` + `KMP_DUPLICATE_LIB_OK=TRUE` | Env |
| 2026-04-12 | ML | BẤT KỲ transform nào (STL/PCA) đều fit trên TRAIN ONLY → Tránh Look-Ahead Bias | Data |
| 2026-04-12 | ML | Fourier đã khử mùa vụ → Deseasonalizing explicit = redundant & add noise | Data |
| 2026-04-12 | ML | Outlier IQR*3 quá gắt cho PM2.5 (chặt mất unhealthy) → Dùng domain [0,500] | Data |
| 2026-04-19 | DEPLOY | "1 task running" gây lỗi → LUÔN kill background tasks sau debug/audit | Workflow |
| 2026-04-19 | DATA | Sensor ở Sa Đéc, Đồng Tháp → Đừng nhầm là Cần Thơ (chỉ là tên trường) | Context |
| 2026-04-28 | DEPLOY | Snapshot v7_pre_cqr thừa → Xóa, đổi v8 thành v7, chỉ lưu khi có experiment MỚI | Versioning |
| 2026-04-28 | DEPLOY | "No label associated with form field" / Summarizer API → Framework issue, IGNORE | DevTools |
| 2026-04-28 | UI | Hardcode HEX color cho Streamlit card gây lỗi tương phản → Dùng Theme Inversion | CSS |
| 2026-04-29 | UI | Plotly SVG text KHÔNG nhận CSS Var, `get_option` trả None → Dùng mã Kẽm 500 (`#71717A`) | Plotly |
| 2026-04-29 | UI | Hardcode style rải rác → Gom tất cả rcParams, bbox, Plotly_template vào VTF | Theme |
| 2026-04-29 | BUILD | Early stopping block NGOÀI epoch loop = không hoạt động → LUÔN verify indent level | DL |
| 2026-04-29 | BUILD | results dict trong for arch_name bị ghi đè nếu assign nằm sai indent → LSTM ghi đè GRU | DL |
| 2026-04-29 | BUILD | `_load_latest("lightgbm")` glob match `lightgbm_preds_*` → Filter by timestamp digit | IO |
| 2026-04-30 | UI | Hardcode text/insight trong UI làm break layout khi update, khó maintain → Zero-Hardcode: Tách ra `dashboard_content.json` và gọi bằng `ContentManager` | Architecture |
| 2026-04-30 | TEST | JSON content có thể chứa số liệu cũ (stale MASE/MAE) SAI với snapshot thực → LUÔN chạy audit script cross-ref JSON vs snapshot data trước khi phát hành | Content Audit |
| 2026-04-30 | API | Python `.format()` placeholder `{len(snapshots)}` ≠ `{len}` → KeyError runtime. JSON template phải dùng simple key names | Template |
| 2026-04-30 | BUILD | ContentManager chưa init trong function mới → NameError. LUÔN verify import + init khi thêm page function | Runtime |
| 2026-04-30 | ARCH | SKILL.md quá dài (>900 dòng) làm loãng context agent → Tối ưu: Dùng SKILL.md làm L0 Index Router, dời toàn bộ details sang .agent/guides/ | Architecture |
| 2026-05-02 | TEST | Thesis ghi data split 80/20 nhưng `splitter.py` dùng 80/10/10 → LUÔN `grep` code thực tế (splitter, config) trước khi sửa thesis. Thesis = reference document, code = source of truth | Data Split |
| 2026-05-02 | BUILD | DL/ARIMA/TFT pred files có test set size KHÁC sklearn (688 vs 600) → LUÔN check len() alignment trước khi tính metrics. Dùng tail-alignment hoặc Actuals riêng mỗi source | Metrics |
| 2026-05-02 | ML | MASE cũ dùng out-of-sample naive MAE → SAI chuẩn Hyndman 2006 (in-sample). Kết quả MASE thay đổi đáng kể khi fix | Metrics |
| 2026-05-03 | API | Table `info_cards` không tự update khi restart Docker do volume cache & lifespan check `count == 0` → Chạy tay `seed_info_cards.py` hoặc dùng cờ `FORCE_SEED` | DB Seed |
| 2026-05-03 | UI | `stMarkdownContainer` có z-index cao vẫn bị `stVerticalBlock` đè tooltip → Dùng `:has(.cite-tooltip:hover)` nâng z-index cho TẤT CẢ các thẻ cha (`stColumn`, v.v.) | CSS |
| 2026-05-03 | ARCH | Code đánh giá model bị hardcode tìm best theo MAE và so sánh Baseline `MASE < 1.0` → Gây sai lệch kết quả do `MASE_UNIFIED`. Bắt buộc phải compare trực tiếp theo MASE | Reporting |
| 2026-05-03 | UI | Hardcode `7,742 rows` ở 5+ files (app.py, citations, explainability_hub) — số liệu V7 (1h only). V9 multi-res: 15m ~110K, 30m ~55K, 1h ~27K. LUÔN grep trước khi release | Content Audit |
| 2026-05-03 | UI | Sankey diagram gộp 3 resolutions thành 1 flow → sai logic V9. Cần tách 3 nhánh song song sau bước Clean | Visualization |
| 2026-05-03 | UI | SHAP cache 24h có thể bị overwrite bởi 1h → lag1h xuất hiện top 24h. LUÔN verify cache file per-horizon | Cache |
| 2026-05-03 | UI | Chỉ số trên chart dùng màu trắng → invisible trên Light mode. Dùng Kẽm 500 `#71717A` (VTF standard) | Theme |
| 2026-05-03 | DEPLOY | Khi chuyển máy: Plan/artifacts ở `~/.gemini/antigravity/brain/` KHÔNG đi theo source code → Lưu plan vào `docs/PENDING_PLAN.md` trong project | Portability |
| 2026-05-04 | BUILD | macOS `._*` metadata files (Apple Double resource fork) gây `UnicodeDecodeError` khi glob `*.json` trên Windows → Xóa `._*` + thêm `._*` vào `.gitignore` | Cross-OS |
| 2026-05-04 | BUILD | `open()` thiếu `encoding="utf-8"` trên Windows → Python dùng `cp1252` gây crash. Mọi `open()` text mode PHẢI có `encoding="utf-8"` | Encoding |
| 2026-05-04 | API | APIClient log spam 404 khi API offline (expected fallback behavior) → Thêm `quiet=True` flag cho fallback calls, chỉ log lỗi khi unexpected | API Design |
| 2026-05-04 | DOCS | Trích dẫn (IEEE_REFS) thiếu tính xác thực học thuật → LUÔN thêm `quote` (từ abstract/sách gốc) và `location` (chương/trang) sau khi verify web search, tuyệt đối không hallucinate. | Citation |
| 2026-05-05 | UI | Tr?c Y log scale (Matplotlib/Plotly) t? d?ng ?n di?m c� gi� tr? 0.0 (do log(0) undefined) ? G�y hi?u l?m m?t d? li?u. C?n gi?i h?n du?i (floor e.g. 1e-15) ho?c th�m ghi ch� gi?i th�ch r� tr�n UI | Visualization |
| 2026-05-05 | DATA | Bi?n t?o ra ? tab tru?c (raw_sm) kh�ng t?n t?i trong scope tab sau n?u b? b?c trong if block ? LU�N init bi?n ? parent scope ho?c load l?i data t? ngu?n chu?n x�c (scatter_df) | Streamlit |
| 2026-05-05 | UI | Raw HTML từ hàm cite() hiển thị dưới dạng text thay vì tooltip | Khi chèn HTML động vào st.markdown(), LUÔN PHẢI kèm theo unsafe_allow_html=True, nếu không Streamlit sẽ tự động escape (mặc định là False). | Streamlit |
| 2026-05-05 | UI | Markdown (như **bold**) truyền vào thẻ div HTML không hiển thị đúng mà bị in ra dạng text thô | st.markdown KHÔNG tự parse markdown bên trong các khối HTML thô (raw div). Cần dùng regex (ví dụ: re.sub) để parse text trước khi đưa vào template HTML | Streamlit |
| 2026-05-08 | DATA | Bẫy Outlier Removal: Áp dụng IQR cho PM2.5 (đặc tính fat-tailed) sẽ xóa nhầm các đỉnh ô nhiễm thật sự. BẮT BUỘC dùng Domain Bounds (0-500) theo chuẩn WHO để giữ lại cảnh báo. | Preprocessing |
