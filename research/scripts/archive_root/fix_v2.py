import json
path = 'research/experiments/dashboard_runs/v2_enhanced_20260411.json'
with open(path) as f:
    data = json.load(f)

if 'changes' not in data:
    data['changes'] = {
        'what': data.get('description', ''),
        'why': 'Kiểm tra tác động của feature engineering và log1p transform (giảm skewness) lên các mô hình. Thêm các mô hình tuyến tính làm baseline mới.',
        'result': 'Các mô hình tuyến tính (RidgeCV, LassoCV) đạt kết quả tốt với feature mới. LightGBM được hưởng lợi từ rolling & lag features.',
        'conclusion': 'Feature engineering và target transform là cần thiết để xử lý biến động cực đoan của PM2.5. Các mô hình truyền thống hưởng lợi lớn từ bộ tính năng này.'
    }

with open(path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
