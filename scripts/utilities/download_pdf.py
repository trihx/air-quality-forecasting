import urllib.request
import urllib.error
import re
import os
import ssl

# Bỏ qua xác thực SSL (Hữu ích khi một số mirror của Sci-hub bị lỗi chứng chỉ HTTPS)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Danh sách các tên miền dự phòng của Sci-hub (cập nhật 2026)
SCI_HUB_MIRRORS = [
    'https://sci-hub.ru',
    'https://sci-hub.st',
    'https://sci-hub.se',
    'https://sci-hub.wf',
    'https://sci-hub.ee'
]

# Danh sách 14 DOIs từ 14 bài báo
DOIS = [
    # Quốc tế (10 papers)
    "10.1016/j.chemosphere.2022.136180", # [39] Zhang Jiaxuan
    "10.1016/j.envres.2024.120363",      # [40] Shetty et al.
    "10.3390/app14125062",               # [41] Tian et al.
    "10.1016/j.ecoinf.2023.102067",      # [42] Gokul et al.
    "10.1007/s44163-024-00184-7",        # [43] Inam et al.
    "10.1016/j.jclepro.2023.139278",     # [44] Shakya et al.
    "10.17977/um018v5i12022p53-66",      # [45] Pranolo et al.
    "10.3390/atmos14060968",             # [46] Kim et al.
    "10.1016/j.enceco.2025.07.001",      # [47] Patel et al.
    "10.3390/ijgi14020042",              # [48] Kaveh et al.
    
    # Việt Nam (4 papers có DOI)
    "10.3846/jeelm.2024.22361",          # [49] Nguyễn T.N.T. et al.
    "10.52939/ijg.v19i12.2975",          # [50] Hải P.H. et al.
    "10.1016/j.uclim.2022.101315",       # [51] Rakholia et al.
    "10.1016/j.atmosenv.2023.120161"     # [52] Tran et al.
] Zhang & Li
    "10.4209/aaqr.220355",               # [40] Zhao et al.
    "10.1016/j.atmosenv.2023.119852",    # [41] Bi et al.
    "10.1007/978-981-99-6547-2",         # [42] Bhardwaj et al.
    "10.3390/s24051523",                 # [43] Park & Kim
    "10.1016/j.scitotenv.2024.170245",   # [44] Tsai et al.
    "10.1016/j.envres.2024.120363",      # [45] S-MESH Team
    "10.3390/app14125062",               # [46] Lee et al.
    "10.1016/j.envpol.2024.125630",      # [47] Shen et al.
    "10.1007/s40808-025-02214-5",        # [48] Yekenov et al.
    
    # 🇻🇳 Việt Nam (4 papers có DOI)
    "10.3846/jeelm.2024.22361",          # [49] Nguyễn T.N.T.
    "10.52939/ijg.v19i12.2975",          # [50] Hải P.H.
    "10.4209/aaqr.230155",               # [51] Trần V.A.
    "10.3390/atmos13111822"              # [52] Võ T.T.M.
]

def download_paper(doi, output_dir="literature_pdfs"):
    # Tạo thư mục chứa file
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuyển đổi dấu '/' trong DOI thành '_' để làm tên file hợp lệ
    safe_doi = doi.replace('/', '_')
    filename = os.path.join(output_dir, f"{safe_doi}.pdf")
    
    if os.path.exists(filename):
        print(f"✅ Đã tải: {filename}")
        return True

    print(f"\n📥 Đang tìm kiếm DOI: {doi}")
    for mirror in SCI_HUB_MIRRORS:
        url = f"{mirror}/{doi}"
        print(f"   🔄 Thử truy cập: {mirror}...")
        
        # Fake User-Agent để tránh bị block
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        try:
            response = urllib.request.urlopen(req, context=ctx, timeout=15)
            html = response.read().decode('utf-8', errors='ignore')

            # Regex tìm thẻ iframe hoặc embed chứa link PDF (cấu trúc chuẩn của Sci-hub)
            match = re.search(r'id="pdf".*?src="(.*?)"', html)
            if match:
                pdf_url = match.group(1)
                
                # Xử lý URL tương đối từ Sci-hub
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                elif pdf_url.startswith('/'):
                    pdf_url = mirror + pdf_url

                print(f"   🔗 Tìm thấy link PDF: {pdf_url}")
                
                # Bắt đầu tải file PDF
                pdf_req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                pdf_res = urllib.request.urlopen(pdf_req, context=ctx, timeout=30)
                
                with open(filename, 'wb') as f:
                    f.write(pdf_res.read())
                    
                print(f"   🎉 Thành công! Lưu tại: {filename}")
                return True
            else:
                print(f"   ❌ Không tìm thấy link tải trong HTML (Có thể bị Captcha).")
                
        except Exception as e:
            print(f"   ⚠️ Lỗi kết nối: {e}")
            
    print(f"❌ THẤT BẠI: Đã thử hết các Domain nhưng không tải được {doi}.")
    return False

if __name__ == "__main__":
    print(f"🚀 KHỞI ĐỘNG CÔNG CỤ XUẤT {len(DOIS)} BÀI BÁO SCIENTIFIC BENCHMARK...\n")
    success_count = 0
    
    for doi in DOIS:
        if download_paper(doi):
            success_count += 1
            
    print(f"\n==================================================")
    print(f"📊 KẾT QUẢ TỔNG QUAN:")
    print(f"   - Tải thành công: {success_count}/{len(DOIS)}")
    print(f"   - Lưu tại thư mục: ./literature_pdfs/")
    print(f"==================================================")
