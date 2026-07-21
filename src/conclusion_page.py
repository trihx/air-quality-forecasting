"""Page: Kết Luận & Hướng Phát Triển — Thesis conclusion dashboard.

Designed for academic presentation per CTU-QD1799 standards.
Sections:
    1. Research Summary (achievements timeline)
    2. Key Findings (quantitative)
    3. Limitations
    4. Future Directions (with feasibility & impact matrix)
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

from src.frontend.citations import cite, render_references_section


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def page_conclusion(results: dict):
    """Main entry point for the Conclusion & Future Work page."""
    from app import section_header, insight_card

    st.markdown("""
    <h1 style="text-align:center; font-size:1.8rem; margin-bottom:0.2rem;">
        📝 Kết Luận & Hướng Phát Triển
    </h1>
    <p style="text-align:center; opacity:0.6; font-size:0.95rem; margin-bottom:2rem;">
        Tổng hợp kết quả nghiên cứu và đề xuất hướng phát triển — Chương 5 Luận văn Thạc sĩ
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "1. Tổng Hợp Kết Quả",
        "2. Hạn Chế",
        "3. Hướng Phát Triển",
    ])

    with tab1:
        _render_summary(section_header, insight_card)

    with tab2:
        _render_limitations(section_header, insight_card)

    with tab3:
        _render_future_work(section_header, insight_card)

    # ── References section ──
    render_references_section()


# ──────────────────────────────────────────────────────────────
# Tab 1: Tổng Hợp Kết Quả
# ──────────────────────────────────────────────────────────────


def _render_summary(section_header, insight_card):
    """Research summary — what was accomplished."""
    section_header("🎯", "Tóm Tắt Kết Quả Nghiên Cứu")

    st.markdown("""
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1.5rem; border-left: 4px solid #00D4AA; margin-bottom: 1.5rem;">
        <div style="font-size: 0.95rem; line-height: 1.7;">
            Nghiên cứu đã xây dựng thành công hệ thống dự báo nồng độ bụi mịn PM2.5
            sử dụng dữ liệu cảm biến IoT tại Sa Đéc, Đồng Tháp — khu vực
            <b>chưa có nghiên cứu tiền lệ</b> về giám sát chất lượng không khí bằng
            Machine Learning. Pipeline trải qua <b>9 phiên bản</b> cải tiến liên tục,
            đánh giá <b>30+ mô hình</b> trên 3 độ phân giải × 3 tầm dự báo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key quantitative findings
    section_header("📊", "Các Phát Hiện Chính")

    findings = [
        {
            "title": "1. Độ phân giải 30 phút là tối ưu",
            "detail": (
                f"Thí nghiệm Multi-Resolution Ablation (15m vs 30m vs 1h) chứng minh "
                f"30 phút là điểm cân bằng tối ưu giữa nhiễu vi mô (15m) và "
                f"bẫy autocorrelation (1h) {cite('hyndman2021')}. Ensemble_Weighted_30m đạt "
                f"<b>MASE = 0.382 (6h)</b> {cite('hyndman2006')} và <b>MASE = 0.469 (24h)</b>."
            ),
        },
        {
            "title": "2. Không có mô hình duy nhất tốt nhất",
            "detail": (
                f"Autocorrelation giảm dần: 0.99 (1h) → 0.85 (6h) → 0.45 (24h) {cite('joseph2022')}. "
                f"Ở 1h, Persistence gần như bất khả chiến bại (chỉ GRU_15m phá vỡ, "
                f"MASE=0.667) {cite('hyndman2006')}. Ở 6h-24h, Ensemble và ML vượt trội {cite('peixeiro2022')}."
            ),
        },
        {
            "title": "3. Fair Pipeline > Expert Pipeline cho IoT data",
            "detail": (
                f"Deep Learning sử dụng 119 tabular features (Fair) cho kết quả "
                f"tốt hơn DL tự trích xuất từ raw data (Expert) {cite('hyndman2021')}. "
                f"Feature Engineering có giá trị thực tế cho dữ liệu IoT thưa."
            ),
        },
        {
            "title": "4. SHAP giải mã thành công cơ chế dự báo",
            "detail": (
                f"<code>pm25_lag_1</code> chi phối ở horizon 1h (autocorrelation) {cite('lundberg2017')}. "
                f"Fourier features nổi bật ở 24h (chu kỳ ngày/đêm). "
                f"Các biến khí tượng (nhiệt độ, độ ẩm) đóng vai trò phụ trợ {cite('zhang2017')}."
            ),
        },
        {
            "title": "5. Anti-Leakage Pipeline đảm bảo tính toàn vẹn",
            "detail": (
                f"Phát hiện và xử lý 4 nguồn rò rỉ dữ liệu (diff, pct_change, "
                f"ratio, domain knowledge features) {cite('hyndman2021')}. 188+ automated tests. "
                f"Test-on-Real-Only policy (is_imputed == 0) {cite('tashman2000')}."
            ),
        },
    ]

    for f in findings:
        st.markdown(f"""
        <div style="background: var(--secondary-background-color); border-radius: 10px;
                    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
                    border-left: 3px solid #00D4AA;">
            <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem;">
                {f['title']}
            </div>
            <div style="font-size: 0.88rem; opacity: 0.85; line-height: 1.6;">
                {f['detail']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Contribution summary
    section_header("🏆", "Đóng Góp Của Nghiên Cứu")

    c1, c2, c3 = st.columns(3)
    contributions = [
        ("🔬", "Khoa học", "Phương pháp luận Multi-Resolution × Multi-Horizon đầu tiên cho IoT PM2.5 tại ĐBSCL"),
        ("🛡️", "Kỹ thuật", "Pipeline Anti-Leakage 4 tầng + Tiered Imputation + Test-on-Real-Only"),
        ("🖥️", "Ứng dụng", "Dashboard toàn diện: EDA → Train → Evaluate → Explain → Forecast"),
    ]
    for col, (icon, label, desc) in zip([c1, c2, c3], contributions):
        with col:
            st.markdown(f"""
            <div style="background: var(--secondary-background-color); border-radius: 12px;
                        padding: 1.2rem; text-align: center; min-height: 180px;
                        border: 1px solid rgba(0,212,170,0.15);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: 700; color: #00D4AA; margin-bottom: 0.5rem;">
                    {label}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.8; line-height: 1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Tab 2: Hạn Chế
# ──────────────────────────────────────────────────────────────


def _render_limitations(section_header, insight_card):
    """Research limitations — honest scientific assessment."""
    section_header("⚠️", "Hạn Chế Của Nghiên Cứu")

    limitations = [
        {
            "cat": "Dữ liệu & Tiền xử lý",
            "icon": "📉",
            "items": [
                "Data Sparsity: 89 ngày/năm bị 'mù' hoàn toàn (Tháng 2 & 9), giảm khả năng học chu kỳ mùa",
                "Đơn trạm: Chỉ 1 vị trí đo (Sa Đéc), không đại diện cho toàn ĐBSCL",
                "4 biến phụ giới hạn: Thiếu biến khí tượng quan trọng (tốc độ gió, áp suất, lượng mưa)",
                "Outlier Removal Trap: Áp dụng phương pháp loại nhiễu thống kê thuần túy (IQR) cho PM2.5 đã vô tình xóa bỏ các đỉnh ô nhiễm thật (fat-tailed). Bắt buộc sử dụng Domain Bounds (0 - 500) thay thế để giữ nguyên cảnh báo.",
            ],
        },
        {
            "cat": "Mô hình",
            "icon": "🤖",
            "items": [
                "Horizon 1h: Autocorrelation ~0.97 khiến Persistence baseline rất mạnh, chỉ GRU_15m phá vỡ",
                "TFT_1h thất bại: Kiến trúc Transformer không phù hợp dữ liệu autocorrelation cực cao",
                "Batch processing: Chưa có online learning — mô hình không tự cập nhật khi có data mới",
            ],
        },
        {
            "cat": "Hạ tầng",
            "icon": "🔧",
            "items": [
                "Apple Silicon (MPS): Một số PyTorch ops chưa hỗ trợ đầy đủ, phải fallback CPU",
                "Thời gian huấn luyện: 30+ models × 3 resolutions × 3 horizons yêu cầu tài nguyên lớn",
            ],
        },
    ]

    for lim in limitations:
        items_html = "".join(
            f'<li style="margin-bottom: 0.4rem;">{item}</li>'
            for item in lim["items"]
        )
        st.markdown(f"""
        <div style="background: var(--secondary-background-color); border-radius: 10px;
                    padding: 1.2rem; margin-bottom: 1rem;
                    border-left: 4px solid #EF4444;">
            <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem;">
                {lim['icon']} {lim['cat']}
            </div>
            <ul style="font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 1.2rem; opacity: 0.9;">
                {items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Ablation Study: Outlier Removal Trap
    section_header("🧪", "Ablation Study: Bẫy Outlier Removal")
    
    try:
        import json
        import pandas as pd
        
        comp_path = PROJECT_ROOT / "research" / "experiments" / "v10_ablation" / "comparison_table.json"
        if comp_path.exists():
            with open(comp_path, "r", encoding="utf-8") as f:
                ablation_data = json.load(f)
                
            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.05); padding: 1rem; border-left: 3px solid #EF4444; border-radius: 4px; margin-bottom: 1rem;">
                    <span style="font-size: 0.95em;"><b>"False Sense of Accuracy" (Ảo giác chính xác):</b> Thí nghiệm v10 (Ablation) cố tình dùng thuật toán IQR thay cho Domain Bounds. Kết quả: IQR đã cắt mất 66 đợt ô nhiễm nghiêm trọng (> 54 µg/m³). Mô hình v10 trông <b>chính xác hơn (MASE thấp hơn)</b> ở các horizon ngắn, nhưng thực chất đã bị "mù" trước các đợt bùng phát ô nhiễm thật sự.</span>
                </div>
                """, unsafe_allow_html=True
            )
            
            rows = []
            for h, models_data in ablation_data.items():
                for model_name, metrics in models_data.items():
                    if "v9_mase" in metrics:
                        delta = metrics["delta_mase"]
                        note = metrics.get("note", "")
                        if note == "FALSE ACCURACY":
                            status = "🚨 Ảo giác (MASE giảm ảo)"
                        else:
                            status = "✅ Domain tốt hơn"
                            
                        rows.append({
                            "Horizon": h,
                            "Mô hình": model_name,
                            "v9 MASE (Domain - Đúng)": f"{metrics['v9_mase']:.3f}",
                            "v10 MASE (IQR - Lỗi)": f"{metrics['v10_mase']:.3f}",
                            "Δ MASE": f"{delta:+.3f}",
                            "Đánh giá": status
                        })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            fig_path = PROJECT_ROOT / "research" / "figures" / "ablation_outlier_impact.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Biểu đồ MASE: Ở horizon 1h, mô hình lỗi (v10) có vẻ sai số thấp hơn, tạo ra ảo giác an toàn.")
                
    except Exception as e:
        st.info("💡 Chạy script `v10_ablation_compare.py` để xem kết quả Ablation Study.")

    # External data analysis
    section_header("🌍", "Phân Tích Dữ Liệu Ngoại Lai (Open-Meteo)")

    insight_card(
        "📋 Kết quả thử nghiệm thu thập dữ liệu ngoại lai",
        "Đã thu thập 20.520 dòng dữ liệu từ Open-Meteo API (ERA5 Reanalysis + CAMS PM2.5) "
        "cho các khoảng trống của cảm biến IoT. Tuy nhiên, phát hiện <b>bias hệ thống</b> "
        "nghiêm trọng giữa hai nguồn:<br>"
        "• Nhiệt độ: IoT ~29°C vs Open-Meteo ~27°C (bias ~2°C — do đo indoor vs outdoor)<br>"
        "• Độ ẩm: IoT ~75.6% vs Open-Meteo ~80.8% (bias ~5%)<br>"
        "• PM2.5: IoT ~13.7 vs CAMS ~22.2 µg/m³ (<b>bias ~62%</b> — cảm biến vs mô phỏng vệ tinh)<br><br>"
        "**Quyết định:** Không merge vào pipeline chính để tránh distribution shift. "
        "Lưu trữ tại <code>dataset/external/</code> làm tài liệu tham khảo cho nghiên cứu tiếp theo.",
        card_type="warning",
    )


# ──────────────────────────────────────────────────────────────
# Tab 3: Hướng Phát Triển
# ──────────────────────────────────────────────────────────────


def _render_future_work(section_header, insight_card):
    """Future research directions with feasibility assessment."""
    section_header("🚀", "Đề Xuất Hướng Phát Triển")

    st.markdown("""
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1rem 1.2rem; margin-bottom: 1.5rem;
                border-left: 4px solid #00D4AA; font-size: 0.9rem; line-height: 1.6;">
        Các hướng phát triển được đề xuất dựa trên kết quả thực nghiệm và hạn chế
        đã xác định. Mỗi hướng được đánh giá theo <b>Mức độ khả thi</b> (dựa trên
        hạ tầng hiện có) và <b>Tác động kỳ vọng</b> (dựa trên literature review).
    </div>
    """, unsafe_allow_html=True)

    future_directions = [
        {
            "id": "FD-1",
            "title": "Tích hợp dữ liệu ngoại lai (External Data Fusion)",
            "icon": "🌐",
            "feasibility": "Trung bình",
            "impact": "Cao",
            "f_score": 3,
            "i_score": 5,
            "desc": (
                "Sử dụng kỹ thuật Domain Adaptation hoặc Bias Correction "
                "để hiệu chỉnh phân phối giữa dữ liệu IoT cục bộ và "
                "dữ liệu reanalysis (Open-Meteo ERA5). Xây dựng mô hình "
                "hồi quy trung gian để map Open-Meteo → IoT scale trước khi merge."
            ),
            "prereq": "Đã có: dataset/external/open_meteo_missing_periods.csv (20,520 rows)",
        },
        {
            "id": "FD-2",
            "title": "Multi-Station Network",
            "icon": "📡",
            "feasibility": "Thấp",
            "impact": "Rất cao",
            "f_score": 2,
            "i_score": 5,
            "desc": (
                "Mở rộng mạng lưới cảm biến IoT ra nhiều vị trí trong ĐBSCL "
                "(Cần Thơ, Long Xuyên, Vĩnh Long). Áp dụng Spatial-Temporal GNN "
                "hoặc ConvLSTM để học tương quan không gian giữa các trạm."
            ),
            "prereq": "Cần: Phần cứng IoT bổ sung + triển khai thực địa",
        },
        {
            "id": "FD-3",
            "title": "Online Learning & Model Update",
            "icon": "🔄",
            "feasibility": "Cao",
            "impact": "Cao",
            "f_score": 4,
            "i_score": 4,
            "desc": (
                "Chuyển từ batch processing sang online/incremental learning. "
                "Mô hình tự cập nhật khi nhận data mới từ cảm biến, phát hiện "
                "concept drift và retrain tự động. Phù hợp cho triển khai production."
            ),
            "prereq": "Đã có: Pipeline v9 + Docker architecture + streaming data",
        },
        {
            "id": "FD-4",
            "title": "Kiến trúc CNN-BiLSTM-Attention",
            "icon": "🧠",
            "feasibility": "Cao",
            "impact": "Trung bình",
            "f_score": 4,
            "i_score": 3,
            "desc": (
                "Kết hợp CNN (trích xuất local patterns) + Bidirectional LSTM "
                "(bối cảnh hai chiều) + Attention (tập trung vào timesteps quan trọng). "
                "Literature cho thấy CNN-BiLSTM-Attn đạt R²=0.96 trên dữ liệu tương tự "
                "(Patel et al., 2025)."
            ),
            "prereq": "Đã có: Fair Pipeline features + training infrastructure",
        },
        {
            "id": "FD-5",
            "title": "Transfer Learning từ Dữ liệu Vệ tinh",
            "icon": "🛰️",
            "feasibility": "Trung bình",
            "impact": "Cao",
            "f_score": 3,
            "i_score": 4,
            "desc": (
                "Pre-train mô hình trên dữ liệu PM2.5 quy mô lớn (CAMS Global, "
                "Copernicus Atmosphere), sau đó fine-tune trên dữ liệu IoT cục bộ. "
                "Giải quyết bài toán thiếu hụt data mà không gây bias trực tiếp."
            ),
            "prereq": "Đã có: Phân tích bias Open-Meteo vs IoT → biết rõ domain gap",
        },
        {
            "id": "FD-6",
            "title": "Tích hợp biến khí tượng mở rộng",
            "icon": "🌦️",
            "feasibility": "Cao",
            "impact": "Trung bình",
            "f_score": 5,
            "i_score": 3,
            "desc": (
                "Bổ sung các biến: tốc độ gió, hướng gió, áp suất khí quyển, "
                "lượng mưa, bức xạ mặt trời từ trạm khí tượng Đồng Tháp. "
                "Granger causality test đã xác nhận ảnh hưởng của khí tượng lên PM2.5."
            ),
            "prereq": "Cần: Liên hệ Đài Khí tượng Thủy văn Đồng Tháp",
        },
    ]

    # Render as cards
    for fd in future_directions:
        f_bar = "█" * fd["f_score"] + "░" * (5 - fd["f_score"])
        i_bar = "█" * fd["i_score"] + "░" * (5 - fd["i_score"])

        # Color based on feasibility
        border_color = (
            "#00D4AA" if fd["f_score"] >= 4
            else "#F59E0B" if fd["f_score"] >= 3
            else "#EF4444"
        )

        st.markdown(f"""
        <div style="background: var(--secondary-background-color); border-radius: 12px;
                    padding: 1.2rem 1.5rem; margin-bottom: 1rem;
                    border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;
                        margin-bottom: 0.5rem;">
                <div style="font-weight: 700; font-size: 1rem;">
                    {fd['icon']} {fd['id']}: {fd['title']}
                </div>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; opacity: 0.9;
                        margin-bottom: 0.8rem;">
                {fd['desc']}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem;
                        font-size: 0.82rem;">
                <div>
                    <span style="opacity: 0.6;">Khả thi:</span>
                    <span style="font-family: monospace; color: {border_color};"> {f_bar}</span>
                    <span style="opacity: 0.5;"> ({fd['feasibility']})</span>
                </div>
                <div>
                    <span style="opacity: 0.6;">Tác động:</span>
                    <span style="font-family: monospace; color: #00D4AA;"> {i_bar}</span>
                    <span style="opacity: 0.5;"> ({fd['impact']})</span>
                </div>
                <div>
                    <span style="opacity: 0.6;">Tiền đề:</span>
                    <span style="opacity: 0.7;"> {fd['prereq']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Priority matrix
    section_header("📋", "Ma Trận Ưu Tiên")

    st.markdown("""
    <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem;">
        <thead>
            <tr style="border-bottom: 2px solid rgba(0,212,170,0.3);">
                <th style="text-align: left; padding: 0.6rem;">Mã</th>
                <th style="text-align: left; padding: 0.6rem;">Hướng phát triển</th>
                <th style="text-align: center; padding: 0.6rem;">Khả thi</th>
                <th style="text-align: center; padding: 0.6rem;">Tác động</th>
                <th style="text-align: center; padding: 0.6rem;">Ưu tiên</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-3</td>
                <td style="padding: 0.5rem;">Online Learning</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center;"><span style="background: #00D4AA; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P1</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-4</td>
                <td style="padding: 0.5rem;">CNN-BiLSTM-Attention</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center;"><span style="background: #00D4AA; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P1</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-1</td>
                <td style="padding: 0.5rem;">External Data Fusion</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center;"><span style="background: #F59E0B; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P2</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-5</td>
                <td style="padding: 0.5rem;">Transfer Learning</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center;"><span style="background: #F59E0B; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P2</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-6</td>
                <td style="padding: 0.5rem;">Biến khí tượng mở rộng</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center;"><span style="background: #F59E0B; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P2</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">FD-2</td>
                <td style="padding: 0.5rem;">Multi-Station Network</td>
                <td style="text-align: center; color: #EF4444;">★★☆☆☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center;"><span style="background: #71717A; color: #FAFAFA;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P3</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.caption("*P1 = Ưu tiên cao (có thể thực hiện ngay), P2 = Ưu tiên trung bình, P3 = Dài hạn*")
