"""Page: Kết Luận & Hướng Phát Triển — Thesis conclusion dashboard.

Designed for academic presentation per CTU-QD1799 standards.
Sections:
    1. 5.1 Kết luận chính của Đề án
    2. 5.2 Khả năng ứng dụng thực tế
    3. 5.3 Hạn chế của nghiên cứu
    4. 5.4 Hướng phát triển (với ma trận khả thi & tác động)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.frontend.citations import cite, render_references_section

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def page_conclusion(results: dict):
    """Main entry point for the Conclusion & Future Work page."""
    from app import insight_card, section_header

    st.markdown(
        """
    <h1 style="text-align:center; font-size:1.8rem; margin-bottom:0.2rem;">
        📝 Kết Luận & Hướng Phát Triển
    </h1>
    <p style="text-align:center; opacity:0.6; font-size:0.95rem; margin-bottom:2rem;">
        Tổng hợp kết quả thực nghiệm và lộ trình triển khai hệ thống dự báo PM2.5 đa độ phân giải
    </p>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. 5.1 Kết Luận Kỹ Thuật",
            "2. 5.2 Ứng Dụng Vận Hành",
            "3. 5.3 Hạn Chế Hệ Thống",
            "4. 5.4 Lộ Trình Nâng Cấp",
        ]
    )

    with tab1:
        _render_summary(section_header, insight_card)

    with tab2:
        _render_practical_applications(section_header, insight_card)

    with tab3:
        _render_limitations(section_header, insight_card)

    with tab4:
        _render_future_work(section_header, insight_card)

    # ── References section ──
    render_references_section()


# ──────────────────────────────────────────────────────────────
# Tab 1: 5.1 Kết Luận Kỹ Thuật Hệ Thống
# ──────────────────────────────────────────────────────────────


def _render_summary(section_header, insight_card):
    """Research summary — technical outcomes and verified performance."""
    section_header("🎯", "5.1 Kết Luận Kỹ Thuật Hệ Thống")

    st.markdown(
        f"""
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1.5rem; border-left: 4px solid #00D4AA; margin-bottom: 1.5rem;">
        <div style="font-size: 0.95rem; line-height: 1.7;">
            Hệ thống dự báo nồng độ bụi mịn PM2.5 được xây dựng và kiểm chứng trên dữ liệu cảm biến IoT thực tế
            tại Sa Đéc, Đồng Tháp. Quy trình hoàn thiện qua <b>9 phiên bản</b> cải tiến liên tục,
            đánh giá <b>41 cấu hình mô hình thực nghiệm</b> trên 3 độ phân giải
            (15 phút, 30 phút, 1 giờ) × 3 tầm dự báo (1 giờ, 6 giờ, 24 giờ) {cite("tashman2000")}.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 6 Quantitative Findings matching Chapter 5.1 verbatim
    section_header("📊", "6 Phát Hiện & Kết Quả Kỹ Thuật Cốt Lõi")

    findings = [
        {
            "num": "1",
            "title": "Thiết lập quy trình kỹ nghệ dữ liệu chống rò rỉ (Anti-Leakage Pipeline)",
            "badge": "Chuẩn 250 Tests",
            "detail": (
                f"Xây dựng pipeline 7 bước tiền xử lý đạt chuẩn kiểm thử <b>250 bài kiểm thử tự động</b> "
                f"(unit & integration tests). Việc áp dụng nghiêm ngặt phép trễ <code>shift(1)</code> giúp "
                f"triệt tiêu hoàn toàn rò rỉ dữ liệu, đưa chỉ số đánh giá về giá trị thực nghiệm trung thực "
                f"(<b>R² đạt 0,11–0,27</b>), loại bỏ triệt để hiện tượng R² ảo xấp xỉ 1,0 do nhìn trộm tương lai {cite('hyndman2021')}."
            ),
        },
        {
            "num": "2",
            "title": "Triển khai chiến lược nội suy phân tầng hiệu quả (Tiered Imputation)",
            "badge": "PCHIP + KNN + Drop",
            "detail": (
                f"Kết hợp <b>Cubic Spline (≤ 6h)</b>, <b>KNN (6 – 24h)</b> {cite('troyanskaya2001')} "
                f"và loại bỏ khoảng trống mất tín hiệu dài (<b>gap > 24h</b>) kèm đánh số <code>segment_id</code> "
                f"{cite('moritz2015')}. Chiến lược này phục hồi tối đa tín hiệu vật lý mà không làm méo mó "
                f"phân phối chuỗi thời gian nồng độ bụi."
            ),
        },
        {
            "num": "3",
            "title": "Xác lập điểm ngọt độ phân giải 30 phút (Sweet Spot Resolution)",
            "badge": ">80% Top Cases",
            "detail": (
                "Chứng minh bằng thực nghiệm: độ phân giải <b>30 phút</b> đạt hiệu năng dự báo tối ưu "
                "trên <b>> 80% các trường hợp đánh giá</b> ở mốc 6 giờ và 24 giờ. Tần suất 30 phút mang lại "
                "tỷ lệ tín hiệu trên nhiễu (SNR) lý tưởng, loại bỏ nhiễu ngẫu nhiên vi mô của chuỗi 15 phút, "
                "vượt thoát bẫy tự tương quan của chuỗi 1 giờ, đồng thời tiết kiệm 50% chi phí tính toán."
            ),
        },
        {
            "num": "4",
            "title": "Nâng cao độ chính xác dự báo bằng Ensemble (Ensemble Weighted v9)",
            "badge": "MASE = 0,382 (6h)",
            "detail": (
                f"Mô hình <b>Ensemble_Weighted_v9_30m</b> đạt <b>MASE = 0,382</b> tại 6 giờ "
                f"(<b>giảm 49,6% MAE</b> so với mô hình cơ sở quán tính Persistence: 3,493 vs 6,932 µg/m³) "
                f"và <b>MASE = 0,469</b> tại 24 giờ {cite('hyndman2006')}. Sự vượt trội có ý nghĩa thống kê "
                f"rõ rệt theo kiểm định Diebold-Mariano (<b>p < 0,001</b>) {cite('diebold1995')}."
            ),
        },
        {
            "num": "5",
            "title": "Cung cấp khoảng tin cậy và cảnh báo vượt ngưỡng tin cậy",
            "badge": "F1-Score = 0,782",
            "detail": (
                f"Phương pháp <b>Conformal Quantile Regression (CQR)</b> {cite('romano2019')} kết hợp hiệu chuẩn "
                f"thích ứng <b>ACI</b> {cite('gibbs2021')} cung cấp khoảng tin cậy 90% hợp lệ dưới biến động phân phối (concept drift). "
                f"Mô hình đạt <b>F1-Score = 0,782</b> (Precision = 0,812; Recall = 0,754) trong nhiệm vụ "
                f"cảnh báo sớm trước 6 giờ các đợt ô nhiễm vượt ngưỡng khuyến nghị WHO (45 µg/m³) {cite('who2021')}."
            ),
        },
        {
            "num": "6",
            "title": "Diễn giải cơ chế vật lý bằng XAI & Ngưỡng bùng phát",
            "badge": "Tipping Point 14–17 µg/m³",
            "detail": (
                f"Lượng hóa thành công cơ chế 2 cấp độ nồng độ nền gồm: <b>ngưỡng chuyển pha 14–17 µg/m³</b> "
                f"(trùng khớp khuyến nghị 24h của WHO 15 µg/m³) và <b>ngưỡng kích hoạt bùng phát khi vượt 17 µg/m³</b> "
                f"thông qua phân tích SHAP Dependence Plot {cite('lundberg2017')}. Xác lập tầm quan trọng biến số dựa trên "
                f"nền tảng lý thuyết Permutation Feature Importance của Fisher et al. (2019) {cite('fisher2019')}, "
                f"phù hợp với các nghiên cứu giải thích máy học ô nhiễm không khí gần đây {cite('gu2021')}."
            ),
        },
    ]

    for f in findings:
        st.markdown(
            f"""
        <div style="background: var(--secondary-background-color); border-radius: 10px;
                    padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
                    border-left: 4px solid #00D4AA;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div style="font-weight: 700; font-size: 1rem; color: var(--text-color);">
                    <span style="background: rgba(0,212,170,0.15); color: #00D4AA; padding: 0.15rem 0.55rem;
                                 border-radius: 4px; margin-right: 0.5rem;">{f["num"]}</span>
                    {f["title"]}
                </div>
                <span style="background: rgba(0,212,170,0.12); color: #00D4AA; font-size: 0.78rem;
                             font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 6px;">
                    {f["badge"]}
                </span>
            </div>
            <div style="font-size: 0.88rem; opacity: 0.88; line-height: 1.65; color: var(--text-color);">
                {f["detail"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Contribution summary
    section_header("🏆", "Giá Trị Kỹ Thuật & Khả Năng Ứng Dụng")

    c1, c2, c3 = st.columns(3)
    contributions = [
        (
            "🔬",
            "Giá Trị Nghiên Cứu",
            "Phương pháp luận Multi-Resolution × Multi-Horizon đầu tiên cho bụi mịn PM2.5 IoT tại ĐBSCL. "
            "Chứng minh thực nghiệm điểm ngọt 30 phút và cơ chế vượt bẫy tự tương quan.",
        ),
        (
            "🛡️",
            "Quy Trình & Kỹ Thuật Hệ Thống",
            "Pipeline kỹ nghệ dữ liệu chống rò rỉ nghiêm ngặt 7 bước (shift-1, tiered imputation, test-on-real-only), "
            "được bảo vệ bởi 250 bài kiểm thử tự động đạt độ tin cậy tái lập 100%.",
        ),
        (
            "🏛️",
            "Ứng Dụng Vận Hành",
            "Xác lập cơ chế 2 cấp độ nồng độ nền (14–17 µg/m³ & >17 µg/m³) và kiến trúc phần mềm 3 tầng "
            "đóng gói Docker sẵn sàng tích hợp các trung tâm điều hành đô thị thông minh (IOC).",
        ),
    ]
    for col, (icon, label, desc) in zip([c1, c2, c3], contributions, strict=False):
        with col:
            st.markdown(
                f"""
            <div style="background: var(--secondary-background-color); border-radius: 12px;
                        padding: 1.3rem; text-align: center; min-height: 200px;
                        border: 1px solid rgba(0,212,170,0.25);">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-weight: 700; color: #00D4AA; font-size: 1.02rem; margin-bottom: 0.6rem;">
                    {label}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.85; line-height: 1.6; color: var(--text-color);">
                    {desc}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────
# Tab 2: 5.2 Ứng Dụng Vận Hành
# ──────────────────────────────────────────────────────────────


def _render_practical_applications(section_header, insight_card):
    """Practical operational applications per Section 5.2."""
    section_header("🏛️", "5.2 Ứng Dụng Vận Hành")

    st.markdown(
        """
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1.3rem 1.5rem; margin-bottom: 1.5rem;
                border-left: 4px solid #00D4AA; font-size: 0.92rem; line-height: 1.65;">
        Hệ thống được thiết kế hướng tới <b>3 kịch bản vận hành thực tế</b> phục vụ công tác giám sát chất lượng không khí,
        bảo vệ sức khỏe cộng đồng và sẵn sàng tích hợp trung tâm điều hành đô thị thông minh tại khu vực Đồng bằng sông Cửu Long.
    </div>
    """,
        unsafe_allow_html=True,
    )

    app_columns = st.columns(3)

    with app_columns[0]:
        st.markdown(
            f"""
        <div style="background: var(--secondary-background-color); border-radius: 12px;
                    padding: 1.4rem; border: 1px solid rgba(0,212,170,0.25); height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⏱️</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #00D4AA; margin-bottom: 0.6rem;">
                1. Cảnh Báo Sớm 6h – 24h
            </div>
            <div style="font-size: 0.87rem; line-height: 1.6; opacity: 0.9; color: var(--text-color);">
                <b>Đối tượng hưởng lợi:</b> Chi cục Bảo vệ Môi trường, Sở Nông nghiệp & Môi trường,
                các cơ quan y tế dự phòng tại Đồng Tháp và ĐBSCL.<br><br>
                <b>Giá trị nghiệp vụ:</b> Cung cấp dự báo nồng độ bụi trước 6 đến 24 giờ với
                <b>F1-Score = 0,782</b>, giúp chính quyền chuyển dịch từ trạng thái <i>"ứng phó thụ động"</i>
                khi ô nhiễm đã xảy ra sang <i>"chủ động phát đi thông điệp khuyến cáo"</i> cho người dân {cite("who2021")}.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with app_columns[1]:
        st.markdown(
            """
        <div style="background: var(--secondary-background-color); border-radius: 12px;
                    padding: 1.4rem; border: 1px solid rgba(0,212,170,0.25); height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚨</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #00D4AA; margin-bottom: 0.6rem;">
                2. Cơ Chế 2 Cấp Độ Nền
            </div>
            <div style="font-size: 0.87rem; line-height: 1.6; opacity: 0.9; color: var(--text-color);">
                <b>Chỉ báo ra quyết định điều hành:</b><br>
                • <b>Ngưỡng chuyển pha 14–17 µg/m³:</b> Làm chỉ báo cảnh báo sớm (Early Warning)
                cho các nhóm dân cư nhạy cảm (trẻ em, người già, người mắc bệnh hô hấp).<br><br>
                • <b>Ngưỡng kích hoạt > 17 µg/m³:</b> Làm chỉ báo can thiệp khẩn cấp (Emergency Intervention)
                phục vụ kiểm soát nguồn phát thải (đốt phụ phẩm nông nghiệp) và điều phối phân luồng giao thông đô thị.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with app_columns[2]:
        st.markdown(
            """
        <div style="background: var(--secondary-background-color); border-radius: 12px;
                    padding: 1.4rem; border: 1px solid rgba(0,212,170,0.25); height: 100%;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏙️</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #00D4AA; margin-bottom: 0.6rem;">
                3. Tích Hợp Smart City IOC
            </div>
            <div style="font-size: 0.87rem; line-height: 1.6; opacity: 0.9; color: var(--text-color);">
                <b>Kiến trúc phần mềm chuẩn hóa:</b><br>
                Hệ thống phần mềm 3 tầng (FastAPI Backend + Streamlit Dashboard + Supabase/PostgreSQL)
                đã được <b>đóng gói hoàn chỉnh bằng Docker Compose</b>.<br><br>
                Sẵn sàng kết nối qua RESTful API tiêu chuẩn để tích hợp dữ liệu vào
                <b>Trung tâm Điều hành Đô thị Thông minh (IOC)</b> của tỉnh Đồng Tháp và các đô thị trong vùng.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Decision Matrix Table
    section_header("📋", "Ma Trận Hành Động Ứng Phó Dựa Trên Ngưỡng Nồng Độ Nền")

    st.markdown(
        """
    <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem;">
        <thead>
            <tr style="border-bottom: 2px solid rgba(0,212,170,0.3);">
                <th style="text-align: left; padding: 0.7rem;">Cấp độ Nồng độ</th>
                <th style="text-align: center; padding: 0.7rem;">Khoảng PM2.5</th>
                <th style="text-align: left; padding: 0.7rem;">Trạng thái Khí quyển</th>
                <th style="text-align: left; padding: 0.7rem;">Khuyến nghị Y tế Cộng đồng</th>
                <th style="text-align: left; padding: 0.7rem;">Hành động Quản lý Đô thị</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.6rem; font-weight: 600; color: #10B981;">🟢 Cấp 1: An toàn</td>
                <td style="text-align: center; font-family: monospace;">&lt; 14 µg/m³</td>
                <td style="padding: 0.6rem;">Khí quyển tự làm sạch hiệu quả (SHAP &lt; 0)</td>
                <td style="padding: 0.6rem;">Hoạt động ngoài trời bình thường</td>
                <td style="padding: 0.6rem;">Duy trì giám sát thường quy</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); background: rgba(245,158,11,0.05);">
                <td style="padding: 0.6rem; font-weight: 600; color: #F59E0B;">🟡 Cấp 2: Cảnh báo sớm</td>
                <td style="text-align: center; font-family: monospace;">14 – 17 µg/m³</td>
                <td style="padding: 0.6rem;">Vùng chuyển pha (Tipping Point), SHAP đảo chiều</td>
                <td style="padding: 0.6rem;">Nhóm nhạy cảm hạn chế thể thao ngoài trời</td>
                <td style="padding: 0.6rem;">Phát cảnh báo sớm trên app đô thị / đài truyền thanh</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06); background: rgba(239,68,68,0.05);">
                <td style="padding: 0.6rem; font-weight: 600; color: #EF4444;">🔴 Cấp 3: Can thiệp khẩn</td>
                <td style="text-align: center; font-family: monospace;">&gt; 17 µg/m³</td>
                <td style="padding: 0.6rem;">Bùng phát tích tụ phi tuyến, gia tốc khi độ ẩm thấp</td>
                <td style="padding: 0.6rem;">Toàn dân đeo khẩu trang lọc bụi khi ra ngoài</td>
                <td style="padding: 0.6rem;">Kiểm soát đốt rơm rạ, phân luồng xe tải giờ cao điểm</td>
            </tr>
        </tbody>
    </table>
    """,
        unsafe_allow_html=True,
    )

    insight_card(
        "💡 Tính Thực Tiễn Của Khung Dự Báo",
        "Việc tích hợp đồng thời <b>dự báo điểm</b> (Point Forecast) từ mô hình Ensemble v9, "
        "<b>khoảng tin cậy thích ứng</b> (ACI 90%) và <b>cơ chế phân cấp nồng độ nền</b> "
        "giúp lãnh đạo đô thị có đầy đủ thông tin cả về giá trị dự kiến lẫn mức độ bất định, "
        "tránh hiện tượng báo động giả và nâng cao hiệu quả các quyết định can thiệp môi trường.",
        card_type="info",
    )


# ──────────────────────────────────────────────────────────────
# Tab 3: 5.3 Hạn Chế Kỹ Thuật Hệ Thống
# ──────────────────────────────────────────────────────────────


def _render_limitations(section_header, insight_card):
    """System limitations — practical operational assessment per Section 5.3."""
    section_header("⚠️", "5.3 Hạn Chế Kỹ Thuật Hệ Thống")

    st.markdown(
        """
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
                border-left: 4px solid #EF4444; font-size: 0.92rem; line-height: 1.65;">
        Để phản ánh trung thực năng lực vận hành thực tế của hệ thống, ghi nhận <b>5 giới hạn kỹ thuật khách quan</b>
        trong quá trình khai thác dữ liệu cảm biến IoT và triển khai mô hình:
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 5 Official limitations matching thesis page 61 verbatim
    official_limitations = [
        {
            "id": "1",
            "icon": "📅",
            "title": "Gián đoạn dữ liệu quan trắc do đặc thù thiết bị IoT",
            "desc": (
                "Dữ liệu quan trắc bị gián đoạn trung bình <b>89 ngày/năm</b> do sự cố mất điện, lỗi đường truyền kết nối "
                "và bảo trì thiết bị cảm biến ngoài trời, đặc biệt tập trung tại <b>tháng 02 và tháng 09</b>. "
                "Sự gián đoạn này làm hạn chế khả năng học chu kỳ mùa vụ dài hạn của các mô hình học sâu."
            ),
        },
        {
            "id": "2",
            "icon": "📍",
            "title": "Phạm vi thực nghiệm trên một trạm quan trắc đơn lẻ",
            "desc": (
                "Nghiên cứu mới thực nghiệm trên <b>một trạm quan trắc đơn lẻ tại Sa Đéc</b>. "
                "Cần mở rộng kiểm chứng khả năng tổng quát hóa (generalization) sang các khu vực lân cận "
                "và các tiểu vùng sinh thái khác nhau của ĐBSCL."
            ),
        },
        {
            "id": "3",
            "icon": "🔬",
            "title": "Sai số phần cứng nội tại của cảm biến chi phí thấp",
            "desc": (
                "Cảm biến chi phí thấp có sai số phần cứng nội tại khoảng <b>±3 µg/m³</b> "
                "so với thiết bị quan trắc chuẩn tham chiếu (BAM-1020). Điều này tạo ra một "
                "<b>giới hạn sai số tối thiểu (error floor)</b> tự nhiên mà không một thuật toán học máy nào có thể vượt qua."
            ),
        },
        {
            "id": "4",
            "icon": "💨",
            "title": "Chưa tích hợp biến gió và áp suất khí quyển",
            "desc": (
                "Chưa tích hợp trực tiếp các biến <b>hướng gió, tốc độ gió và áp suất khí quyển</b> "
                "do trạm quan trắc hiện tại chưa trang bị các module cảm biến này. Đây là các biến số động học "
                "khí quyển có ảnh hưởng mạnh tới cơ chế khuếch tán ô nhiễm theo không gian."
            ),
        },
        {
            "id": "5",
            "icon": "⚖️",
            "title": "Độ lệch dự báo nhẹ của mô hình Ensemble",
            "desc": (
                "Mô hình Ensemble có độ lệch dự báo nhẹ (<b>Forecast Bias = +1,30 µg/m³ tại 6 giờ</b>), "
                "có xu hướng thiên về dự báo an toàn (hơi cao hơn thực tế nhằm tránh bỏ sót đỉnh ô nhiễm nguy hại). "
                "Cần bổ sung mô-đun hiệu chỉnh bias động (dynamic bias correction) trong tương lai."
            ),
        },
    ]

    for lim in official_limitations:
        st.markdown(
            f"""
        <div style="background: var(--secondary-background-color); border-radius: 10px;
                    padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
                    border-left: 4px solid #EF4444;">
            <div style="font-weight: 700; font-size: 0.98rem; margin-bottom: 0.35rem; color: var(--text-color);">
                <span style="font-size: 1.1rem; margin-right: 0.4rem;">{lim["icon"]}</span>
                <span style="background: rgba(239,68,68,0.15); color: #EF4444; padding: 0.15rem 0.5rem;
                             border-radius: 4px; font-size: 0.85rem; margin-right: 0.5rem;">Hạn chế #{lim["id"]}</span>
                {lim["title"]}
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; opacity: 0.88; color: var(--text-color); padding-left: 1.8rem;">
                {lim["desc"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Ablation Study: Outlier Removal Trap
    section_header("🧪", "Thực Nghiệm Bóc Tách: Bẫy Cắt Lọc Ngoại Lai (Ablation Study - Hình 4.12)")

    try:
        comp_path = PROJECT_ROOT / "research" / "experiments" / "v10_ablation" / "comparison_table.json"
        if comp_path.exists():
            with open(comp_path, encoding="utf-8") as f:
                ablation_data = json.load(f)

            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.05); padding: 1rem; border-left: 3px solid #EF4444; border-radius: 4px; margin-bottom: 1rem;">
                    <span style="font-size: 0.95em;"><b>"False Sense of Accuracy" (Ảo giác chính xác):</b> Thực nghiệm bóc tách (Ablation Study) cố tình dùng thuật toán IQR thay cho Domain Bounds. Kết quả: IQR đã cắt mất 66 đợt ô nhiễm nghiêm trọng (> 54 µg/m³). Mô hình trông <b>chính xác hơn (MASE thấp hơn)</b> ở các horizon ngắn, nhưng thực chất đã bị "mù" trước các đợt bùng phát ô nhiễm thật sự.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rows = []
            for h, models_data in ablation_data.items():
                for model_name, metrics in models_data.items():
                    if "v9_mase" in metrics:
                        delta = metrics["delta_mase"]
                        note = metrics.get("note", "")
                        status = "🚨 Ảo giác (MASE giảm ảo)" if note == "FALSE ACCURACY" else "✅ Domain tốt hơn"

                        rows.append(
                            {
                                "Horizon": h,
                                "Mô hình": model_name,
                                "v9 MASE (Domain - Đúng)": f"{metrics['v9_mase']:.3f}",
                                "v10 MASE (IQR - Lỗi)": f"{metrics['v10_mase']:.3f}",
                                "Δ MASE": f"{delta:+.3f}",
                                "Đánh giá": status,
                            }
                        )

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            fig_path = PROJECT_ROOT / "research" / "figures" / "thesis" / "Hinh_4.12_Ablation_Outlier_Impact.png"
            if not fig_path.exists():
                fig_path = PROJECT_ROOT / "research" / "figures" / "ablation_outlier_impact.png"
            if fig_path.exists():
                st.image(
                    str(fig_path),
                    caption="Hình 4.12: Đánh giá mức độ ảnh hưởng của loại bỏ ngoại lai đến hiệu năng mô hình (Ablation Study)",
                    use_container_width=True,
                )

    except Exception:
        st.info("💡 Chạy script `v10_ablation_compare.py` để xem kết quả Ablation Study.")

    # External data analysis
    section_header("🌍", "Phân Tích Dữ Liệu Ngoại Lai (Open-Meteo & CAMS)")

    insight_card(
        "📋 Kết quả thử nghiệm thu thập dữ liệu ngoại lai",
        "Đã thu thập 20.520 dòng dữ liệu từ Open-Meteo API (ERA5 Reanalysis + CAMS PM2.5) "
        "cho các khoảng trống của cảm biến IoT. Tuy nhiên, phát hiện <b>bias hệ thống</b> "
        "nghiêm trọng giữa hai nguồn:<br>"
        "• Nhiệt độ: IoT ~29°C vs Open-Meteo ~27°C (bias ~2°C — do đo indoor vs outdoor)<br>"
        "• Độ ẩm: IoT ~75.6% vs Open-Meteo ~80.8% (bias ~5%)<br>"
        "• PM2.5: IoT ~13.7 vs CAMS ~22.2 µg/m³ (<b>bias ~62%</b> — cảm biến vs mô phỏng vệ tinh)<br><br>"
        "**Quyết định:** Không merge trực tiếp vào pipeline chính để tránh hiện tượng phân phối bị dịch chuyển (distribution shift). "
        "Lưu trữ tại <code>dataset/external/</code> làm tài liệu tham khảo cho các nghiên cứu tiếp theo.",
        card_type="warning",
    )

    # Sensitivity Analysis Block
    section_header("🔍", "Kiểm Định Độ Nhạy Thực Nghiệm (Sensitivity Analysis)")
    try:
        sens_path = PROJECT_ROOT / "research" / "diagnostics" / "sensitivity_analysis.json"
        if sens_path.exists():
            with open(sens_path, encoding="utf-8") as f:
                sens_d = json.load(f)
            knn_k = sens_d.get("knn_k_sensitivity", {})
            aci_g = sens_d.get("aci_gamma_sensitivity", {})

            col_k, col_g = st.columns(2)
            with col_k:
                st.markdown(f"**KNN Imputation $k$-value {cite('troyanskaya2001')}**", unsafe_allow_html=True)
                rows1 = [{"k": k.replace("k_", "k="), "MAE": v["mae"], "RMSE": v["rmse"]} for k, v in knn_k.items()]
                st.dataframe(pd.DataFrame(rows1), use_container_width=True, hide_index=True)
            with col_g:
                st.markdown(f"**ACI Adaptation Rate $\\gamma$ {cite('gibbs2021')}**", unsafe_allow_html=True)
                rows2 = [
                    {
                        "Gamma (γ)": v["gamma"],
                        "Coverage": f"{v['empirical_coverage'] * 100:.1f}%",
                        "Stability": v["stability_score"],
                    }
                    for k, v in aci_g.items()
                ]
                st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

            insight_card(
                "💡 Kết luận kiểm định độ nhạy",
                f"Thử nghiệm quét siêu tham số xác nhận: "
                f"(1) $k=5$ (KNN) cho sai số MAE tối ưu ($32,25\\,\\mu\\text{{g/m}}^3$) {cite('troyanskaya2001')}. "
                f"(2) $\\gamma=0,005$ (ACI) duy trì độ phủ $91,0\\%$ (mục tiêu $90\\%$) với chỉ số ổn định cao nhất ($0,988$) {cite('gibbs2021')}.",
                card_type="info",
            )
    except Exception:  # noqa: S110
        pass


# ──────────────────────────────────────────────────────────────
# Tab 4: 5.4 Lộ Trình Nâng Cấp Hệ Thống
# ──────────────────────────────────────────────────────────────


def _render_future_work(section_header, insight_card):
    """Future technical roadmap with feasibility assessment per Section 5.4."""
    section_header("🚀", "5.4 Lộ Trình Nâng Cấp Hệ Thống")

    st.markdown(
        """
    <div style="background: var(--secondary-background-color); border-radius: 12px;
                padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
                border-left: 4px solid #00D4AA; font-size: 0.92rem; line-height: 1.65;">
        Dựa trên các kết quả thực nghiệm và những giới hạn kỹ thuật đã ghi nhận, hệ thống đề xuất
        <b>4 hướng nâng cấp trọng tâm</b> kết hợp đánh giá tính khả thi và tác động vận hành:
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 4 Core Directions matching Chapter 5.4 verbatim
    core_directions = [
        {
            "id": "FW-1",
            "icon": "🛰️",
            "title": "Mô Hình Dự Báo Không - Thời Gian (Spatiotemporal Forecasting) Đa Điểm & Vệ Tinh CAMS",
            "desc": (
                "Mở rộng mạng lưới quan trắc đa điểm trên địa bàn tỉnh Đồng Tháp và khu vực ĐBSCL. "
                "Tích hợp dữ liệu ảnh viễn thám vệ tinh CAMS (Copernicus Atmosphere Monitoring Service) "
                "kết hợp các thuật toán học sâu không gian - thời gian tiên tiến (Spatiotemporal GNN, ConvLSTM) "
                "để xây dựng mô hình dự báo diện rộng có độ phân giải không gian cao."
            ),
            "priority": "P1 (Chiến lược)",
        },
        {
            "id": "FW-2",
            "icon": "💨",
            "title": "Bổ Sung Trạm Đo Gió Tự Động & Trường Vận Tốc Gió Vào Mô Hình",
            "desc": (
                "Bổ sung trạm quan trắc hướng gió và tốc độ gió tự động nhằm đưa trường vận tốc gió "
                "(wind velocity vector field) vào mô hình lan truyền và phân tán ô nhiễm khí quyển. "
                "Điều này giúp lượng hóa chính xác dòng bụi dịch chuyển từ các nguồn phát thải lân cận."
            ),
            "priority": "P1 (Thực thi ngay)",
        },
        {
            "id": "FW-3",
            "icon": "⚡",
            "title": "Thử Nghiệm Các Kiến Trúc Transformer Mới (PatchTST & iTransformer)",
            "desc": (
                "Thử nghiệm và tối ưu hóa các kiến trúc Transformer thế hệ mới chuyên biệt cho chuỗi thời gian "
                "như PatchTST (Patch Time Series Transformer) và iTransformer (Inverted Transformer) "
                "trên tập dữ liệu đa trạm có độ dài chuỗi lớn hơn."
            ),
            "priority": "P2 (Nghiên cứu sâu)",
        },
        {
            "id": "FW-4",
            "icon": "🔄",
            "title": "Triển Khai Học Thích Ứng Trực Tuyến (Online Learning) Trên Thiết Bị Biên (Edge Computing)",
            "desc": (
                "Triển khai mô hình học thích ứng trực tuyến (Online/Incremental Learning) trực tiếp trên "
                "thiết bị biên (Edge Computing) nhằm tự động cập nhật trọng số mô hình khi luồng dữ liệu mới "
                "liên tục đổ về, tự động phát hiện và thích ứng với hiện tượng trôi dạt khái niệm (Concept Drift)."
            ),
            "priority": "P2 (Kỹ thuật biên)",
        },
    ]

    for cd in core_directions:
        st.markdown(
            f"""
        <div style="background: var(--secondary-background-color); border-radius: 10px;
                    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
                    border-left: 4px solid #00D4AA;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div style="font-weight: 700; font-size: 1.02rem; color: var(--text-color);">
                    <span style="font-size: 1.15rem; margin-right: 0.4rem;">{cd["icon"]}</span>
                    {cd["title"]}
                </div>
                <span style="background: rgba(0,212,170,0.15); color: #00D4AA; font-size: 0.8rem;
                             font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 6px;">
                    {cd["priority"]}
                </span>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.65; opacity: 0.88; color: var(--text-color); padding-left: 1.8rem;">
                {cd["desc"]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Priority matrix
    section_header("📋", "Ma Trận Đánh Giá Tính Khả Thi & Tác Động (Priority Matrix)")

    future_directions = [
        {
            "id": "FD-1",
            "title": "Online Learning trên Edge Computing",
            "icon": "🔄",
            "feasibility": "Cao",
            "impact": "Cao",
            "f_score": 4,
            "i_score": 4,
            "desc": "Tự động cập nhật trọng số mô hình khi có dữ liệu mới, bù đắp concept drift.",
            "prereq": "Đã có: Pipeline v9 + Docker architecture + streaming data",
        },
        {
            "id": "FD-2",
            "title": "Bổ sung trạm đo gió tự động",
            "icon": "💨",
            "feasibility": "Rất cao",
            "impact": "Cao",
            "f_score": 5,
            "i_score": 4,
            "desc": "Bổ sung sensor tốc độ & hướng gió để đưa trường vector gió vào phân tán bụi.",
            "prereq": "Cần: Module anemometer chi phí thấp tích hợp IoT trạm hiện tại",
        },
        {
            "id": "FD-3",
            "title": "Kiến trúc PatchTST & iTransformer",
            "icon": "⚡",
            "feasibility": "Cao",
            "impact": "Trung bình",
            "f_score": 4,
            "i_score": 3,
            "desc": "PatchTST chia subseries thành các patch nhỏ, giảm độ phức tạp O(L²) và bắt chu kỳ tốt hơn.",
            "prereq": "Đã có: Tabular pipeline + GPU server/hạ tầng huấn luyện",
        },
        {
            "id": "FD-4",
            "title": "Đồng hóa dữ liệu viễn thám vệ tinh CAMS",
            "icon": "🛰️",
            "feasibility": "Trung bình",
            "impact": "Rất cao",
            "f_score": 3,
            "i_score": 5,
            "desc": "Sử dụng Domain Adaptation để kết hợp dữ liệu toàn cầu CAMS với cảm biến mặt đất Sa Đéc.",
            "prereq": "Đã có: Dataset external CAMS + hiểu rõ bias hệ thống ~62%",
        },
        {
            "id": "FD-5",
            "title": "Mạng lưới quan trắc đa điểm ĐBSCL",
            "icon": "📡",
            "feasibility": "Thấp",
            "impact": "Rất cao",
            "f_score": 2,
            "i_score": 5,
            "desc": "Mở rộng ra Cần Thơ, Cao Lãnh, Long Xuyên; phát triển Spatiotemporal GNN.",
            "prereq": "Cần: Nguồn lực mở rộng phần cứng thực địa đa địa phương",
        },
    ]

    for fd in future_directions:
        f_bar = "█" * fd["f_score"] + "░" * (5 - fd["f_score"])
        i_bar = "█" * fd["i_score"] + "░" * (5 - fd["i_score"])
        border_color = "#00D4AA" if fd["f_score"] >= 4 else "#F59E0B" if fd["f_score"] >= 3 else "#EF4444"

        st.markdown(
            f"""
        <div style="background: var(--secondary-background-color); border-radius: 12px;
                    padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
                    border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;
                        margin-bottom: 0.4rem;">
                <div style="font-weight: 700; font-size: 0.95rem;">
                    {fd["icon"]} {fd["id"]}: {fd["title"]}
                </div>
            </div>
            <div style="font-size: 0.86rem; line-height: 1.55; opacity: 0.88; margin-bottom: 0.6rem;">
                {fd["desc"]}
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1.5fr; gap: 0.8rem;
                        font-size: 0.8rem;">
                <div>
                    <span style="opacity: 0.6;">Khả thi:</span>
                    <span style="font-family: monospace; color: {border_color};"> {f_bar}</span>
                    <span style="opacity: 0.5;"> ({fd["feasibility"]})</span>
                </div>
                <div>
                    <span style="opacity: 0.6;">Tác động:</span>
                    <span style="font-family: monospace; color: #00D4AA;"> {i_bar}</span>
                    <span style="opacity: 0.5;"> ({fd["impact"]})</span>
                </div>
                <div>
                    <span style="opacity: 0.6;">Tiền đề:</span>
                    <span style="opacity: 0.7;"> {fd["prereq"]}</span>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
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
                <td style="padding: 0.5rem;">FD-2</td>
                <td style="padding: 0.5rem;">Bổ sung trạm đo gió tự động</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center;"><span style="background: #00D4AA; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P1</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-1</td>
                <td style="padding: 0.5rem;">Online Learning trên Edge Computing</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center;"><span style="background: #00D4AA; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P1</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-3</td>
                <td style="padding: 0.5rem;">Kiến trúc PatchTST & iTransformer</td>
                <td style="text-align: center; color: #00D4AA;">★★★★☆</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center;"><span style="background: #00D4AA; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P2</span></td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.06);">
                <td style="padding: 0.5rem;">FD-4</td>
                <td style="padding: 0.5rem;">Đồng hóa dữ liệu viễn thám vệ tinh CAMS</td>
                <td style="text-align: center; color: #F59E0B;">★★★☆☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center;"><span style="background: #F59E0B; color: #0E1117;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P2</span></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem;">FD-5</td>
                <td style="padding: 0.5rem;">Mạng lưới quan trắc đa điểm ĐBSCL</td>
                <td style="text-align: center; color: #EF4444;">★★☆☆☆</td>
                <td style="text-align: center; color: #00D4AA;">★★★★★</td>
                <td style="text-align: center;"><span style="background: #71717A; color: #FAFAFA;
                    padding: 0.2rem 0.6rem; border-radius: 4px; font-weight: 700;">P3</span></td>
            </tr>
        </tbody>
    </table>
    """,
        unsafe_allow_html=True,
    )

    st.caption("*P1 = Ưu tiên cao (thực hiện ngay), P2 = Ưu tiên trung hạn, P3 = Chiến lược dài hạn*")
