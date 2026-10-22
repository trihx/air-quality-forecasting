"""Kiểm thử tự động cho bộ slide bảo vệ đề án thạc sĩ v3 (cấu trúc 04 phần theo QĐ 1799/QĐ-ĐHCT)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[2]
PPTX_PATH = ROOT / "docs" / "defense-2026" / "BAO_VE_DE_AN_HOANG_XUAN_TRI_v3_CTU.pptx"


@pytest.fixture(scope="module")
def prs() -> Presentation:
    assert PPTX_PATH.exists(), f"Không tìm thấy tệp trình chiếu: {PPTX_PATH}"
    return Presentation(str(PPTX_PATH))


def _slide_text(slide) -> str:
    parts = []
    for shp in slide.shapes:
        if shp.has_text_frame:
            parts.append(shp.text_frame.text)
        if shp.has_table:
            for row in shp.table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
    return "\n".join(parts)


def _notes_text(slide) -> str:
    if not slide.has_notes_slide:
        return ""
    return slide.notes_slide.notes_text_frame.text


class TestDefenseDeckV3Structure:
    """Kiểm tra cấu trúc 04 phần chuẩn Quyết định 1799/QĐ-ĐHCT và thời lượng 5–10 phút."""

    def test_slide_counts(self, prs: Presentation) -> None:
        assert len(prs.slides) == 27, (
            "Bộ slide phải gồm đúng 27 slide (12 chính + 1 mục lục dự phòng + 14 dự phòng B1–B14)"
        )

    def test_four_part_qd1799_structure(self, prs: Presentation) -> None:
        s2_text = _slide_text(prs.slides[1])
        assert "04 phần" in s2_text or "PHẦN 1" in s2_text
        for part_label in ("PHẦN 1", "PHẦN 2", "PHẦN 3", "PHẦN 4"):
            assert part_label in s2_text, f"Slide 2 (Cấu trúc báo cáo) thiếu {part_label}"

        main_texts = "\n".join(_slide_text(prs.slides[i]) for i in range(2, 11))
        for tag_label in (
            "PHẦN 1 · GIỚI THIỆU ĐỀ ÁN",
            "PHẦN 2 · CƠ SỞ LÝ LUẬN & PHƯƠNG PHÁP",
            "PHẦN 3 · KẾT QUẢ & THẢO LUẬN",
            "PHẦN 4 · KẾT LUẬN & ĐỀ XUẤT",
        ):
            assert tag_label in main_texts, f"Các slide nội dung chính thiếu nhãn phân phần: {tag_label}"

    def test_speaking_duration_5_to_10_minutes(self, prs: Presentation) -> None:
        main_words = sum(len(_notes_text(prs.slides[i]).split()) for i in range(12))
        # Tốc độ nói chuẩn bảo vệ học thuật tiếng Việt ~130–150 từ/phút -> 5–10 phút ứng với 750–1.500 từ
        assert 850 <= main_words <= 1450, f"Số từ speaker notes 12 slide chính ({main_words}) nằm ngoài khung 5–10 phút"

    def test_backup_hyperlinks_and_index(self, prs: Presentation) -> None:
        index_slide = prs.slides[12]
        for idx in range(1, 15):
            code = f"B{idx}"
            target_slide = prs.slides[12 + idx]
            btn = next((s for s in index_slide.shapes if s.name == f"BTN_{code}"), None)
            assert btn is not None, f"Thiếu nút {code} trên slide mục lục dự phòng"
            assert btn.click_action.target_slide == target_slide, f"Nút {code} trỏ sai slide đích"

            back_btn = next((s for s in target_slide.shapes if s.name == "BTN_back"), None)
            assert back_btn is not None, f"Slide dự phòng {code} thiếu nút quay về mục lục"
            assert back_btn.click_action.target_slide == index_slide


class TestDefenseDeckV3ContentAndAntiSlop:
    """Kiểm tra độ khớp số liệu báo cáo đề án và quy tắc Anti-AI Slop."""

    def test_no_emoji_no_draft_flags_no_luan_van(self, prs: Presentation) -> None:
        emoji_ranges = [(0x1F300, 0x1FAFF), (0x2600, 0x26FF), (0x2700, 0x27BF)]
        for idx, slide in enumerate(prs.slides, 1):
            combined = _slide_text(slide) + "\n" + _notes_text(slide)
            assert "CHỜ XÁC MINH" not in combined, f"Slide {idx} còn chứa cờ nháp nội bộ"
            assert "luận văn" not in combined.lower(), f"Slide {idx} dùng từ 'luận văn' thay vì 'đề án'"
            for ch in combined:
                code = ord(ch)
                assert not any(lo <= code <= hi for lo, hi in emoji_ranges), f"Slide {idx} chứa ký tự emoji: {ch!r}"

    def test_key_report_tables_and_metrics_present(self, prs: Presentation) -> None:
        all_text = "\n".join(_slide_text(s) for s in prs.slides)
        expected_tokens = [
            "209.594",
            "19.810",
            "103,5",
            "3,493",
            "0,382",
            "3,417",
            "0,469",
            "2,944",
            "2,596",
            "9,148",
            "89,5%",
            "89,6%",
            "89,4%",
            "0,782",
            "14–17",
            "+1,30",
            "−8,452",
            "−5,891",
            "−2,134",
        ]
        for tok in expected_tokens:
            assert tok in all_text, f"Thiếu số liệu trọng tâm của báo cáo trên bộ slide: {tok}"

    def test_officecli_zero_issues(self) -> None:
        officecli_bin = shutil.which("officecli") or "/opt/homebrew/bin/officecli"
        if not Path(officecli_bin).exists():
            pytest.skip("Không tìm thấy công cụ officecli trên hệ thống")
        proc = subprocess.run(
            [officecli_bin, "view", str(PPTX_PATH), "issues"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Found 0 issue(s)" in proc.stdout, f"officecli phát hiện lỗi định dạng:\n{proc.stdout}"

    def test_master_script_document_completeness(self) -> None:
        script_path = ROOT / "docs" / "defense-2026" / "12_KICH_BAN_THUYET_TRINH_VA_PHAN_BIEN_CHI_TIET_v3_CTU.md"
        assert script_path.exists(), f"Không tìm thấy tệp kịch bản thuyết trình chi tiết: {script_path}"
        content = script_path.read_text(encoding="utf-8")
        for i in range(1, 14):
            assert f"### SLIDE {i}:" in content, f"Kịch bản thiếu phần trình bày cho SLIDE {i}"
        for idx in range(1, 15):
            slide_no = 13 + idx
            assert f"### SLIDE {slide_no} (`B{idx}`):" in content, (
                f"Kịch bản thiếu phần giải thích cho SLIDE {slide_no} (B{idx})"
            )
        assert "Lời thoại Bản Chuẩn" in content
        assert "Lời thoại Bản Cô Đọng" in content
        assert "Kịch bản trả lời 30 giây" in content
        assert "Kịch bản giải thích chi tiết" in content
        assert "luận văn" not in content.lower(), "Kịch bản dùng từ 'luận văn' thay vì 'đề án'"
        assert "CHỜ XÁC MINH" not in content
