const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        Header, Footer, AlignmentType, PageOrientation, LevelFormat, 
        HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak } = require('docx');

const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } }, // 12pt default
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal",
        run: { size: 44, bold: true, color: "1155cc", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 240 }, alignment: AlignmentType.CENTER } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "333333", font: "Arial" },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "444444", font: "Arial", italics: true },
        paragraph: { spacing: { before: 120, after: 120 }, outlineLevel: 2 } }
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list-1",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-list-1",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ 
        alignment: AlignmentType.CENTER,
        children: [new TextRun("Trang "), new TextRun({ children: [PageNumber.CURRENT] }), new TextRun(" / "), new TextRun({ children: [PageNumber.TOTAL_PAGES] })]
      })] })
    },
    children: [
      new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("HƯỚNG DẪN KIỂM CHỨNG KHOA HỌC: DASHBOARD DỰ BÁO PM2.5")] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Quy trình Đánh giá Tính chính xác & Chống rò rỉ dữ liệu (Anti-Leakage)", italics: true, size: 26 })] }),
      
      new Paragraph({ children: [new PageBreak()] }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("TỔNG QUAN VÀ MỤC ĐÍCH")] }),
      new Paragraph({ children: [new TextRun("Dashboard này không chỉ là công cụ hiển thị dự báo, mà là một hệ thống kiểm chứng khoa học. Tài liệu này hướng dẫn người dùng thao tác theo đúng luồng nghiên cứu (scientific workflow) để tự mình kiểm chứng các phát hiện cốt lõi của đề tài: (1) Bẫy tự tương quan, (2) Điểm ngọt độ phân giải 30 phút, và (3) Độ tin cậy của mô hình Ensemble kết hợp Conformal Quantile Regression.")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("LUỒNG KIỂM CHỨNG (WORKFLOW) TỪNG BƯỚC")] }),
      
      // BƯỚC 1
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Bước 1: Kiểm định \"Bẫy tự tương quan\" và rò rỉ dữ liệu (Anti-Leakage Audit)")] }),
      new Paragraph({ children: [new TextRun("Mục đích: Chứng minh các mô hình có R² cao bất thường (~1.0) là do mắc lỗi học thuộc lòng (rò rỉ dữ liệu từ tương lai) chứ không phải kỹ năng thật.", { italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "Thao tác trên Sidebar:", bold: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Tại mục "), new TextRun({ text: "Select Version", bold: true }), new TextRun(", chọn: "), new TextRun({ text: "v1_baseline", italics: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Tại mục "), new TextRun({ text: "Forecasting Horizon", bold: true }), new TextRun(", chọn: "), new TextRun({ text: "1h", italics: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Tại mục "), new TextRun({ text: "Select Model", bold: true }), new TextRun(", chọn: "), new TextRun({ text: "Ridge", italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "⚠️ Kết quả cần lưu ý:", bold: true, color: "D93025" })] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun("Nhìn vào thẻ KPI R²: Nếu thấy R² = 1.000 và MAE ≈ 0.004, hệ thống đang báo động về tình trạng Data Leakage. Điều này giải thích tại sao luận văn phải áp dụng cơ chế dịch chuyển mục tiêu (shift) triệt để (192 unit tests) để trả R² về mức thực tế khoa học (0.11 - 0.27).")] }),

      // BƯỚC 2
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Bước 2: Phân tích Đa độ phân giải & Khám phá \"Điểm ngọt 30m\"")] }),
      new Paragraph({ children: [new TextRun("Mục đích: Xác nhận tại sao độ phân giải 30 phút là tối ưu so với 1 giờ (bỏ lỡ tín hiệu) và 15 phút (nhiễu quá cao).", { italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "Thao tác trên Giao diện:", bold: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Chuyển Version về "), new TextRun({ text: "v9_multi_resolution", italics: true }), new TextRun(" (phiên bản ổn định chính).")] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Nhấp sang Tab: "), new TextRun({ text: "Scientific Benchmark", bold: true })] }),
      new Paragraph({ children: [new TextRun({ text: "⚠️ Kết quả cần lưu ý:", bold: true, color: "D93025" })] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun("Tại \"Multi-Resolution Evaluation Table\", hãy tập trung vào cột "), new TextRun({ text: "MASE (Sai số chuẩn hoá)", bold: true }), new TextRun(". Khác với RMSE, MASE đánh giá kỹ năng của mô hình so với Baseline ngây ngô. MASE < 1.0 là tốt.")] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun("Quan sát thấy các mô hình hậu tố "), new TextRun({ text: "_30m", bold: true }), new TextRun(" (đặc biệt là Ensemble_Weighted_v9_30m) luôn chiếm ưu thế với MASE thấp nhất ở các mốc 6h và 24h, chứng minh luận điểm Điểm ngọt 30 phút.")] }),

      // BƯỚC 3
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Bước 3: Kiểm chứng dự báo thực tế và Khoảng tin cậy (CQR)")] }),
      new Paragraph({ children: [new TextRun("Mục đích: Xem xét tính ứng dụng thực tiễn của mô hình tốt nhất.", { italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "Thao tác trên Giao diện:", bold: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Nhấp sang Tab: "), new TextRun({ text: "Main Dashboard", bold: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Horizon chọn "), new TextRun({ text: "6h", italics: true }), new TextRun(", Model chọn "), new TextRun({ text: "Ensemble_Weighted_v9_30m", italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "⚠️ Kết quả cần lưu ý:", bold: true, color: "D93025" })] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun({ text: "Biểu đồ Dự báo:", bold: true }), new TextRun(" Chú ý dải màu mờ bao quanh đường dự báo. Đó là khoảng tin cậy 90% (Conformal Quantile Regression). Phải đảm bảo đường thực tế (Actual) đa số nằm gọn trong dải này.")] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun({ text: "KPI Thực tiễn:", bold: true }), new TextRun(" RMSE 6h đạt 5.079 μg/m³ (thuộc nhóm Top 15% quốc tế). Chú ý tỷ lệ Directional Accuracy (DA) cho biết hệ thống dự đoán xu hướng tăng/giảm chính xác đến mức nào.")] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun({ text: "Forecast Bias (Độ lệch dư):", bold: true }), new TextRun(" Hãy nhìn vào mục Metric Context. Nếu Bias dương (+1.30 μg/m³), mô hình đang dự báo hơi cao hơn thực tế (thiên hướng cẩn trọng, an toàn cho cảnh báo sức khỏe).")] }),

      // BƯỚC 4
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Bước 4: Minh bạch hoá Mô hình học máy (Experiment Tracking)")] }),
      new Paragraph({ children: [new TextRun("Mục đích: Xóa bỏ tính \"hộp đen\" của AI, đảm bảo mọi kết quả đều có nguồn gốc rõ ràng.", { italics: true })] }),
      new Paragraph({ children: [new TextRun({ text: "Thao tác trên Giao diện:", bold: true })] }),
      new Paragraph({ numbering: { reference: "numbered-list-1", level: 0 },
        children: [new TextRun("Nhấp sang Tab: "), new TextRun({ text: "Experiment Tracking", bold: true })] }),
      new Paragraph({ children: [new TextRun({ text: "⚠️ Kết quả cần lưu ý:", bold: true, color: "D93025" })] }),
      new Paragraph({ numbering: { reference: "bullet-list-1", level: 0 },
        children: [new TextRun("Người dùng có thể tra cứu toàn bộ Siêu tham số (Hyperparameters), hàm loss, số lớp nơ-ron (Layers), Optimizer của từng mô hình được load tự động từ hệ thống quản lý thay vì bị ẩn giấu đi. Khẳng định tính minh bạch và khả năng tái lập (reproducibility) của đề tài.")] }),

      new Paragraph({ children: [new PageBreak()] }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("TỔNG KẾT")] }),
      new Paragraph({ children: [new TextRun("Thông qua 4 bước kiểm chứng, người đọc sẽ hiểu được không chỉ \"kết quả cuối cùng\" mà còn là \"hành trình khoa học\" để tìm ra kết quả đó: Từ việc phát hiện lỗi Leakage (v1), đến tối ưu độ phân giải 30m (v9), và cuối cùng là tính ứng dụng của mô hình Ensemble với khoảng tin cậy toán học chuẩn xác.")] })
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("docs/Huong_dan_su_dung_Dashboard.docx", buffer);
  console.log("Done. Saved to docs/Huong_dan_su_dung_Dashboard.docx");
}).catch(console.error);
