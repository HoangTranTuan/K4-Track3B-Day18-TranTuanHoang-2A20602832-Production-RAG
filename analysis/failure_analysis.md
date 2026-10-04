# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Tuấn Hoàng  
**Khóa:** K4 - Track 3B  
**Mã học viên:** 2A202602832  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.7351 | 0.7753 | +0.0402 |
| Answer Relevancy | 0.7691 | 0.7314 | -0.0378 |
| Context Precision | 0.9118 | 0.8981 | -0.0136 |
| Context Recall | 0.9216 | 0.7576 | -0.1640 |

### Nhận xét tổng quan về điểm số:
- **Faithfulness (+0.0402):** Production RAG tăng độ trung thực đáng kể (từ 0.7351 lên 0.7753) nhờ tầng Cross-Encoder Reranking chọn lọc đúng 3 context phù hợp nhất và kỹ thuật Contextual Prepend bổ sung nguồn tài liệu giúp LLM bám sát thực tế, hạn chế hallucination.
- **Answer Relevancy (-0.0378):** Giảm nhẹ do context giàu thông tin hơn khiến LLM trả lời chi tiết và kèm theo các điều kiện quy định bổ sung, trong khi câu trả lời chuẩn (ground truth) rất ngắn gọn.
- **Context Precision (-0.0136):** Duy trì ở mức rất cao (~0.90), chứng minh sự kết hợp giữa BM25 tiếng Việt và Dense Search qua RRF xếp đúng các đoạn trích quan trọng lên đầu.
- **Context Recall (-0.1640):** Giảm do Production RAG chỉ lấy `top_k=3` đoạn sau reranking (thay vì lấy diện rộng), khiến một số câu hỏi phức tạp yêu cầu tổng hợp thông tin từ nhiều nguồn bị thiếu hụt một phần thông tin phụ.

---

## Bottom-5 Failures

### #1. Câu hỏi nghỉ phép kết hôn
- **Question:** "Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?"
- **Expected:** "Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm."
- **Got:** Mô hình trả lời đúng 3 ngày làm việc nhưng bị trừ điểm Faithfulness do tự động suy diễn thêm các thủ tục hồ sơ (nộp giấy đăng ký kết hôn trong 5 ngày) không hoàn toàn khớp với context ngắn trích xuất được.
- **Worst metric:** Faithfulness (thấp do mô hình bổ sung thông tin ngoài đoạn trích rút gọn).
- **Error Tree:** Output thừa điều kiện → Context đúng? Có chứa `nghi_phep_dac_biet.md` nhưng bị chia cắt giữa danh sách và điều kiện → Query OK? Query tốt.
- **Root cause:** Kỹ thuật Hierarchical chunking chia child nhỏ (256 ký tự) làm phần liệt kê "- Kết hôn: 3 ngày" bị tách rời khỏi đoạn ghi chú "không trừ vào phép năm".
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng về số lượng ngày (3 ngày làm việc), nhưng bị xem là suy diễn khi mở rộng thông tin chứng từ.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đáp án trong file `nghi_phep_dac_biet.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không cần viết lại, câu hỏi tự nhiên và thực tế.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 1 (Chunking): sử dụng Structure-Aware chunking để bảo toàn nguyên khối section chế độ nghỉ phép đặc biệt; Module 5 (Enrichment): tóm tắt bối cảnh toàn bộ chính sách trước khi chunk.
- **Suggested fix:** Thắt chặt system prompt để LLM chỉ trả lời đúng số ngày và điều kiện có trong ngữ cảnh, không suy diễn thêm thủ tục chứng từ.

---

### #2. Hạn mức bảo hiểm sức khỏe PVI
- **Question:** "Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?"
- **Expected:** "Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa."
- **Got:** Mô hình trả lời đúng hạn mức 200.000.000 VNĐ nhưng thiếu các phạm vi chi trả (nội trú, ngoại trú, nha khoa) hoặc bị phân tâm bởi quy định bảo hiểm trong file `thu_viec.md`.
- **Worst metric:** Faithfulness / Context Recall
- **Error Tree:** Output thiếu phạm vi chi trả → Context chứa cả `bao_hiem_suc_khoe.md` và `thu_viec.md` → Reranker xếp đoạn thử việc lên cao do có từ khóa "bảo hiểm sức khỏe PVI".
- **Root cause:** Hiện tượng tranh chấp từ khóa giữa tài liệu chính sách bảo hiểm và tài liệu thử việc. Cả hai đều nhắc đến PVI khiến LLM bị chia sẻ sự chú ý (attention dilution).
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng số tiền nhưng chưa đầy đủ phạm vi quyền lợi.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đầy đủ trong `bao_hiem_suc_khoe.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có thể mở rộng thành: "Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là bao nhiêu và bao gồm những quyền lợi gì?".
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 3 (Reranker): Điều chỉnh ngưỡng điểm ưu tiên tài liệu gốc có tiêu đề trùng khớp danh mục chính sách.
- **Suggested fix:** Cải thiện metadata category ("insurance" vs "probation") ở Module 5 để lọc đúng tài liệu bảo hiểm trước khi rerank.

---

### #3. Phụ cấp ăn trưa hàng tháng
- **Question:** "Phụ cấp ăn trưa hàng tháng là bao nhiêu?"
- **Expected:** "Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương."
- **Got:** Mô hình trả lời: "Phụ cấp ăn trưa là 1.000.000 VNĐ/tháng. Phụ cấp này áp dụng cho toàn bộ nhân viên bao gồm nhân viên thử việc từ ngày đầu tiên, không áp dụng cho ngày nghỉ phép, nghỉ ốm hay công tác ngoài."
- **Worst metric:** Answer Relevancy (0.8156)
- **Error Tree:** Output quá dài dòng → Context chứa cả `phu_cap.md` và `thu_viec.md` → LLM cố gắng giải thích toàn bộ ngữ cảnh thay vì trả lời trực diện.
- **Root cause:** Prompt tạo sinh chưa đủ chặt chẽ về tính súc tích, dẫn đến việc mô hình tổng hợp mọi thông tin liên quan trong context.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Hoàn toàn đúng về mặt ngữ nghĩa nhưng điểm Relevancy giảm do thừa thông tin ngoài câu hỏi.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa đầy đủ trong `phu_cap.md`.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không cần, câu hỏi rất tường minh.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module LLM Answer Generation (Prompt Template trong `pipeline.py`).
- **Suggested fix:** Bổ sung hướng dẫn vào system prompt: "Trả lời ngắn gọn, trực diện trong 1 câu, chỉ nêu con số và điều kiện thanh toán được hỏi, không liệt kê ngoại lệ nếu không có yêu cầu."

---

### #4. Xung đột phiên bản: Số ngày nghỉ phép năm
- **Question:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** "Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế."
- **Got:** Mô hình trả lời 12 ngày hoặc 15 ngày nhưng không nêu rõ bối cảnh thay thế chính sách cũ v2023 bằng chính sách mới v2024.
- **Worst metric:** Faithfulness
- **Error Tree:** Xung đột tài liệu (Document Conflict) → Context chứa cả `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày) → BM25 và Dense Search đều thấy cả 2 văn bản có độ tương đồng cực cao.
- **Root cause:** Kho dữ liệu tồn tại văn bản đã hết hiệu lực (superseded document) mà pipeline chưa có cơ chế lọc theo thời gian hoặc trạng thái tài liệu.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Chưa đạt vì không phân biệt được phiên bản nào đang có hiệu lực.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có chứa cả 2 phiên bản mâu thuẫn nhau.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Không, người dùng thực tế không bao giờ biết số phiên bản để chỉ định.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 5 (Enrichment) và Module 2 (Search): Cần trích xuất trường `effective_date` và `status` (active / superseded) vào metadata để Qdrant thực hiện Metadata Filtering.
- **Suggested fix:** Thêm metadata filter loại bỏ các file có tiền tố hoặc nội dung ghi nhận đã bị thay thế, hoặc đưa quy tắc ưu tiên phiên bản mới nhất vào prompt LLM.

---

### #5. Xung đột phiên bản: Thâm niên công tác cộng ngày phép
- **Question:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"
- **Expected:** "Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm."
- **Got:** Mô hình trả lời theo quy định cũ (5 năm) hoặc nêu mâu thuẫn "cứ mỗi 3 năm hoặc 5 năm".
- **Worst metric:** Faithfulness / Context Precision
- **Error Tree:** Xung đột phiên bản → Reranker xếp đoạn văn bản cũ lên trên bản mới do mật độ từ khóa trùng khớp → LLM lấy thông tin từ đoạn văn đứng đầu.
- **Root cause:** Cross-Encoder chỉ đánh giá mức độ tương quan câu chữ (relevance) chứ không đánh giá được tính cập nhật thời gian (temporality).
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Sai hoặc không chính xác do lấy số liệu từ chính sách 2023 đã hết hạn.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có cả hai nguồn thông tin xung đột.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Người dùng hỏi thông thường, hệ thống phải tự đảm bảo tính thời sự của tri thức.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module 2 (Hybrid Search - Metadata filtering) và Module 3 (Reranker có trọng số thời gian Recency-weighted Reranking).
- **Suggested fix:** Áp dụng bộ lọc phiên bản tại Qdrant (`filter={"status": "active"}`) hoặc bổ sung bước Recency Boosting trong công thức RRF.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?"

**Error Tree walkthrough:**
1. *Output đúng?* → Không hoàn toàn: Mô hình dễ trả lời 12 ngày (theo bản v2023) thay vì 15 ngày (theo bản v2024 hiện hành) nếu tài liệu cũ được trích xuất.
2. *Context đúng?* → Bị nhiễu: Context đưa vào LLM bị pha trộn giữa 2 văn bản có nội dung trái ngược nhau do cả hai đều chứa từ khóa "nghỉ phép năm", "nhân viên chính thức".
3. *Query rewrite OK?* → Query nguyên bản đã rõ ràng, nhưng hệ thống chưa tự động nhận diện ý định "áp dụng chính sách hiện hành".
4. *Fix ở bước:* **Module 5 (Auto Metadata Enrichment) kết hợp Module 2 (Qdrant Metadata Filtering).**

**Nếu có thêm 1 giờ, tôi sẽ optimize:**
- **Giải pháp 1 (Metadata Filtering):** Ở Module 5, tự động trích xuất `version` và `superseded_by`. Khi tìm kiếm ở Module 2, cấu hình bộ lọc Qdrant để tự động loại bỏ các chunk có `is_deprecated: true`.
- **Giải pháp 2 (Recency Re-weighting trong RRF):** Trong hàm `reciprocal_rank_fusion()`, nhân thêm hệ số recency boost ($1.2 \times$) cho các văn bản có năm ban hành gần nhất (2024 > 2023).
- **Giải pháp 3 (Temporal Conflict-Aware Prompt):** Thêm vào system prompt của LLM: "Nếu phát hiện hai văn bản có thông tin xung đột, luôn ưu tiên áp dụng chính sách có phiên bản cao hơn hoặc ngày ban hành muộn hơn".
