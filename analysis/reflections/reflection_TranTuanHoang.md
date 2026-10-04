# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Tuấn Hoàng  
**Khóa:** K4 - Track 3B  
**Mã học viên:** 2A202602832  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu chi tiết giữa các khái niệm lý thuyết cốt lõi trong bài giảng Production RAG và các module, hàm cụ thể đã được triển khai trong bài thực hành:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Sử dụng Cosine Distance giữa các câu liên tiếp (tính bằng embedding model `all-MiniLM-L6-v2`) với ngưỡng `breakpoint_percentile_threshold=0.85` hoặc absolute distance `0.3`. Giữ trọn vẹn ngữ nghĩa mạch lạc của từng ý niệm, tránh cắt ngang câu hoặc đoạn luận điểm so với Fixed-size chunking. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo cấu trúc cây Parent-Child (Parent: 1024 ký tự, overlap 128; Child: 256 ký tự, overlap 32). Cho phép retrieve chính xác ở mức độ chi tiết nhỏ (child) nhưng khi feed vào LLM context thì truyền parent rộng hơn để LLM không bị thiếu ngữ cảnh. |
| **Structure-Aware Chunking** | M1 | `chunk_markdown()` | Phân tách theo cấu trúc tiêu đề Markdown (`#`, `##`, `###`), bảng Markdown và code block. Giữ nguyên toàn bộ một bảng hoặc một điều khoản chính sách trong cùng một chunk kèm theo breadcrumbs tiêu đề cha trong metadata. |
| **Vietnamese BM25** | M2 | `BM25Retriever` | Tích hợp thư viện `underthesea` để tách từ ghép tiếng Việt (ví dụ: `nghỉ_phép`, `bảo_hiểm_y_tế`, `thử_việc`) trước khi đưa vào rank-bm25, khắc phục triệt để điểm yếu tách từ theo khoảng trắng của tokenizer tiếng Anh. |
| **Dense Vector Search** | M2 | `DenseRetriever` | Sử dụng thư viện `qdrant-client` ở chế độ in-memory (`:memory:`) hoặc local vector database. Tối ưu hóa cosine similarity search cho các truy vấn diễn đạt tương đồng về ngữ nghĩa nhưng khác từ khóa bề mặt. |
| **Hybrid Search & Fusion** | M2 | `reciprocal_rank_fusion()` | Áp dụng thuật toán Reciprocal Rank Fusion (RRF) với hằng số chuẩn $k=60$. Kết hợp danh sách xếp hạng từ Lexical (BM25) và Semantic (Dense) theo công thức $RRF\_score(d) = \sum \frac{1}{k + rank(d)}$, giúp bù trừ điểm mù của cả 2 phương pháp. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Sử dụng mô hình Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) để đồng thời chấm điểm cặp `(query, document)`. Tích hợp bộ nhớ đệm LRU Cache (`hash(query + text)`) và cơ chế fallback sang `Flashrank` để giảm thiểu độ trễ inference. Lọc từ 20 candidate xuống top 3-5 tài liệu chất lượng nhất. |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `build_failure_analysis()` | Đánh giá định lượng 4 chiều: Faithfulness (độ trung thực), Answer Relevancy (mức độ liên quan của câu trả lời), Context Precision (tỷ lệ context đúng ở thứ hạng cao), Context Recall (độ bao phủ context so với ground truth). Xây dựng cây chẩn đoán (Decision Tree) tự động phân loại nguyên nhân lỗi. |
| **Document Enrichment (Contextual Retrieval)** | M5 | `enrich_chunks_batch()`, `contextual_prepend()` | Thực hiện kỹ thuật Contextual Prepend của Anthropic: LLM tự động tóm tắt 1-2 câu bối cảnh của toàn bộ tài liệu nguồn rồi gắn vào đầu mỗi chunk (`[Nguồn: ... \| Bối cảnh: ...]`). Giúp retrieval recall tăng vọt khi chunk đứng độc lập. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình thực hành Lab 18, tôi đã đối mặt và giải quyết các bài toán kỹ thuật thực tế sau:

### 1. Sự cố phình to thư viện PyTorch GPU trên môi trường WSL CPU
- **Exact error message / Hiện tượng:**  
  Khi chạy `pip install -r requirements.txt`, pip tự động tải `torch` kèm gói NVIDIA CUDA/CUDNN có dung lượng hơn 2.5 GB (`nvidia_cudnn_cu13... 553MB`, `torch-2.14.1... 554MB`) khiến quá trình cài đặt kéo dài hàng chục phút, tràn ổ đĩa và nghẽn băng thông mạng trên máy cá nhân vốn chỉ có CPU.
- **Nguyên nhân gốc rễ:**  
  Cấu hình mặc định của pip tìm kiếm bánh xe nhị phân (wheel) mới nhất của PyTorch trên PyPI vốn tích hợp sẵn CUDA driver.
- **Cách debug & khắc phục:**  
  Tôi đã cập nhật `requirements.txt` bằng cách bổ sung dòng chỉ thị tải bản CPU trực tiếp:  
  `--extra-index-url https://download.pytorch.org/whl/cpu` và chỉ định `torch>=2.2.0+cpu`. Điều này giảm kích thước gói từ 2.5GB xuống còn ~150MB, quá trình cài đặt hoàn thành chỉ trong 1 phút và chạy cực kỳ ổn định trên WSL CPU.

### 2. Hiện tượng pytest bị treo (freeze) khi chạy Module 1
- **Exact error message / Hiện tượng:**  
  Khi gõ lệnh `pytest tests/test_m1.py -v`, pytest dừng lại rất lâu ở `tests/test_m1.py::test_semantic_returns_chunks .` mà không in thêm bất kỳ log nào, trông giống như tiến trình bị deadlock.
- **Nguyên nhân gốc rễ:**  
  Module Semantic Chunking cần mô hình HuggingFace `all-MiniLM-L6-v2` để tính sentence embedding. Lần đầu tiên chạy, thư viện `sentence-transformers` tự động download model weights qua mạng mà không có progress bar hiển thị trong pytest captured output.
- **Cách debug & khắc phục:**  
  Kiểm tra đường dẫn cache tại `~/.cache/huggingface/hub` thấy tiến trình đang ghi đĩa. Tôi đã tách bước khởi tạo mô hình singleton, thêm cơ chế log rõ ràng hoặc sử dụng mock embedding trong unit tests để kiểm tra logic thuật toán chia đoạn độc lập với tốc độ mạng.

### 3. Tích hợp OpenRouter API và xử lý lỗi Rate Limit / Base URL
- **Exact error message / Hiện tượng:**  
  Khi chuyển đổi mô hình LLM sang OpenRouter (`sk-or-v1-...`), thư viện `openai` mặc định gọi đến endpoint `api.openai.com/v1`, dẫn đến lỗi `401 Unauthorized` hoặc `404 Model Not Found`.
- **Nguyên nhân gốc rễ:**  
  OpenRouter yêu cầu `base_url="https://openrouter.ai/api/v1"` và header tùy biến `HTTP-Referer`. Nếu chỉ truyền API key mà không cấu hình base URL thì client sẽ gửi request sai cổng.
- **Cách debug & khắc phục:**  
  Tôi đã cập nhật `config.py` để tự động nhận diện `OPENROUTER_BASE_URL` từ file `.env`, truyền đúng `base_url` và model id (ví dụ `google/gemini-2.0-flash-001`), đồng thời bổ sung `tenacity` retry với exponential backoff để xử lý êm đẹp các lỗi `429 Too Many Requests`.

### 4. Lỗi định dạng JSON Report do giá trị NaN trong RAGAS
- **Exact error message / Hiện tượng:**  
  Khi chạy `python check_lab.py`, script báo lỗi không đọc được `reports/ragas_report.json` (`json.decoder.JSONDecodeError: Invalid number 'NaN'`).
- **Nguyên nhân gốc rễ:**  
  RAGAS tính toán điểm Context Recall trên một số câu hỏi không trích xuất được ground-truth token và trả về giá trị `float('nan')`. Hàm `json.dump()` tiêu chuẩn của Python đôi khi ghi trực tiếp chữ `NaN` không hợp lệ theo chuẩn RFC 8259 của JSON.
- **Cách debug & khắc phục:**  
  Viết một hàm chuẩn hóa `sanitize_floats()` trong `m4_eval.py` để duyệt qua dictionary kết quả, chuyển toàn bộ giá trị `math.isnan(v)` thành `0.0` trước khi serialize ra đĩa, giúp `check_lab.py` đọc dữ liệu trơn tru 100%.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên toàn bộ kiến thức và trải nghiệm thực chiến từ Lab 18, tôi xây dựng kế hoạch ứng dụng kiến trúc Production RAG vào bài toán thực tế của mình:

### Project: Hệ thống Trợ lý Pháp lý & Chính sách Nhân sự Doanh nghiệp (Enterprise Policy & Legal Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản: Chia đoạn văn bản theo Fixed-size (500 ký tự, overlap 50) bằng LangChain RecursiveCharacterTextSplitter, lưu vào ChromaDB, tìm kiếm Semantic thuần túy qua OpenAI embeddings và prompt trực tiếp GPT-3.5-turbo.
- **Vấn đề / Bottlenecks đang gặp:**
  1. *Retrieval Precision thấp (~62%):* Người dùng hỏi các điều khoản chi tiết (ví dụ: "Điều 15 khoản 2 quy định mức phạt bao nhiêu?") thì ChromaDB trả về các điều khoản chung chung không chứa thông tin cụ thể do khoảng cách ngữ nghĩa bị loãng.
  2. *Xung đột phiên bản chính sách:* Doanh nghiệp cập nhật quy chế hàng năm (v2023, v2024). Naive RAG thường xuyên trả về quy định cũ đã hết hiệu lực vì từ khóa trùng khớp cao.
  3. *Hiện tượng Hallucination và dài dòng:* Khi context đưa vào chứa các bảng phụ lục dài, LLM dễ tự bịa đặt các con số hoặc liệt kê toàn bộ văn bản gây quá số token cho phép.

#### 2. Kế hoạch cải tiến toàn diện
1. **Chunking Strategy:**
   - Chuyển từ Fixed-size sang **Structure-Aware Chunking kết hợp Hierarchical Chunking**:
   - Sử dụng regex bóc tách cấu trúc văn bản pháp quy: `Chương > Mục > Điều > Khoản > Điểm`. Mỗi "Điều" đóng vai trò là Parent Chunk (1000 - 1500 ký tự), các "Khoản/Điểm" là Child Chunks (200 - 300 ký tự).
   - Bảo toàn nguyên vẹn cấu trúc bảng biểu phụ cấp bằng Markdown Table chunking.
2. **Search Retrieval (Hybrid RRF):**
   - Triển khai **Hybrid Search** kết hợp giữa:
     - Lexical Search: **BM25 Tiếng Việt** có phân đoạn từ (`underthesea`) để bắt chính xác các danh từ riêng, số hiệu văn bản (ví dụ: `Thông tư 08/2024/TT-BGDĐT`, `PVI`, `1.000.000 VNĐ`).
     - Dense Search: **Qdrant Vector Database** sử dụng embedding tiếng Việt chuyên dụng (`bkai-foundation-models/vietnamese-bi-encoder`).
   - Kết hợp kết quả bằng **Reciprocal Rank Fusion (RRF)** với $k=60$.
3. **Reranking Layer:**
   - Đưa tầng **Cross-Encoder Reranker** vào sau bước retrieval.
   - Lấy top 25 ứng viên từ Hybrid search, sử dụng Cross-Encoder đa ngữ (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) để rerank và chỉ giữ lại top 4 văn bản có điểm số cao nhất chuyển cho LLM.
   - Thêm LRU cache 2048 entries để tối ưu latency cho các câu hỏi phổ biến của nhân viên.
4. **Metadata Enrichment & Contextual Retrieval:**
   - Áp dụng kỹ thuật **Contextual Prepend** từ Module 5: Tự động bổ sung tiền tố ngữ cảnh `[Văn bản: Quy chế Tài chính v2024 | Trạng thái: Có hiệu lực | Ban hành: 15/01/2024]`.
   - Lưu các trường metadata: `effective_date`, `status`, `department`, `doc_type` lên Qdrant để thực hiện **Pre-filtering**, loại bỏ 100% tài liệu đã bị thay thế trước khi tìm kiếm.
5. **Evaluation & Continuous Monitoring:**
   - Xây dựng bộ testset chuẩn gồm 60 câu hỏi có nhãn ground-truth đại diện cho các phòng ban (HR, Pháp chế, Kế toán).
   - Chạy pipeline đánh giá tự động bằng **RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)** định kỳ mỗi tuần hoặc sau mỗi lần cập nhật kho tài liệu.
   - Mục tiêu: Nâng Faithfulness từ 0.72 lên > 0.90 và Context Precision > 0.92.

#### 3. Timeline triển khai (4 tuần)
- **Tuần 1 (Data Processing & Ingestion):**
  - Xây dựng module parser văn bản pháp lý theo cấu trúc Điều/Khoản.
  - Tích hợp `underthesea` và cấu hình Qdrant collection với đầy đủ payload schema metadata.
- **Tuần 2 (Retrieval & Reranking Optimization):**
  - Cài đặt Hybrid Search (BM25 + Dense) và tinh chỉnh trọng số RRF.
  - Triển khai Cross-Encoder reranker, đo đạc latency p95 và tích hợp bộ nhớ đệm cache.
- **Tuần 3 (Enrichment & Version Filtering):**
  - Xây dựng pipeline tự động gắn Contextual Header và lọc trạng thái tài liệu (`is_active: True`).
  - Tối ưu Prompt template cho LLM: yêu cầu dẫn nguồn chính xác theo Điều/Khoản và trả lời súc tích.
- **Tuần 4 (Evaluation & Deployment):**
  - Chạy benchmark RAGAS trên 60 câu hỏi kiểm thử, so sánh số liệu giữa Baseline và Production.
  - Đóng gói Docker container, xây dựng giao diện Streamlit/FastAPI nội bộ và bàn giao nghiệm thu.

