# ragEnlighten

基于 **LangChain**、**LangGraph** 和 **FastAPI** 构建的 Agentic RAG（检索增强生成）系统。

默认使用 **DeepSeek** 作为问答大模型，使用阿里云百炼兼容 OpenAI 接口的 Embedding 模型进行向量化。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 文档上传与索引 | 支持 PDF、DOCX、TXT、Markdown 文件，自动切分并向量化存储（Chroma） |
| 文本直接索引 | 可将纯文本片段直接写入知识库 |
| Agentic RAG 问答 | 基于 LangGraph 的多步流水线：问题分类 → 规划 → 路由 → 查询改写 → 检索 → 相关性评分 → 多跳补充检索 → 生成 → 校验 |
| 多集合管理 | 支持按集合（collection）组织不同领域的知识库 |
| 校验与人审 | 低置信度答案会重试一次；高风险问题或无证据场景会标记人工复核 |
| OpenAPI 文档 | FastAPI 自动生成交互式 Swagger UI（`/docs`）和 ReDoc（`/redoc`） |

---

## 项目结构

```
ragEnlighten/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── api/
│   │   └── routes.py        # API 路由定义
│   ├── core/
│   │   ├── config.py        # Pydantic 配置管理
│   │   └── graph.py         # LangGraph RAG 工作流
│   ├── models/
│   │   └── schemas.py       # Pydantic 请求/响应模型
│   └── services/
│       └── indexing.py      # 文档加载、切分、向量化服务
├── requirements.txt
├── .env.example
└── README.md
```

---

## LangGraph 工作流

```
START
  │
  ▼
[classify_question] ── 问题分类、风险识别
  │
  ▼
[plan_question] ── 生成执行计划与工具选择
  │
  ├─ internal_api 路由 ──▶ [call_internal_api] ──▶ [generate]
  │
  ├─ rag 路由 ──▶ [rewrite_query] ──▶ [retrieve] ──▶ [grade_docs]
  │                                 │
  │                                 ├─ 相关文档不足且未达到最大 hop ──▶ [rewrite_query]
  │                                 ├─ 无相关文档 ──▶ [no_answer]
  │                                 └─ 相关文档充分 ──▶ [generate]
  │
  └─ direct 路由 ──▶ [generate]
  │
  ▼
[validate_answer] ── 检查置信度与证据支撑
  │
  ├─ 低置信且可重试 ──▶ [reflect_and_retry] ──▶ [rewrite_query]
  ├─ 高风险或需人工确认 ──▶ [human_review]
  └─ 通过校验 ──▶ [finalize] ──▶ END
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 LLM_API_KEY / EMBEDDING_API_KEY 等配置
```

### 3. 启动服务

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 4. 启动 Streamlit 前端

```bash
c:/Users/liman/githubProject/ragEnlighten/.venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

如果你已经先激活了 `.venv`，也可以直接执行 `streamlit run streamlit_app.py`。

默认会连接 `http://127.0.0.1:8004/api/v1`，也可以在页面侧边栏里改成你当前 FastAPI 服务的地址。

前端现在会显示：问题分类、路由结果、执行计划、工具调用记录、校验报告、人工复核标记，以及每个检索片段的 hop 和 score。

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/documents/upload` | 上传文件并建立索引 |
| POST | `/api/v1/documents/text` | 索引纯文本 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除指定文档块 |
| GET  | `/api/v1/collections` | 列出所有知识库集合 |
| POST | `/api/v1/query` | RAG 问答 |

### 示例：上传文档

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf"
```

### 示例：问答

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "文档中提到了哪些核心概念？"}'
```

### `/query` 响应中的关键字段

| 字段 | 说明 |
|------|------|
| `question_type` | 当前问题被归类到哪一类任务 |
| `route` | 当前请求走的是 `rag`、`internal_api` 还是 `direct` |
| `plan` | 后端规划出的执行步骤 |
| `tool_calls` | 本次问答里实际调用过的工具及摘要 |
| `confidence_score` | 答案校验阶段给出的置信度 |
| `validation` | 校验是否通过、是否有证据支撑、存在什么问题 |
| `needs_human_review` | 是否需要人工复核 |
| `sources[].retrieval_score` | 向量召回分数 |
| `sources[].retrieval_hop` | 这个 chunk 是第几跳检索到的 |

---

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | _(必填)_ | DeepSeek API 密钥，用于问答和相关性评分 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | DeepSeek OpenAI-compatible API 地址 |
| `LLM_MODEL` | `deepseek-chat` | 问答模型 |
| `EMBEDDING_API_KEY` | _(索引时必填)_ | 阿里云百炼 API Key；也兼容读取 `DASHSCOPE_API_KEY` |
| `EMBEDDING_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 百炼 OpenAI 兼容接口地址 |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 文本向量模型 |
| `EMBEDDING_DIMENSIONS` | `1024` | 向量维度；可按百炼支持范围调整 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Chroma 向量库持久化目录 |
| `RETRIEVER_TOP_K` | `4` | 每次检索返回的最大文档块数 |
| `RETRIEVAL_MAX_HOPS` | `2` | 最多允许多少跳补充检索 |
| `MIN_RELEVANT_CHUNKS_TO_ANSWER` | `2` | 至少保留多少相关 chunk 才直接进入生成 |
| `MAX_VALIDATION_RETRIES` | `1` | 校验不通过时最多追加多少次自反思重试 |
| `ANSWER_VALIDATION_MIN_CONFIDENCE` | `0.65` | 答案校验阶段的最低通过置信度 |
| `APP_HOST` | `0.0.0.0` | 服务监听地址 |
| `APP_PORT` | `8000` | 服务监听端口 |
| `APP_RELOAD` | `false` | 是否开启热重载（开发模式） |
