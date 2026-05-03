# ragEnlighten

基于 **LangChain**、**LangGraph** 和 **FastAPI** 构建的 Agentic RAG（检索增强生成）系统。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 文档上传与索引 | 支持 PDF、DOCX、TXT、Markdown 文件，自动切分并向量化存储（Chroma） |
| 文本直接索引 | 可将纯文本片段直接写入知识库 |
| Agentic RAG 问答 | 基于 LangGraph 的三步流水线：检索 → 相关性评分 → 生成 |
| 多集合管理 | 支持按集合（collection）组织不同领域的知识库 |
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
[retrieve]   ── 从向量数据库检索 top-k 相关文档
  │
  ▼
[grade_docs] ── 使用 LLM 过滤不相关文档
  │
  ├─ 无相关文档 ──▶ [no_answer] ──▶ END
  │
  ▼
[generate]   ── 基于相关文档生成回答
  │
  ▼
 END
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
# 编辑 .env，填写 OPENAI_API_KEY 等配置
```

### 3. 启动服务

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

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

---

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | _(必填)_ | OpenAI API 密钥 |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | 生成模型 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型 |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Chroma 向量库持久化目录 |
| `RETRIEVER_TOP_K` | `4` | 每次检索返回的最大文档块数 |
| `APP_HOST` | `0.0.0.0` | 服务监听地址 |
| `APP_PORT` | `8000` | 服务监听端口 |
| `APP_RELOAD` | `false` | 是否开启热重载（开发模式） |
