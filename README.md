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
| React Agent 聊天接口 | 新增 `/api/v1/chat/react-agent`，基于 ReAct 模式按需调用知识库检索工具完成问答，支持传入多轮历史消息 |
| React Agent 流式接口 | 新增 `/api/v1/chat/react-agent/stream`，以 SSE（`text/event-stream`）持续返回 token、trace 和最终结果，适合前端逐步渲染 |
| Plan Execute Agent 接口 | 新增 `/api/v1/chat/plan-execute-agent` 和 `/api/v1/chat/plan-execute-agent/stream`，先生成完整计划，再按计划顺序执行 |
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

## React Agent 聊天接口

新增接口：`POST /api/v1/chat/react-agent`

流式接口：`POST /api/v1/chat/react-agent/stream`

Plan Execute Agent 接口：`POST /api/v1/chat/plan-execute-agent`

Plan Execute Agent 流式接口：`POST /api/v1/chat/plan-execute-agent/stream`

适用场景：

- 希望使用 ReAct Agent 模式，由模型自主决定何时调用知识库检索工具
- 希望使用 Plan Execute Agent 模式，让模型先产出完整计划，再按计划执行
- 需要让 Agent 通过 `web_search` 工具获取外部或较新的 Web 信息
- 需要传入历史消息，支持多轮问答上下文
- 不想替换现有 LangGraph `/query` 流程，而是并行保留两种问答模式

请求示例：

```json
{
  "question": "总结一下这个候选人的项目经历",
  "collection_name": "resume_bank",
  "history": [
    {
      "role": "user",
      "content": "先告诉我他的主要技术方向"
    },
    {
      "role": "assistant",
      "content": "他主要集中在后端服务和 RAG 应用开发。"
    }
  ]
}
```

返回结构与 `/api/v1/query` 基本一致，仍然包含：

- `answer`
- `tool_calls`
- `sources`
- `trace`
- `confidence_score`
- `validation`

Agent 模式现在可用的工具包括：

- `knowledge_base_search`：检索已索引的向量知识库内容，支持可选 `source_types`（如 `pdf`、`docx`、`txt`、`md`）和 `top_k` 过滤
- `collection_overview`：查看当前模型、集合等服务元信息
- `web_search`：检索外部 Web 信息，默认使用 `mock` provider，返回确定性的标题、URL 和摘要，便于本地开发和测试；也可以配置 `tavily` provider 调用 Tavily Search API 获取真实 Web 结果

这些工具现在通过共享 Agent 工具服务层执行，并返回统一结构化结果（`status`、`content`、`sources`、`error`）。本地 LangChain 工具会把 `sources` 继续映射到 `QueryResponse.sources`，供 grounding 校验和前端展示使用。

### Agent MCP 工具模式（方案 C）

默认情况下，ReAct Agent 仍使用进程内本地工具（`MCP_ENABLED=false`）。如果希望把同一套工具暴露为 MCP server，并让 Agent 作为 MCP client 调用，可以启用 MCP 双路径模式：

```env
MCP_ENABLED=true
MCP_TOOLS_URL=http://127.0.0.1:8001/mcp
MCP_TOOL_NAME_PREFIX=true
MCP_FALLBACK_TO_LOCAL=true
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001
AGENT_TOOL_TIMEOUT_SECONDS=15
AGENT_TOOL_DEFAULT_MAX_RETRIES=1
```

启动 MCP tool server：

```bash
python -m app.mcp.server
```

行为说明：

- `MCP_ENABLED=false`：ReAct Agent 使用本地 `@tool` 包装器，行为与现有开发模式兼容。
- `MCP_ENABLED=true`：ReAct Agent 通过 MCP 协议加载 `knowledge_base_search`、`collection_overview`、`web_search`。
- MCP 加载失败且 `MCP_FALLBACK_TO_LOCAL=true` 时，会自动回退到本地工具，并在 `debug_events` 中记录 `mcp_load_error`。
- MCP 工具返回的结构化 `sources` 会被 React Agent 重新合并进 grounding 上下文，保证 `sources` 和 validation 不丢失。

`web_search` 不会新增 API schema；调用记录仍进入 `tool_calls`、`trace` 和 `debug`，搜索结果会以 source-compatible 形式进入 `sources`。

默认配置使用 `mock` provider，不需要外部凭据。要启用 Tavily：

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
WEB_SEARCH_API_KEY=tvly-...
WEB_SEARCH_TOP_K=3
WEB_SEARCH_TIMEOUT_SECONDS=10
WEB_SEARCH_TAVILY_SEARCH_DEPTH=basic
WEB_SEARCH_TAVILY_INCLUDE_RAW_CONTENT=false
WEB_SEARCH_TAVILY_MAX_RAW_CONTENT_CHARS=1000
```

Tavily provider 会把 Tavily `results` 映射为现有的 `title`、`url`、`snippet` 结构。缺少 API key、HTTP 错误或无可用结果时，错误会记录为失败的 `web_search` 工具输出，不会改变 Agent endpoint 的响应 schema。

### ReAct Agent 与 Plan Execute Agent 的区别

- ReAct Agent 是“边想边做”：模型在循环中动态决定是否调用工具、调用什么工具以及下一步怎么做。
- Plan Execute Agent 是“先规划再执行”：模型先输出完整 `plan`，后续执行阶段只按这个计划顺序调用工具并汇总答案，不再让模型临时追加工具决策。
- 如果需要更强适应性，使用 ReAct Agent；如果需要更容易审计、展示和复盘的执行过程，使用 Plan Execute Agent。

### LangSmith 可观测（React / Plan-Execute）

可为 React Agent 与 Plan-Execute Agent 启用 [LangSmith](https://smith.langchain.com/) tracing，在 UI 中查看 LLM 调用、工具链路与请求元数据。默认关闭，不影响现有 API 行为。

在 `.env` 中配置（由应用的 `LANGSMITH_*` 变量控制；启用后应用会写入 `LANGCHAIN_TRACING_V2` 等 LangChain 兼容环境变量）：

```env
LANGSMITH_TRACING_ENABLED=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ragEnlighten
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

`LANGSMITH_API_KEY` 为必填项：若只开启 `LANGSMITH_TRACING_ENABLED` 而未配置 API key，服务会记录警告且不会导出 trace。

启动服务后调用 `/api/v1/chat/react-agent`、`/api/v1/chat/react-agent/stream`、`/api/v1/chat/plan-execute-agent` 或 `/api/v1/chat/plan-execute-agent/stream`，trace 会写入上述 project。

每次 API 请求在 LangSmith 中应呈现为一棵层级 trace：根 run 为 `react-agent-query` / `react-agent-stream` / `plan-execute-query` / `plan-execute-stream`，其下嵌套 planning、execution、step、synthesis 或 LangChain agent 子 run。

在 LangSmith 中可按 metadata 筛选（根 run 与 execution step 子 run 均包含这些字段）：

| 字段 | 含义 |
|------|------|
| `route` | `react_agent` 或 `plan_execute_agent` |
| `endpoint` | 具体 API 路径 |
| `streaming` | `true` / `false` |
| `collection_name` | 使用的向量集合 |
| `phase` | Plan-Execute：`planning` / `execution` / `synthesis` |

常见 run 名称：`react-agent-query`、`react-agent-stream`、`plan-execute-query`、`plan-execute-stream`、`plan-execute-planning`、`plan-execute-execution`、`plan-execute-synthesis`，以及 `plan-execute-step-{id}` 逐步执行子 run。

未启用 tracing 时，Agent 响应与 SSE 事件格式与启用前一致。

流式接口返回 SSE 协议，响应类型为 `text/event-stream`。每个事件使用 `event:` 标识事件类型，`data:` 携带 JSON 数据，常见事件类型：

- `plan`：Plan Execute Agent 规划完成后的有序步骤
- `trace`：执行轨迹更新
- `debug`：Agent 内部阶段、工具调用和 grounding 状态
- `token`：增量文本片段
- `final`：最终完整响应对象
- `error`：流执行错误

SSE 事件示例：

```text
event: trace
data: "1. Query accepted by React Agent stream API"

event: plan
data: ["1. Search the knowledge base (knowledge_base_search)", "2. Synthesize answer (answer_synthesis)"]

event: token
data: "这是增量输出片段"

event: final
data: {"question":"...","answer":"...","sources":[]}
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

默认会连接 `http://127.0.0.1:8000/api/v1`，也可以在页面侧边栏里改成你当前 FastAPI 服务的地址。

前端现在会显示：问题分类、路由结果、执行计划、工具调用记录、校验报告、人工复核标记，以及每个检索片段的 hop 和 score。

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/documents/upload` | 上传文件并建立索引 |
| POST | `/api/v1/documents/text` | 索引纯文本 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除指定文档块 |
| DELETE | `/api/v1/documents/source/delete` | 按 source 删除该文档的全部向量块 |
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
| `MCP_ENABLED` | `false` | 是否让 ReAct Agent 通过 MCP client 加载工具 |
| `MCP_TOOLS_URL` | `http://127.0.0.1:8001/mcp` | Agent MCP client 连接地址 |
| `MCP_FALLBACK_TO_LOCAL` | `true` | MCP 加载失败时是否回退到本地工具 |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `15` | 共享 Agent 工具执行超时预算（秒） |
| `AGENT_TOOL_DEFAULT_MAX_RETRIES` | `1` | 共享 Agent 工具默认重试次数 |
| `APP_HOST` | `0.0.0.0` | 服务监听地址 |
| `APP_PORT` | `8000` | 服务监听端口 |
| `APP_RELOAD` | `false` | 是否开启热重载（开发模式） |
