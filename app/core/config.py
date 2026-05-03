from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # Embeddings
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    )
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices(
            "EMBEDDING_BASE_URL",
            "DASHSCOPE_BASE_URL",
            "OPENAI_EMBEDDING_BASE_URL",
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-v4",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL"),
    )
    embedding_dimensions: int | None = Field(
        default=1024,
        validation_alias=AliasChoices("EMBEDDING_DIMENSIONS"),
    )

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Retriever
    retriever_top_k: int = 4
    retrieval_max_hops: int = 2
    min_relevant_chunks_to_answer: int = 2
    max_validation_retries: int = 1
    answer_validation_min_confidence: float = 0.65

    # Content preview lengths
    grade_context_chars: int = 500   # chars of each chunk sent to the relevance grader
    rewrite_context_chars: int = 800
    source_preview_chars: int = 300  # chars shown per source in query responses

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False


settings = Settings()
