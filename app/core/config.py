from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Retriever
    retriever_top_k: int = 4

    # Content preview lengths
    grade_context_chars: int = 500   # chars of each chunk sent to the relevance grader
    source_preview_chars: int = 300  # chars shown per source in query responses

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = False


settings = Settings()
