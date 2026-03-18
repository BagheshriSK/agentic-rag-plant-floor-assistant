from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq (takes priority if set)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenAI fallback
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = "./chroma_db"
    collection_safety: str = "safety_procedures"
    collection_maintenance: str = "maintenance_manuals"
    collection_quality: str = "quality_control_standards"

    top_k: int = 5
    eval_threshold: float = 0.6
    max_retries: int = 1
    conversation_history_window: int = 5

    @property
    def active_chat_model(self) -> str:
        return self.groq_model if self.groq_api_key else self.chat_model

    @property
    def active_judge_model(self) -> str:
        return self.groq_model if self.groq_api_key else self.judge_model

    class Config:
        env_file = ".env"


settings = Settings()
