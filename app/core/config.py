from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ML Model Serving Platform"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    postgres_db: str = "ml_serving"
    postgres_user: str = "postgres"
    postgres_password: str = "ml_dev_password"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()