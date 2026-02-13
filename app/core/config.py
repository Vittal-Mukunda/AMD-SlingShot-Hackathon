from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "MCP Project Manager"
    DEBUG: bool = True

settings = Settings()
