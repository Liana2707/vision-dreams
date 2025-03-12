from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_dir: str
    num_cores: int
    url: str
    
    model_config = SettingsConfigDict(env_file='.env')