"""Configuration management module"""
import os
from typing import Optional
from dotenv import load_dotenv

# Try to import BaseSettings from pydantic-settings (required for pydantic v2)
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # If pydantic-settings is not available, try pydantic v1 (deprecated)
    try:
        from pydantic import BaseSettings
    except (ImportError, AttributeError):
        # Last resort: use BaseModel and manually handle env file loading
        from pydantic import BaseModel
        class BaseSettings(BaseModel):
            class Config:
                env_file = ".env"
                case_sensitive = False

load_dotenv()


class Settings(BaseSettings):

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")    
    
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: Optional[str] = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Image generation configuration
    image_gen_provider: str = os.getenv("IMAGE_GEN_PROVIDER", "sd_local")  # "openai" or "sd_local"
    image_gen_model: str = os.getenv("IMAGE_GEN_MODEL", "dall-e-3")
    
    # Stable Diffusion local configuration
    sd_model_path: str = os.getenv("SD_MODEL_PATH", "stabilityai/stable-diffusion-3.5-large")
    sd_device: str = os.getenv("SD_DEVICE", "cuda")  # "cuda" or "cpu"
    sd_dtype: str = os.getenv("SD_DTYPE", "float16")  # "float16" or "float32"
    sd_num_inference_steps: int = int(os.getenv("SD_NUM_INFERENCE_STEPS", "28"))
    sd_guidance_scale: float = float(os.getenv("SD_GUIDANCE_SCALE", "7.0"))
    
    # Output configuration
    output_dir: str = os.getenv("OUTPUT_DIR", "./sample_dataset_news")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Dataset configuration
    # Path to the dataset directory. Can be:
    # - "fake_news_dataset/sample_dataset" - Use sample dataset (50 articles)
    # - "fake_news_dataset" - Use full dataset (all subfolders)
    # - "fake_news_dataset/snopes_downloaded" - Use specific source
    # If None, will try to auto-detect from project structure
    dataset_path: Optional[str] = os.getenv("DATASET_PATH", "fake_news_dataset/sample_dataset")
    
    # Poster configuration
    poster_width: int = 1024
    poster_height: int = 1024
    max_poster_text_length: int = 300  # Maximum length of poster text (must include title + news summary with 5W1H)
    
    # Post text generation method
    # 1: Modified facts (randomly modify extracted facts to introduce errors)
    # 2: Modified evidence (modify evidence/arguments to create false evidence supporting opposite claims)
    post_method: int = int(os.getenv("POST_METHOD", "1"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

