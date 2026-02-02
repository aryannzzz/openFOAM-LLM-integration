"""
Configuration management for the application
"""
import os
from typing import Optional
from pathlib import Path


class Config:
    """Application configuration"""
    # OpenFOAM settings
    OPENFOAM_PATH: str = os.getenv("OPENFOAM_PATH", "/opt/openfoam")
    OPENFOAM_VERSION: str = os.getenv("OPENFOAM_VERSION", "v2312")
    WORKDIR: str = os.getenv("WORKDIR", "/tmp/foam_simulations")

    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_DEBUG: bool = os.getenv("API_DEBUG", "true").lower() == "true"

    # LLM settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Simulation settings
    MAX_SIMULATION_TIME: int = int(os.getenv("MAX_SIMULATION_TIME", 3600))
    MAX_PARALLEL_SIMULATIONS: int = int(os.getenv("MAX_PARALLEL_SIMULATIONS", 4))
    ENABLE_VISUALIZATION: bool = os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/foam_orchestrator.log")

    # Ensure workdir exists
    Path(WORKDIR).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get configuration instance"""
    return Config()
