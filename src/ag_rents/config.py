"""Configuration management using Pydantic settings."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NASS API Configuration
    nass_api_key: str = Field(default="", description="NASS QuickStats API key")
    nass_base_url: str = Field(
        default="https://quickstats.nass.usda.gov/api",
        description="NASS QuickStats API base URL"
    )
    nass_rate_limit_delay: float = Field(
        default=0.5,
        description="Delay between API requests in seconds"
    )

    # Data paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Project root directory"
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    # Data years
    start_year: int = Field(default=2008, description="First year of cash rents data")
    end_year: int = Field(default=2024, description="Last year of cash rents data")


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
