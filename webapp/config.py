"""WebApp configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WebAppConfig:
    """Configuration for the web application."""

    title: str = "Satellites Admin"
    debug: bool = False
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent / "templates")
    static_dir: Path = field(default_factory=lambda: Path(__file__).parent / "static")
