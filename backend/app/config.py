import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _parse_hours(value: str) -> tuple[int, int]:
    start, end = value.split("-")
    return int(start), int(end)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", BACKEND_DIR / "data")).resolve())
    database_url: str = ""
    github_tokens: tuple[str, ...] = ()
    batch_size: int = int(os.getenv("BATCH_SIZE", "200"))
    work_hours: tuple[int, int] = _parse_hours(os.getenv("WORK_HOURS", "9-18"))
    core_threshold: float = float(os.getenv("CORE_THRESHOLD", "0.8"))
    network_window_months: int = int(os.getenv("NETWORK_WINDOW_MONTHS", "3"))
    abandon_months: int = int(os.getenv("ABANDON_MONTHS", "12"))
    pseudonym_salt: str | None = os.getenv("PSEUDONYM_SALT")
    show_real_identities: bool = os.getenv("SHOW_REAL_IDENTITIES", "false").lower() in ("1", "true", "yes")
    heartbeat_timeout_s: int = int(os.getenv("HEARTBEAT_TIMEOUT_S", "120"))
    # ToxiCR has not been validated on GitHub issue/PR comments (in-domain precision was near zero on a
    # sample), so its stage is off until enabled explicitly.
    enable_toxicity: bool = os.getenv("ENABLE_TOXICITY", "false").lower() in ("1", "true", "yes")
    # External text classifiers (installed outside the repository, see README "Text classification").
    external_dir: Path = field(default_factory=lambda: Path(os.getenv("EXTERNAL_DIR", BACKEND_DIR.parent / "external")).resolve())

    def __post_init__(self):
        if not self.database_url:
            object.__setattr__(
                self, "database_url", os.getenv("DATABASE_URL", f"sqlite:///{self.data_dir / 'strainscope.db'}")
            )
        if not self.github_tokens:
            tokens = [t.strip() for t in os.getenv("GITHUB_TOKENS", "").split(",") if t.strip()]
            object.__setattr__(self, "github_tokens", tuple(tokens))

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.repos_dir.mkdir(parents=True, exist_ok=True)
