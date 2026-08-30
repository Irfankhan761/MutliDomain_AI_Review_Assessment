"""Project-local model registry for offline execution.

Runtime agents must resolve model folders inside the project (or from explicit
``.env`` overrides) rather than silently downloading models during a demo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(env_path: str | Path = ".env") -> None:
    env_path = Path(env_path)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def enforce_offline_mode(env_path: str | Path = ".env") -> None:
    """Force Hugging Face libraries to use local files only."""
    load_env_file(env_path)
    if os.getenv("LOCAL_MODELS_ONLY", "true").lower() in {"1", "true", "yes"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_model_path(env_key: str, default_relative_path: str) -> Path:
    load_env_file()
    return resolve_project_path(os.getenv(env_key, default_relative_path))


def get_distilbert_sentiment_path() -> Path:
    return get_model_path(
        "DISTILBERT_SENTIMENT_MODEL_PATH", "outputs/models/distilbert_sentiment"
    )


def get_minilm_path() -> Path:
    return get_model_path("MINILM_MODEL_PATH", "outputs/models/all-MiniLM-L6-v2")


def get_rating_model_path() -> Path:
    return get_model_path(
        "RATING_MODEL_PATH", "outputs/models/nlptown_bert_rating"
    )


def _has_any(folder: Path, relative_names: tuple[str, ...]) -> bool:
    return any((folder / name).is_file() for name in relative_names)


def local_model_status() -> Dict[str, dict]:
    """Return detailed, lightweight completeness checks for all three models."""
    enforce_offline_mode()
    paths = {
        "distilbert_sentiment": get_distilbert_sentiment_path(),
        "minilm": get_minilm_path(),
        "rating_model": get_rating_model_path(),
    }

    status: Dict[str, dict] = {}
    for name, path in paths.items():
        missing: list[str] = []
        if not path.is_dir():
            missing.append("model folder")
        elif name == "minilm":
            if not (path / "modules.json").is_file():
                missing.append("modules.json")
            if not _has_any(
                path,
                (
                    "0_Transformer/model.safetensors",
                    "0_Transformer/pytorch_model.bin",
                    "model.safetensors",
                    "pytorch_model.bin",
                ),
            ):
                missing.append("model weights")
        else:
            if not (path / "config.json").is_file():
                missing.append("config.json")
            if not _has_any(path, ("model.safetensors", "pytorch_model.bin")):
                missing.append("model weights")

        status[name] = {
            "path": str(path),
            "ready": not missing,
            "missing": missing,
        }
    return status


def require_local_model(path: str | Path, model_name: str) -> Path:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} local folder not found: {path}\n"
            "Run: python scripts/download_local_models.py --cache-only\n"
            "If cache-only fails, run once online: python scripts/download_local_models.py"
        )
    return path


def require_all_local_models() -> dict[str, Path]:
    """Validate folders, configuration and weight files for every runtime model."""
    status = local_model_status()
    missing = [
        f"{name}: {', '.join(info['missing'])} missing -> {info['path']}"
        for name, info in status.items()
        if not info["ready"]
    ]
    if missing:
        raise FileNotFoundError(
            "Some local models are missing or incomplete:\n"
            + "\n".join(missing)
            + "\n\nRun:\npython scripts/download_local_models.py --cache-only\n"
            + "If cache-only fails, run once online:\npython scripts/download_local_models.py"
        )
    return {name: Path(info["path"]) for name, info in status.items()}
