from __future__ import annotations

"""Build a clean, reproducible delivery ZIP without private/runtime files."""

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIR_NAMES = {
    "venv",
    ".venv",
    "__pycache__",
    ".browser_profiles",
    "uploads",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
EXCLUDED_OUTPUT_DIRS = {
    "ui_runs",
    "google_maps_debug",
    "reports",
    "charts",
    "qa_runs",
    "final_orchestrator_runs",
    "google_maps_scraper_test",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def is_allowed(path: Path, include_models: bool) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    parts = relative.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    if parts and parts[0] == "outputs":
        if len(parts) >= 2 and parts[1] == "models":
            return include_models
        if len(parts) >= 2 and parts[1] in EXCLUDED_OUTPUT_DIRS:
            return False
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean Review Trust AI delivery ZIP")
    parser.add_argument("--output", default=str(PROJECT_ROOT.parent / "Review_Trust_AI_Delivery.zip"))
    parser.add_argument("--include-models", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    files = [
        path for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and is_allowed(path, include_models=args.include_models)
    ]
    files.sort(key=lambda path: str(path.relative_to(PROJECT_ROOT)).lower())

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": "Review Trust AI",
        "models_included": bool(args.include_models),
        "file_count": len(files),
        "files": [],
    }

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            archive.write(path, relative)
            manifest["files"].append(
                {"path": relative, "size": path.stat().st_size, "sha256": sha256(path)}
            )
        archive.writestr(
            "DELIVERY_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    print(f"Delivery ZIP: {output}")
    print(f"Files: {len(files)}")
    print(f"Models included: {args.include_models}")
    print(f"Size: {output.stat().st_size / (1024 * 1024):.2f} MB")
    if not args.include_models:
        print("Note: local AI models must be delivered separately or rebuilt using the setup script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
