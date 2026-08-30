from __future__ import annotations

"""Reliable subprocess bridge between the Flask UI and Google Maps collector CLI.

The browser scraper remains independently testable through
``scripts/test_google_maps_scraper.py``.  The web application invokes that same
working command with the current virtual-environment interpreter, validates its
structured result contract, and then sends the generated prepared CSV into the
existing CSV analysis pipeline.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable
import pandas as pd

from services.google_maps_url_utils import validate_google_maps_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GoogleMapsCliBridgeError(RuntimeError):
    """Raised when the collector command or its result contract fails."""


class GoogleMapsCliBridgeService:
    ALLOWED_SORTS = {
        "most_relevant",
        "newest",
        "highest_rating",
        "lowest_rating",
    }
    ALLOWED_DOMAINS = {"auto", "hotel", "restaurant"}
    REQUIRED_COMMON_COLUMNS = {
        "review_id",
        "domain",
        "entity_id",
        "entity_name",
        "review_text",
        "rating",
        "rating_original",
        "review_date",
        "source",
        "raw_source_path",
    }

    def __init__(
        self,
        project_root: str | Path = PROJECT_ROOT,
        timeout_seconds: int | None = None,
        cdp_url: str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.script_path = self.project_root / "scripts" / "test_google_maps_scraper.py"
        configured_timeout = timeout_seconds or int(
            os.environ.get("GOOGLE_MAPS_BRIDGE_TIMEOUT_SECONDS", "1800")
        )
        self.timeout_seconds = max(60, min(int(configured_timeout), 7200))
        self.cdp_url = (
            cdp_url
            or os.environ.get("GOOGLE_MAPS_CDP_URL")
            or "http://127.0.0.1:9222"
        ).rstrip("/")

    @staticmethod
    def validate_place_url(value: str) -> str:
        return validate_google_maps_url(value)

    @classmethod
    def normalize_sort(cls, value: str) -> str:
        aliases = {
            "relevant": "most_relevant",
            "relevance": "most_relevant",
            "highest": "highest_rating",
            "lowest": "lowest_rating",
        }
        normalized = aliases.get(str(value or "most_relevant").strip().lower(), str(value or "most_relevant").strip().lower())
        if normalized not in cls.ALLOWED_SORTS:
            raise ValueError(f"Unsupported Google Maps sort order: {value}")
        return normalized

    @classmethod
    def normalize_domain(cls, value: str) -> str:
        normalized = str(value or "auto").strip().lower().replace("-", "_")
        if normalized not in cls.ALLOWED_DOMAINS:
            raise ValueError(f"Unsupported Google Maps domain: {value}")
        return normalized

    @staticmethod
    def normalize_max_reviews(value: int | str) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Max reviews must be a whole number.") from exc
        if count < 1:
            raise ValueError("Max reviews must be at least 1.")
        return min(count, 1000)

    def cdp_is_ready(self, timeout: float = 1.5) -> bool:
        try:
            with urllib.request.urlopen(f"{self.cdp_url}/json/version", timeout=timeout) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError, ValueError):
            return False

    @staticmethod
    def _tail(text: str, lines: int = 18) -> str:
        chunks = [line for line in str(text or "").splitlines() if line.strip()]
        return "\n".join(chunks[-lines:])

    @staticmethod
    def _is_within(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def _resolve_artifact_path(self, value: Any, output_dir: Path, label: str) -> Path:
        if not value:
            raise GoogleMapsCliBridgeError(f"Collector result did not include {label}.")
        path = Path(str(value))
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        if not self._is_within(path, output_dir):
            raise GoogleMapsCliBridgeError(
                f"Collector returned an unsafe {label} path outside its job folder: {path}"
            )
        if not path.exists() or not path.is_file():
            raise GoogleMapsCliBridgeError(f"Collector {label} was not created: {path}")
        return path

    def _validate_prepared_csv(self, csv_path: Path, expected_count: int | None = None) -> int:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise GoogleMapsCliBridgeError(
                f"Prepared Google Maps CSV could not be read: {exc}"
            ) from exc

        missing = sorted(self.REQUIRED_COMMON_COLUMNS.difference(df.columns))
        if missing:
            raise GoogleMapsCliBridgeError(
                "Prepared Google Maps CSV does not match the common schema. "
                f"Missing columns: {', '.join(missing)}"
            )

        if df.empty:
            raise GoogleMapsCliBridgeError("Prepared Google Maps CSV contains no usable reviews.")

        text_count = int(df["review_text"].fillna("").astype(str).str.strip().ne("").sum())
        if text_count < 1:
            raise GoogleMapsCliBridgeError("Prepared Google Maps CSV contains no review text.")

        if expected_count is not None and int(expected_count) != len(df):
            # Do not fail for a harmless metadata mismatch; report the validated row count.
            return int(len(df))
        return int(len(df))

    def collect(
        self,
        place_url: str,
        output_dir: str | Path,
        max_reviews: int = 100,
        sort_order: str = "most_relevant",
        domain: str = "auto",
        use_cdp: bool = True,
    ) -> Dict[str, Any]:
        url = self.validate_place_url(place_url)
        count = self.normalize_max_reviews(max_reviews)
        selected_sort = self.normalize_sort(sort_order)
        selected_domain = self.normalize_domain(domain)

        if not self.script_path.exists():
            raise GoogleMapsCliBridgeError(
                f"Google Maps collector script was not found: {self.script_path}"
            )

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        result_path = output_path / "collector_result.json"
        stdout_path = output_path / "collector_stdout.log"
        stderr_path = output_path / "collector_stderr.log"

        if use_cdp and not self.cdp_is_ready():
            raise GoogleMapsCliBridgeError(
                "The signed-in Google Maps Chrome session is not available at "
                f"{self.cdp_url}. Run `python scripts/start_google_maps_chrome.py`, "
                "sign in once, keep that Chrome window open, and run the analysis again."
            )

        command = [
            sys.executable,
            str(self.script_path),
            url,
            "--max-reviews",
            str(count),
            "--sort",
            selected_sort,
            "--domain",
            selected_domain,
            "--output-dir",
            str(output_path),
            "--result-json",
            str(result_path),
        ]
        if use_cdp:
            command.extend(["--cdp", "--cdp-url", self.cdp_url])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["GOOGLE_MAPS_CDP_URL"] = self.cdp_url if use_cdp else env.get("GOOGLE_MAPS_CDP_URL", "")

        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            stdout_path.write_text(str(stdout), encoding="utf-8", errors="ignore")
            stderr_path.write_text(str(stderr), encoding="utf-8", errors="ignore")
            raise GoogleMapsCliBridgeError(
                f"Google Maps collection exceeded {self.timeout_seconds} seconds and was stopped."
            ) from exc

        duration_seconds = round(time.monotonic() - started, 2)
        stdout_path.write_text(completed.stdout or "", encoding="utf-8", errors="ignore")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8", errors="ignore")

        result_payload: Dict[str, Any] = {}
        if result_path.exists():
            try:
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise GoogleMapsCliBridgeError(
                    "Google Maps collector created an invalid result JSON file."
                ) from exc

        if completed.returncode != 0 or result_payload.get("status") == "failed":
            detail = (
                result_payload.get("error")
                or self._tail(completed.stderr)
                or self._tail(completed.stdout)
                or f"collector exited with code {completed.returncode}"
            )
            raise GoogleMapsCliBridgeError(f"Google Maps collector failed: {detail}")

        if result_payload.get("status") != "completed":
            raise GoogleMapsCliBridgeError(
                "Google Maps collector finished without a completed result contract."
            )

        prepared_path = self._resolve_artifact_path(
            result_payload.get("prepared_csv"), output_path, "prepared CSV"
        )
        raw_path = self._resolve_artifact_path(
            result_payload.get("raw_csv"), output_path, "raw CSV"
        )
        metadata_path = self._resolve_artifact_path(
            result_payload.get("metadata_json"), output_path, "metadata JSON"
        )
        validated_count = self._validate_prepared_csv(
            prepared_path, result_payload.get("scraped_count")
        )

        result_payload.update(
            {
                "prepared_csv": str(prepared_path),
                "raw_csv": str(raw_path),
                "metadata_json": str(metadata_path),
                "result_json": str(result_path.resolve()),
                "stdout_log": str(stdout_path.resolve()),
                "stderr_log": str(stderr_path.resolve()),
                "scraped_count": validated_count,
                "duration_seconds": duration_seconds,
                "collector_python": sys.executable,
                "collection_method": "subprocess_cli_bridge",
                "cdp_url": self.cdp_url if use_cdp else "",
            }
        )
        return result_payload
