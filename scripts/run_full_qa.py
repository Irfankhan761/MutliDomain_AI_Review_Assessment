from __future__ import annotations

import argparse
import ast
import compileall
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Development-only/generated folders are deliberately excluded from source QA scans.
# They remain available in the working project but must not be packaged for delivery.
IGNORED_SCAN_DIRS = {
    "venv", ".venv", ".browser_profiles", ".git", "outputs", "uploads",
    "__pycache__", "node_modules",
}

def _is_ignored_for_scan(path: Path) -> bool:
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return True
    if relative.name == ".env":
        return True
    return any(part in IGNORED_SCAN_DIRS for part in relative.parts)

def iter_source_files(pattern: str = "*"):
    for path in PROJECT_ROOT.rglob(pattern):
        if not _is_ignored_for_scan(path):
            yield path


class QAReport:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        status = status.upper()
        self.items.append({"name": name, "status": status, "detail": str(detail)})
        print(f"[{status:4}] {name}" + (f" — {detail}" if detail else ""))

    def run(self, name: str, fn, warning: bool = False) -> Any:
        try:
            value = fn()
            detail = "" if value is None or value is True else str(value)
            self.add(name, "PASS", detail)
            return value
        except Exception as exc:
            self.add(name, "WARN" if warning else "FAIL", f"{type(exc).__name__}: {exc}")
            return None

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["status"] == "FAIL"]

    @property
    def warnings(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["status"] == "WARN"]

    def save(self, output_dir: Path, started_at: str) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "project_root": str(PROJECT_ROOT),
            "summary": {
                "pass": sum(i["status"] == "PASS" for i in self.items),
                "warn": len(self.warnings),
                "fail": len(self.failures),
            },
            "checks": self.items,
        }
        json_path = output_dir / "full_qa_report.json"
        md_path = output_dir / "full_qa_report.md"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = [
            "# Review Trust AI — Full QA Report",
            "",
            f"- Started: {payload['started_at']}",
            f"- Completed: {payload['completed_at']}",
            f"- Python: `{sys.executable}`",
            f"- PASS: {payload['summary']['pass']}",
            f"- WARN: {payload['summary']['warn']}",
            f"- FAIL: {payload['summary']['fail']}",
            "",
            "| Status | Check | Detail |",
            "|---|---|---|",
        ]
        for item in self.items:
            detail = item["detail"].replace("|", "\\|").replace("\n", "<br>")
            lines.append(f"| {item['status']} | {item['name']} | {detail} |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, md_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_required_files() -> str:
    required = [
        "app.py",
        "main.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "templates/index.html",
        "static/app.js",
        "static/app.css",
        "pipeline/final_orchestrator.py",
        "services/google_maps_cli_bridge_service.py",
        "services/google_maps_scraper_service.py",
        "scripts/test_google_maps_scraper.py",
    ]
    missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
    require(not missing, f"Missing files: {missing}")
    return f"{len(required)} required files present"


def check_python_compile() -> str:
    checked = 0
    failures: list[str] = []
    for path in iter_source_files("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            checked += 1
        except Exception as exc:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {type(exc).__name__}: {exc}")
    require(not failures, "Python syntax failures: " + "; ".join(failures[:10]))
    return f"{checked} active Python files compiled without creating bytecode"


def check_js_syntax() -> str:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js not installed; JavaScript syntax check skipped")
    checked = []
    for relative in ("static/app.js", "static/js/app.js"):
        path = PROJECT_ROOT / relative
        if path.exists():
            completed = subprocess.run(
                [node, "--check", str(path)], capture_output=True, text=True, check=False
            )
            require(completed.returncode == 0, completed.stderr or completed.stdout)
            checked.append(relative)
    return ", ".join(checked)


def check_local_imports() -> str:
    py_files = list(iter_source_files("*.py"))
    modules: set[str] = set()
    for path in py_files:
        parts = list(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.add(".".join(parts))

    missing: list[str] = []
    for path in py_files:
        if "legacy" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in {"agents", "services", "pipeline", "scripts"}:
                        candidate = alias.name
                        if candidate not in modules and not any(
                            item.startswith(candidate + ".") for item in modules
                        ):
                            missing.append(f"{path.name}:{node.lineno}:{candidate}")
            if module and module.split(".")[0] in {"agents", "services", "pipeline", "scripts"}:
                if module not in modules and not any(item.startswith(module + ".") for item in modules):
                    missing.append(f"{path.name}:{node.lineno}:{module}")
    require(not missing, "Missing local imports: " + ", ".join(missing))
    return f"{len(py_files)} Python files scanned"


def check_delivery_cleanliness() -> str:
    # This check reports development-only top-level items without recursively walking
    # thousands of virtual-environment or browser-profile files. It is a warning in
    # the working project and becomes relevant only when preparing the delivery ZIP.
    forbidden: list[str] = []
    for name in (".browser_profiles", "venv", ".venv", ".env"):
        if (PROJECT_ROOT / name).exists():
            forbidden.append(name)
    for path in iter_source_files("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            forbidden.append(str(relative))
    require(
        not forbidden,
        "Development/private items must be excluded from the final delivery ZIP: "
        + ", ".join(forbidden[:20]),
    )
    return "No delivery-only exclusions detected"


def check_secret_scan() -> str:
    import re

    patterns = [
        re.compile(r"gsk_[A-Za-z0-9_-]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    ]
    hits = []
    for path in iter_source_files("*"):
        if not path.is_file() or path.suffix.lower() in {".csv", ".pyc", ".png", ".jpg", ".zip"}:
            continue
        if "__pycache__" in path.parts or "legacy" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            if pattern.search(text):
                hits.append(str(path.relative_to(PROJECT_ROOT)))
                break
    require(not hits, f"Possible hard-coded secret found in: {hits}")
    return "No common API-key patterns found in active source"


def check_asset_consistency() -> str:
    app_js = (PROJECT_ROOT / "static/app.js").read_bytes()
    nested_js = (PROJECT_ROOT / "static/js/app.js").read_bytes()
    require(app_js == nested_js, "static/app.js and static/js/app.js differ")
    app_css = (PROJECT_ROOT / "static/app.css").read_bytes()
    nested_css = (PROJECT_ROOT / "static/css/app.css").read_bytes()
    require(app_css == nested_css, "static/app.css and static/css/app.css differ")
    return "Duplicate compatibility assets are synchronised"


def check_sample_datasets() -> str:
    import pandas as pd

    hotel = pd.read_csv(PROJECT_ROOT / "data/processed/orchestrator_hotel_sample.csv")
    require(len(hotel) == 50, f"Hotel sample expected 50 rows, found {len(hotel)}")
    require(hotel["domain"].eq("hotel").all(), "Hotel sample contains blank/non-hotel domains")
    multi = pd.read_csv(PROJECT_ROOT / "data/processed/orchestrator_multidomain_sample.csv")
    require(set(multi["domain"].dropna()) == {"mobile_app", "ecommerce", "hotel", "restaurant"}, "Multi-domain sample is incomplete")
    return "Hotel and multi-domain demonstration samples are valid"


def check_raw_standardisation() -> str:
    import pandas as pd
    import tempfile
    from pipeline.final_orchestrator import FinalOrchestrator

    sources = {
        "mobile_app": PROJECT_ROOT / "data/raw/mobile_app/google_play_reviews.csv",
        "ecommerce": PROJECT_ROOT / "data/raw/ecommerce_amazon/amazon_all_beauty_sample.csv",
        "hotel": PROJECT_ROOT / "data/raw/hotel_europe/Hotel_Reviews.csv",
        "restaurant": PROJECT_ROOT / "data/raw/yelp_restaurant/Yelp Restaurant Reviews.csv",
    }
    with tempfile.TemporaryDirectory() as temp:
        orchestrator = FinalOrchestrator(base_output_dir=temp, use_rag=False, use_groq=False)
        for domain, path in sources.items():
            require(path.exists(), f"Raw source missing: {path}")
            raw = pd.read_csv(path, nrows=20, low_memory=False)
            standard = orchestrator.standardize_known_dataframe(raw, domain, str(path))
            require(len(standard) > 0, f"{domain} standardisation returned no rows")
            require(standard["domain"].eq(domain).all(), f"{domain} labels incorrect")
            require(pd.to_numeric(standard["rating"], errors="coerce").between(1, 5).all(), f"{domain} ratings outside 1-5")
    return "All four raw-domain formats standardised into the common schema"


def check_dependencies() -> str:
    modules = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "flask": "flask",
        "dotenv": "python-dotenv",
        "google_play_scraper": "google-play-scraper",
        "playwright": "playwright",
        "torch": "torch",
        "transformers": "transformers",
        "sentence_transformers": "sentence-transformers",
        "faiss": "faiss-cpu",
        "openai": "openai",
    }
    missing = [package for module, package in modules.items() if importlib.util.find_spec(module) is None]
    require(not missing, "Missing packages: " + ", ".join(missing))
    return f"{len(modules)} runtime dependency imports available"


def check_models() -> str:
    from services.local_model_registry import require_all_local_models

    paths = require_all_local_models()
    return "; ".join(f"{name}={path}" for name, path in paths.items())


def check_common_dataset() -> str:
    import pandas as pd

    path = PROJECT_ROOT / "data/processed/combined_multidomain_reviews.csv"
    require(path.exists(), f"Dataset missing: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {
        "review_id", "domain", "entity_id", "entity_name", "review_text",
        "rating", "rating_original", "review_date", "source", "raw_source_path",
    }
    require(required.issubset(df.columns), f"Missing columns: {sorted(required - set(df.columns))}")
    require(len(df) == 20_000, f"Expected 20,000 rows, found {len(df)}")
    counts = df["domain"].value_counts().to_dict()
    require(set(counts) == {"mobile_app", "ecommerce", "hotel", "restaurant"}, counts)
    require(df["review_text"].fillna("").astype(str).str.strip().ne("").all(), "Blank review text found")
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    require(ratings.notna().all() and ratings.between(1, 5).all(), "Invalid rating found")
    require(not df["review_id"].astype(str).duplicated().any(), "Duplicate review IDs found")
    return str(counts)


def check_preprocessing_regressions() -> str:
    import pandas as pd
    from agents.preprocessing_agent import PreprocessingAgent

    agent = PreprocessingAgent("review_text", "rating")
    df = pd.DataFrame(
        {
            "review_text": ["acceptable", "average", "quite good", "بہت اچھا ہوٹل"],
            "rating": [2.5, 3.0, 3.9, 4.0],
        }
    )
    out = agent.process(df)
    require(len(out) == 4, f"Fractional/Unicode rows were dropped: {len(out)}/4")
    require(out["sentiment_label"].tolist() == ["neutral", "neutral", "neutral", "positive"], out["sentiment_label"].tolist())
    require(bool(out.iloc[-1]["clean_review"]), "Unicode text was erased")
    return "Fractional ratings and Unicode text retained"


def check_orchestrator_regressions() -> str:
    import pandas as pd
    from pipeline.final_orchestrator import FinalOrchestrator

    with tempfile.TemporaryDirectory() as temp:
        orchestrator = FinalOrchestrator(
            base_output_dir=temp, sample_size=10, use_rag=False, use_groq=False
        )
        raw_hotel = pd.DataFrame(
            {
                "Hotel_Name": ["H", "H"],
                "Positive_Review": ["good", "great"],
                "Negative_Review": ["bad", ""],
                "Reviewer_Score": [2.0, 10.0],
                "Average_Score": [8.4, 8.4],
                "Review_Date": ["2020-01-01", "2020-01-02"],
            }
        )
        standard = orchestrator.standardize_hotel(raw_hotel)
        require(standard["rating"].tolist() == [1.0, 5.0], standard["rating"].tolist())

        partial = pd.DataFrame(
            {
                "review_id": ["1"],
                "domain": ["hotel"],
                "entity_id": ["h"],
                "entity_name": ["Hotel"],
                "review_text": ["good"],
                "rating": [5],
            }
        )
        completed = orchestrator.standardize_known_dataframe(partial, "hotel")
        require(set(orchestrator.STANDARD_COLUMNS) == set(completed.columns), completed.columns.tolist())

        unbalanced = pd.DataFrame(
            {
                "domain": ["a"] + ["b"] * 10 + ["c"] * 10,
                "row": list(range(21)),
            }
        )
        sample = orchestrator.sample_dataframe(unbalanced, 9)
        require(len(sample) == 9, f"Expected 9, got {len(sample)}")
        require(not sample["row"].duplicated().any(), "Duplicate sample rows found")

    return "Hotel scores, partial schema and exact stratified sampling passed"


def check_url_security() -> str:
    from services.google_maps_cli_bridge_service import GoogleMapsCliBridgeService
    from services.google_maps_scraper_service import GoogleMapsScraperService

    valid = [
        "https://www.google.com/maps/place/Test/@1,2,17z",
        "https://www.google.co.uk/maps/place/Test/@1,2,17z",
        "https://www.google.com.pk/maps/place/Test/@1,2,17z",
        "https://maps.app.goo.gl/abc123",
    ]
    invalid = [
        "https://evilgoogle.com/maps/place/Test",
        "https://google.com.evil.example/maps/place/Test",
        "file:///etc/passwd",
        "https://example.com/maps/place/Test",
    ]
    for url in valid:
        GoogleMapsCliBridgeService.validate_place_url(url)
        GoogleMapsScraperService.validate_place_url(url)
    for url in invalid:
        for validator in (
            GoogleMapsCliBridgeService.validate_place_url,
            GoogleMapsScraperService.validate_place_url,
        ):
            try:
                validator(url)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsafe URL accepted: {url}")
    return f"{len(valid)} valid accepted; {len(invalid)} unsafe rejected"


def check_pipeline_smoke(output_dir: Path) -> str:
    import pandas as pd
    from pipeline.multidomain_review_analysis_pipeline import MultiDomainReviewAnalysisPipeline

    sample = pd.read_csv(
        PROJECT_ROOT / "data/processed/orchestrator_multidomain_sample.csv"
    ).head(20)
    pipeline = MultiDomainReviewAnalysisPipeline(
        model_path=PROJECT_ROOT / "outputs/models/distilbert_sentiment",
        use_transformer=False,
        use_discrepancy_model=False,
        use_semantic_issue_model=False,
        use_rag=False,
        output_dir=output_dir / "pipeline_smoke",
    )
    result = pipeline.analyze(sample, save_outputs=True)
    review_df = result["review_level_results"]
    require(len(review_df) == 20, f"Expected 20 processed rows, got {len(review_df)}")
    require(review_df["trust_score"].between(20, 100).all(), "Trust score outside 20-100")
    require((review_df["risk_score"] == 100 - review_df["trust_score"]).all(), "Risk/trust mismatch")
    required_files = [
        "multidomain_review_level_results.csv",
        "multidomain_entity_level_summary.csv",
        "multidomain_pipeline_trace.json",
    ]
    missing = [name for name in required_files if not (output_dir / "pipeline_smoke" / name).exists()]
    require(not missing, f"Missing output files: {missing}")
    return f"{len(review_df)} rows; {len(result['entity_level_summary'])} entities"


def check_flask_routes() -> str:
    from app import app

    client = app.test_client()
    health = client.get("/api/health")
    require(health.status_code == 200, f"Health status {health.status_code}")
    bad_mode = client.post("/api/analyze", data={"mode": "unsupported"})
    require(bad_mode.status_code == 400, f"Invalid mode status {bad_mode.status_code}")
    bad_number = client.post(
        "/api/analyze", data={"mode": "csv", "sample_size": "not-a-number"}
    )
    require(bad_number.status_code == 400, f"Invalid number status {bad_number.status_code}")
    return "Health 200; invalid requests 400"


def run_command(command: list[str], timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
    return (completed.stdout or "").strip()[-2000:]


def live_google_maps(url: str, output_dir: Path, max_reviews: int) -> str:
    result_json = output_dir / "live_google_maps_result.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/test_google_maps_scraper.py"),
        url,
        "--max-reviews", str(max_reviews),
        "--sort", "most_relevant",
        "--domain", "auto",
        "--cdp",
        "--output-dir", str(output_dir / "live_google_maps"),
        "--result-json", str(result_json),
    ]
    run_command(command, timeout=1800)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    require(payload.get("status") == "completed", payload)
    require(int(payload.get("scraped_count", 0)) >= 1, payload)
    return f"{payload.get('entity_name')}: {payload.get('scraped_count')} reviews"


def live_google_play(value: str, output_dir: Path, max_reviews: int) -> str:
    from services.google_play_scraper_service import GooglePlayScraperService

    service = GooglePlayScraperService()
    df = service.scrape_reviews(value, count=max_reviews, sort_order="newest")
    require(not df.empty, "No Google Play reviews returned")
    path = service.save_reviews(df, str(output_dir / "live_google_play"))
    return f"{len(df)} reviews saved to {path}"


def live_groq() -> str:
    from services.groq_client import GroqClient

    result = GroqClient(env_path=PROJECT_ROOT / ".env").test_connection()
    require(result.get("status") == "success", result)
    return f"Model {result.get('model')} replied successfully"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-delivery QA for Review Trust AI")
    parser.add_argument("--google-maps-url", default="")
    parser.add_argument("--google-play-input", default="")
    parser.add_argument("--max-reviews", type=int, default=5)
    parser.add_argument("--run-groq", action="store_true")
    parser.add_argument("--deep-model-test", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs/qa_runs" / stamp
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = QAReport()
    report.add("Python interpreter", "PASS", f"{sys.executable} — {sys.version.split()[0]}")
    report.run("Required project files", check_required_files)
    report.run("Delivery package cleanliness", check_delivery_cleanliness, warning=True)
    report.run("Hard-coded secret scan", check_secret_scan)
    report.run("Python compilation", check_python_compile)
    report.run("JavaScript syntax", check_js_syntax, warning=True)
    report.run("Local import integrity", check_local_imports)
    report.run("Static asset consistency", check_asset_consistency)
    report.run("Runtime dependencies", check_dependencies)
    report.run("Local model completeness", check_models)
    report.run("Common dataset integrity", check_common_dataset)
    report.run("Demonstration sample integrity", check_sample_datasets)
    report.run("Raw-domain standardisation", check_raw_standardisation)
    report.run("Preprocessing regression tests", check_preprocessing_regressions)
    report.run("Orchestrator regression tests", check_orchestrator_regressions)
    report.run("Google Maps URL security", check_url_security)
    report.run("Offline pipeline smoke test", lambda: check_pipeline_smoke(output_dir))
    report.run("Flask route smoke test", check_flask_routes)

    if args.deep_model_test:
        report.run(
            "Deep local model inference",
            lambda: run_command(
                [sys.executable, str(PROJECT_ROOT / "scripts/test_local_offline_models.py")],
                timeout=1200,
            ),
        )
    else:
        report.add("Deep local model inference", "WARN", "Not requested; use --deep-model-test")

    if args.google_maps_url:
        report.run(
            "Live Google Maps acceptance test",
            lambda: live_google_maps(args.google_maps_url, output_dir, args.max_reviews),
        )
    else:
        report.add("Live Google Maps acceptance test", "WARN", "No URL supplied")

    if args.google_play_input:
        report.run(
            "Live Google Play acceptance test",
            lambda: live_google_play(args.google_play_input, output_dir, args.max_reviews),
        )
    else:
        report.add("Live Google Play acceptance test", "WARN", "No app input supplied")

    if args.run_groq:
        report.run("Live Groq connection", live_groq)
    else:
        report.add("Live Groq connection", "WARN", "Not requested; use --run-groq")

    json_path, md_path = report.save(output_dir, started_at)
    print("\nQA JSON:", json_path)
    print("QA Markdown:", md_path)
    print(f"Summary: PASS={sum(i['status']=='PASS' for i in report.items)} WARN={len(report.warnings)} FAIL={len(report.failures)}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
