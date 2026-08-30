from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.google_maps_scraper_agent import GoogleMapsScraperAgent


def _write_result(path: Path | None, payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _success_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "completed",
        "prepared_csv": result.get("prepared_csv"),
        "raw_csv": result.get("raw_csv"),
        "metadata_json": result.get("metadata_json"),
        "domain": result.get("domain"),
        "entity_id": result.get("entity_id"),
        "entity_name": result.get("entity_name"),
        "category": result.get("category"),
        "overall_rating": result.get("overall_rating"),
        "displayed_review_count": result.get("displayed_review_count"),
        "scraped_count": result.get("scraped_count"),
        "resolved_url": result.get("resolved_url"),
        "sort_order_applied": result.get("sort_order_applied"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Google Maps reviews and create a common-schema CSV."
    )
    parser.add_argument("url", help="Exact public Google Maps place URL")
    parser.add_argument("--max-reviews", type=int, default=10)
    parser.add_argument(
        "--sort",
        default="most_relevant",
        choices=["most_relevant", "newest", "highest_rating", "lowest_rating"],
    )
    parser.add_argument(
        "--domain",
        default="auto",
        choices=["auto", "hotel", "restaurant"],
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show a locally launched browser instead of headless mode.",
    )
    parser.add_argument(
        "--cdp",
        action="store_true",
        help="Attach to the signed-in Chrome debugging session.",
    )
    parser.add_argument(
        "--cdp-url",
        default=os.environ.get("GOOGLE_MAPS_CDP_URL", "http://127.0.0.1:9222"),
        help="Chrome DevTools endpoint used with --cdp.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "google_maps_scraper_test"),
        help="Job-specific directory for raw, prepared and metadata files.",
    )
    parser.add_argument(
        "--result-json",
        default="",
        help="Optional machine-readable result contract written after the run.",
    )
    args = parser.parse_args()

    result_json = Path(args.result_json).resolve() if args.result_json else None

    try:
        if args.headed:
            os.environ["GOOGLE_MAPS_HEADLESS"] = "false"
        if args.cdp:
            os.environ["GOOGLE_MAPS_CDP_URL"] = str(args.cdp_url).rstrip("/")

        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print("collector_status: starting", flush=True)
        print(f"collector_output_dir: {output_dir}", flush=True)

        agent = GoogleMapsScraperAgent()
        result = agent.run(
            place_url=args.url,
            output_dir=output_dir,
            max_reviews=args.max_reviews,
            sort_order=args.sort,
            domain=args.domain,
        )

        payload = _success_payload(result)
        _write_result(result_json, payload)

        print("Google Maps scraper completed.", flush=True)
        for key, value in payload.items():
            if key != "status":
                print(f"{key}: {value}", flush=True)
        return 0

    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if os.environ.get("GOOGLE_MAPS_INCLUDE_TRACEBACK", "false").lower() in {"1", "true", "yes"}:
            payload["traceback"] = traceback.format_exc()
        _write_result(result_json, payload)
        print(f"Google Maps scraper failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
