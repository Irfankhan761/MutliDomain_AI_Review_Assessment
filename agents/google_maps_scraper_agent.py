from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from services.google_maps_scraper_service import GoogleMapsScraperService


class GoogleMapsScraperAgent:
    """Specialised collection agent for exact Google Maps place URLs."""

    def __init__(
        self,
        headless: Optional[bool] = None,
        timeout_ms: int = 45_000,
    ) -> None:
        self.service = GoogleMapsScraperService(
            headless=headless,
            timeout_ms=timeout_ms,
        )

    def run(
        self,
        place_url: str,
        output_dir: str | Path,
        max_reviews: int = 100,
        sort_order: str = "most_relevant",
        domain: str = "auto",
    ) -> Dict[str, Any]:
        return self.service.scrape_to_files(
            place_url=place_url,
            output_dir=output_dir,
            max_reviews=max_reviews,
            sort_order=sort_order,
            domain=domain,
        )
