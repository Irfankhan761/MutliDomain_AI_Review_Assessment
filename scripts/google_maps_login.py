from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.google_maps_scraper_service import GoogleMapsScraperService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the dedicated Google Maps scraper profile for one-time manual sign-in."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.google.com/maps?hl=en",
        help="Optional Google Maps place URL to open after the browser starts.",
    )
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise SystemExit("Playwright is not installed. Run: pip install playwright") from exc

    service = GoogleMapsScraperService(headless=False, slow_mo_ms=100)

    print("\nA dedicated scraper Chrome profile will open.")
    print("1. Click Sign in inside that browser window.")
    print("2. Log in manually. Do NOT put your email/password in .env or Python code.")
    print("3. Confirm Google Maps is working and the Reviews panel opens.")
    print("4. Return to this terminal and press Enter.\n")
    print(f"Profile directory: {service._profile_dir()}")

    with sync_playwright() as playwright:
        context = service._launch_persistent_context(playwright)
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(service.timeout_ms)
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=service.timeout_ms)
        except Exception:
            page.goto("https://www.google.com/maps?hl=en", wait_until="domcontentloaded")

        input("\nAfter login is complete, press Enter here to save and close the profile... ")
        context.close()

    print("\nLogin session saved. The Google Maps scraper will reuse this profile.")
    print("Keep .browser_profiles/ private and do not upload it or commit it to Git.")


if __name__ == "__main__":
    main()
