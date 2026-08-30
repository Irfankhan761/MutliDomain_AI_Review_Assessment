from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_chrome() -> Path:
    configured = os.environ.get("GOOGLE_CHROME_PATH", "").strip()
    candidates = [
        configured,
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        str(Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise SystemExit(
        "Chrome/Edge was not found. Set GOOGLE_CHROME_PATH to the browser executable."
    )


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def endpoint_ready(cdp_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{cdp_url}/json/version", timeout=1) as response:
            json.loads(response.read().decode("utf-8"))
            return response.status == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the private signed-in Chrome session used by the Google Maps collector."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument(
        "--profile-dir",
        default=str(PROJECT_ROOT / ".browser_profiles" / "google_maps_cdp"),
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    cdp_url = f"http://{args.host}:{args.port}"

    if port_open(args.host, args.port) and endpoint_ready(cdp_url):
        print(f"Google Maps Chrome is already ready at {cdp_url}")
        print("Keep that browser open. The Flask UI can now run Google Maps jobs.")
        return 0

    chrome = find_chrome()
    command = [
        str(chrome),
        f"--remote-debugging-address={args.host}",
        f"--remote-debugging-port={args.port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://www.google.com/maps?hl=en",
    ]
    subprocess.Popen(command, close_fds=True)

    for _ in range(60):
        if endpoint_ready(cdp_url):
            break
        time.sleep(0.5)
    else:
        raise SystemExit(
            "Chrome opened, but its debugging endpoint did not become ready. "
            "Close duplicate scraper Chrome windows and try again."
        )

    print("\nDedicated Google Maps Chrome is ready.")
    print("1. Sign in manually once if Google asks.")
    print("2. Keep this Chrome window open while using the Flask UI.")
    print("3. Start the application normally with: python app.py")
    print(f"CDP endpoint: {cdp_url}")
    print(f"Private session profile: {profile_dir}")
    print("No email or password is stored in project code or .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
