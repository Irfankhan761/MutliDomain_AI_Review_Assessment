from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.google_maps_cli_bridge_service import GoogleMapsCliBridgeService


def main() -> int:
    bridge = GoogleMapsCliBridgeService(project_root=PROJECT_ROOT)
    package_ready = importlib.util.find_spec("playwright") is not None
    script_ready = bridge.script_path.exists()
    cdp_ready = bridge.cdp_is_ready()

    print(f"python_executable: {sys.executable}")
    print(f"playwright_ready: {package_ready}")
    print(f"collector_script_ready: {script_ready}")
    print(f"cdp_url: {bridge.cdp_url}")
    print(f"signed_in_chrome_ready: {cdp_ready}")

    if not package_ready or not script_ready:
        print("status: bridge_not_ready")
        return 1
    if not cdp_ready:
        print("status: start_chrome_session")
        print("command: python scripts/start_google_maps_chrome.py")
        return 2

    print("status: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
