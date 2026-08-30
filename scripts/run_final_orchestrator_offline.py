from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from services.local_model_registry import enforce_offline_mode, require_all_local_models

enforce_offline_mode(PROJECT_ROOT / ".env")
require_all_local_models()

# Import after offline mode is enforced.
from scripts.run_final_orchestrator import main  # noqa: E402


if __name__ == "__main__":
    main()
