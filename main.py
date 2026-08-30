"""Compatibility entry point for the Review Trust AI Flask application.

The original phase-by-phase monolithic research script is preserved at
``legacy/phase1_monolithic_pipeline.py``. The maintained application entry point
is ``app.py``; ``python main.py`` remains supported for convenience.
"""

from __future__ import annotations

import os

from app import app


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    print("\nReview Trust AI Flask UI running")
    print("Open: http://127.0.0.1:5000")
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=debug_mode,
        use_reloader=False,
        threaded=True,
    )
