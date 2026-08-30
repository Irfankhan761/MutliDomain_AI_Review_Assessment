from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.issue_mining_agent import IssueMiningAgent


CASES = [
    {
        "name": "mixed_positive_wording_with_explicit_content_concern",
        "domain": "mobile_app",
        "rating": 1.0,
        "predicted_sentiment": "positive",
        "text": (
            "I want watch interesting things but not about sex or shamefull pictures. "
            "Knowledge is great like science, new discoveries etc."
        ),
        "expected": "inappropriate_content",
    },
    {
        "name": "protective_absence_should_not_be_issue",
        "domain": "mobile_app",
        "rating": 5.0,
        "predicted_sentiment": "positive",
        "text": "Great science app. It has no sexual or inappropriate pictures.",
        "expected": "no_issue",
    },
    {
        "name": "clean_positive_review",
        "domain": "mobile_app",
        "rating": 5.0,
        "predicted_sentiment": "positive",
        "text": "Great app for learning science and discovering interesting facts.",
        "expected": "no_issue",
    },
]


def main():
    df = pd.DataFrame(
        [
            {
                "domain": case["domain"],
                "rating": case["rating"],
                "score": case["rating"],
                "predicted_sentiment": case["predicted_sentiment"],
                "clean_review": case["text"],
                "review_text": case["text"],
            }
            for case in CASES
        ]
    )

    agent = IssueMiningAgent()
    out = agent.process(df, text_column="clean_review")

    columns = [
        "primary_issue",
        "issue_label",
        "primary_issue_similarity",
        "whole_review_issue_similarity",
        "issue_semantic_margin",
        "issue_threshold_used",
        "issue_acceptance_threshold_used",
        "matched_taxonomy_cues",
        "issue_detection_reason",
        "issue_evidence_segment",
        "issue_severity_level",
    ]

    failures = []
    for i, case in enumerate(CASES):
        row = out.iloc[i]
        actual = str(row["primary_issue"])
        print("\n" + "=" * 88)
        print(case["name"])
        print("expected:", case["expected"])
        print("actual  :", actual)
        for col in columns:
            if col in out.columns:
                print(f"{col:34} = {row[col]}")
        if actual != case["expected"]:
            failures.append((case["name"], case["expected"], actual))

    print("\n" + "=" * 88)
    if failures:
        print("REGRESSION RESULT: FAIL")
        for name, expected, actual in failures:
            print(f"- {name}: expected={expected}, actual={actual}")
        raise SystemExit(1)

    print("REGRESSION RESULT: PASS")
    print("All context-aware issue-mining checks passed.")


if __name__ == "__main__":
    main()
