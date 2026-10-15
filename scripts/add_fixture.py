"""Log a bad production result as a fixture for the eval harness.

Usage:
    python scripts/add_fixture.py \
      --query "caseoh guesses burgers correct stream" \
      --expected exact_match \
      --root_cause weak_clip_verb \
      [--video_id abc123] [--timestamp 1606] [--notes "wrong clip exported"]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = ROOT / "data" / "fixtures.json"

VALID_ROOT_CAUSES = (
    "entity_only_passed",
    "false_narration_classification",
    "weak_clip_verb",
    "lexicon_gap",
    "search_recall_miss",
    "window_misaligned",
    "other",
)

VALID_EXPECTED = ("exact_match", "subject_only", "wrong", "no_clip")


def load_fixtures() -> dict:
    if FIXTURES_PATH.exists():
        return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return {"version": 1, "description": "Fixture set for eval harness.", "fixtures": []}


def save_fixtures(data: dict) -> None:
    FIXTURES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Add a fixture to the eval set")
    parser.add_argument("--query", required=True, help="The query text")
    parser.add_argument("--expected", required=True, choices=VALID_EXPECTED,
                        help="Expected behavior")
    parser.add_argument("--root_cause", required=True, choices=VALID_ROOT_CAUSES,
                        help="Root cause category")
    parser.add_argument("--video_id", default=None, help="Known video ID (if any)")
    parser.add_argument("--timestamp", type=float, default=None,
                        help="Known timestamp (if any)")
    parser.add_argument("--notes", default="", help="Free-text notes")
    args = parser.parse_args()

    data = load_fixtures()

    # Check for duplicate query
    for f in data["fixtures"]:
        if f["query"] == args.query:
            print(f"WARNING: Fixture for this query already exists (expected={f['expected']}). Updating.")
            f["expected"] = args.expected
            f["root_cause"] = args.root_cause
            f["known_video_id"] = args.video_id
            f["known_timestamp"] = args.timestamp
            f["notes"] = args.notes
            save_fixtures(data)
            print(f"Updated fixture for: {args.query}")
            return

    fixture = {
        "query": args.query,
        "expected": args.expected,
        "root_cause": args.root_cause,
        "known_video_id": args.video_id,
        "known_timestamp": args.timestamp,
        "notes": args.notes,
    }
    data["fixtures"].append(fixture)
    save_fixtures(data)
    print(f"Added fixture #{len(data['fixtures'])}: {args.query}")
    print(f"  expected: {args.expected}")
    print(f"  root_cause: {args.root_cause}")


if __name__ == "__main__":
    main()
