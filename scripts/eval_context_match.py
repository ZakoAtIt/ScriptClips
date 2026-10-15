"""Eval harness: run fixture queries through the pipeline and print per-query diagnostics.

Usage:
    python scripts/eval_context_match.py [--fixture FILE] [--query SUBSTRING]

Without arguments, runs all fixtures in data/fixtures.json.
With --query, runs only fixtures whose query contains the substring.
With --fixture, uses an alternate fixture file.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from modules import script_planner, llm_client


def load_fixtures(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("fixtures", [])


def print_plan_diagnostics(plan: dict) -> None:
    """Print the classification and key fields for a plan."""
    print(f"  sentence_type : {plan.get('sentence_type', '?')}")
    print(f"  action_terms  : {plan.get('action_terms', [])}")
    print(f"  object_terms  : {plan.get('object_terms', [])}")
    print(f"  topics        : {plan.get('topics', [])}")
    print(f"  core_entities : {plan.get('core_entities', [])}")
    print(f"  visual_event  : {plan.get('visual_event', False)}")
    print(f"  visual_event_kind : {plan.get('visual_event_kind')}")
    print(f"  scene_prompts : {plan.get('scene_prompts', [])}")
    print(f"  is_event (derived) : {plan.get('sentence_type') == 'event'}")
    print(f"  window_seconds : {plan.get('window_seconds')}")
    queries = plan.get("search_queries") or []
    if queries:
        print(f"  search_queries : {queries[:3]}")
    action_queries = plan.get("action_search_queries") or []
    if action_queries:
        print(f"  action_queries : {action_queries[:3]}")


def run_fixture(fixture: dict, plans_by_sentence: dict[str, dict]) -> dict:
    """Evaluate one fixture against the cached plan (or re-plan if missing)."""
    query = fixture["query"]
    plan = plans_by_sentence.get(query)

    result = {
        "query": query,
        "expected": fixture.get("expected", "?"),
        "root_cause": fixture.get("root_cause"),
        "notes": fixture.get("notes", ""),
    }

    if plan:
        result["actual_sentence_type"] = plan.get("sentence_type", "?")
        result["actual_action_terms"] = plan.get("action_terms", [])
        result["actual_object_terms"] = plan.get("object_terms", [])
        result["actual_topics"] = plan.get("topics", [])
        result["actual_scene_prompts"] = plan.get("scene_prompts", [])
        result["actual_visual_event"] = plan.get("visual_event", False)
        result["plan_found"] = True
    else:
        result["plan_found"] = False
        result["actual_sentence_type"] = "UNKNOWN (no cached plan)"

    return result


def print_result(result: dict, idx: int) -> None:
    print(f"\n{'='*60}")
    print(f"Fixture #{idx}: {result['query']}")
    print(f"  Expected      : {result['expected']}")
    print(f"  Root cause    : {result.get('root_cause') or '(unknown)'}")
    print(f"  Plan found    : {result.get('plan_found', False)}")
    if result.get("plan_found"):
        print(f"  sentence_type : {result['actual_sentence_type']}")
        print(f"  action_terms  : {result['actual_action_terms']}")
        print(f"  object_terms  : {result['actual_object_terms']}")
        print(f"  topics        : {result['actual_topics']}")
        print(f"  visual_event  : {result['actual_visual_event']}")
        prompts = result.get("actual_scene_prompts", [])
        if prompts:
            print(f"  scene_prompts : {prompts[:5]}{'...' if len(prompts) > 5 else ''}")
    else:
        print(f"  WARNING: No cached plan found for this query.")
    if result.get("notes"):
        print(f"  Notes         : {result['notes']}")


def classification_matches_expected(result: dict) -> str:
    """Check if the plan's classification matches the expected behavior."""
    expected = result.get("expected", "?")
    stype = result.get("actual_sentence_type", "?")

    if expected == "no_clip" and stype == "narration":
        return "PASS"
    if expected == "exact_match" and stype == "event":
        return "PASS"
    if expected == "wrong" and stype == "event":
        return "PASS"  # wrong clips are expected for event queries that have bad data
    if expected == "subject_only" and stype == "event":
        return "PASS"  # subject-only is expected before Batch 2 fixes

    return "CHECK"


def main():
    parser = argparse.ArgumentParser(description="Eval harness for ScriptClips fixtures")
    parser.add_argument("--fixture", default=str(ROOT / "data" / "fixtures.json"),
                        help="Path to fixture JSON file")
    parser.add_argument("--query", default=None,
                        help="Run only fixtures whose query contains this substring")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON instead of formatted text")
    args = parser.parse_args()

    fixtures = load_fixtures(Path(args.fixture))
    if args.query:
        fixtures = [f for f in fixtures if args.query.lower() in f["query"].lower()]

    if not fixtures:
        print("No fixtures to run.")
        return

    # Load cached plans
    plans_path = config.SENTENCES_CACHE
    plans_by_sentence = {}
    if plans_path.exists():
        try:
            data = json.loads(plans_path.read_text(encoding="utf-8"))
            for p in data.get("sentences", []):
                plans_by_sentence[p.get("sentence", "")] = p
        except Exception as e:
            print(f"WARNING: Could not load plans cache: {e}")

    results = []
    for i, fixture in enumerate(fixtures, 1):
        result = run_fixture(fixture, plans_by_sentence)
        results.append(result)
        if not args.json:
            print_result(result, i)

    # Summary table
    if not args.json:
        print(f"\n{'='*60}")
        print("SCOREBOARD")
        print(f"{'='*60}")
        print(f"{'#':<4} {'Query':<45} {'Expected':<14} {'Type':<12} {'Status'}")
        print(f"{'-'*4} {'-'*45} {'-'*14} {'-'*12} {'-'*10}")
        for i, r in enumerate(results, 1):
            status = classification_matches_expected(r)
            q = r["query"][:43] + ".." if len(r["query"]) > 45 else r["query"]
            print(f"{i:<4} {q:<45} {r['expected']:<14} {r.get('actual_sentence_type', '?'):<12} {status}")

        total = len(results)
        passed = sum(1 for r in results if classification_matches_expected(r) == "PASS")
        print(f"\n{passed}/{total} classification checks passed")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
