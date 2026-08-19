import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.json"
QA_SET_PATH = REPO_ROOT / "eval" / "qa_set.jsonl"

REQUIRED_FIELDS = {"id", "question", "bucket", "reference_answer", "citations", "unanswerable", "notes"}
VALID_BUCKETS = {"easy", "medium", "hard-multihop", "hard-unanswerable"}


def load_qa_set():
    pairs = []
    with open(QA_SET_PATH) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"eval/qa_set.jsonl line {lineno}: invalid JSON: {e}")
    return pairs


def main():
    manifest = json.load(open(MANIFEST_PATH))
    file_headings = {f["path"]: f["headings"] for f in manifest["files"]}
    pairs = load_qa_set()

    errors = []
    ids = []
    seen_questions = {}
    bucket_counts = Counter()

    for d in pairs:
        pid = d.get("id", "<missing id>")
        ids.append(pid)
        bucket_counts[d.get("bucket")] += 1

        missing = REQUIRED_FIELDS - d.keys()
        if missing:
            errors.append(f"{pid}: missing fields {missing}")

        if d.get("bucket") not in VALID_BUCKETS:
            errors.append(f"{pid}: invalid bucket {d.get('bucket')!r}")

        is_unanswerable_bucket = d.get("bucket") == "hard-unanswerable"
        if d.get("unanswerable") != is_unanswerable_bucket:
            errors.append(
                f"{pid}: unanswerable={d.get('unanswerable')} inconsistent with bucket {d.get('bucket')!r}"
            )

        citations = d.get("citations", [])
        if is_unanswerable_bucket and citations:
            errors.append(f"{pid}: hard-unanswerable but has citations {citations}")
        if not is_unanswerable_bucket and not citations:
            errors.append(f"{pid}: {d.get('bucket')} but has no citations")
        if d.get("bucket") == "medium" and len(citations) < 2:
            errors.append(f"{pid}: medium bucket but only {len(citations)} citation(s)")
        if d.get("bucket") == "hard-multihop":
            distinct_files = {c.get("file") for c in citations}
            if len(distinct_files) < 2:
                errors.append(f"{pid}: hard-multihop but citations span only {len(distinct_files)} file(s)")

        for c in citations:
            heads = file_headings.get(c.get("file"))
            if heads is None:
                errors.append(f"{pid}: cited file not found in corpus: {c.get('file')}")
                continue
            valid_slugs = {h["slug"] for h in heads}
            if c.get("heading") not in valid_slugs:
                errors.append(f"{pid}: heading slug {c.get('heading')!r} not found in {c.get('file')}")

        q_key = d.get("question", "").strip().lower()
        if q_key in seen_questions:
            errors.append(f"{pid}: near-duplicate question of {seen_questions[q_key]}")
        else:
            seen_questions[q_key] = pid

    dup_ids = {i for i in ids if ids.count(i) > 1}
    if dup_ids:
        errors.append(f"duplicate ids: {dup_ids}")

    print(f"loaded {len(pairs)} pairs")
    print(f"bucket counts: {dict(bucket_counts)}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(" -", e)
        sys.exit(1)

    print("all citations resolve cleanly against corpus/manifest.json")


if __name__ == "__main__":
    main()
