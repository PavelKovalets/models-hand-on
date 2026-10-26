"""Ad-hoc tries of GLiNER2.5-Decide: edit CASES, then `uv run try.py`."""

import argparse
import json
import time

import torch
from gliner2 import AutoExtractor

DEFAULT_MODEL = "fastino/GLiNER2.5-Decide-1B"

# (text, tasks). A task is a label list, a {label: description} dict, or a dict with
# "labels" plus options: multi_label, cls_threshold, top_k, prompt, examples.
CASES = [
    (
        "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop.",
        {"sentiment": ["positive", "negative", "mixed", "neutral"]},
    ),
    (
        "Hi, I was charged twice for my March invoice and I'd like the duplicate refunded.",
        {
            "intent": {
                "labels": {
                    "refund_request": "wants money back",
                    "billing_question": "asks about a charge without asking for money back",
                    "cancel_subscription": "wants to stop the service",
                    "technical_issue": "something is broken",
                },
            },
            "urgency": ["low", "medium", "high"],
        },
    ),
    (
        "The deploy failed because the database migration timed out; on-call rolled back and paged the DBA.",
        {
            "topics": {
                "labels": ["deployment", "database", "incident", "billing", "hiring"],
                "multi_label": True,
                "cls_threshold": 0.5,
            },
        },
    ),
    (
        "Policy: refunds are allowed within 30 days of purchase. The customer bought the item 45 days ago.",
        {
            "refund_allowed": {
                "labels": ["yes", "no"],
                "prompt": "Is the customer eligible for a refund under the policy?",
            },
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="e.g. fastino/GLiNER2.5-Decide for the 340M")
    parser.add_argument("--fp32", action="store_true", help="keep fp32 weights (default casts to fp16)")
    parser.add_argument("--confidence", action="store_true", help="include label confidences")
    args = parser.parse_args()

    start = time.perf_counter()
    model = AutoExtractor.from_pretrained(args.model, map_location="cuda", quantize=not args.fp32)
    print(f"loaded {args.model} in {time.perf_counter() - start:.1f}s on {torch.cuda.get_device_name()}")
    model.classify_text(*CASES[0])  # warm-up, so the timings below are steady-state

    for text, tasks in CASES:
        start = time.perf_counter()
        result = model.classify_text(text, tasks, include_confidence=args.confidence)
        ms = (time.perf_counter() - start) * 1000
        print(f"\n{text}\n  {json.dumps(result)}  ({ms:.0f} ms)")

    print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
