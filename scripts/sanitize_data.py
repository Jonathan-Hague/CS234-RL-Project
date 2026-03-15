#!/usr/bin/env python3
"""Sanitize episode data files for public release.

Strips user_content and assistant_content (which contain proprietary
diagram logic and production query text) from episodes.json and
episodes_scored.json. All numeric training fields are preserved intact.

Writes sanitized output to:
  data/episodes.public.json
  data/episodes_scored.public.json

The original files are not modified.
"""

import json
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    ("episodes.json", "episodes.public.json"),
    ("episodes_scored.json", "episodes_scored.public.json"),
]

REDACT_FIELDS = {"user_content", "assistant_content"}


def anonymize_conversation_id(raw_id: str) -> str:
    """Replace raw conversation ID with a stable anonymous hash."""
    return "conv_" + hashlib.sha256(raw_id.encode()).hexdigest()[:24]


def sanitize_episode(episode: dict) -> dict:
    anon_id = anonymize_conversation_id(episode.get("conversation_id", ""))
    sanitized_steps = []
    for step in episode.get("steps", []):
        clean = {
            k: (None if k in REDACT_FIELDS else v)
            for k, v in step.items()
        }
        sanitized_steps.append(clean)
    return {"conversation_id": anon_id, "steps": sanitized_steps}


def sanitize_file(src_name: str, dst_name: str) -> None:
    src = DATA_DIR / src_name
    dst = DATA_DIR / dst_name

    if not src.exists():
        print(f"[skip] {src_name} not found")
        return

    with open(src) as f:
        episodes = json.load(f)

    sanitized = [sanitize_episode(ep) for ep in episodes]

    with open(dst, "w") as f:
        json.dump(sanitized, f, indent=2)

    total_steps = sum(len(ep["steps"]) for ep in sanitized)
    print(f"[done] {src_name} -> {dst_name}  ({len(sanitized)} episodes, {total_steps} steps)")


def main() -> None:
    print(f"Writing sanitized files to {DATA_DIR}\n")
    for src, dst in FILES:
        sanitize_file(src, dst)
    print("\nDone. Verify the .public.json files, then add the originals to .gitignore.")


if __name__ == "__main__":
    main()
