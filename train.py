"""
train.py — CleanStream model training script.

Usage:
    # Retrain the audio classifier from labeled examples
    py -3.13 train.py audio

    # Add a labeled example manually
    py -3.13 train.py add "what the hell" mild

    # Review flagged detections and label them
    py -3.13 train.py review
"""

import sys
import json
from pathlib import Path

TRAINING_DATA = Path("models/audio/training_data.jsonl")
FLAGGED_LOG   = Path("flagged_audio.jsonl")


def cmd_train_audio():
    from models.audio.classifier import train
    print("\nTraining audio classifier...")
    stats = train()
    if "error" in stats:
        print(f"Error: {stats['error']}")
    else:
        print(f"Done! Accuracy: {stats['accuracy']*100:.1f}%")
        print(f"Examples: {stats['examples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Model saved to: {stats['model_path']}")


def cmd_add(text: str, label: str):
    valid = ("clean", "mild", "moderate", "severe")
    if label not in valid:
        print(f"Label must be one of: {valid}")
        sys.exit(1)
    from models.audio.classifier import add_example
    add_example(text, label)
    print(f"Added: [{label}] {text!r}")


def cmd_review():
    """Interactive CLI to review flagged detections and label them."""
    if not FLAGGED_LOG.exists():
        print("No flagged detections to review yet.")
        return

    from models.audio.classifier import add_example

    with open(FLAGGED_LOG) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    unreviewed = [e for e in entries if not e.get("reviewed")]
    if not unreviewed:
        print("All detections have been reviewed.")
        return

    print(f"\n{len(unreviewed)} unreviewed detections. Commands: c=clean, m=mild, o=moderate, s=severe, skip=skip\n")

    reviewed_indices = []
    for i, entry in enumerate(unreviewed):
        print(f"[{i+1}/{len(unreviewed)}] Detected as: {entry['severity'].upper()}")
        print(f"  Text: {entry['text']!r}")
        print(f"  Hits: {entry['hits']}")
        choice = input("  Label (c/m/o/s/skip): ").strip().lower()

        label_map = {"c": "clean", "m": "mild", "o": "moderate", "s": "severe"}
        if choice in label_map:
            add_example(entry["text"], label_map[choice])
            entry["reviewed"] = True
            reviewed_indices.append(i)
            print(f"  → Labeled as {label_map[choice]}")
        elif choice == "skip":
            print("  → Skipped")
        else:
            print("  → Invalid input, skipping")

    # Update flagged log with reviewed status
    all_entries = entries.copy()
    reviewed_count = 0
    for entry in all_entries:
        if entry["text"] in [unreviewed[i]["text"] for i in reviewed_indices]:
            entry["reviewed"] = True
            reviewed_count += 1

    with open(FLAGGED_LOG, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nReviewed {reviewed_count} detections.")
    if reviewed_count > 0:
        retrain = input("Retrain classifier now? (y/n): ").strip().lower()
        if retrain == "y":
            cmd_train_audio()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]

    if cmd == "audio":
        cmd_train_audio()
    elif cmd == "add" and len(args) == 3:
        cmd_add(args[1], args[2])
    elif cmd == "review":
        cmd_review()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
