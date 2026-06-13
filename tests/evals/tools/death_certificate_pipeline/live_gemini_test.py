import json
import sys
from pathlib import Path

from death_certificate_consistency import analyze_death_certificate_consistency

# --- Config ---
IMAGE_PATH = "image.png"  # Change to your local image path

CHAT_HISTORY = [
    {"role": "user", "content": "My father John Smith passed away on March 3rd, 2021."},
    {"role": "assistant", "content": "I'm sorry for your loss. Can you tell me more about what happened?"},
    {"role": "user", "content": "He was 74 years old and died of heart failure at St. Mary's Hospital in Chicago."},
]

# --- Load image ---
image_path = Path(IMAGE_PATH)
if not image_path.exists():
    print(f"Error: image not found at '{IMAGE_PATH}'")
    print("Update IMAGE_PATH at the top of this script to point to your file.")
    sys.exit(1)

image_bytes = image_path.read_bytes()
print(f"Loaded image: {image_path} ({len(image_bytes):,} bytes)")

# --- Run ---
print("Calling analyze_death_certificate_consistency...\n")
result = analyze_death_certificate_consistency(
    chat_history=CHAT_HISTORY,
    image_bytes=image_bytes,
)

# --- Print results ---
print(json.dumps(result, indent=2))

print(f"\n{'='*50}")
print(f"Consistency: {result['consistency_label'].upper()} ({result['consistency_score']:.2f})")
print(f"Confidence:  {result['confidence']:.2f}")
if result["matches"]:
    print(f"\nMatches ({len(result['matches'])}):")
    for m in result["matches"]:
        print(f"  ✓ {m}")
if result["mismatches"]:
    print(f"\nMismatches ({len(result['mismatches'])}):")
    for m in result["mismatches"]:
        print(f"  ✗ {m}")
if result["uncertain_points"]:
    print(f"\nUncertain ({len(result['uncertain_points'])}):")
    for u in result["uncertain_points"]:
        print(f"  ? {u}")
print(f"\nSummary: {result['summary']}")