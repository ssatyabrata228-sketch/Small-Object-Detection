import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Remove invalid bounding boxes from a COCO-format annotation file."
)

parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Path to the input annotation JSON file."
)

parser.add_argument(
    "--output",
    type=Path,
    required=True,
    help="Path to save the cleaned annotation JSON file."
)

args = parser.parse_args()

input_path = args.input
output_path = args.output

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

valid_annotations = []
valid_image_ids = set()

for ann in data["annotations"]:
    bbox = ann.get("bbox", [])

    # Keep only valid bounding boxes
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and bbox[2] > 0
        and bbox[3] > 0
    ):
        valid_annotations.append(ann)
        valid_image_ids.add(ann["image_id"])

# Keep only images that have valid annotations
valid_images = [
    img for img in data["images"]
    if img["id"] in valid_image_ids
]

data["annotations"] = valid_annotations
data["images"] = valid_images

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Dataset cleaned successfully!")
print("Images:", len(valid_images))
print("Annotations:", len(valid_annotations))
print("Output:", output_path)