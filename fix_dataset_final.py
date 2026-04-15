import json

input_path = "dataset/SODA-D/divData/Annotations/train_fixed.json"
output_path = "dataset/SODA-D/divData/Annotations/train_clean_final.json"

with open(input_path, "r") as f:
    data = json.load(f)

valid_annotations = []
valid_image_ids = set()

for ann in data["annotations"]:
    bbox = ann.get("bbox", [])

    # ✅ KEEP ONLY VALID BBOX
    if (
        isinstance(bbox, list) and
        len(bbox) == 4 and
        all(isinstance(x, (int, float)) for x in bbox) and
        bbox[2] > 0 and bbox[3] > 0
    ):
        valid_annotations.append(ann)
        valid_image_ids.add(ann["image_id"])

# ✅ KEEP ONLY IMAGES WITH VALID ANNOTATIONS
valid_images = [img for img in data["images"] if img["id"] in valid_image_ids]

data["annotations"] = valid_annotations
data["images"] = valid_images

with open(output_path, "w") as f:
    json.dump(data, f)

print("✅ FINAL CLEAN DONE")
print("Images:", len(valid_images))
print("Annotations:", len(valid_annotations))