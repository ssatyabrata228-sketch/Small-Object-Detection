import json

input_path = r"C:\Users\sahoo\CFINet\dataset\SODA-D\divData\Annotations\train.json"
output_path = r"C:\Users\sahoo\CFINet\dataset\SODA-D\divData\Annotations\train_fixed.json"

with open(input_path, 'r') as f:
    data = json.load(f)

valid_annotations = []
valid_image_ids = set()

for ann in data['annotations']:
    bbox = ann.get('bbox', [])

    # ✅ keep only valid boxes
    if (
        isinstance(bbox, list) and
        len(bbox) == 4 and
        bbox[2] > 0 and
        bbox[3] > 0
    ):
        valid_annotations.append(ann)
        valid_image_ids.add(ann['image_id'])

# ✅ keep only images that have valid boxes
valid_images = [img for img in data['images'] if img['id'] in valid_image_ids]

data['annotations'] = valid_annotations
data['images'] = valid_images

with open(output_path, 'w') as f:
    json.dump(data, f)

print("✅ Dataset cleaned!")
print("Images:", len(valid_images))
print("Annotations:", len(valid_annotations))