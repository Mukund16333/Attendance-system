import os
from PIL import Image

BASE_DIR = "faces"   # root folder containing person folders

def convert_to_rgb(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                img.save(image_path)
                print(f"✅ Converted to RGB: {image_path}")
            else:
                print(f"✔ Already RGB: {image_path}")
    except Exception as e:
        print(f"❌ Failed: {image_path} | {e}")

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(root, file)
            convert_to_rgb(full_path)

print("\n🎯 All images processed successfully.")
