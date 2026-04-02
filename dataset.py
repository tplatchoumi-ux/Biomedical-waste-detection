import os
import cv2
import numpy as np
import pandas as pd
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR = "Biomedical-Waste-Sample"
IMG_DIR = os.path.join(BASE_DIR, "images")
ANN_DIR = os.path.join(BASE_DIR, "annotations")

NUM_IMAGES = 30
IMG_SIZE = 224

classes = ["Organic_Waste", "Non_Organic_Waste", "Mixed_Waste"]

# -------------------------------
# Create folders
# -------------------------------
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)

# Save classes
with open(os.path.join(BASE_DIR, "classes.txt"), "w") as f:
    for c in classes:
        f.write(c + "\n")

# -------------------------------
# Metadata storage
# -------------------------------
metadata = []
start_time = datetime(2025, 1, 1, 10, 0, 0)

# -------------------------------
# Generate images
# -------------------------------
for i in range(NUM_IMAGES):

    # Create blank image
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    # Random object count
    num_objects = random.randint(1, 3)

    objects = []

    for _ in range(num_objects):
        x1 = random.randint(10, 100)
        y1 = random.randint(10, 100)
        x2 = random.randint(120, 210)
        y2 = random.randint(120, 210)

        label = random.choice(classes)

        # Different colors for classes
        if label == "Organic_Waste":
            color = (0, 180, 0)  # green
        elif label == "Non_Organic_Waste":
            color = (180, 0, 0)  # blue/red
        else:
            color = (0, 180, 180)  # mixed

        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        objects.append((label, x1, y1, x2, y2))

    # Save image
    filename = f"img_{i:03d}.jpg"
    cv2.imwrite(os.path.join(IMG_DIR, filename), img)

    # -------------------------------
    # Create XML Annotation
    # -------------------------------
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(IMG_SIZE)
    ET.SubElement(size, "height").text = str(IMG_SIZE)
    ET.SubElement(size, "depth").text = "3"

    for obj in objects:
        label, x1, y1, x2, y2 = obj

        obj_tag = ET.SubElement(annotation, "object")
        ET.SubElement(obj_tag, "name").text = label

        bbox = ET.SubElement(obj_tag, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(x1)
        ET.SubElement(bbox, "ymin").text = str(y1)
        ET.SubElement(bbox, "xmax").text = str(x2)
        ET.SubElement(bbox, "ymax").text = str(y2)

    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(ANN_DIR, filename.replace(".jpg", ".xml")))

    # -------------------------------
    # Metadata (timestamp + location)
    # -------------------------------
    timestamp = start_time + timedelta(minutes=10 * i)
    metadata.append([filename, timestamp, "Chennai"])

# -------------------------------
# Save metadata
# -------------------------------
df = pd.DataFrame(metadata, columns=["image_id", "timestamp", "location"])
df.to_csv(os.path.join(BASE_DIR, "metadata.csv"), index=False)

print("Sample Biomedical Waste Dataset Created Successfully!")