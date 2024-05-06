from PIL import Image
import numpy as np
import sys
import glob
Image.MAX_IMAGE_PIXELS = 933120000

files = glob.glob("../samples/files/*.png")

for file in files:
    img = Image.open(file)
    img = img.convert("L")
    img.save(file)
    print(f"Converted {file}")