import sys
from PIL import Image
import glob
import os
from tqdm import tqdm

images = []

for f in tqdm(glob.glob(os.path.join(sys.argv[1], "*","*.png"))):
    with Image.open(f) as img:
        images.append(img.copy())

input("Press Enter to continue...")
