from PIL import Image
import numpy as np
import sys
import glob
import time

Image.MAX_IMAGE_PIXELS = 933120000
input("Press Enter to continue...")

files = glob.glob("../samples/files/*.png")
images = []

for file in files:
    start_time = time.time()  # Save the start time
    images.append(np.array(Image.open(file).convert("L")))
    end_time = time.time()  # Save the end time
    
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to load {file}: {elapsed_time:.2f} seconds")

input("Press Enter to continue...")
