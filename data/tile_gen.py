import argparse
import os
import sys
import PIL
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def compute_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + args.tile_size, rect1[1] + args.tile_size
    xx1, yy1, xx2, yy2 = rect2[0], rect2[1], rect2[0] + args.tile_size, rect2[1] + args.tile_size

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1, xx1)
    y_top = max(y1, yy1)
    x_right = min(x2, xx2)
    y_bottom = min(y2, yy2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (xx2 - xx1) * (yy2 - yy1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

parser = argparse.ArgumentParser(description='Generate tiles from an image')
parser.add_argument('--image', type=str,required=True, help='Image to coverage map')
parser.add_argument('--tile_size', type=int, default=512, help='Tile size')
parser.add_argument('--threshold', type=float, default=.3, help='Threshold for coverage map')
parser.add_argument('--iou_threshold', type=float, default=.5, help='IoU Threshold for filtering rectangles')
args = parser.parse_args()

image = Image.open(args.image)
image = image.convert("L")
image.save("gray.png")
width, height = image.size
print("Image size: {}x{}".format(width, height))

fig, ax = plt.subplots()

plt.imshow(image)
rectangles = []
while len(rectangles) < 2000:
    x = random.randint(0, width - args.tile_size)
    y = random.randint(0, height - args.tile_size)
    new_rect = (x, y)
    
    # Check IoU with existing rectangles
    duplicate = False
    for rect in rectangles:
        if compute_iou(new_rect, rect) > args.iou_threshold:
            duplicate = True
            break
    
    if not duplicate:
        rectangles.append(new_rect)
        tile = image.crop((x, y, x + args.tile_size, y + args.tile_size))
        numpy_tile = np.array(tile)
        binary_tile = numpy_tile > 0
        pixel_sum = np.sum(binary_tile)
        pixel_value = pixel_sum / (args.tile_size * args.tile_size)
        if pixel_value > args.threshold:
            print("Tile position: {}x{}, num:{}".format(x, y, len(rectangles)))
            rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
            ax.add_patch(rectangle)

plt.show()

with open("tiles.json", "w") as f:
    json.dump({"tile_size": args.tile_size ,"tiles": rectangles}, f)
    #tile.save("tile_{}_{}.png".format(x, y))
