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

def generate_tiles(w,h, tile_size):
    tiles = []
    for x in range(0, w-tile_size, tile_size):
        for y in range(0, h-tile_size, tile_size):
            tiles.append((x, y))
    base_tiles = np.array(tiles)
    vertical_halves = np.array([tiles + [0, tile_size/2] for tiles in base_tiles])
    horizontal_halves = np.array([tiles + [tile_size/2, 0] for tiles in base_tiles])
    intersections = np.array([tiles + [tile_size/2, tile_size/2] for tiles in base_tiles])
    #clear those tiles which gors out of the scope
    vertical_halves = vertical_halves[vertical_halves[:,1] < h]
    horizontal_halves = horizontal_halves[horizontal_halves[:,0] < w]
    intersections = intersections[(intersections[:,0] < w) & (intersections[:,1] < h)]
    tiles = np.concatenate((base_tiles, vertical_halves, horizontal_halves, intersections))
    return base_tiles, vertical_halves, horizontal_halves, intersections, tiles
    
    
parser = argparse.ArgumentParser(description='Generate tiles from an image')
parser.add_argument('--image', type=str,required=True, help='Image to coverage map')
parser.add_argument('--tile_size', type=int, default=1024, help='Tile size')
args = parser.parse_args()

image = Image.open(args.image)
image = image.convert("L")
image.save("gray.png")
width, height = image.size
print("Image size: {}x{}".format(width, height))

base_tiles, vertical_halves, horizontal_halves, intersections, tiles = generate_tiles(width, height, args.tile_size)

fig, ((ax00, ax0, ax1), (ax2, ax3, ax4)) = plt.subplots(2, 3)

#fig, ax = plt.subplots()
#plt.imshow(image)

ax00.imshow(image)
ax00.set_title('original')

ax0.imshow(image)   
ax0.set_title('base_tiles')

ax1.imshow(image)
ax1.set_title('vertical_halves')

ax2.imshow(image)
ax2.set_title('horizontal_halves')

ax3.imshow(image)
ax3.set_title('intersections')

ax4.imshow(image)
ax4.set_title('tiles')

for x, y in base_tiles:
    rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
    ax0.add_patch(rectangle)

for x, y in vertical_halves:
    rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
    ax1.add_patch(rectangle)

for x, y in horizontal_halves:
    rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
    ax2.add_patch(rectangle)

for x, y in intersections:
    rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
    ax3.add_patch(rectangle)
    

def filter_tiles(_image, _tiles, args, axis):
    final_tiles = []
    for x, y in _tiles:
        tile = _image.crop((x, y, x + args.tile_size, y + args.tile_size))
        numpy_tile = np.array(tile)
        binary_tile = numpy_tile > 0
        pixel_sum = np.sum(binary_tile)
        if pixel_sum > 0:
            final_tiles.append((int(x), int(y)))
            if axis is not None:
                rectangle = patches.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color='red')
                axis.add_patch(rectangle)
    return final_tiles

final_tiles = filter_tiles(image, tiles, args, ax4)
base_tiles = filter_tiles(image, base_tiles, args, None)
vertical_halves = filter_tiles(image, vertical_halves, args, None)
horizontal_halves = filter_tiles(image, horizontal_halves, args, None)
intersections = filter_tiles(image, intersections, args, None)

print("Number of tiles: {}".format(len(final_tiles)))

#plt.show()

with open("inference_tiles.json", "w") as f:
    json.dump({
        "canvas_size": [width, height],
        "tile_size": args.tile_size ,
        "tiles": final_tiles,
        "base_tiles": base_tiles,
        "vertical_halves": vertical_halves,
        "horizontal_halves": horizontal_halves,
        "intersections": intersections
        }, f , indent=4)
