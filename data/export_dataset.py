import argparse
from PIL import Image
import glob
import json
import os 
import numpy as np
import tqdm
import concurrent.futures

Image.MAX_IMAGE_PIXELS = 933120000

def process_file(args):
    file, tiles, tile_size, output_path, threshold, skip = args  # unpacking the tuple
    if os.path.exists(os.path.join(output_path, f"{tiles[-1][0]}_{tiles[-1][1]}", os.path.basename(file))):
        if skip:
            return
    try:
        image = Image.open(file).convert("L")
    except OSError:
        print(f"Failed to open {file}")
        return
    for tile in tiles:
        os.makedirs(os.path.join(output_path, f"{tile[0]}_{tile[1]}"), exist_ok=True)
        if os.path.exists(os.path.join(output_path, f"{tile[0]}_{tile[1]}", os.path.basename(file))):
            if skip:
                return
            continue
        img = image.crop((tile[0], tile[1], tile[0]+tile_size, tile[1]+tile_size))
        np_img = np.array(img)
        np_binary = np_img > 0
        threshold_val = threshold * tile_size * tile_size
        if np.sum(np_binary) < threshold_val:
            continue
        img.save(os.path.join(output_path, f"{tile[0]}_{tile[1]}", os.path.basename(file)))

def remove_empty_dirs(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            if not os.listdir(item_path):  # Check if directory is empty
                os.rmdir(item_path)
                print(f'Removed empty directory: {item_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dataset tiles to PNG files")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold")
    parser.add_argument("--tiles", type=str, default=None, help="Tiles metadata file")
    parser.add_argument("--skip", action="store_true", help="Skip file processing if some  tile is already processed")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.input, "*.png"))
    if args.tiles is None:
        tiles_meta_path = os.path.join(args.input, "tiles.json")
    else:
        tiles_meta_path = args.tiles
    with open(tiles_meta_path) as f:
        tiles_meta = json.load(f)

    tile_size = tiles_meta["tile_size"]
    tiles = tiles_meta["tiles"]
    os.makedirs(args.output, exist_ok=True)

    # Packing parameters into a tuple for each file
    map_args = [(file, tiles, tile_size, args.output, args.threshold, args.skip) for file in files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm.tqdm(
            executor.map(process_file, map_args),
            total=len(files)
        ))

    remove_empty_dirs(args.output)
