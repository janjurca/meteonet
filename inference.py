import argparse
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
from io import BytesIO
from time import sleep
import logging
import os 
from PIL import Image
import torch
from einops import rearrange
import torchvision.transforms as T
from autoencoder.autoencoder_vit import ViTAutoencoder 
import numpy as np
from io import BytesIO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#check for mps
if torch.backends.mps.is_available():
    device = "mps"


config = {
    "embed_dim" : 4,
    "double_z" : False,
    "channels" : 384,
    "resolution" : 256,
    "timesteps" : 12,
    "skip" : 1,
    "in_channels" : 1,
    "out_ch" : 1,
    "num_res_blocks" : 2,
    "attn_resolutions" : [],
    "splits" : 1,
}
MODEL = None

Image.MAX_IMAGE_PIXELS = 933120000

logger = logging.getLogger(__name__)


class ImageContainer:
    def __init__(self, timestamp:int, image:bytes) -> None:
        self.timestamp = timestamp
        self.image = image
    def __call__(self, resolution: tuple[int, int] | None=None) -> Image.Image:
        print(f"extracting {self.timestamp}")
        if resolution:
            return Image.open(BytesIO(self.image)).convert("L").resize(resolution)
        return Image.open(BytesIO(self.image)).convert("L")

class ImageLoader:
    def __init__(self, url="https://api.rainviewer.com/public/weather-maps.json", n_images=12) -> None:
        self.url : str = url
        self.host : str | None= None
        self.available_timestamps : dict[int, str]  = dict()
        self.data : dict[int,ImageContainer] = dict()
        self.n_images = n_images

    def fetch_available_timestamps(self) -> None:
        response = requests.get(self.url)
        response.raise_for_status()
        data = response.json()
        self.host = data["host"]
        for x in data["radar"]["past"]:
            self.available_timestamps[int(x["time"])] = f"{self.host}{x['path']}/8000/0/0_1.png"
 
    def download_image(self, url: str, path: str) -> bytes:
        print(f"Downloading {url}")
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    
    def fetch_new_images(self) -> bool:
        ret = False
        self.fetch_available_timestamps()
        for timestamp, path in self.available_timestamps.items():
            if timestamp not in self.data:
                self.data[timestamp] = ImageContainer(timestamp,self.download_image(path, str(timestamp)))
                print(f"Loaded {timestamp}")
                ret = True
        all_timestamps = sorted(list(self.data.keys()))
        for timestamp in all_timestamps[:-self.n_images]:
            del self.data[timestamp]
        return ret
    
    def __len__(self) -> int:
        return len(self.data)
    
    def max_timestamp(self) -> int:
        return max(self.data.keys())
    
    def __str__(self) -> str:
        return f"ImageLoader({len(self.data)}, {self.max_timestamp()})"

def predict_batch(batch, iteration = 1) -> torch.Tensor:
    x = rearrange(batch / 127.5 - 1, 'b t c h w -> b c t h w')
    with torch.no_grad():
        x = x.to(device)
        predicted = MODEL(x)
    predicted = predicted[0]
    predicted = rearrange(predicted, '(b t) c h w -> b t c h w', b=x.shape[0], t=x.shape[2])
    predicted = (predicted + 1) * 127.5
    iteration -= 1
    if iteration > 0:
        next_iteration_prediction = predict_batch(predicted)
        # append next iteration to original prediction
        predicted = torch.cat((predicted, next_iteration_prediction), dim=1)
    return predicted

def predict(tile_dict, batch_size: int):
    ret = {}
    batch = []
    batch_tilecords = []
    for tilecoord, sequence in tqdm(tile_dict.items(), desc="Predicting"):
        array_sequence = []
        for image in sequence:
            array_sequence.append(T.ToTensor()(image) * 255)
        array_sequence = torch.stack(array_sequence)
        if array_sequence.view(-1, array_sequence.shape[-1]).sum() <= 50:
            ret[tilecoord] = sequence
            continue
        
        batch.append(array_sequence)
        batch_tilecords.append(tilecoord)
        if len(batch) == batch_size or tilecoord == list(tile_dict.keys())[-1]:
            batch = torch.stack(batch)
            predicted = predict_batch(batch)
            for i, (tilecoord, sequence) in enumerate(zip(batch_tilecords, predicted)):
                #sequence is in this format t c h w
                sequence = sequence.squeeze(1)
                ret[tilecoord] = [Image.fromarray(img.cpu().numpy().astype(np.uint8)) for img in sequence]
            batch = []  
            batch_tilecords = []
    return ret

def get_tile_sequences(image_sequence, tiles, resolution):
    ret_tiles = {}
    tile_size = tiles["tile_size"]
    print("#tiles: ", len(tiles["tiles"]))

    for tile in tiles["tiles"]:
        x, y = tile[0], tile[1]
        ret_tiles[tuple(tile)] = [
            image.crop((x, y, x + tile_size, y + tile_size)).resize(
                (resolution, resolution)
            )
            for image in image_sequence
        ]

    return ret_tiles

def get_nth_images(tiles, n):
    return {tilecoord: sequence[n] for tilecoord, sequence in tiles.items()}

def merge_tiles(tiles, image_tiles, index):
    tile_groups = [
        "base_tiles",
        "vertical_halves",
        "horizontal_halves",
        "intersections",
    ]

    canvases = {key: Image.new("L", tiles["canvas_size"]) for key in tile_groups}

    for group in tile_groups:
        for tilecoord in tiles[group]:
            image = np.array(image_tiles[tuple(tilecoord)], dtype=np.uint8)
            image = Image.fromarray(image).convert("L").resize(
                (tiles["tile_size"], tiles["tile_size"]), resample=Image.NEAREST
            )
            canvases[group].paste(image, tilecoord)

    # Merge canvases
    np_canvases = np.median([np.array(canvases[group]) for group in tile_groups], axis=0)
    np_canvases = np_canvases.astype(np.uint8)
    return index, Image.fromarray(np_canvases)

def run(tiles, args, loader: ImageLoader) -> None:
    query = []
    print("Running for", loader)
    max_timestamp = loader.max_timestamp()
    
    #for i in range(loader.n_images):
    #    query.append(loader.data[max_timestamp - 600 * (loader.n_images - i - 1)]())
    for timestamp in sorted(loader.data.keys(), reverse=True)[:loader.n_images]:
        query.append(loader.data[timestamp](resolution=tuple(tiles["canvas_size"])))
    
    print("Getting tile sequences")
    tile_sequences = get_tile_sequences(query, tiles, tiles["tile_size"])
    del query
    print("Got tile sequences")
    results = []    
    predicted_tiles_sequences = predict(tile_sequences, args.batch_size)
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(loader.n_images):
            image_tiles = get_nth_images(predicted_tiles_sequences, i)
            with open(args.tiles, "r") as f:
                tiles = json.load(f)
            futures.append(
                executor.submit(
                    merge_tiles, tiles, image_tiles, i
                )
            )

        for future in tqdm(as_completed(futures), desc="Waiting for blended images", total=loader.n_images):
            i, image = future.result()
            results.append(image)
            if args.save:
                os.makedirs(args.save, exist_ok=True)
                output_path = os.path.join("output", f"{i}.png")
                image.save(output_path)
    


def main(args):
    with open(args.tiles, "r") as f:
        tiles = json.load(f)

    loader = ImageLoader()
    if loader.fetch_new_images() and len(loader) >= loader.n_images:
        print("New images loaded")
        run(tiles, args, loader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=str, default="inference_tiles.json")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--save", type=str, default="output")
    parser.add_argument("--log", type=str, default="INFO")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="model_80000.pth")
    args = parser.parse_args()

    if MODEL is None:
        # Initialize the model only if it hasn't been initialized yet
        MODEL = ViTAutoencoder(embed_dim=4, ddconfig=config)
        MODEL.load_state_dict(torch.load(args.checkpoint, map_location=device))
        MODEL.to(device)

    logging.basicConfig(level=args.log.upper())

    main(args)
