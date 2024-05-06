import argparse
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import torch
from einops import rearrange
import torchvision.transforms as T
from autoencoder.autoencoder_vit import ViTAutoencoder 
import matplotlib.pyplot as plt
import imageio
import cv2


Image.MAX_IMAGE_PIXELS = 933120000

MODEL = None
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

def tensor_to_list(tensor):
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) * 127.5
    tensor = tensor.squeeze(1)
    images = []
    for image in tensor:
        image = image.numpy().astype(np.uint8).tolist()
        images.append(image)
    return images

def meteonet_predict(images_list, iterations=2):
    query = []
    for image in images_list:
        image = np.array(image, dtype=np.uint8)
        image = Image.fromarray(image).convert("L")
        image = T.ToTensor()(image) * 255
        query.append(image)
    query = torch.stack(query).unsqueeze(0)
    if query.view(-1, query.shape[-1]).sum() <= 12:
        print("SKIPPING PROCESSING")    
        return images_list
    x = rearrange(query / 127.5 - 1, 'b t c h w -> b c t h w')
    with torch.no_grad():
        predicted = MODEL(x)
    print(f"Predicting it: {iterations} images: {len(images_list)}")
    p = predicted[0].squeeze(1)
    images = tensor_to_list(p)
    iterations -= 1
    if iterations > 0:
        images.extend(meteonet_predict(images, iterations))
    return images

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


def main(args):
    with open(args.tiles, "r") as f:
        tiles = json.load(f)

    input_timestamp = int(args.timestamp)
    query = []

    n_images = 12
    for i in range(n_images):
        file = os.path.join(args.path, f"{input_timestamp - 600 * (n_images - i)}.png")
        print(file)
        with Image.open(file).convert("L") as img:
            img = img.resize((6000, 3000)) 
            #save the image to same file
            img.save(file)
            query.append(img)
    
    ground_truth = []
    for i in range(n_images*args.iterations):
        file = os.path.join(args.path, f"{input_timestamp + 600 * (1+i)}.png")
        print(file)
        with Image.open(file).convert("L") as img:
            img = img.resize((6000, 3000)) 
            #save the image to same file
            img.save(file)
            ground_truth.append(img)


    tile_sequences = get_tile_sequences(query, tiles, args.resolution)
    gt_tile_sequences = get_tile_sequences(ground_truth, tiles, args.resolution)
    print("tile_sequences: ", len(tile_sequences))
    print("gt_tile_sequences: ", len(gt_tile_sequences))

    concentrated = []
    for tilecoord, sequence in tqdm(tile_sequences.items(), desc="Counting concentrated"):
        array_sequence = [np.array(image).tolist() for image in sequence]
        suma = int(np.sum(array_sequence))  
        concentrated.append({"sum": suma, "sequence": array_sequence, "tile": tilecoord, "ground_truth": gt_tile_sequences[tilecoord]})

    # sort by sum
    concentrated.sort(key=lambda x: x["sum"], reverse=True)
    #get the top 10 most concentrated
    concentrated = concentrated[:50]
    samples = random.sample(concentrated, args.samples)

    results = []    
    for sample in samples:
        result = meteonet_predict(sample["sequence"], iterations=args.iterations)
        results.append(result)

    os.makedirs("gifs", exist_ok=True)
    fig, axes = plt.subplots(args.samples)
    TEXT_SIZE = 40
    for i, ax in enumerate(axes):
        gts = [np.array(image).tolist() for image in samples[i]["ground_truth"]]
        with imageio.get_writer(f'gifs/{i}.gif', mode='I', duration=500) as writer:
            for k, in_image in enumerate(samples[i]["sequence"]):
                in_image = np.array(in_image, dtype=np.uint8)
                black_image = np.zeros_like(in_image)
                label_left = np.zeros((TEXT_SIZE, in_image.shape[1]), dtype=np.uint8)
                label_left = cv2.putText(label_left, f'input {(k - len(samples[i]["sequence"]) + 1)*10}m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                label_right = np.zeros((TEXT_SIZE, in_image.shape[1]), dtype=np.uint8)
                label_right = cv2.putText(label_right, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                labels = np.hstack([label_left, label_right])
                data = np.hstack([in_image, black_image])
                image = np.vstack([labels, data])
                writer.append_data(image)
            for j, (gt, result) in enumerate(zip(gts, results[i])):
                gt = np.array(gt, dtype=np.uint8)
                result = np.array(result, dtype=np.uint8)
                label_left = np.zeros((TEXT_SIZE, in_image.shape[1]), dtype=np.uint8)
                label_left = cv2.putText(label_left, f"gt {(j+1)*10}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                label_right = np.zeros((TEXT_SIZE, in_image.shape[1]), dtype=np.uint8)
                label_right = cv2.putText(label_right, "prediciton", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                labels = np.hstack([label_left, label_right])
                data = np.hstack([gt, result])  # Change vertical stacking to horizontal stacking
                image = np.vstack([labels, data])
                writer.append_data(image)
        im_gts = np.hstack(gts)
        im = np.hstack(results[i])
        im = np.vstack([im_gts, im])
        ax.imshow(im, cmap="gray")
    plt.show()


def list_gifs_in_dir(directory):
    gif_files = [f for f in os.listdir(directory) if f.endswith('.gif')]
    return gif_files


def grid_stack_images(directory, output_path, cols=3):
    gif_files = list_gifs_in_dir(directory)
    gif_files = [os.path.join(directory, f) for f in gif_files]
    gif_files = sorted(gif_files)
    gif_files_loaded = [imageio.get_reader(gif) for gif in gif_files]
    frames = []
    for i in range(len(gif_files_loaded[0])):
        frame = None
        row = []
        for gif in gif_files_loaded:
            current_gif_frame = gif.get_data(i)
            row.append(current_gif_frame)
            if len(row) == cols:
                if frame is None:
                    frame = np.hstack(row)
                else:
                    row = np.hstack(row)
                    frame = np.vstack([frame, row])
                row = []

        frames.append(frame)
    
    imageio.mimsave(output_path, frames, duration=500, loop=0)
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--tiles", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="model_80000.pth")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    MODEL = ViTAutoencoder(embed_dim=4, ddconfig=config)
    MODEL.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    #main(args)

    grid_stack_images("gifs", "matrix.gif")