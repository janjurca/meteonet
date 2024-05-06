import argparse
from diffusers.models import AutoencoderKL
import torch
import os 
from PIL import Image
import numpy as np
import sys
from matplotlib import pyplot as plt
import os.path
import glob
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True) 
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--resolution", type=int, default=512) 
parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
parser.add_argument("--device", type=str, default=device)
args = parser.parse_args()

vae = AutoencoderKL.from_pretrained(args.vae)
vae.to(device)

for f in tqdm(sorted(list(glob.glob(os.path.join(args.input, "*/*.png"))))):
    output = f.replace(".png", ".pt").replace(args.input, args.output)
    if os.path.exists(output):
        continue
    image = Image.open(f)
    image = image.resize((args.resolution, args.resolution))
    image = image.convert("RGB")
    image = np.array(image)
    image = image.transpose(2,0,1)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.unsqueeze(0)
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    image = image.to(device)
    z = vae.encode(image).latent_dist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    z = z.sample().cpu().detach()
    torch.save(z, output)
