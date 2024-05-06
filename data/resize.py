from PIL import Image
import argparse
import concurrent.futures
Image.MAX_IMAGE_PIXELS = 933120000

parser = argparse.ArgumentParser()
parser.add_argument("--file", nargs="+", type=str, required=True)
parser.add_argument("--factor", type=float, default=0.25)
args = parser.parse_args()


def process_file(file):
    print(file)
    with Image.open(file) as img:
        width, height = img.size
        img = img.resize((int(width * args.factor), int(height * args.factor)))
        img.save(file)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_file, args.file)
