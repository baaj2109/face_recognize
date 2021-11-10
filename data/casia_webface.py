
import os
import numpy as np
import argparse


def process(args):
    folders = [f for f in os.listdir(args.image_path)]

    with open(args.output_file, "w") as writefile:
        label_count = 0
        for folder in folders:
            for file in os.listdir(os.path.join(args.image_path, folder)):
                print(f"{args.image_path}/{folder}/{file} {label_count}", file = writefile)
            label_count += 1


def parse_args():
    parser = argparse.ArgumentParser(description='handler asica web face dataset')

    parser.add_argument("--image-path", 
                        type=str,
                        help='casia webface dataset path')

    parser.add_argument("--output-file",
                        type = str,
                        default = "./face_emore_align_112.txt",
                        help = """output file list 
                        example: path label
                            path/name_1.jpg 0
                            path/name_2.jpg 0
                            path/name_3/jpg 1
                        """)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    process(args)
    print(f"save to {args.output_file}")

