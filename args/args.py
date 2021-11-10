
import argparse

parser = argparse.ArgumentParser(description='training mobilefacenet with casia dataset')


"""
Data
"""
parser.add_argument("--file-list",
                    type = str,
                    default = "./face_emore_align_112.txt",
                    help = """training file list 
                    example: path label
                        path/name_1.jpg 0
                        path/name_2.jpg 0
                        path/name_3/jpg 1
                    """)

parser.add_argument('--validation-file-list',
                    type = str,
                    default = './cfp_fp_align_122.txt',
                    help = """validation file list
                    example: path path label
                        path/name_1.jpg path/name_11.jpg 0
                        path/name_2.jpg path/name_22.jpg 0
                        path/name_3.jpg path/name_33.jpg 1
                    """)

parser.add_argument("--output-path",
                    type = str,
                    default = "./log")

"""
Model
"""
parser.add_argument("--embedding-size", 
                    type = int,
                    default = 512,
                    help = 'embedding vector')

parser.add_argument('--model-weights-path',
                    type = str,
                    default = "./",
                    help = 'trained model weights')
"""
Training
"""

parser.add_argument("--num-workers", 
                    type = int,
                    default = 0,
                    help = 'number of workers in dataset')

parser.add_argument("--weight-decay",
                    type = float,
                    default = 5e-4,
                    help = "learning rate weight decay")

parser.add_argument("--learning-rate",
                    type = float,
                    default = 0.01,
                    help = "learning rate")

parser.add_argument("--momentum",
                    type = float,
                    default = 0.9,
                    help = "learning momentum")

parser.add_argument("--epochs",
                    type = int,
                    default = 12,
                    help = "number of epoch")

parser.add_argument("--batch-size",
                    type = int,
                    default = 128,
                    help = 'training batch size')

parser.add_argument('--flip',
                    type = str, 
                    default = True, 
                    help = 'if flip the image with time augmentation')

parser.add_argument('--measure-method',
                    type = str,
                    default = 'l2_distance',
                    choices = ["l2_distance", "cos_distance"],
                    help = "measure method support two method `l2_distance` and `cos_distance`")








