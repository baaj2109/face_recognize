import os

import torchvision.transforms as transforms
import torch.utils.data as data
from model import MobileFaceNet, ArcFace
from data_loader import CASIAWebFace, CFPFP
from args import parser
from trainer import ModelTrainer


def main(args):

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]

    dataset = CASIAWebFace(args.file_list, transform = transform) 

    train_loader = data.DataLoader(dataset,
                                  batch_size = args.batch_size, 
                                  shuffle = True, 
                                  num_workers = args.num_workers,
                                  drop_last = False)

    model = MobileFaceNet(args.embedding_size)

    if args.pretrained_model_path != "./":
        try: 
        detect_model.load_state_dict(torch.load(args.model_weights_path, 
                                                map_location=lambda storage, loc: storage))
        except Exception as e:
            print(f"failed to load weithts, {e}")


    margin = ArcFace(embedding_size = args.embedding_size, 
                     classnum = dataset.class_nums,
                     s = 32., 
                     m = 0.5)

    dataset = CFPFP(args.validation_file_list, transform = transform)
    validation_loader = data.DataLoader(dataset,
                                        batch_size = args.batch_size,
                                        shuffle = False,
                                        num_workers = 2,
                                        drop_last = False)
    

    trainer = ModelTrainer(args, model, margin, train_loader, validation_loader)
    trainer.train()

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)

