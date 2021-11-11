import os
import logging
import torchvision.transforms as transforms
import torch.utils.data as data
from model import MobileFaceNet, ArcFace
from data_loader import CASIAWebFace, CFPFP
from args import parser
from trainer import ModelTrainer


def create_log_workspace(path):
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(path, log_dir)
    models_dir = os.path.join(log_dir, 'models')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok = True)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    return log_dir, models_dir

def write_args(path):
    with open(os.path.join(path, "args.txt"), "w") as writefile:
        for k, v in sorted(self.args.__dict__.items()):
            print(f"{k}: {v}", file = writefile)


def main(args):
    log_dir, models_dir = create_log_workspace(args.output_path)
    write_args(log_dir)

    logging.basicConfig(
        stream = sys.stdout, 
        level = logging.INFO,
        filename = os.path.join(log_dir, "training_log.log"), 
        filemode = 'a',
        format = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean = (0.5, 0.5, 0.5), 
                             std = (0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]

    dataset = CASIAWebFace(args.file_list, transform = transform) 

    train_loader = data.DataLoader(dataset,
                                  batch_size = args.batch_size, 
                                  shuffle = True, 
                                  num_workers = args.num_workers,
                                  drop_last = False)

    model = MobileFaceNet(args.embedding_size)

    if args.model_weights_path != "./":
        try: 
            detect_model.load_state_dict(
                torch.load(args.model_weights_path, 
                           map_location=lambda storage, loc: storage))
        except Exception as e:
            logging.warning(f"failed to load weithts, {e}")


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
    

    trainer = ModelTrainer(args, model, margin, train_loader,
                           validation_loader, log_dir, models_dir)
    trainer.train()

if __name__ == "__main__":

    args = parser.parse_args()
    main(args)

