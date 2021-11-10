import os
import numpy as np
import datetime
import time

import torch 
import torch.optim as optim
from torch.optim import lr_scheduler
from .evaluation import getFeature, evaluation_10_fold

class ModelTrainer(object):

    def __init__(self, args, model, margin, train_loader, validation_loader):
        self.args = args
        self.model = model
        self.margin = margin
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self._init_workspace()

    def _create_log_workspace(sef, path):
        log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(path, log_dir)
        models_dir = os.path.join(log_dir, 'models')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok = True)
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        return log_dir, models_dir

    def _write_args(self, path,):
        with open(os.path.join(path, "args.txt"), "w") as writefile:
            for k, v in sorted(self.args.__dict__.items()):
                print(f"{k}: {v}", file = writefile)

    def _init_workspace(self):
        self.log_dir, self.models_dir = self._create_log_workspace(self.args.output_path)
        self._write_args(self.log_dir)
        self.train_logging_file = os.path.join(self.log_dir, "training_log.txt") 
        self.test_logging_file = os.path.join(self.log_dir, "test_log.txt")


    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD([
            {'params': self.model.parameters(), 'weight_decay': self.args.weight_decay},
            {'params': self.margin.parameters(), 'weight_decay': self.args.weight_decay}], 
            lr = self.args.learning_rate,
            momentum = self.args.momentum,
            nesterov = True)

        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,
                                                    milestones=[6, 8, 10],
                                                    gamma=0.3) 
        total_iters = 0
        for epoch in range(self.args.epochs):
            # train model
            exp_lr_scheduler.step()
            self.model.train()     
            since = time.time()
            for det in self.train_loader: 
                img, label = det[0], det[1]
                optimizer_ft.zero_grad()
                print(f"iteraters: {total_iters}", end = "\r")
                with torch.set_grad_enabled(True):
                    raw_logits = self.model(img)
                    output = self.margin(raw_logits, label)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer_ft.step()
                    
                    total_iters += 1

                    # print train information
                    if total_iters % 100 == 0:

                        # current training accuracy 
                        _, preds = torch.max(output.data, 1)
                        total = label.size(0)
                        correct = (np.array(preds.cpu()) == np.array(label.data.cpu())).sum()                  
                        time_cur = (time.time() - since) / 100
                        since = time.time()

                        for p in  optimizer_ft.param_groups:
                            lr = p['lr']
                        training_info = f"Epoch {epoch}/{self.args.epochs - 1}, "
                        training_info += f"Iters: {total_iters:0>6d}, "
                        training_info += f"loss: {loss.item():.4f}, "
                        training_info += f"train_accuracy: {correct/total:.4f}, "
                        training_info += f"time: {time_cur:.2f}, "
                        training_info += f"learning rate: {lr}"
                        print(training_info)

                        with open(self.train_logging_file, "a") as writefile:

                            logging_info = f"Epoch {epoch}/{self.args.epochs - 1}, "
                            logging_info += f"Iters: {total_iters: 0>6d}, "
                            logging_info += f"loss: {loss.item():.4f}, "
                            logging_info += f"train_accuracy: {correct/total:.4f}, "
                            logging_info += f"time: {time_cur:.2f} s/iter, "
                            logging_info += f"learning rate: {lr}"
                            print(logging_info, file = train_logging_file)

                if total_iters % 3000 == 0:
                    torch.save({
                        'iters': total_iters,
                        'net_state_dict': model.state_dict()},
                        os.path.join(models_dir, f'Iter_{total_iters:06d}_model.ckpt'))
                    torch.save({
                        'iters': total_iters,
                        'net_state_dict': margin.state_dict()},
                        os.path.join(models_dir, f'Iter_{total_iters:06d}_margin.ckpt'))
                    
                # evaluate accuracy
                if total_iters % 3000 == 0:
                    self.model.eval()
                    for phase in ['LFW', 'CFP_FP', 'AgeDB30']:                 
                        featureLs, featureRs = getFeature(model, dataloaders[phase], device, flip = self.args.flip)
                        ACCs, threshold = evaluation_10_fold(featureLs, featureRs, dataset[phase], method = self.args.measure_method)
                        
                        test_info = f"Epoch {epoch}/{self.args.epochs - 1} {phase}"
                        test_info += f"average acc: {np.mean(ACCs) * 100:.4f} "
                        test_info += f"average threshold: {np.mean(threshold):.f4}"
                        print(test_info)
                        
                        if best_acc[phase] <= np.mean(ACCs) * 100:
                            best_acc[phase] = np.mean(ACCs) * 100
                            best_iters[phase] = total_iters

                        with open(self.test_logging_file, 'a') as writefile:
                            logging_info = f"Epoch {epoch}/{self.args.epochs - 1}, {phase} "
                            logging_info += f"average acc: {100 * np.mean(ACCs):.4f} "
                            logging_info += f"average threshold: {np.mean(threshold):.4f}"
                            print(logging_info, file = test_logging_file)
                    self.model.train()

        time_elapsed = time.time() - start  
        print(f"""Finally Best Accuracy: {best_acc['LFW']:.4f} in iters: {best_iters['LFW']}
                               CFP_FP: {best_acc['CFP_FP']:.4f} in iters: {best_iter['CFP_FP']}
                               AgeDB-30: {best_acc['AgeDB30']:.4f} in iters: {best_iters['AgeDB30']}""")

        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')


