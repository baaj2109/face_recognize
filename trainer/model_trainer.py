import os
import numpy as np
import time
import logging 

import torch 
import torch.optim as optim
from torch.optim import lr_scheduler
from .evaluation import getFeature, evaluation_10_fold

class ModelTrainer(object):

    def __init__(self, 
                 args, 
                 model, 
                 margin, 
                 train_loader, 
                 validation_loader,
                 log_dir,
                 models_dir):
        self.args = args
        self.model = model
        self.margin = margin
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.log_dir = log_dir
        self.models_dir = models_dir

    def train(self):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD([
            {'params': self.model.parameters(),'weight_decay': self.args.weight_decay},
            {'params': self.margin.parameters(),'weight_decay': self.args.weight_decay}], 
            lr = self.args.learning_rate,
            momentum = self.args.momentum,
            nesterov = True)

        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,
                                                    milestones = [6, 8, 10],
                                                    gamma = 0.3) 
        total_iters = 0
        best_acc = 0.0
        best_iters = 0
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
                        logging.info(
                            f"Epoch {epoch}/{self.args.epochs - 1}, " +
                            f"Iters: {total_iters:06d}, " +
                            f"loss: {loss.item():.4f}, " +
                            f"train_accuracy: {correct/total:.4f}, " +
                            f"time: {time_cur:.2f}, " +
                            f"learning rate: {lr}"
                        )
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
                    featureLs, featureRs = getFeature(
                        self.model,  
                        self.validation_loader, 
                        flip = self.args.flip)
                    ACCs, threshold = evaluation_10_fold(
                        featureLs, 
                        featureRs, 
                        self.validation_loader.dataset, 
                        method = self.args.measure_method)
                    logging.info(
                        f"Epoch {epoch}/{self.args.epochs - 1} CFP_FP " +
                        f"average acc: {np.mean(ACCs) * 100:.4f} " +
                        f"average threshold: {np.mean(threshold):.4f}"
                    )
                    if best_acc <= np.mean(ACCs) * 100:
                        best_acc = np.mean(ACCs) * 100
                        best_iters = total_iters
                    self.model.train()

        time_elapsed = time.time() - start  
        logging.info(
            f"Finally Best Accuracy CFP_FP: {best_acc:.4f} " +
            f"in iters: {best_iter}"
        )
        logging.info(
            f"Training complete in {time_elapsed//60:.0f}m " +
            f"{time_elapsed%60:.0f}s"
        )


