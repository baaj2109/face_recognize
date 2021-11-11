
import os
import cv2
import numpy as np
import torch.utils.data as data


class CFPFP(data.Dataset):
    def __init__(self, file_list, transform = None):

        self.file_list = file_list
        self.transform = transform
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // 700
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):
        img_l = self._load_image(self.nameLs[index])
        img_r = self._load_image(self.nameRs[index])
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])
            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        return img

if __name__ == "__main__":

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])
    file_list_path = "./cfp_fp_align_112.txt"
    dataset = CASIAWebFAce(file_list_path, transform = transform)
    trainloader = data.DataLoader(dataset,
                                  batch_size = 128, 
                                  shuffle = False,
                                  num_worker = 0,
                                  drop_last = False)
    for data in trainloader:
        print(f"data shape: {data[0].shape}")
        print(f"label shape: {data[1].shape}")
        break



