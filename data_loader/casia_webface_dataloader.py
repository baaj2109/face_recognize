
import os
import cv2
import numpy as np
import logging
import torch.utils.data as data
import torchvision.transforms as transforms


class CASIAWebFace(data.Dataset):

    def __init__(self, file_document, transform = None):
        self.transform = transform
        image_list = []
        label_list = []
        with open(file_document , "r") as readfile:
            img_label_list = readfile.read().splitlines()
            
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        logging.info(
            f"CASIA dataset size: {len(self.image_list)} / {self.class_nums}"  
        )

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112,112))
        # random flip with ratio of 0.5
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
    file_list_path = "./face_emore_align_112.txt"
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
















