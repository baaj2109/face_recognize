# face recognize

## train
1. create training dataset file list
```
python casia_webface.py --image-path ./CASIA-WebFace \
                        --output-file ./face_emore_align_112.txt
```
2. create validation dataset file list
```
python cfp_fp.py --folder-path ./CASIA-cfp-dataset \
                 --output-file ./cfp_fp_align_112.txt
```
3. training model
```
python main.py --file-list ./data/face_emore_align_112.txt \
               --validation-file-list ./data/cfp_fp_align_112.txt 
```

## folder structure
```sh
.
├───data
│   ├───casia_webface
│   └───cfp_fp
├───data_loader
│   ├───casia_webface_dataloader
│   └───cfp_fp_dataloader
├───model
│   ├───arcface
│   ├───mobilefacenet
│   └───model_utils
├───trainer
│   ├───model_trainer
│   └───evaluation
├───args
│   └───args
└───log
    └───datetime_log
```

## reference
- [MobileFaceNet_Tutorial_Pytorch](https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch)
