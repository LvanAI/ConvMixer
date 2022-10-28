# MindSpore ConvMixer


## Introduction

This work is used for reproduce ConvMixer based on NPU(Ascend 910)


## Data preparation

Download and extract [ImageNet](https://image-net.org/).

The directory structure is the standard layout for the MindSpore [`dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/dataset/mindspore.dataset.ImageFolderDataset.html?highlight=imagefolderdataset), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training


```
mpirun -n 8 python train.py --config <config path> > train.log 2>&1 &
```

## Evaluation 


```
python eval.py --config <config path>
```

## Acknowledgement

We heavily borrow the code from [convmixer](https://github.com/locuslab/convmixer) and [swin_transformer](https://gitee.com/mindspore/models/tree/master/research/cv/swin_transformer)
We thank the authors for the nicely organized code!
