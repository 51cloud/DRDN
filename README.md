# Dual-Recommendation Disentanglement Network for View Fuzz in Action Recognition
This is the official PyTorch implementation of our work: "Dual-Recommendation Disentanglement Network for View Fuzz in Action Recognition".

In this paper, we present a novel approach and we define a new problem for Multi-view action recognition. We asses the performance of our method and previous state-of-the-art methods on N-UCLA and NTU-RGB+D datasets, We do some experimental analysis at IXMAS dataset.


# Requirements
This repository uses the following libraries:
- Python >= 3.6
- Numpy
- PyTorch >= 1.3
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`

# How to download data
In this project we use three dataset, N-UCLA, NTU-RGB+D and IXMAS. 
We provide the scripts to download them in 'data/download_\<dataset_name\>.sh'.
The script takes no inputs but use it in the target directory (where you want to download data). 

# How to perform training
The most important file is run.py, that is in charge to start the training or test procedure.
To run it, simpy use the following command:

> python tools/run_net.py --cfg \<cfg_path\> DATA_LOADER.NUM_WORKERS 0 NUM_GPUS \<GPU_NUM\> BATCHH_SIZE \<BATCHH_SIZE_NUM\> SOLVER.BASE_LR \<LR_NUM\> SOLVER.MAX_EPOCH \<EPOCH_NUM\> SOLVER.WEIGHT_DECAY \<WEIGHT_DECAY_NUM\> SOLVER.WARMUP_EPOCHS 0.0 DATA.PATH_TO_DATA_DIR \<DATA_PATH\>

The default is to use a pretraining for the backbone used, that is searched in the pretrained folder of the project. 
We used the pretrained model released by the Kinetics (as said in the paper), that can be found here:
 [link](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md). 
 
## Train a Standard Model from Scratch

Here we can start with training a simple C2D models by running:

```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply add

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```
To the yaml configs file, then you do not need to pass it to the command line every time.


You may also want to add:
```
  DATA_LOADER.NUM_WORKERS 0 \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
```

If you want to launch a quick job for debugging on your local machine.

## Resume from an Existing Checkpoint
If your checkpoint is trained by PyTorch, then you can add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
```

If the checkpoint in trained by Caffe2, then you can do the following:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_Caffe2_checkpoint \
TRAIN.CHECKPOINT_TYPE caffe2
```

If you need to performance inflation on the checkpoint, remember to set `TRAIN.CHECKPOINT_INFLATE` to True.


## Perform Test
We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. If only testing is preferred, you can set the `TRAIN.ENABLE` to False, and do not forget to pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

### Run command
```
python \tools\run_net.py --cfg path/to/<pretrained_model_config_file>.yaml
```
## Contributors
PySlowFast is written and maintained by [Wenxuan Liu](https://orcid.org/0000-0002-4417-6628), [Xian Zhong](https://orcid.org/0000-0002-5242-0467), [Zhuo Zhou](https://orcid.org/0000-0003-4620-4378), [Kui Jiang](https://orcid.org/0000-0002-4055-7503), [Zheng Wang](https://orcid.org/0000-0003-3846-9157), [Chia-Wen Lin}](https://orcid.org/0000-0002-9097-2318).

## Citing PySlowFast
If you find DRDN useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@article{DBLP:journals/tip/LiuZZJWL23,
  author       = {Wenxuan Liu and
                  Xian Zhong and
                  Zhuo Zhou and
                  Kui Jiang and
                  Zheng Wang and
                  Chia{-}Wen Lin},
  title        = {Dual-Recommendation Disentanglement Network for View Fuzz in Action
                  Recognition},
  journal      = {{IEEE} Trans. Image Process.},
  volume       = {32},
  pages        = {2719--2733},
  year         = {2023},
}
```
