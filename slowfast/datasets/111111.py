#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from fvcore.common.file_io import PathManager
import torch.nn.functional as F

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from slowfast.models.sf_model import SFNet

import time

logger = logging.get_logger(__name__)


def load_SF_model(feature_h=14, feature_W=14, beta=50, kernel_sigma=5):
    net = SFNet(feature_h, feature_W, beta = beta, kernel_sigma = kernel_sigma)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net = net.cuda()
    #net = nn.DataParallel(net)
    net.to(device)

    best_weights = torch.load("/public/home/jiaxm/perl5/SlowFast-master/pretrained/best_checkpoint.pt")
    #best_weights = "/public/home/liuwx/perl5/SFnet/pretrained/best_checkpoint.pt"
    adap3_dict = best_weights['state_dict1']
    adap4_dict = best_weights['state_dict2']
    net.adap_layer_feat3.load_state_dict(adap3_dict, strict=False)
    net.adap_layer_feat4.load_state_dict(adap4_dict, strict=False)
    #net.load_state_dict({k.replace('.', ''): v for k, v in torch.load(best_weights).items()})

    return net


def feature_align(frames, align_num, feature_w=16, feature_h=16):
    net = load_SF_model()
    net.eval()
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #net = net.cuda()
        net = net.to(device)
        middle_frame_num = (align_num - 1) // 2
        result_frame = None
        datas = []
        for i in range(frames.size()[0]):
            datas.append(frames[i])
            if (i + 1) % align_num == 0:
                src_image = datas[middle_frame_num].unsqueeze(0).to(device)
                #src_image = datas[middle_frame_num].unsqueeze(0).cuda()
                for m in range(len(datas)):
                    if m != middle_frame_num:
                        tgt_image = datas[m].unsqueeze(0).to(device)
                        #tgt_image = datas[m].unsqueeze(0).cuda()
                        output = net(src_image, tgt_image, train=False)

                        small_grid = output['grid_S2T'][:,1:-1,1:-1,:]
                        small_grid[:,:,:,0] = small_grid[:,:,:,0] * (feature_w//2)/(feature_w//2 - 1)
                        small_grid[:,:,:,1] = small_grid[:,:,:,1] * (feature_h//2)/(feature_h//2 - 1)
                        tgt_image_H = datas[m].size()[0]
                        tgt_image_W = datas[m].size()[1]
                        small_grid = small_grid.permute(0,3,1,2)
                        grid = F.interpolate(small_grid, size = (tgt_image_H,tgt_image_W), mode='bilinear', align_corners=True)
                        grid = grid.permute(0,2,3,1)

                        tgt_image_real = datas[m].unsqueeze(0).float().to(device).permute(0,3,1,2)
                        #tgt_image_real = datas[m].unsqueeze(0).float().cuda().permute(0, 3, 1, 2)
                        aligned_image = torch.nn.functional.grid_sample(tgt_image_real, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                        aligned_image = aligned_image.permute(0,2,3,1)
                        if result_frame == None:
                            result_frame = aligned_image
                        else:
                            result_frame = torch.cat([result_frame, aligned_image], 0)
                    else:
                        result_frame = torch.cat([result_frame, src_image], 0)
                datas = []
        #res = frames[27:32, :, :, :].cuda()
        res = frames[27:32, :, :, :].to(device)
        result_frame = torch.cat([result_frame, res], 0)
        return result_frame.cpu()


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=max_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            

            frames[1] = feature_align(frames[1].permute(1, 0, 2, 3), align_num=9).permute(1, 0, 2, 3)
            #for i in range(len(frames)):
                #frames[i] = feature_align(frames[i].permute(1, 0, 2, 3), num).permute(1, 0, 2, 3)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
