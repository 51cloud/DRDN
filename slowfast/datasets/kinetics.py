import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


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
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._path_to_videos2 = []
        self._path_to_videos3 = []
        self._path_to_videos4 = []
        self._labels = []
        self._label3 = []
        self._label4 = []
        self._spatial_temporal_idx = []
        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))== 2)
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                # print('path:', path) # /public/home/jiaxm/perl5/datasets/N-UCLA/view1/v01_s10_e05_a11.avi
                if self.mode in ["train"]:
                    for idx in range(self._num_clips):
                        # print('path:', path) # /public/home/jiaxm/perl5/datasets/N-UCLA/view1/v01_s01_e00_a02.avi
                        # print('--------')
                        path2 = path.split('view1')[0]
                        action2 = path.split('/')[-1][3:]
                        #video2 = str(path2) + 'view2/' + 'v02' + str(action2)
                        video2 = str(path2) + 'view3/' + 'v03' + str(action2)
                        # print('video2:', video2, os.path.exists(video2))

                        path3 = path.split('_a')[0]
                        actions3 = list(range(1, 7)) + list(range(8, 10)) + list(range(11, 13))
                        action3 = random.choice(actions3)
                        label3 = action3
                        if action3 > 7 & action3 < 10:
                            label3 = action3 - 1
                        elif action3 > 10 & action3 < 13:
                            label3 = action3 - 2
                        # video2 = str(path2) + 'view2/' + 'v02' + str(action2)
                        if action3 < 10 :
                            video3 = str(path3) + '_a0' + str(action3) + '.avi'
                        else:
                            video3 = str(path3) + '_a' + str(action3) + '.avi'
                        # print('video3:', video3, os.path.exists(video3))

                        path4 = video2.split('_a')[0]
                        actions4 = list(range(1, 7)) + list(range(8, 10)) + list(range(11, 13))
                        action4 = random.choice(actions4)
                        label4 = action4
                        if action4 > 7 & action4 < 10:
                            label4 = action4 - 1
                        elif action4 > 10 & action4 < 13:
                            label4 = action4 - 2
                        # video2 = str(path2) + 'view2/' + 'v02' + str(action2)
                        if action4 < 10:
                            video4 = str(path4) + '_a0' + str(action4) + '.avi'
                        else:
                            video4 = str(path4) + '_a' + str(action4) + '.avi'
                        # print('video4:', video4, os.path.exists(video4))

                        # print('os.path.exists(video2):', os.path.exists(video2))
                        if os.path.exists(video2) & os.path.exists(video3) & os.path.exists(video4):
                            # print('------------------------Enter--------------------')
                            self._path_to_videos.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                            )
                            self._path_to_videos2.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, video2)
                            )
                            self._path_to_videos3.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, video3)
                            )
                            self._path_to_videos4.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, video4)
                            )
                            self._labels.append(int(label))
                            self._label3.append(int(label3))
                            self._label4.append(int(label4))
                            self._spatial_temporal_idx.append(idx)
                            self._video_meta[clip_idx * self._num_clips + idx] = {}
                        else:
                            self._path_to_videos.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                            )
                            self._path_to_videos2.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                            )
                            self._path_to_videos3.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                            )
                            self._path_to_videos4.append(
                                os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                            )
                            self._labels.append(int(label))
                            self._label3.append(int(label))
                            self._label4.append(int(label))
                            self._spatial_temporal_idx.append(idx)
                            self._video_meta[clip_idx * self._num_clips + idx] = {}
                else:
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics  from {}".format(
            path_to_file
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

        if self.mode in ["train"]:
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
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
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
        frames = []
        frames5 = []
        label = []
        label3 = []
        label4 = []
        for i_try in range(self._num_retries):
            # print('self.mode:', self.mode)
            if self.mode in ["train"]:
                # print('--------------------train-----------------')
                '''----------------new------------------------'''
                video1 = self._path_to_videos[index]
                video2 = self._path_to_videos2[index]
                video3 = self._path_to_videos3[index]
                video4 = self._path_to_videos4[index]
                #print('video1', video1)
                #print('video2', video2)
                video1_container = None
                video2_container = None
                video3_container = None
                video4_container = None
                try:
                    video1_container = container.get_video_container(
                        #self._path_to_videos[index],
                        video1,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video1 from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                try:
                    video2_container = container.get_video_container(
                        video2,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video2 from {} with error {}".format(
                            video2, e
                        )
                    )
                try:
                    video3_container = container.get_video_container(
                        video3,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video2 from {} with error {}".format(
                            video3, e
                        )
                    )
                try:
                    video4_container = container.get_video_container(
                        video4,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video2 from {} with error {}".format(
                            video4, e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video1_container is None:
                    logger.warning(
                        "Failed to meta load video1 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if video2_container is None:
                    logger.warning(
                        "Failed to meta load video2 idx {} from {}; trial {}".format(
                            index, self._path_to_videos2[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if video3_container is None:
                    logger.warning(
                        "Failed to meta load video1 idx {} from {}; trial {}".format(
                            index, self._path_to_videos3[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if video4_container is None:
                    logger.warning(
                        "Failed to meta load video2 idx {} from {}; trial {}".format(
                            index, self._path_to_videos4[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                # Decode video. Meta info is used to perform selective decoding.
                # print('self._video_meta[]:', self._video_meta)
                # print('self._video_meta[].len:', len(self._video_meta))
                frames1 = decoder.decode(
                    video1_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,

                )
                frames2 = decoder.decode(
                    video2_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,

                )
                frames3 = decoder.decode(
                    video3_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,

                )
                frames4 = decoder.decode(
                    video4_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,

                )
                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames1 is None:
                    logger.warning(
                        "Failed to decode video1 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if frames2 is None:
                    logger.warning(
                        "Failed to decode video2 idx {} from {}; trial {}".format(
                            index, self._path_to_videos2[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if frames3 is None:
                    logger.warning(
                        "Failed to decode video2 idx {} from {}; trial {}".format(
                            index, self._path_to_videos3[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if frames4 is None:
                    logger.warning(
                        "Failed to decode video2 idx {} from {}; trial {}".format(
                            index, self._path_to_videos4[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                if self.aug:
                    if self.cfg.AUG.NUM_SAMPLE > 1:

                        frame_list = []
                        label_list = []
                        index_list = []
                        for _ in range(self.cfg.AUG.NUM_SAMPLE):
                            new_frames1 = self._aug_frame(
                                frames1,
                                spatial_sample_index,
                                min_scale,
                                max_scale,
                                crop_size,
                            )
                            label = self._labels[index]
                            new_frames1 = utils.pack_pathway_output(
                                self.cfg, new_frames1
                            )
                            frame_list.append(new_frames1)
                            label_list.append(label)
                            index_list.append(index)
                        return frame_list, label_list, index_list, {}

                    else:
                        frames1 = self._aug_frame(
                            frames1,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                    if self.cfg.AUG.NUM_SAMPLE > 1:

                        frame_list = []
                        label_list = []
                        index_list = []
                        for _ in range(self.cfg.AUG.NUM_SAMPLE):
                            new_frames2 = self._aug_frame(
                                frames2,
                                spatial_sample_index,
                                min_scale,
                                max_scale,
                                crop_size,
                            )
                            label = self._labels[index]
                            new_frames2 = utils.pack_pathway_output(
                                self.cfg, new_frames2
                            )
                            frame_list.append(new_frames2)
                            label_list.append(label)
                            index_list.append(index)
                        return frame_list, label_list, index_list, {}

                    else:
                        frames2 = self._aug_frame(
                            frames2,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )

                else:
                    frames1 = utils.tensor_normalize(
                        frames1, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    # T H W C -> C T H W.
                    frames1 = frames1.permute(3, 0, 1, 2)
                    # Perform data augmentation.
                    frames1 = utils.spatial_sampling(
                        frames1,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                    frames2 = utils.tensor_normalize(
                        frames2, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    # T H W C -> C T H W.
                    frames2 = frames2.permute(3, 0, 1, 2)
                    # Perform data augmentation.
                    frames2 = utils.spatial_sampling(
                        frames2,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                    frames3 = utils.tensor_normalize(
                        frames3, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    # T H W C -> C T H W.
                    frames3 = frames3.permute(3, 0, 1, 2)
                    # Perform data augmentation.
                    frames3 = utils.spatial_sampling(
                        frames3,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                    frames4 = utils.tensor_normalize(
                        frames4, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    # T H W C -> C T H W.
                    frames4 = frames4.permute(3, 0, 1, 2)
                    # Perform data augmentation.
                    frames4 = utils.spatial_sampling(
                        frames4,
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                label = self._labels[index]
                label3 = self._label3[index]
                label4 = self._label4[index]
                frames = utils.pack_pathway_output(self.cfg, frames1, frames2)
                frames5 = utils.pack_pathway_output(self.cfg, frames3, frames4)
            elif self.mode in ["val", "test"]:
                # print('-----------------val-----------------')
                video_container = None
                try:
                    video_container = container.get_video_container(
                        self._path_to_videos[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video3 from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video3 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
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
                    max_spatial_scale=min_scale,
                )

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    logger.warning(
                        "Failed to decode video3 idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                if self.aug:
                    if self.cfg.AUG.NUM_SAMPLE > 1:

                        frame_list = []
                        label_list = []
                        index_list = []
                        for _ in range(self.cfg.AUG.NUM_SAMPLE):
                            new_frames = self._aug_frame(
                                frames,
                                spatial_sample_index,
                                min_scale,
                                max_scale,
                                crop_size,
                            )
                            label = self._labels[index]
                            new_frames = utils.pack_pathway_output(
                                self.cfg, new_frames
                            )
                            frame_list.append(new_frames)
                            label_list.append(label)
                            index_list.append(index)
                        return frame_list, label_list, index_list, {}

                    else:
                        frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )

                else:
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
                label3 = self._labels[index]
                label4 = self._labels[index]
                frames = utils.pack_pathway_output(self.cfg, frames, frames)
                frames5 = utils.pack_pathway_output(self.cfg, frames, frames)
            return frames, frames5, label, label3, label4, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video3 after {} retries.".format(
                    self._num_retries
                )
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)