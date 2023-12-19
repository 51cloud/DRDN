#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY
from torch.nn import functional as F

import torchsnooper
# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
            self,
            dim_in,
            fusion_conv_channel_ratio,
            fusion_kernel,
            alpha,
            eps=1e-5,
            bn_mmt=0.1,
            inplace_relu=True,
            norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
                cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, x2, tymodel, bboxes=None):
        if tymodel == 'train':
            # torch.Size([1, 3, 8, 224, 224])
            # torch.Size([1, 3, 32, 224, 224])
            x = self.s1(x)
            x2 = self.s1(x2)
            # torch.Size([1, 64, 8, 56, 56])
            # torch.Size([1, 8, 32, 56, 56])
            x = self.s1_fuse(x)
            x2 = self.s1_fuse(x2)
            # torch.Size([1, 80, 8, 56, 56])
            # torch.Size([1, 8, 32, 56, 56])
            x = self.s2(x)
            x2 = self.s2(x2)
            # torch.Size([1, 256, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            x = self.s2_fuse(x)
            x2 = self.s2_fuse(x2)
            # torch.Size([1, 320, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x2[pathway] = pool(x2[pathway])
            # torch.Size([1, 320, 8, 56, 56])
            # torch.Size([1, 32, 32, 56, 56])
            x = self.s3(x)
            x2 = self.s3(x2)
            # torch.Size([1, 512, 8, 28, 28])
            # torch.Size([1, 64, 32, 28, 28])
            x = self.s3_fuse(x)
            x2 = self.s3_fuse(x2)
            # torch.Size([1, 640, 8, 28, 28])
            # torch.Size([1, 64, 32, 28, 28])
            x = self.s4(x)
            x2 = self.s4(x2)
            # torch.Size([1, 1024, 8, 14, 14])
            # torch.Size([1, 128, 32, 14, 14])
            x = self.s4_fuse(x)
            x2 = self.s4_fuse(x2)
            # torch.Size([1, 1280, 8, 14, 14])
            # torch.Size([1, 128, 32, 14, 14])
            x = self.s5(x)
            x2 = self.s5(x2)
            x1 = (x + x2) / 2
            # torch.Size([1, 2048, 8, 7, 7])
            # torch.Size([1, 256, 32, 7, 7])
            if self.enable_detection:
                x1 = self.head(x1, bboxes)
            else:
                x1 = self.head(x1)
            # torch.Size([1, 12])
        else:
            x = self.s1(x)
            x = self.s1_fuse(x)
            x = self.s2(x)
            x = self.s2_fuse(x)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            x = self.s3(x)
            x = self.s3_fuse(x)
            x = self.s4(x)
            x = self.s4_fuse(x)
            x = self.s5(x)
            if self.enable_detection:
                x = self.head(x, bboxes)
            else:
                x = self.head(x)
            x1 = x
        return x1


class Uncertain():
    def __init__(self, alpha, cfg):
        super(Uncertain, self).__init__()
        self.views = 2
        # self.num_classes = params.num_classes
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.alpha = alpha

    def DS_Combin_two(self, alpha1, alpha2):
        # alpha1 = alpha1.cpu()
        # alpha1 = nn.BatchNorm1d(self.num_classes)(alpha1)
        # alpha1 = nn.ReLU()(alpha1).cuda()
        # print('alpha1: ', alpha1)
        # alpha2 = alpha2.cpu().detach().numpy()
        # alpha2 = nn.BatchNorm1d(self.num_classes)(alpha2)
        # alpha2 = nn.ReLU()(alpha2).cuda()
        # print('alpha2: ', alpha2)
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(self.views):
            S[v] = torch.sum(alpha[v], dim=-1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.num_classes / S[v]
            # print('u[v]:', u[v])
        return u

    def __iter__(self):
        return self

    def __next__(self):
        alpha = self.alpha
        # print('alpha2:', alpha)
        for v_num in range(len(alpha)):
            # step two
            alpha[v_num] = alpha[v_num] + 1
        # print('alphs[0]: ', alpha[0])
        # print('alphs[1]: ', alpha[1])
        u = self.DS_Combin_two(alpha[0], alpha[1])

        return u


def GAP(r1, r2, alpha, cfg):
    # print('alpha1:', alpha)
    u = Uncertain(alpha, cfg).__next__()
    # print('u:', u)
    # (1-u1)/(3-u1-u2-u3)
    u1 = u[0]
    u2 = u[1]
    sum = 2 - (u1 + u2)
    if u1 > u2:
        ri = (1 - u1)/sum * r1
        r = (1 - u2)/sum * (r2 + ri)
    else:
        ri = (1 - u2) / sum * r2
        r = (1 - u1) / sum * (r1 + ri)
    '''
    print('sum: ', sum)
    print('(1-u[0])/sum: ', (1 - u[0]) / sum)
    print('(1-u[1])/sum: ', (1 - u[1]) / sum)
    print('(1-u[2])/sum: ', (1 - u[2]) / sum)

    b, c, t, w, h = r1.size()
    r = np.empty((b, c, t, w, h))
    r1 = r1.cpu().detach().numpy()
    r2 = r2.cpu().detach().numpy()
    r3 = r3.cpu().detach().numpy()
    for i in range(b):
        u1 = u[0][i][0].cpu().detach().numpy()
        u2 = u[1][i][0].cpu().detach().numpy()
        u3 = u[2][i][0].cpu().detach().numpy()
        sum = 3 - u1 - u2 - u3
        # print('r1[i]:', r1[i])
        # print('u1:', u1)
        print('sum: ', sum)
        print('((1-u1)/sum: ', ((1-u1)/sum))
        print('((1-u2)/sum: ', ((1-u2)/sum))
        print('((1-u3)/sum: ', ((1-u3)/sum))
        r[i] = r1[i]*((1-u1)/sum) + r2[i]*((1-u2)/sum) + r3[i]*((1-u3)/sum)
    r = torch.from_numpy(r).cuda()
    r = torch.tensor(r, dtype=torch.float32)
    return r

    uu = dict()
    for v in range(params.num_views):
        uu[v] = torch.sum(u[v], dim=0, keepdim=True)/float(params.batch_size)
    print('uu:', uu)
    u1 = uu[0]
    u2 = uu[1]
    u3 = uu[2]
    sum = 3-u1-u2-u3
    '''
    # u1 = 0.5
    # u2 = 0.4
    # u3 = 0.3
    # sum = 3-(u1+u2+u3)
    # r = r1*((1-u1)/sum) + r2*((1-u2)/sum) + r3*((1-u3)/sum)
    # r = (r1 + r2) / 2
    return r


class guide():
    def __init__(self, r, r1, b, t):
        super(guide, self).__init__()
        self.r = r.cuda()
        self.r1 = r1.cuda()
        self.b = b
        self.t = t
        # nn.Sequential()创建一个容器
        self.glo_fc = nn.Sequential(nn.Linear(2048, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU()).cuda()

        self.corr_atte = nn.Sequential(
            # nn.Conv2d(2048 + 1024, 1024, 1, 1, bias=False),
            nn.Conv3d(2048 + 2048, 1024, 1, 1, bias=False),
            nn.BatchNorm3d(1024),
            nn.Conv3d(1024, 256, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 1, 1, 1, bias=False),
            nn.BatchNorm3d(1),
        ).cuda()

    def __iter__(self):
        return self

    def __next__(self):
        r_glo = self.r.mean(dim=-1).mean(dim=-1).mean(dim=-1)  # r_glo = tensor<(32, 2048), float32, cuda:0, grad>
        # r1_glo = self.r1.mean(dim=-1).mean(dim=-1).mean(dim=1)  # r1_glo = tensor<(32, 2048), float32, cuda:0, grad>
        r_glo = r_glo.cuda()
        # r1_glo = r1_glo.cuda()
        # view():改变形状 contiguous():tensor开辟新内存空间
        '''
        glo = self.glo_fc(r_glo).view(self.b, 1, 1024, 1, 1).contiguous().expand(self.b, self.t, 1024, 16, 8).contiguous().view(self.b * self.t,
                                                                                                                 1024,
                                                                                                                 16, 8)
        # glo = tensor<(32, 1024, 16, 8), float32, cuda:0, grad>
        glo1 = self.glo_fc(r1_glo).view(self.b, 1, 1024, 1, 1).contiguous().expand(self.b, self.t, 1024, 16, 8).contiguous().view(
            self.b * self.t,
            1024,
            16, 8)
        # glo1 = tensor<(32, 1024, 16, 8), float32, cuda:0, grad>
        '''

        glo = self.glo_fc(r_glo).view(self.b, 2048, 1, 1, 1).contiguous().expand(self.b, 2048, self.t, 7, 7)
        # print('glo.size()', glo.size()) # glo.size() torch.Size([8, 1024, 8, 7, 7])
        # print('r1.size()', self.r1.size()) # r1.size() torch.Size([8, 2048, 8, 7, 7])

        # glo = tensor<(32, 1, 1024, 4, 4)
        # glo1 = self.glo_fc(r1_glo).view(self.b, 1, 1024, 1, 1).contiguous().expand(self.b, self.t, 1024, 4, 4)
        # glo1 = tensor<(32, 1, 1024, 4, 4), float32, cuda:0, grad>
        r_corr = torch.cat((self.r1, glo), dim=1)  # r_corr = tensor<(32, 1, 2048, 4, 4), float32, cuda:0, grad>

        # r_corr = r_corr.permute(0, 2, 1, 3, 4)  # r_corr = tensor<(32, 2048, 1, 4, 4), float32, cuda:0, grad>
        corr_map = self.corr_atte(r_corr)  # corr_map = tensor<(32, 1, 1, 4, 4), float32, cuda:0, grad>
        # corr_map = F.sigmoid(corr_map).view(self.b * self.t, 1, 4, 4).contiguous()
        corr_map = F.sigmoid(corr_map).view(self.b, 1, self.t, 7, 7).contiguous()  # torch.Size([32, 1, 1, 4, 4])
        # print('corr_map.size()', corr_map.size())
        # r_uncorr = tensor<(32, 2048, 1, 4, 4)
        r_uncorr = self.r1 * corr_map
        r_corr = self.r1 * (1 - corr_map)

        return r_uncorr, r_corr

# @torchsnooper.snoop()
class DAL_regularizer(nn.Module):
    '''
    Disentangled Feature Learning module in paper.
    '''

    def __init__(self, ps, ns):
        super().__init__()
        self.discrimintor = nn.Sequential(nn.Linear(4096, 2048)
                                          , nn.ReLU(inplace=True)
                                          , nn.Linear(2048, 2048)
                                          , nn.ReLU(inplace=True)
                                          , nn.Linear(2048, 1)
                                          , nn.Sigmoid()).cuda()
        self.ps = ps.cuda()
        self.ns = ns.cuda()

    def __next__(self):
        # ps1 = self.ps.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        # ps1 = ps1.cuda()
        # ns1 = self.ns.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        # ns1 = ns1.cuda()
        ps_scores = self.discrimintor(self.ps)
        # torch.backends.cudnn.enabled = False
        ns_scores = self.discrimintor(self.ns)

        return ps_scores, ns_scores

@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.data_bn = nn.BatchNorm3d(2048)

        self.data_rule = nn.ReLU()

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )


    def forward(self, x, x2, x3, x4, tymode, bboxes=None):
        if tymode == 'train':
            x = self.s1(x)  # torch.Size([1, 3, 8, 224, 224])
            x2 = self.s1(x2)
            x3 = self.s1(x3)
            x4 = self.s1(x4)

            x = self.s2(x)
            x2 = self.s2(x2)
            x3 = self.s2(x3)
            x4 = self.s2(x4)

            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x2[pathway] = pool(x2[pathway])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x3[pathway] = pool(x3[pathway])
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x4[pathway] = pool(x4[pathway])

            x = self.s3(x)
            x2 = self.s3(x2)
            x3 = self.s3(x3)
            x4 = self.s3(x4)

            x = self.s4(x)
            x2 = self.s4(x2)
            x3 = self.s4(x3)
            x4 = self.s4(x4)

            x = self.s5(x)  # torch.Size([1, 2048, 8, 7, 7])
            x2 = self.s5(x2)
            x3 = self.s5(x3)
            x4 = self.s5(x4)

            #GAP
            # xbn = self.data_bn(x[0])
            # xbn = self.data_rule(xbn)
            # x2bn = self.data_bn(x2[0])
            # x2bn = self.data_rule(x2bn)
            # x = [x, ]
            # x2 = [x2, ]
            _, alpha1, _, _= self.head(x)
            # alph1 = nn.BatchNorm2d(alph1)
            # alph1 = nn.ReLU(alph1)
            _, alpha2, _, _ = self.head(x2)
            # alph2 = nn.BatchNorm2d(alph2)
            # alph2 = nn.ReLU(alph2)
            alphas = [alpha1[0], alpha2[0]]

            x1 = GAP(x[0], x2[0], alphas, self.cfg)
            # x1 = (x[0] + x2[0])/2

            # guide
            b, c, t, h, w = x[0].size()
            wr1, sr1 = guide(x1, x[0], b, t).__next__()
            wr2, sr2 = guide(x1, x2[0], b, t).__next__()

            # Multi-Net
            x5 = (x[0] + x3[0])/2
            wr3, sr3 = guide(x5, x3[0], b, t).__next__()
            wr3 = [wr3, ]
            y3, _, _, _ = self.head(wr3)

            x6 = (x[0] + x4[0]) / 2
            wr4, sr4 = guide(x6, x4[0], b, t).__next__()
            wr4 = [wr4, ]
            y4, _, _, _ = self.head(wr4)

            a_wr1 = [wr1, ]
            v_sr1 = [sr1, ]
            v_sr2 = [sr2, ]

            _, _, a, _ = self.head(a_wr1)
            _, _, v1, _ = self.head(v_sr1)
            _, _, v2, _ = self.head(v_sr2)

            ps = torch.cat((a, v1), 1)  # torch.Size([8, 4096, 8, 7, 7])
            ns = torch.cat((a, v2), 1)

            ps_score, ns_score = DAL_regularizer(ps, ns).__next__()
            # wr1 = [wr1, ]
            # wr2 = [wr2, ]
            #GAP
            # wr1bn = self.data_bn(wr1)
            # wr1bn = self.data_rule(wr1bn)
            # wr2bn = self.data_bn(wr2)
            # wr2bn = self.data_rule(wr2bn)

            wr1bn = [wr1, ]
            wr2bn = [wr2, ]

            _, alpha3, _, _ = self.head(wr1bn)
            # alph3 = nn.BatchNorm2d(alph3)
            # alph3 = nn.ReLU(alph3)
            _, alpha4, _, _ = self.head(wr2bn)
            # alph4 = nn.BatchNorm2d(alph4)
            # alph4 = nn.ReLU(alph4)

            alphas2 = [alpha3[0], alpha4[0]]
            # wr = (wr1 + wr2)/2
            wr = GAP(wr1, wr2, alphas2, self.cfg)
            wr = [wr, ]
            # if self.enable_detection:
            #     x = self.head(x, bboxes)
            # else:
            #     x = self.head(x)
            if self.enable_detection:
                y, _, _, feature = self.head(wr, bboxes)
            else:
                y, _, _, feature = self.head(wr)
        else:
            x = self.s1(x)

            x = self.s2(x)

            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x[pathway] = pool(x[pathway])

            x = self.s3(x)

            x = self.s4(x)

            x = self.s5(x)

            if self.enable_detection:
                y, _, _, feature = self.head(x, bboxes)
            else:
                y, _, _, feature = self.head(x)
            y3 = 0
            y4 = 0
            ps_score = 0
            ns_score = 0

        return y, y3, ps_score, ns_score, feature
