import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from op_flow.update import BasicUpdateBlock, SmallUpdateBlock
from op_flow.corr import CorrBlock, AlternateCorrBlock
from op_flow.utils import bilinear_sampler, coords_grid, upflow8,upConf
from op_flow.extractor import BasicEncoder, SmallEncoder
from visualize_utils import channel_features_to_RGB, pcl_features_to_RGB, show_feature_map
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
       
        self.fnet1 = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet1 = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1#[B,2,H//8, W//8]

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8*H, 8*W)


    def forward(self, image1, image2, mask, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0#归一化
        # image2 = 2 * (image2 / 255.0) - 1.0

        # image1 = image1.contiguous()
        # image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network提取特征
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet1([image1, image2]) #[B,256,H/8,W/8]    ,[B,256,H/8,W/8]
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        #
        result = torch.cat((fmap1, fmap2), dim=3)
        pcl_features_to_RGB([nn.Upsample(scale_factor=8, mode='nearest')(result)], 0, "result_visualize/")
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)#金字塔结构(batch*H//8*W//8, dim, H//8, W//8)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet1(image1)#[b,hdim+cdim,H/8,W/8]
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)#[b,hdim,H/8,W/8],[b,cdim,H/8,W/8] hdim = 128
                                                            #因为这里用的是和上面拼接后特征提取一样的结构，所以要对其进行分离

   
            net = torch.tanh(net)
            inp = torch.relu(inp)#激活
            

        coords0, coords1 = self.initialize_flow(image1)#[B,2,H//8, W//8] ,[B,2,H//8, W//8] 初始化生成对应的图像，均分生成
                                                        #起初coords0、coords1一样
        # coords0 = coords0 * mask
        # coords1 = coords1 * mask
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_conf = []
        for itr in range(iters):#迭代
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume   #[B，（2*r+1）*（2*r+1）*4,H/8，W/8]#金字塔四层中收集信息
                                    #相似度

            flow = coords1 - coords0#coords1就会变化代表的就是光流与原图像的变化#[B,2,H//8, W//8]
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, delta_conf = self.update_block(net, inp, corr, flow)#生成下一状态和光流变化

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            coords1 = coords1*mask
            delta_conf = delta_conf*mask
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                delta_conf = upConf(delta_conf)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                delta_conf = upConf(delta_conf)
            
            flow_predictions.append(flow_up)
            flow_conf.append(delta_conf)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions, flow_conf
