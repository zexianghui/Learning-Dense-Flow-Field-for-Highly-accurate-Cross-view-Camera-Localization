import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import gen_BEV.utils as utils
import os
import torchvision.transforms.functional as TF
from visualize_utils import channel_features_to_RGB, pcl_features_to_RGB, show_feature_map
from op_flow.raft import RAFT
from op_flow.utils import bilinear_sampler, coords_grid, upflow8
from RANSAC_lib.euclidean_trans import Least_Squares_weight, rt2edu_matrix, Least_Squares
# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from gen_BEV.VGG import VGGUnet, BasicEncoder, RefineNet
from gen_BEV.ResNet import ResNet18,ResNet50

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
#from models_ford import loss_func
#from RNNs import NNrefine

EPS = utils.EPS

class BEV_corr(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(BEV_corr, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args
        
        self.level = args.level
        self.N_iters = args.N_iters
        self.using_weight = args.using_weight
        self.grd_height = -2
        self.rotation_range = args.rotation_range

        self.SatFeatureNet = ResNet18()
        self.GrdFeatureNet = ResNet18()
        

        self.damping = nn.Parameter(self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True))

        self.meters_per_pixel = []

        self.op_flow = RAFT(args)
        #{k.replace('module.',''):v for k,v in torch.load('op_flow/raft-kitti.pth').items()}
        # self.op_flow.load_state_dict({k.replace('module.',''):v for k,v in torch.load('op_flow/raft-kitti.pth').items()}, strict=False)

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def sat2grd_uv(self, rot, shift_u, shift_v, level, H, W, meter_per_pixel):
        '''
        rot.shape = [B]
        shift_u.shape = [B]
        shift_v.shape = [B]
        H: scalar  height of grd feature map, from which projection is conducted
        W: scalar  width of grd feature map, from which projection is conducted
        '''

        B = shift_u.shape[0]

        shift_u = shift_u / np.power(2, 3 - level)
        shift_v = shift_v / np.power(2, 3 - level)

        S = 64 / np.power(2, 3 - level)

        shift_u = shift_u / 64 * S
        shift_v = shift_v / 64 * S

        ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=shift_u.device),
                                torch.arange(0, S, dtype=torch.float32, device=shift_u.device))
        ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] v dimension
        jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # [B, S, S] u dimension

        radius = torch.sqrt((ii-(S/2-0.5 + shift_v.reshape(-1, 1, 1)))**2 + (jj-(S/2-0.5 + shift_u.reshape(-1, 1, 1)))**2)

        theta = torch.atan2(ii - (S / 2 - 0.5 + shift_v.reshape(-1, 1, 1)), jj - (S / 2 - 0.5 + shift_u.reshape(-1, 1, 1)))
        theta = (-np.pi / 2 + (theta) % (2 * np.pi)) % (2 * np.pi)
        theta = (theta + rot[:, None, None] * self.args.rotation_range / 180 * np.pi) % (2 * np.pi)

        theta = theta / 2 / np.pi * W

        # meter_per_pixel = self.meter_per_pixel_dict[city] * 512 / S
        meter_per_pixel = meter_per_pixel * 8 #level
        phimin = torch.atan2(radius * meter_per_pixel[:, None, None], torch.tensor(self.grd_height))
        phimin = phimin / np.pi * H

        uv = torch.stack([theta, phimin.float()], dim=-1)

        return uv

    def grid_sample(self, image, optical):
        # values in optical within range of [0, H], and [0, W]
        N, C, IH, IW = image.shape
        _, H, W, _ = optical.shape

        ix = optical[..., 0].view(N, 1, H, W)
        iy = optical[..., 1].view(N, 1, H, W)

        with torch.no_grad():
            ix_nw = torch.floor(ix)  # north-west  upper-left-x
            iy_nw = torch.floor(iy)  # north-west  upper-left-y
            ix_ne = ix_nw + 1        # north-east  upper-right-x
            iy_ne = iy_nw            # north-east  upper-right-y
            ix_sw = ix_nw            # south-west  lower-left-x
            iy_sw = iy_nw + 1        # south-west  lower-left-y
            ix_se = ix_nw + 1        # south-east  lower-right-x
            iy_se = iy_nw + 1        # south-east  lower-right-y

            torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

            torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

            torch.clamp(ix_se, 0, IW -1, out=ix_se)
            torch.clamp(iy_se, 0, IH -1, out=iy_se)

        mask_x = (ix >= 0) & (ix <= IW - 1)
        mask_y = (iy >= 0) & (iy <= IH - 1)
        mask = mask_x * mask_y

        assert torch.sum(mask) > 0

        nw = (ix_se - ix) * (iy_se - iy) * mask
        ne = (ix - ix_sw) * (iy_sw - iy) * mask
        sw = (ix_ne - ix) * (iy - iy_ne) * mask
        se = (ix - ix_nw) * (iy - iy_nw) * mask

        image = image.view(N, C, IH * IW)

        nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
        se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

        out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

        return out_val, mask
    
    def grd_f2BEV(self, grd_f, rot, shift_u, shift_v, level, meter_per_pixel):
        '''
        grd_f.shape = [B, C, H, W]
        shift_u.shape = [B]
        shift_v.shape = [B]
        '''
        B, C, H, W = grd_f.size()
        uv = self.sat2grd_uv(rot, shift_u, shift_v, level, H, W, meter_per_pixel)  # [B, S, S, 2]
        grd_f_trans, mask = self.grid_sample(grd_f, uv)
        return grd_f_trans, mask
    
    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v):
        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 + gt_shift_u[:, 0] * self.args.shift_range_lon / meter_per_pixel)
            h = torch.round(corr_H / 2 - gt_shift_v[:, 0] * self.args.shift_range_lat / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))
    
    #Least-Squares Fitting of Two 3-D Point Sets
    def Least_Squares_weight(self, pstA, pstB, weight):
            #pstA [B,num,2]
            #pstB [B,num,2]
            #x [B,num]
            B, num, _ = pstA.size()
            # weight = weight.permute(0,2,3,1).view(B,num,-1)

            # ws = weight.sum((1,2))[:,None]+1e-6
            # weight = weight.repeat(1,1,2)
            weight = weight.permute(0,2,3,1).view(B,num,-1)
            ws = (weight.sum((1,2))[:,None]+1e-6)
            G_A = (pstA * weight).sum(axis=1)/(ws)
            G_B = (pstB * weight).sum(axis=1)/(ws)

            Am = (pstA - G_A[:,None,:])*weight
            Bm = (pstB - G_B[:,None,:])*weight

            H = torch.bmm(Am.permute(0,2,1) ,Bm)
            H = H + (torch.eye(2, device=H.device)*1e-5).view(1,2,2).repeat(B,1,1)
            # print("min:",H.min(),"max:",H.max())
            # H = H + 1e-6
            # U, S, V = torch_batch_svd.svd(H)
            U, S, V = torch.svd(H)

            # print(torch.det(H))
            # print(torch.isnan(S))
            # print(S.grad)
            # print(U@torch.diag(S[0])@V.permute(0,2,1))
            R = torch.bmm(V, U.permute(0,2,1))
            # theta = torch.zeros((B,1),device = pstA.device)
            # for i in range(B):
            #     if torch.det(R[i]) < 0:
            #         print("det(R) < R, reflection detected!, correcting for it ...")
            #         V[i,:,1] *= -1
            #         R[i] = V[i] @ U[i].T 
            #     theta[i,0] = torch.arccos((torch.trace(R[i]))/2)*R[i,1,0]/(torch.abs(R[i,0,1]) + 1e-4)
                # print("R:",torch.abs(R[i,0,1]))
            # print("R",R)
            there_cos = torch.clamp(torch.einsum('bii->b', R)/2,-0.99999,0.99999)#https://github.com/pytorch/pytorch/issues/61810
            theta = torch.arccos(there_cos)*R[:,1,0]/(torch.abs(R[:,0,1]) + 1e-6)
            theta = theta[:,None]
            G_A = G_A.unsqueeze(-1)
            G_B = G_B.unsqueeze(-1)
            t = -torch.bmm(R,G_A) + G_B
            return theta, t[:,0], t[:,1]
    
    def rt2edu_matrix(self, rot, u, v):
        B = rot.size()[0]
        cos = torch.cos(rot)
        sin = torch.sin(rot)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        Euclidean_matrix = torch.cat([cos, -sin, u, sin, cos, v, zeros, zeros, ones], dim=-1)
        Euclidean_matrix = Euclidean_matrix.view(B, 3, 3)
        # Euclidean_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), u],
        #                                 [torch.sin(rot), torch.cos(rot), v],
        #                                 [torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)]], device=rot.device)
        return Euclidean_matrix
    
    def forward(self, sat_map, grd_img_left, meter_per_pixel, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                mode='train', file_name=None, gt_depth=None):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat = self.SatFeatureNet(sat_map)
        
        grd_feat = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        _,C, H, W = grd_feat.size()
        A = sat_feat.shape[-1]
        
        grd_feat_proj,mask = self.grd_f2BEV(grd_feat, heading, shift_u, shift_v, level=3, meter_per_pixel=meter_per_pixel)
        mask = mask.detach()
        
        #g2s_feat_list.append(grd_feat_proj)
        #grd_feat_proj
        
        # pcl_features_to_RGB([nn.Upsample(scale_factor=8, mode='nearest')(grd_feat_list[0])], 0, "result_visualize/BEV/PCA/grd/")
        # pcl_features_to_RGB([nn.Upsample(scale_factor=8, mode='nearest')(grd_feat_proj)], 0, "result_visualize/BEV/PCA/BEV/")
        # pcl_features_to_RGB([nn.Upsample(scale_factor=8, mode='nearest')(sat_feat)], 0, "result_visualize/BEV/PCA/sat/")
        # show_feature_map(mask[0], "result_visualize/BEV/mask/")
        # show_feature_map(grd_feat_proj[0], "result_visualize/g2s_channel_64*64/")
        # show_feature_map(sat_feat[0], "result_visualize/sat_channel_64*64/")
        flow_predictions, flow_conf = self.op_flow(grd_feat_proj, sat_feat.float(), mask, iters=self.args.iters)
        
        if self.args.end2end == 0:
            return flow_predictions, flow_conf, mask
        else:
            #mask
            mask = mask.float()
            mask = F.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=True)
            mask = mask.bool()
            coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
            B,C,H,W = coords0.size()
            coor_points = coords0 + flow_predictions[-1]
            #mask
            ptsA  = coords0.permute(0,2,3,1).view(B,H*W,-1)#!
            ptsB = coor_points.permute(0,2,3,1).view(B,H*W,-1)
            B,C,H,W = coords0.size()
            _,_,sat_H,sat_W = sat_map.size()
            # flow_conf = torch.ones_like(ptsA, device=ptsA.device)
            pre_theta1, pre_u1, pre_v1 = self.Least_Squares_weight(ptsA, ptsB, flow_conf[-1])
            # pre_theta1, pre_u1, pre_v1 = Least_Squares_weight(ptsA, ptsB, flow_conf)
            edu_matrix = self.rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
            R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
            rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
            # T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
            T1= edu_matrix@torch.inverse(rol_center)@torch.inverse(R)@rol_center
            pre_theta = pre_theta1/3.14*180
            # pre_u = T1[:,0,2][:,None]*meter_per_pixel[:,None]
            # pre_v = -T1[:,1,2][:,None]*meter_per_pixel[:,None]
            pre_u = T1[:,0,2][:,None]
            pre_v = T1[:,1,2][:,None]
            
            return flow_predictions, flow_conf, mask, pre_u, pre_v, pre_theta



