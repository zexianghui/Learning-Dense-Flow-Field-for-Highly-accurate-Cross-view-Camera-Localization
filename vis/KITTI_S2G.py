import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import gen_BEV.utils as utils
import os
import torchvision.transforms.functional as TF

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

class KITTI_S2G(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(KITTI_S2G, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args
        
        self.level = args.level
        self.N_iters = args.N_iters
        self.using_weight = args.using_weight

        self.SatFeatureNet = ResNet18()
        self.GrdFeatureNet = ResNet18()
        

        self.damping = nn.Parameter(self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True))

        self.meters_per_pixel = []
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))

        self.op_flow = RAFT(args)
        #{k.replace('module.',''):v for k,v in torch.load('op_flow/raft-kitti.pth').items()}
        # self.op_flow.load_state_dict({k.replace('module.',''):v for k,v in torch.load('op_flow/raft-kitti.pth').items()}, strict=False)

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def get_E(self,shift_u, shift_v, heading):
        #shift_u:[B, 1]
        #shift_v:[B, 1]
        #heading：[B, 1]
        B,_ = shift_u.size()
        shift_u_meters = self.args.shift_range_lat * shift_u
        shift_v_meters = self.args.shift_range_lon * shift_v
        heading = heading * 10 / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        return torch.cat([R, T], dim=-1)#[B,3,4]

    def get_I(self,ori_camera_k, grd_W, ori_grdW, grd_H, ori_grdH):
        """
        ori_camera_k:[B, 3, 3]
        """
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1, :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        return camera_k #[B, 3, 3]


    def seq_warp_real2camera(self,XYZ_1, E_inv, I_inv):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
 
        P = I_inv @ E_inv 

        #P=torch.ones_like(P)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        #P:[B,1,1,3,4] XYZ_1:[1,H,W,1,4]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)#[B,H,W,3]
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        #print(uv.size())
        return uv
    
    def get_warp_sat2real(self, satmap_sidelength):

        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def grd_f2BEV(self, image, optical):
        # values in optical within range of [0, H], and [0, W]
        #grd_f: B,C,H,W
        #uv:[B, H, W, 2]
        #jac [3, B, H, W, 2]
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
        return out_val,mask

    
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
    def Least_Squares_weight(self, pstA, pstB, weight, mask):
            #pstA [B,num,2]
            #pstB [B,num,2]
            #x [B,num]
            B, num, _ = pstA.size()
            # weight = weight.permute(0,2,3,1).view(B,num,-1)

            # ws = weight.sum((1,2))[:,None]+1e-6
            # weight = weight.repeat(1,1,2)
            weight = weight.permute(0,2,3,1).view(B,num,-1)
            mask = mask.permute(0,2,3,1).view(B,num,-1)
            ws = (mask.sum((1,2))[:,None]+1e-6)
            G_A = (pstA * mask).sum(axis=1)/(ws)
            G_B = (pstB * mask).sum(axis=1)/(ws)

            Am = (pstA - G_A[:,None,:])*weight
            Bm = (pstB - G_B[:,None,:])

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
    
    def forward(self, sat_map, grd_img_left, left_camera_k):
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


        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        
        grd_feat = F.interpolate(grd_img_left, size=(128, 512), mode='bilinear', align_corners=False)
        _,C, H, W = grd_feat.size()
        A = sat_map.shape[-1]
        XYZ_1=self.get_warp_sat2real(A) #[A, A, 4]
        E_inv=self.get_E(shift_u, shift_v, heading) #[B, 3, 4]
        I_inv=self.get_I(left_camera_k, W, ori_grdW, H, ori_grdH)#[B, 3, 3]

        uv= self.seq_warp_real2camera(XYZ_1, E_inv, I_inv)  # [B, S, E, H, W,2]
        
        return uv




