
import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.KITTI_dataset import load_train_data, load_test1_data, load_test2_data
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio
import gen_BEV.utils as utils
import ssl
import math
ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights
from visualize_utils import line_point, save_img, RGB_KITTI_com_pose, line_img_point,show_feature_map, RGB_KITTI_pose
from RANSAC_lib.euclidean_trans import Least_Squares_weight, rt2edu_matrix, Least_Squares
import numpy as np
import os
import argparse
from models_kitti import BEV_corr
import random
from gen_BEV.utils import gps2distance
import time
from op_flow.loss_fun import  fetch_optimizer,corr_test_loss
from vis.S2G import KITTI_S2G
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

#batch_size 1
def test1(net_test, args, save_path, best_rank_result, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test1_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)


    pred_shifts = []
    pred_headings = []

    gt_shifts = []
    gt_headings = []

    # RANSAC_E = RANSAC(0.5)
    # LS_weight = Least_Squares_weight(50)
    #'epe', '5px', '15px', '25px', '50px'
    test_met = [0,0,0,0,0]

    start_time = time.time()
    for i, data in enumerate(dataloader, 0):
        if i<100:
            continue
        sat_map_gt, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, gt_depth = [item.cuda() for item in data[:-1]]

        vis_heading = gt_heading * args.rotation_range
        vis_u = -gt_shift_u * (args.shift_range_lon / utils.get_meter_per_pixel(scale=1))
        vis_v = -gt_shift_v * (args.shift_range_lat / utils.get_meter_per_pixel(scale=1))
        s_gt_u = gt_shift_u * args.shift_range_lon
        s_gt_v = gt_shift_v * args.shift_range_lat
        s_gt_heading = gt_heading * args.rotation_range

        if args.end2end == 0:
            flow_predictions, flow_conf, mask = net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end=0)
            #mask
            mask = mask.float()
            mask = F.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=True)
            mask = mask.bool()

            coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
            # coords_gt = match(mask, vis_heading, vis_u, vis_v, coords0)
            # 
            # save_img(grd_left_imgs[0], 'result_visualize/grd_ori.jpg')
            # save_img(sat_map_gt[0], 'result_visualize/sat_ori.jpg')
            # save_img(sat_map[0], 'result_visualize/sat_noise.jpg')
            # # grd_solve('result_visualize/sat_ori.jpg', 'result_visualize/grd_ori.jpg', left_camera_k[0:1,:,:].cpu(),'result_visualize/Conf/')
            # # show_feature_map(flow_conf[-1][0], 'result_visualize/Conf/')
            # # overly('result_visualize/Conf/BEV.jpg', 'result_visualize/Conf/0.jpg', 'result_visualize/Conf/')
            # # gt
            # coords0_gt = coords0[0].permute(1,2,0)
            # coords1_gt = (coords_gt[0]).permute(1,2,0)
            # match_x = []
            # match_y = []
            # # x = [256, 256+512-vis_u.data.float().cpu()]
            # # y = [256, 256+vis_v.data.float().cpu()]
            # # match_x.append(x)
            # # match_y.append(y)
            # for h in range(coords0_gt.size()[0]):
            #     for w in range(coords1_gt.size()[1]):
            #         if ((h == 270 and w == 340) or \
            #             (random.randint(0,8000) == 1)) and mask[0,0,h,w]:
            #             x = [coords0_gt[h][w][0].data.float().cpu(), coords1_gt[h][w][0].data.float().cpu() + coords1_gt.size()[1]]
            #             y = [coords0_gt[h][w][1].data.float().cpu(), coords1_gt[h][w][1].data.float().cpu()]
            #             match_x.append(x)
            #             match_y.append(y)
            # line_point('result_visualize/sat_ori.jpg', 'result_visualize/sat_noise.jpg', match_x, match_y,None, 'line_gt.jpg')

            coor_points = coords0 + flow_predictions[-1]
            #mask
            ptsA = coords0.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            ptsB = coor_points.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            # ptsB = coords_gt.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            
            B,C,H,W = coords0.size()
            _,_,sat_H,sat_W = sat_map_gt.size()
            # flow_conf = torch.ones_like(ptsA, device=ptsA.device)
            pre_theta1, pre_u1, pre_v1 = Least_Squares_weight(ptsA, ptsB, flow_conf[-1][mask][None,:,None])
            # pre_theta1, pre_u1, pre_v1 = Least_Squares_weight(ptsA, ptsB, flow_conf)
            edu_matrix = rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
            R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
            rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
            T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
            pre_theta = -pre_theta1/3.14*180
            pre_u = T1[:,0,2][:,None]*utils.get_meter_per_pixel(scale=1)
            pre_v = -T1[:,1,2][:,None]*utils.get_meter_per_pixel(scale=1)

        if args.end2end == 1:
            flow_predictions, flow_conf, mask, pre_u, pre_v, pre_theta = net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading,  end2end=1)
        
        s2g = KITTI_S2G(args)
        uv = s2g(sat_map, grd_left_imgs, left_camera_k)
        show_feature_map(flow_conf[-1].permute(0,2,3,1), 'result_visualize/PCA/')
        save_img(grd_left_imgs[0], 'result_visualize/grd_ori.jpg')
        # save_img(sat_map_gt[0], 'result_visualize/sat_ori.jpg')
        save_img(sat_map[0], 'result_visualize/sat_noise.jpg')
        # line_img_point('result_visualize/PCA/grd_feat.jpg', 'result_visualize/PCA/sat_refine.jpg', \
        #         flow_predictions[-1][0].cpu().detach().numpy(), mask[0].cpu().detach().numpy(), \
        #             flow_conf[-1][0].cpu().detach().numpy(),'line_feat.jpg',uv.cpu().detach().numpy())
        
        shifts = torch.cat([pre_v, pre_u], dim=-1)
        gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
        pred_shifts.append(shifts.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())

        pred_headings.append(pre_theta.data.cpu().numpy())
        gt_headings.append(s_gt_heading.data.cpu().numpy())
        
        mcc_v = input("v")
        mcc_v = float(mcc_v)
        mcc_u = input("u")
        mcc_u = float(mcc_u)
        mcc_thate = input("thate")
        mcc_thate = float(mcc_thate)
        
        RGB_KITTI_com_pose(sat_map, grd_left_imgs, gt_u=vis_u, gt_v=vis_v, gt_theta = vis_heading,\
            pre_u = -shifts[:,1]/ utils.get_meter_per_pixel(scale=1), pre_v =  -shifts[:,0]/ utils.get_meter_per_pixel(scale=1),pre_theta=pre_theta[:,0],\
            com_u = -mcc_u/ utils.get_meter_per_pixel(scale=1), com_v = -mcc_v/ utils.get_meter_per_pixel(scale=1),come_theta=-mcc_thate,\
            save_dir='./result_visualize/')
        
        RGB_KITTI_pose(sat_map, grd_left_imgs, gt_u=vis_u, gt_v=vis_v, gt_theta = vis_heading,\
            pre_u = -shifts[:,1]/ utils.get_meter_per_pixel(scale=1), pre_v =  -shifts[:,0]/ utils.get_meter_per_pixel(scale=1),pre_theta=pre_theta[:,0],\
            com_u = -mcc_u/ utils.get_meter_per_pixel(scale=1), com_v = -mcc_v/ utils.get_meter_per_pixel(scale=1),come_theta=-mcc_thate,\
            save_dir='./result_visualize/')
        
        line_img_point('result_visualize/grd_ori.jpg','result_visualize/pos_pre.jpg',\
            flow_predictions[-1][0].cpu().detach().numpy(), mask[0].cpu().detach().numpy(),\
                flow_conf[-1][0].cpu().detach().numpy(),'line_img.jpg',uv.cpu().detach().numpy())
        
        # line_img_point('result_visualize/grd_ori.jpg','result_visualize/pos.jpg',\
        #     flow_predictions[-1][0].cpu().detach().numpy(), mask[0].cpu().detach().numpy(),\
        #         flow_conf[-1][0].cpu().detach().numpy(),'line_img.jpg',uv.cpu().detach().numpy())
        

        if args.test_flow:
            coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
            coords1 = match(mask, vis_heading, vis_u, vis_v, coords0)#gt
            flow_gt = coords1 - coords0
            flow_gt = flow_gt*mask
            flow_predictions = flow_predictions
            loss, metrics = corr_test_loss(flow_predictions, flow_gt, mask.repeat(1,2,1,1), args.gamma)#?
        #++
        # for z in range(len(meters)):
        #     if abs(pre_u - s_gt_u) < meters[z]:
        #         in_lon_num[z] = in_lon_num[z] + 1
        #     if abs(pre_v - s_gt_v) < meters[z]:
        #         in_lat_num[z] = in_lat_num[z] + 1
        #     if abs(pre_theta - s_gt_heading) < angles[z]:
        #         in_angles[z] = in_angles[z] + 1

        if args.test_flow:
            j = 0
            for key in metrics.keys():
                test_met[j] = test_met[j] + metrics[key]
                j = j + 1

        if i % 20 == 0:
            print(i,"/",len(dataloader))

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0)
    pred_headings = np.concatenate(pred_headings, axis=0)
    gt_shifts = np.concatenate(gt_shifts, axis=0)
    gt_headings = np.concatenate(gt_headings, axis=0)

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]
    if args.test_flow:
        print('Time per image (second): ' + str(duration))
        print('epe:{:.3f}'.format(test_met[0]/len(dataloader)))
        print('5px:{:.3f}'.format(test_met[1]/len(dataloader)*100))
        print('15px:{:.3f}'.format(test_met[2]/len(dataloader)*100))
        print('25px:{:.3f}'.format(test_met[3]/len(dataloader)*100))
        print('50px:{:.3f}'.format(test_met[4]/len(dataloader)*100))

    if not os.path.exists(save_path):
            os.makedirs(save_path)

    file_name = save_path+"/Test1_results.txt"
    f = open(os.path.join(file_name), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    f.write('Validation results:' + '\n')
    f.write('Pred distance average: ' + str(np.mean(distance)) + '\n')
    f.write('Pred distance median: ' + str(np.median(distance)) + '\n')
    f.write('Pred angle average: ' + str(np.mean(angle_diff)) + '\n')
    f.write('Pred angle median: ' + str(np.median(angle_diff)) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Validation results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Pred distance median: ', np.median(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))
    print('Pred angle median: ', np.median(angle_diff))

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')
    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()

#batch_size 1
def test2(net_test, args, save_path, best_rank_result, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test2_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

    pred_shifts = []
    pred_headings = []

    gt_shifts = []
    gt_headings = []

    # RANSAC_E = RANSAC(0.5)
    #'epe', '5px', '15px', '25px', '50px'
    test_met = [0,0,0,0,0]

    start_time = time.time()
    # RANSAC_E = RANSAC(0.5)
    for i, data in enumerate(dataloader, 0):
        sat_map_gt, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, gt_depth = [item.cuda() for item
                                                                                               in data[:-1]]
        vis_heading = gt_heading * args.rotation_range
        vis_u = -gt_shift_u * (args.shift_range_lon / utils.get_meter_per_pixel(scale=1))
        vis_v = -gt_shift_v * (args.shift_range_lat / utils.get_meter_per_pixel(scale=1))
        s_gt_u = gt_shift_u * args.shift_range_lon
        s_gt_v = gt_shift_v * args.shift_range_lat
        s_gt_heading = gt_heading * args.rotation_range

        if args.end2end == 0:
            flow_predictions, flow_conf, mask = net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end = 0)
            mask = mask.float()
            mask = F.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=True)
            mask = mask.bool()
            coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
            # coords_gt = match(mask, vis_heading, vis_u, vis_v, coords0)
            # save_img(grd_left_imgs[0], 'result_visualize/grd_ori.jpg')
            # save_img(sat_map_gt[0], 'result_visualize/sat_ori.jpg')
            # save_img(sat_map[0], 'result_visualize/sat_noise.jpg')
            # # grd_solve('result_visualize/sat_ori.jpg', 'result_visualize/grd_ori.jpg', left_camera_k[0:1,:,:].cpu(),'result_visualize/Conf/')
            # # show_feature_map(flow_conf[-1][0], 'result_visualize/Conf/')
            # # overly('result_visualize/Conf/BEV.jpg', 'result_visualize/Conf/0.jpg', 'result_visualize/Conf/')
            # # gt
            # coords0_gt = coords0[0].permute(1,2,0)
            # coords1_gt = (coords_gt[0]).permute(1,2,0)
            # match_x = []
            # match_y = []
            # # x = [256, 256+512-vis_u.data.float().cpu()]
            # # y = [256, 256+vis_v.data.float().cpu()]
            # # match_x.append(x)
            # # match_y.append(y)
            # for h in range(coords0_gt.size()[0]):
            #     for w in range(coords1_gt.size()[1]):
            #         if ((h == 270 and w == 340) or \
            #             (random.randint(0,8000) == 1)) and mask[0,0,h,w]:
            #             x = [coords0_gt[h][w][0].data.float().cpu(), coords1_gt[h][w][0].data.float().cpu() + coords1_gt.size()[1]]
            #             y = [coords0_gt[h][w][1].data.float().cpu(), coords1_gt[h][w][1].data.float().cpu()]
            #             match_x.append(x)
            #             match_y.append(y)
            # line_point('result_visualize/sat_ori.jpg', 'result_visualize/sat_noise.jpg', match_x, match_y,None, 'line_gt.jpg')

            coor_points = coords0 + flow_predictions[-1]
            ptsA = coords0.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            ptsB = coor_points.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            # ptsB = coords_gt.permute(0,2,3,1)[mask.permute(0,2,3,1).repeat(1,1,1,2)].view(1, -1, 2).detach()
            
            B,C,H,W = coords0.size()
            _,_,sat_H,sat_W = sat_map_gt.size()
            # flow_conf = torch.ones_like(ptsA, device=ptsA.device)
            pre_theta1, pre_u1, pre_v1 = Least_Squares_weight(ptsA, ptsB, flow_conf[-1][mask][None,:,None])
            # pre_theta1, pre_u1, pre_v1 = Least_Squares_weight(ptsA, ptsB, flow_conf)
            edu_matrix = rt2edu_matrix(pre_theta1, pre_u1, pre_v1)
            R = edu_matrix * torch.tensor([[[1,1,0],[1,1,0],[0,0,1]]], device=mask.device)
            rol_center = torch.tensor([[[1,0,-sat_H/2],[0,1,-sat_W/2],[0,0,1]]], device=mask.device).repeat(B,1,1)
            T1= torch.inverse(rol_center)@torch.inverse(R)@rol_center@edu_matrix
            pre_theta = -pre_theta1/3.14*180
            pre_u = T1[:,0,2][:,None]*utils.get_meter_per_pixel(scale=1)
            pre_v = -T1[:,1,2][:,None]*utils.get_meter_per_pixel(scale=1)

        if args.end2end == 1:
            flow_predictions, flow_conf, mask, pre_u, pre_v, pre_theta = net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, end2end = 1)

        
        shifts = torch.cat([pre_v, pre_u], dim=-1)
        gt_shift = torch.cat([s_gt_v, s_gt_u], dim=-1)
        pred_shifts.append(shifts.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())

        pred_headings.append(pre_theta.data.cpu().numpy())
        gt_headings.append(s_gt_heading.data.cpu().numpy())
        
        # mcc_v = input("v")
        # mcc_v = float(mcc_v)
        # mcc_u = input("u")
        # mcc_u = float(mcc_u)
        
        RGB_KITTI_com_pose(sat_map, grd_left_imgs, gt_u=vis_u, gt_v=vis_v, gt_theta = vis_heading,\
            pre_u = -shifts[:,1]/ utils.get_meter_per_pixel(scale=1), pre_v =  shifts[:,0]/ utils.get_meter_per_pixel(scale=1),pre_theta=shifts,\
            com_u = -1/ utils.get_meter_per_pixel(scale=1), com_v = 1/ utils.get_meter_per_pixel(scale=1),come_theta=1,\
            save_dir='./result_visualize/')
        

        if args.test_flow:
            coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
            coords1 = match(mask, vis_heading, vis_u, vis_v, coords0)#gt
            flow_gt = coords1 - coords0
            flow_gt = flow_gt*mask
            flow_predictions = flow_predictions
            loss, metrics = corr_test_loss(flow_predictions, flow_gt, mask.repeat(1,2,1,1), args.gamma)#loss

        if args.test_flow:
            j = 0
            for key in metrics.keys():
                test_met[j] = test_met[j] + metrics[key]
                j = j + 1

        if i % 20 == 0:
            print(i,"/",len(dataloader))

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0)
    pred_headings = np.concatenate(pred_headings, axis=0)
    gt_shifts = np.concatenate(gt_shifts, axis=0)
    gt_headings = np.concatenate(gt_headings, axis=0)

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]
    if args.test_flow:
        print('Time per image (second): ' + str(duration))
        print('epe:{:.3f}'.format(test_met[0]/len(dataloader)))
        print('5px:{:.3f}'.format(test_met[1]/len(dataloader)*100))
        print('15px:{:.3f}'.format(test_met[2]/len(dataloader)*100))
        print('25px:{:.3f}'.format(test_met[3]/len(dataloader)*100))
        print('50px:{:.3f}'.format(test_met[4]/len(dataloader)*100))

    if not os.path.exists(save_path):
            os.makedirs(save_path)

    file_name = save_path+"/Test2_results.txt"
    f = open(os.path.join(file_name), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    f.write('Validation results:' + '\n')
    f.write('Pred distance average: ' + str(np.mean(distance)) + '\n')
    f.write('Pred distance median: ' + str(np.median(distance)) + '\n')
    f.write('Pred angle average: ' + str(np.mean(angle_diff)) + '\n')
    f.write('Pred angle median: ' + str(np.median(angle_diff)) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Validation results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Pred distance median: ', np.median(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))
    print('Pred angle median: ', np.median(angle_diff))

    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')
    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()


def match(mask, rot, tran_x, tran_y, coords0):
    B,C,H,W = mask.size()
    coords1 = []

    for i in range(B):
        coords1_i = (coords0.clone().permute(0,2,3,1))[i][None,:]
        ones = torch.ones((1,H,W,1)).to(coords1_i.device)
        coords1_i = torch.cat((coords1_i, ones), dim=-1)
        coords1_i = coords1_i.view(1*H*W, 3, 1)

        rot1 = rot[i][None,:]/180*math.pi
        cos = torch.cos(rot1)
        cos = cos[:,None]
        sin = torch.sin(rot1)
        sin = sin[:,None]
        zero = torch.zeros_like(sin)
        ones = torch.ones_like(sin)
        tran_x1 = tran_x[i][None,:][:,None]
        tran_y1 = tran_y[i][None,:][:,None]

        rol_tra0 = torch.cat((cos,sin,zero),dim=-1)
        rol_tra1 = torch.cat((-sin,cos,zero),dim=-1)
        rol_tra2 = torch.cat((zero,zero,ones),dim=-1)
        rol_tra = torch.cat((rol_tra0,rol_tra1,rol_tra2),dim = 1)
        rol_tra = rol_tra.repeat(H*W, 1, 1)

        rol_center0 = torch.cat((ones,zero,ones*(-H/2)),dim=-1)
        rol_center1 = torch.cat((zero,ones,ones*(-W/2)),dim=-1)
        rol_center2 = torch.cat((zero,zero,ones),dim=-1)
        rol_center = torch.cat((rol_center0,rol_center1,rol_center2),dim = 1)
        rol_center = rol_center.repeat(H*W, 1, 1)

        tra0 = torch.cat((ones,zero,(-tran_x1)),dim=-1)
        tra1 = torch.cat((zero,ones,(tran_y1)),dim=-1)
        tra2 = torch.cat((zero,zero,ones),dim=-1)
        tra = torch.cat((tra0, tra1, tra2),dim = 1)
        tra = tra.repeat(H*W, 1, 1)

        points = torch.rand((B,3)).to(mask.device)
        points_tran = ((torch.inverse(rol_center))@rol_tra@rol_center@tra@coords1_i)
        #points_tran = (tra@coords1)

        coords1_i = (points_tran[:,:2,:]).view(1, H, W, 2).permute(0,3,1,2)
        coords1.append(coords1_i)
    
    coords1 = torch.cat(coords1,dim=0)
    return coords1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=16, help='resume the trained model')
    parser.add_argument('--test', type=int, default=1, help='test with trained model')
    parser.add_argument('--test_flow', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    #DPP
    parser.add_argument('--dpp', type=bool, default=0)
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--n_gpus', type=int, default=2, help='node rank for distributed training')

    parser.add_argument('--end2end', type=bool, default=1)

    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')

    parser.add_argument('--stereo', type=int, default=0, help='use left and right ground image')
    parser.add_argument('--sequence', type=int, default=1, help='use n images merge to 1 ground image')

    parser.add_argument('--rotation_range', type=float, default=10, help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--coe_shift_lat', type=float, default=100., help='meters')
    parser.add_argument('--coe_shift_lon', type=float, default=100., help='meters')
    parser.add_argument('--coe_heading', type=float, default=100., help='degree')
    parser.add_argument('--coe_L1', type=float, default=100., help='feature')
    parser.add_argument('--coe_L2', type=float, default=100., help='meters')
    parser.add_argument('--coe_L3', type=float, default=100., help='degree')
    parser.add_argument('--coe_L4', type=float, default=100., help='feature')

    parser.add_argument('--metric_distance', type=float, default=5., help='meters')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--loss_method', type=int, default=0, help='0, 1, 2, 3')

    parser.add_argument('--level', type=int, default=-1, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=5, help='any integer')
    parser.add_argument('--using_weight', type=int, default=0, help='weighted LM or not')
    parser.add_argument('--damping', type=float, default=0.1, help='coefficient in LM optimization')
    parser.add_argument('--train_damping', type=int, default=0, help='coefficient in LM optimization')

    # parameters below are used for the first-step metric learning traning
    parser.add_argument('--negative_samples', type=int, default=32, help='number of negative samples '
                                                                         'for the metric learning training')
    parser.add_argument('--use_conf_metric', type=int, default=0, help='0  or 1 ')

    parser.add_argument('--direction', type=str, default='S2GP', help='G2SP' or 'S2GP')
    parser.add_argument('--Load', type=int, default=0, help='0 or 1, load_metric_learning_weight or not')
    parser.add_argument('--Optimizer', type=str, default='LM', help='LM or SGD or ADAM')

    parser.add_argument('--level_first', type=int, default=0, help='0 or 1, estimate grd depth or not')
    parser.add_argument('--proj', type=str, default='geo', help='geo, polar, nn')
    parser.add_argument('--use_gt_depth', type=int, default=0, help='0 or 1')

    parser.add_argument('--dropout', type=int, default=0, help='0 or 1')
    parser.add_argument('--use_hessian', type=int, default=0, help='0 or 1')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 0')

    parser.add_argument('--beta1', type=float, default=0.9, help='coefficients for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='coefficients for adam optimizer')

    parser.add_argument('--lr', type=float, default=0.00002, help='learning rate')  # 1e-2
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--num_steps', type=int, default=110000)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gamma', type=int, default=0.8)
    parser.add_argument('--clip', type=float, default=1.0)
    
    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsKitti/KITTI/geometry_opflow'\
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range)

    print('save_path:', save_path)

    return save_path


if __name__ == '__main__':
    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    net = eval("BEV_corr")(args)

    ### cudaargs.epochs, args.debug)
    #net = torch.nn.DataParallel(net)
    net = torch.nn.DataParallel(net.cuda(), device_ids = [0])
    ###########################city; 104.92 MiB already allocated; 14.75 MiB free; 108.00 MiB reser

    if args.test:
        #[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
        test_model = [23]
        for i in test_model:
            print("test"+str(i))
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_'+str(i)+'.pth'), map_location='cpu'))
            test1(net, args, save_path, 0., epoch = str(i))
            test2(net, args, save_path, 0., epoch = str(i))
        
        # net.load_state_dict(torch.load(os.path.join(save_path, 'model_25.pth')))
        # test1(net, args, save_path, 0., epoch=25)
        # test2(net, args, save_path, 0., epoch=25)

    else:
        print(0)

