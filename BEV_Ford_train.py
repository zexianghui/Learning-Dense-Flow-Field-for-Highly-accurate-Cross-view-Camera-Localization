
import os
import matplotlib.pyplot as plt
import skimage.io as io
import torchvision.utils
import cv2
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['MASTER_PORT'] = '7149'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataLoader.Ford_dataset import SatGrdDatasetFord, SatGrdDatasetFordTest, train_logs, train_logs_img_inds, test_logs, test_logs_img_inds
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio
import gen_BEV.utils as utils
import ssl
import math
ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights
from visualize_utils import line_point, save_img,RGB_VIGOR_pose
from models_ford import BEV_corr

import numpy as np
import os
import argparse
import random
from gen_BEV.utils import gps2distance
import time
from op_flow.loss_fun import sequence_loss, fetch_optimizer,corr_test_loss, loss_fun
from RANSAC_lib.RANSAC import RANSAC

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

    
def train(net, lr, args, save_path, train_log_start=1, train_log_end=2):
    bestRankResult = 0.0  # current best, Siam-FCANET18
    # loop over the dataset multiple times
    print(args.resume)
    print(args.epochs)
    optimizer, scheduler = fetch_optimizer(args, net)
    logfile_name = save_path+"/train_log.txt"
    for epoch in range(args.resume, args.epochs):
        net.train()

        # base_lr = 0
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        print(base_lr)

        optimizer.zero_grad()

        ### feeding A and P into train loader
        if args.dpp:
            train_set = SatGrdDatasetFord(logs=train_logs[train_log_start:train_log_end],
                                logs_img_inds=train_logs_img_inds[train_log_start:train_log_end],
                                shift_range_lat=args.shift_range_lat,
                                shift_range_lon=args.shift_range_lon,
                                rotation_range=args.rotation_range,
                                whole=0)
            train_sampler = DistributedSampler(train_set)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, sampler = train_sampler, pin_memory=True,
                                    num_workers=2, drop_last=True)
            train_sampler.set_epoch(epoch)
        else:
            train_set = SatGrdDatasetFord(logs=train_logs[train_log_start:train_log_end],
                                logs_img_inds=train_logs_img_inds[train_log_start:train_log_end],
                                shift_range_lat=args.shift_range_lat,
                                shift_range_lon=args.shift_range_lon,
                                rotation_range=args.rotation_range,
                                whole=0)
            trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                    num_workers=2, drop_last=True)
        scaler = GradScaler(enabled=args.mixed_precision)
        loss_vec = []
        loss_vec_10 = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs
            optimizer.zero_grad()
            if args.dpp:
                sat_map_gt, sat_map, grd_img, gt_shift_u, gt_shift_v, gt_heading, R_FL, T_FL = [item.cuda(args.local_rank) for item in Data[:-1]]
            else:
                sat_map_gt, sat_map, grd_img, gt_shift_u, gt_shift_v, gt_heading, R_FL, T_FL = [item.cuda() for item in Data[:-1]]

            """
            sat_map:[B, 3, 512, 512]
            left_camera_k:[B, 3, 3]
            grd_left_imgs:[B, 3, 256, 1024]
            gt_shift_u:[B, 1]
            gt_shift_v:[B, 1]
            gt_heading::[B, 1]
            len(file_name):B
            """
            vis_heading = gt_heading * args.rotation_range
            vis_u = gt_shift_u * (args.shift_range_lon / 0.22)
            vis_v = -gt_shift_v * (args.shift_range_lat / 0.22)
            s_gt_u = gt_shift_u * args.shift_range_lon
            s_gt_v = gt_shift_v * args.shift_range_lat
            s_gt_theta = gt_heading * args.rotation_range
            # RGB_VIGOR_pose(sat_map, grd_img, -vis_v, vis_u, vis_heading, args, loop=0, save_dir='./result_visualize/')
            file_name = Data[-1]

            # zero the parameter gradients
            # optimizer.zero_grad()
            # if args.end2end == 0 and epoch<=15:
            if args.end2end == 0:
                flow_predictions, flow_conf, mask = net(0, sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, gt_heading, mode='train', file_name=file_name)
                #mask
                mask = mask.float()
                mask = F.interpolate(mask, size=(512, 512), mode='bilinear', align_corners=True)
                mask = mask.bool()
                mask = mask.repeat(1,2,1,1)

                def coords_grid(batch, ht, wd, device):#tensor[B,2,H, W]
                    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
                    coords = torch.stack(coords[::-1], dim=0).float()
                    return coords[None].repeat(batch, 1, 1, 1)
                coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
                coords1 = match(mask, vis_heading, vis_u, vis_v, coords0)

                flow_gt = coords1 - coords0
                flow_gt = flow_gt*mask
                
                flow_predictions = flow_predictions
                for level in range(len(flow_predictions)):
                    flow_predictions[level] = flow_predictions[level]*mask
                    flow_conf[level] = flow_conf[level]*mask
                loss, metrics = sequence_loss(0, flow_predictions, flow_conf, flow_gt, mask, args.gamma)
            # if args.end2end == 1 or epoch>15:
            if args.end2end == 1 :
                flow_predictions, flow_conf, mask, pre_u, pre_v, pre_theta = net(epoch, sat_map, grd_img, R_FL, T_FL, gt_shift_u, gt_shift_v, gt_heading, mode='train', file_name=file_name)
                #mask
                mask = mask.repeat(1,2,1,1)

                def coords_grid(batch, ht, wd, device):#tensor[B,2,H, W]
                    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
                    coords = torch.stack(coords[::-1], dim=0).float()
                    return coords[None].repeat(batch, 1, 1, 1)

                coords0 = coords_grid(mask.size()[0], mask.size()[2], mask.size()[3], device=mask.device)
                coords1 = match(mask, vis_heading, vis_u, vis_v, coords0)

                flow_gt = coords1 - coords0
                flow_gt = flow_gt*mask
                
                flow_predictions = flow_predictions
                for level in range(len(flow_predictions)):
                    flow_predictions[level] = flow_predictions[level]*mask
  
                loss, metrics, flow_loss, dis_loss = loss_fun(epoch, flow_predictions, flow_gt, flow_conf, mask, args.gamma, pre_u, pre_v, pre_theta,\
                    s_gt_u, s_gt_v, s_gt_theta,\
                    coe_shift_lat=10, coe_shift_lon=10, coe_theta=10)
                
                print(flow_loss.data,"   ", dis_loss.data )
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # loss_vec.append(loss)
            loss_vec_10.append(loss)

            # view_batch = 0
            # save_img(sat_map_gt[view_batch], 'result_visualize/sat_ori.jpg')
            # save_img(sat_map[view_batch], 'result_visualize/sat_noise.jpg')
            # coords0_gt = coords0[view_batch].permute(1,2,0)
            # coords1_gt = (coords0[view_batch]+flow_gt[view_batch]).permute(1,2,0)
            # match_x = []
            # match_y = []
            # # x = [256, 256+512-vis_u.data.float().cpu()]
            # # y = [256, 256+vis_v.data.float().cpu()]
            # # match_x.append(x)
            # # match_y.append(y)
            # for h in range(coords0_gt.size()[0]):
            #     for w in range(coords1_gt.size()[1]):
            #         if (random.randint(0,10000) == 1 and mask[view_batch,0,h,w]):
            #             x = [coords0_gt[h][w][0].data.float().cpu(), coords1_gt[h][w][0].data.float().cpu() + coords1_gt.size()[1]]
            #             y = [coords0_gt[h][w][1].data.float().cpu(), coords1_gt[h][w][1].data.float().cpu()]
            #             match_x.append(x)
            #             match_y.append(y)
            # line_point('result_visualize/sat_ori.jpg', 'result_visualize/sat_noise.jpg', match_x, match_y,None, 'line_gt.jpg')
            # coords0_gt = coords0[view_batch].permute(1,2,0)
            # coords1_gt = (coords0[view_batch]+flow_predictions[-1][view_batch]).permute(1,2,0)
            # match_x = []
            # match_y = []
            # # x = [256, 256+512-vis_u.data.float().cpu()]
            # # y = [256, 256+vis_v.data.float().cpu()]
            # # match_x.append(x)
            # # match_y.append(y)
            # for h in range(coords0_gt.size()[0]):
            #     for w in range(coords1_gt.size()[1]):
            #         if (random.randint(0,10000) == 1 and mask[view_batch,0,h,w]):
            #             x = [coords0_gt[h][w][0].data.float().cpu(), coords1_gt[h][w][0].data.float().cpu() + coords1_gt.size()[1]]
            #             y = [coords0_gt[h][w][1].data.float().cpu(), coords1_gt[h][w][1].data.float().cpu()]
            #             match_x.append(x)
            #             match_y.append(y)
            # line_point('result_visualize/sat_ori.jpg', 'result_visualize/sat_noise.jpg', match_x, match_y,None, 'line_pre.jpg')

            print(epoch,'    ',Loop,'    ',loss)
            if Loop%10 == 0:
                print(epoch,'    ',Loop,'    '\
                      '    ',torch.tensor(loss_vec_10).float().mean(), '    ',scheduler.get_last_lr())
                # f = open(os.path.join(logfile_name), 'a')
                # f.write(str(epoch)+'    '+str(Loop)+'    '+\
                #       '    '+str(torch.tensor(loss_vec_10).float().mean())+ '    '+str(scheduler.get_last_lr()))
                # f.close()
                loss_vec_10 = []
                print(metrics)

        compNum = epoch % 100

        if args.dpp:
            if args.local_rank ==0:
                torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))


    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')

    #log
    parser.add_argument('--train_log_start', type=int, default=1, help='')
    parser.add_argument('--train_log_end', type=int, default=2, help='')
    parser.add_argument('--test_log_ind', type=int, default=1, help='')
    
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    #DPP
    parser.add_argument('--dpp', type=bool, default=0)
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--n_gpus', type=int, default=2, help='node rank for distributed training')

    parser.add_argument('--end2end', type=bool, default=1)

    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

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

    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')  # 1e-2 
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--num_steps', type=int, default=30*3500)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gamma', type=int, default=0.8)
    parser.add_argument('--clip', type=float, default=1.0)
    
    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsKitti/Ford/geometry_op_Log2'\
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range)

    print('save_path:', save_path)

    return save_path


if __name__ == '__main__':
    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    if args.dpp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group("nccl", world_size = args.n_gpus, rank = args.local_rank)
        torch.cuda.set_device(args.local_rank)

    net = eval("BEV_corr")(args)

    if args.dpp:
        net = nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids = [args.local_rank],find_unused_parameters=True,broadcast_buffers=False)
    else:
        net = torch.nn.DataParallel(net.cuda(), device_ids = [0])
    ###########################

    if args.test:
        for i in range(8):
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_'+str(i)+'.pth'), map_location='cpu'))
            #test1(net, args, save_path, 0., epoch='test1_model_'+str(i))
            # test2(net, args, save_path, 0., epoch = str(i))

    else:

        if args.resume:
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth'), map_location='cpu'))

        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_1.pth')))

        lr = args.lr
        # net.load_state_dict(torch.load('op_flow/raft-kitti.pth'), strict=False)

        train(net, lr, args, save_path, train_log_start=args.train_log_start, train_log_end=args.train_log_end)

