import torch
import torch.optim as optim
import numpy as np
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
# torch.set_printoptions(profile="full")
def sequence_loss(epoch, flow_preds, flow_conf, flow_gt, mask, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    B,C,H,W = mask.size()
    flow_gt = flow_gt[mask].view(2,-1)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    # if mask != None:
    #     aim_num = mask[:,0,:,:].sum()
    #     no_care = mask[:,0,:,:].size()[0]*mask[:,0,:,:].size()[1]*mask[:,0,:,:].size()[2] - aim_num
    # exlude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt**2, dim=1).sqrt()

    for i in range(n_predictions):
        lev_preds = flow_preds[i][mask].view(2,-1)
        lev_conf = flow_conf[i][:,0,:,:][mask[:,0,:,:]].view(1,-1)#ugly
        i_weight = gamma**(n_predictions - i - 1)
        dis_loss = (lev_preds - flow_gt).abs()

        #loss 12
        dis_loss_conf = dis_loss.sum(axis = 0)
        con_dis_loss = (dis_loss_conf - dis_loss_conf.mean())/dis_loss_conf.std()
        # conf_loss = lev_conf*(1/(1+torch.exp(-0.05 * con_dis_loss))) + (1 - lev_conf)*(1/(1+torch.exp(0.05 * con_dis_loss)))
        if epoch>15:
            conf_loss = lev_conf*(1/(1+torch.exp(-0.5 * con_dis_loss))) + (1 - lev_conf)*(1/(1+torch.exp(0.5 * con_dis_loss)))
        else:
            conf_loss = lev_conf*(1/(1+torch.exp(-0.05 * con_dis_loss))) + (1 - lev_conf)*(1/(1+torch.exp(0.05 * con_dis_loss)))
        flow_loss += i_weight  * (dis_loss + 100*conf_loss).mean()

    epe = torch.sum((lev_preds - flow_gt)**2, dim=0).sqrt()
    if mask != None:
        metrics = {
            'epe': epe.mean().item(),
            '5px': (epe < 5).float().mean().item(),
            '15px': (epe < 15).float().mean().item(),
            '25px': (epe < 25).float().mean().item(),
            '50px': (epe < 50).float().mean().item(),
        }
    else:
        metrics = {
            'epe': epe.mean().item(),
            '5px': (epe < 5).float().mean().item(),
            '15px': (epe < 15).float().mean().item(),
            '25px': (epe < 25).float().mean().item(),
            '50px': (epe < 50).float().mean().item(),
        }

    return flow_loss, metrics
    # return flow_loss
    
def sequence_dis_loss(shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta,gamma,
              coe_shift_lat=10, coe_shift_lon=10, coe_theta=10):

    dis_loss = 0

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat)  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon)  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta)  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]
    
    losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]

    dis_loss = torch.mean(losses)

    # print("shift_lat_delta",shift_lat_delta)
    # print("shift_lon_delta",shift_lon_delta)
    # print("thetas_delta",thetas_delta)
    return dis_loss

def loss_fun(epoch, flow_preds, flow_gt, flow_conf, mask, gamma, pre_u, pre_v, pre_theta,\
              gt_u, gt_v, gt_theta,\
              coe_shift_lat=5, coe_shift_lon=5, coe_theta=5):
    
    flow_loss,metrics = sequence_loss(epoch, flow_preds, flow_conf, flow_gt, mask, gamma)
    dis_loss = sequence_dis_loss(pre_u, pre_v, pre_theta,\
              gt_u, gt_v, gt_theta, gamma,\
              coe_shift_lat, coe_shift_lon, coe_theta)
    loss = (flow_loss) + (50 * dis_loss)
    # print(flow_loss)
    # print(dis_loss)
    return loss, metrics, flow_loss, dis_loss

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def corr_test_loss(flow_preds, flow_conf, flow_gt, mask, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    """ Loss function defined over sequence of flow predictions """
    B,C,H,W = mask.size()
    flow_gt = flow_gt[mask].view(2,-1)
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    
    lev_preds = flow_preds[-1][mask].view(2,-1)
    epe = torch.sum((lev_preds - flow_gt)**2, dim=0).sqrt()
    if mask != None:
        metrics = {
            'epe': epe.mean().item(),
            '5px': (epe < 5).float().mean().item(),
            '15px': (epe < 15).float().mean().item(),
            '25px': (epe < 25).float().mean().item(),
            '50px': (epe < 50).float().mean().item(),
        }
    else:
        metrics = {
            'epe': epe.mean().item(),
            '5px': (epe < 5).float().mean().item(),
            '15px': (epe < 15).float().mean().item(),
            '25px': (epe < 25).float().mean().item(),
            '50px': (epe < 50).float().mean().item(),
        }

    return flow_loss, metrics