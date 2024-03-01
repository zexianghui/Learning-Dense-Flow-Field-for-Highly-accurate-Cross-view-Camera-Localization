import skimage.io as io
import os.path
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import random
def features_to_RGB(sat_feat_list, grd_feat_list, pred_feat_dict, gt_sat_feat_proj, loop=0, save_dir='./visualize/'):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    # sat_shape = []
    # grd_shape = []
    for level in range(len(sat_feat_list)):
    # for level in [len(sat_feat_list)-1]:
        flatten = []

        sat_feat = sat_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        s2g_feat = [feat.data.cpu().numpy() for feat in pred_feat_dict[level]]
        # a list with length iters, each item has shape [B, C, H, W]
        gt_a2g = gt_sat_feat_proj[level].data.cpu().numpy()   # [B, C, H, W]

        B, C, A, _ = sat_feat.shape
        B, C, H, W = grd_feat.shape
        # sat_shape.append([B, C, A, A])
        # grd_shape.append([B, C, H, W])

        flatten.append(reshape_normalize(sat_feat))
        flatten.append(reshape_normalize(grd_feat))
        flatten.append(reshape_normalize(gt_a2g[:, :, H//2:, :]))

        for feat in s2g_feat:
            flatten.append(reshape_normalize(feat[:, :, H//2:, :]))

        flatten = np.concatenate(flatten[:1], axis=0)

        # if level == 0:
        pca = PCA(n_components=3)
        pca.fit(reshape_normalize(sat_feat))

        pca_grd = PCA(n_components=3)
        pca_grd.fit(reshape_normalize(grd_feat))

    # for level in range(len(sat_feat_list)):
        sat_feat = sat_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        s2g_feat = [feat.data.cpu().numpy() for feat in pred_feat_dict[level]]
        # a list with length iters, each item has shape [B, C, H, W]
        gt_s2g = gt_sat_feat_proj[level].data.cpu().numpy()   # [B, C, H, W]

        B, C, A, _ = sat_feat.shape
        B, C, H, W = grd_feat.shape
        sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat[..., :]))) + 1 )/ 2).reshape(B, A, A, 3)
        grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)
        gt_s2g_new = ((normalize(pca.transform(reshape_normalize(gt_s2g[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

        for idx in range(B):
            sat = Image.fromarray((sat_feat_new[idx] * 255).astype(np.uint8))
            sat = sat.resize((512, 512))
            sat.save(os.path.join(save_dir, 'sat_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.jpg'))

            grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
            grd = grd.resize((1024, 128))
            grd.save(os.path.join(save_dir, 'grd_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.jpg'))

            s2g = Image.fromarray((gt_s2g_new[idx] * 255).astype(np.uint8))
            s2g = s2g.resize((1024, 128))
            s2g.save(os.path.join(save_dir, 's2g_gt_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.jpg'))

        # for iter in range(len(s2g_feat)):
        for iter in [len(s2g_feat)-1]:
            feat = s2g_feat[iter]
            feat_new = ((normalize(pca.transform(reshape_normalize(feat[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

            for idx in range(B):
                img = Image.fromarray((feat_new[idx] * 255).astype(np.uint8))
                img = img.resize((1024, 128))
                img.save(os.path.join(save_dir, 's2g_feat_' + str(loop * B + idx) + '_level_' + str(level)
                                      + '_iter_' + str(iter) + '.jpg'))

    return


def RGB_iterative_pose(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 + shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 + gt_shift_u.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lons[idx, :-1], shift_lats[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lons[idx, -1], shift_lats[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lons[idx, -1], shift_lats[idx, -1], 1, 1, angles=thetas[idx, -1], color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx], color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.jpg'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.jpg'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.jpg'))


def RGB_iterative_pose_ford(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 - shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 - gt_shift_u.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lats[idx, :-1], shift_lons[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lats[idx, -1], shift_lons[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=90, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lats[idx, -1], shift_lons[idx, -1], 1, 1, angles=thetas[idx, -1] + 90, color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx] + 90, color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.jpg'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.jpg'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.jpg'))

#pcl_features_to_RGB([feature_map], 0, "result_visualize/")
def pcl_features_to_RGB(grd_feat_list, loop=0, save_dir='./result_vis/'):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    # sat_shape = []
    # grd_shape = []
    for level in range(len(grd_feat_list)):
    # for level in [len(sat_feat_list)-1]:
        flatten = []

        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape

        flatten.append(reshape_normalize(grd_feat))

        flatten = np.concatenate(flatten[:1], axis=0)

        # if level == 0:
        pca_grd = PCA(n_components=3)
        pca_grd.fit(reshape_normalize(grd_feat))

    # for level in range(len(sat_feat_list)):
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape
        grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat))) + 1) / 2).reshape(B, H, W, 3)
        #gt_s2g_new = ((normalize(pca.transform(reshape_normalize(gt_s2g[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

        for idx in range(B):
            # if not os.path.exists(os.path.join(save_dir)):
            #     os.makedirs(os.path.join(save_dir))

            grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
            grd = grd.resize((W,H))
            grd.save(save_dir)

    return

#feature_map [C,H,W]
def show_feature_map(feature_map, save_path):
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #feature_map = feature_map.squeeze(0)
    C = feature_map.shape[0]
    plt.figure()

    for i in range(C):
        if i>10:
            break
        #wid = math.ceil(math.sqrt(C))
        #ax = plt.subplot(int(wid), int(wid), i + 1)
        #ax.set_title('Feature {}'.format(i))
        #ax.axis('off')
        figure_map = feature_map.data.cpu().numpy()[i,:,:]
        # img = transforms.functional.to_pil_image((figure_map*512).astype(np.uint8))
        # img.save(save_path + str(i) + '.png')
        plt.imshow(figure_map, cmap='jet')
        plt.axis("off")
        plt.xticks([])    
        plt.yticks([])   
        plt.savefig(save_path + str(i) + '.jpg',  bbox_inches='tight',pad_inches=0, quality=100, dpi = 138.6)  
        plt.close()

def channel_features_to_RGB(sat_feat_list, grd_feat_list, loop=0, save_dir='./visualize/'):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    
    for level in range(len(sat_feat_list)):
        level = 2

        sat_feat = sat_feat_list[level]
        grd_feat = grd_feat_list[level]

        B, C, A, _ = sat_feat.shape
        B, C, H, W = grd_feat.shape

        for idx in range(B):
            grd_save_path = os.path.join(save_dir, 'level_' + str(level) + '_batch_' + str(idx) + '_grd_feat/')
            sat_save_path = os.path.join(save_dir, 'level_' + str(level) + '_batch_' + str(idx) + '_sat_feat/')
            show_feature_map(sat_feat[idx], sat_save_path)
            show_feature_map(grd_feat[idx], grd_save_path)

    return


def RGB_pose(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./result_visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy()  / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 + shift_lons.data.cpu().numpy()  / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 + gt_shift_u.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        #update = ax.scatter(shift_lons[idx, :-1], shift_lats[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lons[idx, -1], shift_lats[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lons[idx, -1], shift_lats[idx, -1], 1, 1, angles=thetas[idx, -1], color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx], color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.jpg'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        sat_map=sat_img[idx]
        sat_map=sat_map.view(3, 512, 512)
        sat_map=transforms.ToPILImage()(sat_map)
        sat_map.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.jpg'))
        grd_left_imgs=grd_img[idx]
        grd_left_imgs = grd_left_imgs.view(3, 256, 1024)
        grd_left_imgs=transforms.ToPILImage()(grd_left_imgs)
        grd_left_imgs.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.jpg'))

def line_point(img1, img2, match_x, match_y, flow_conf, name):
   
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.zeros((image_1.shape[0], image_1.shape[1] + image_2.shape[1], 3))
    # print(pj1.dtype)


    fig, ax = plt.subplots()
    split[:,:image_1.shape[1],:] = image_1.copy()  
    split[:,image_2.shape[1]:,:] = image_2.copy()  
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)  
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1) 
        if flow_conf != None:
            txt = ax.annotate(str(int(flow_conf[-1][0,0,int(match_x[i][0]), int(match_y[i][0])]*100)), xy=((match_x[i][1], match_y[i][1])), \
                            xytext=(2, 2), textcoords='offset points', weight = 'light', color = 'b')
    
    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

def line_conf_point(img1, img2, match_x, match_y, flow_conf, name):
    
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c],(0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.zeros((image_1.shape[0], image_1.shape[1] + image_2.shape[1], 3))   
    # print(pj1.dtype) 


    fig, ax = plt.subplots()
    split[:,:image_1.shape[1],:] = image_1.copy()  
    split[:,image_2.shape[1]:,:] = image_2.copy() 
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)  
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        if (int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100))>50:
            line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1)   
            if flow_conf != None:
                txt = ax.annotate(str(int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100)), xy=((match_x[i][1], match_y[i][1])), \
                                xytext=(2, 2), textcoords='offset points', weight = 'light', color = 'b')
        else:
            line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'y', alpha = 1)   
            if flow_conf != None:
                txt = ax.annotate(str(int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100)), xy=((match_x[i][1], match_y[i][1])), \
                                xytext=(2, 2), textcoords='offset points', weight = 'light', color = 'y')

    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

def save_img(img, save_path):
    img = transforms.functional.to_pil_image(img, mode='RGB')
    img.save(save_path)




def overly(img_path, heat_path, save_path):
    heat_img = cv2.imread(heat_path, flags=1)
    org_img =  cv2.imread(img_path, flags=1) 
    # heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    cv2.imwrite(save_path+'heat_img.jpg', heat_img)    
    add_img = cv2.addWeighted(org_img, 0.7, heat_img, 0.5, 0)
    cv2.imwrite(save_path+'overly.jpg', add_img )



def oxford_ori_sat(sat_img_path,shift_u, shift_v, thetas,idx):
    SatMap_name = os.path.join(sat_img_path)
    with Image.open(SatMap_name, 'r') as SatMap:
        sat_map = SatMap.convert('RGB')
    #（ax+by+c, dx+ey+f）
    sat_map_ori = sat_map.transform(sat_map.size, Image.AFFINE, (1, 0, -shift_u[idx].data.cpu().item(), 0, 1, shift_v[idx].data.cpu().item()))
    sat_map_ori = sat_map_ori.rotate(thetas[idx].data.cpu().item())
    sat_map_ori.save('result_visualize/sat_ori.jpg')
    

def RGB_KITTI_com_pose(sat_map, grd_img, gt_u, gt_v, gt_theta,\
        pre_u,pre_v, pre_theta,com_u, com_v, come_theta,save_dir='./result_visualize/'):
    
    idx = 0 
    B, _, A, _ = sat_map.shape
    
    sat_map = sat_map[idx]
    # sat_map=sat_map.view(3, 512, 512)
    sat_map=transforms.ToPILImage()(sat_map)
    sat_map.save(os.path.join(save_dir, 'sat.jpg'))
    
    grd_left_imgs=grd_img[idx]
    # grd_left_imgs = grd_left_imgs.view(3, 256, 1024)
    grd_left_imgs=transforms.ToPILImage()(grd_left_imgs)
    grd_left_imgs.save(os.path.join(save_dir, 'grd.jpg'))

    aerial_img1 = cv2.imread(os.path.join(save_dir, 'sat.jpg'))[:,:,::-1]
    fig, ax = plt.subplots()
    ax.imshow(aerial_img1)

    ax_gt_u, ax_gt_v = A/2-gt_u[idx].data.cpu(), A/2+gt_v[idx].data.cpu()
    ax_pre_u, ax_pre_v = A/2-pre_u[idx].data.cpu(), A/2+pre_v[idx].data.cpu()
    ax_com_u, ax_com_v = A/2-com_u, A/2+com_v
    # init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
    gt = ax.scatter(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), marker='*',color='b', s=200, zorder=2)
    gt = ax.quiver(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2, width  = 0.013, scale = 13)
    pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), marker='*',color='r', s=200, zorder=2)
    pre = ax.quiver(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), 1, 1, angles=pre_theta[idx].cpu().item(), color='r', zorder=2, width  = 0.013, scale = 13)
    # pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), s=200, marker='*', facecolor='y', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, s=200, marker='*', facecolor='g', label='Ours', edgecolors='white')
    com = ax.scatter(ax_com_u, ax_com_v, color='g', marker='*', s=200, zorder=2)
    com = ax.quiver(ax_com_u, ax_com_v, 1, 1, angles=come_theta, color='g', zorder=2, width  = 0.013, scale = 13)
    # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
    # gt = ax.quiver(ax_u.cpu().item(), ax_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2)
    
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, 'pos.jpg'),
                transparent=True, dpi=A, bbox_inches='tight',pad_inches=0)
    plt.close()
    
    # pos_img1 = cv2.imread(os.path.join(save_dir, 'pos.jpg'))

def RGB_KITTI_pose(sat_map, grd_img, gt_u, gt_v, gt_theta,\
        pre_u,pre_v, pre_theta,com_u, com_v, come_theta,save_dir='./result_visualize/'):
    
    idx = 0 
    B, _, A, _ = sat_map.shape
    
    sat_map = sat_map[idx]
    # sat_map=sat_map.view(3, 512, 512)
    sat_map=transforms.ToPILImage()(sat_map)
    sat_map.save(os.path.join(save_dir, 'sat.jpg'))
    
    grd_left_imgs=grd_img[idx]
    # grd_left_imgs = grd_left_imgs.view(3, 256, 1024)
    grd_left_imgs=transforms.ToPILImage()(grd_left_imgs)
    grd_left_imgs.save(os.path.join(save_dir, 'grd.jpg'))

    aerial_img1 = cv2.imread(os.path.join(save_dir, 'sat.jpg'))[:,:,::-1]
    fig, ax = plt.subplots()
    ax.imshow(aerial_img1)

    ax_gt_u, ax_gt_v = A/2-gt_u[idx].data.cpu(), A/2+gt_v[idx].data.cpu()
    ax_pre_u, ax_pre_v = A/2-pre_u[idx].data.cpu(), A/2+pre_v[idx].data.cpu()
    ax_com_u, ax_com_v = A/2-com_u, A/2+com_v
    # init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
    # gt = ax.scatter(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), marker='*',color='b', s=200, zorder=2)
    # gt = ax.quiver(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2, width  = 0.013, scale = 13)
    pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), marker='*',color='r', s=200, zorder=2)
    pre = ax.quiver(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), 1, 1, angles=pre_theta[idx].cpu().item(), color='r', zorder=2, width  = 0.013, scale = 13)
    # pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), s=200, marker='*', facecolor='y', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, s=200, marker='*', facecolor='g', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, color='g', marker='*', s=200, zorder=2)
    # com = ax.quiver(ax_com_u, ax_com_v, 1, 1, angles=come_theta, color='g', zorder=2, width  = 0.013, scale = 13)
    # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
    # gt = ax.quiver(ax_u.cpu().item(), ax_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2)
    
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, 'pos_pre.jpg'),
                transparent=True, dpi=138.6, bbox_inches='tight',pad_inches=0)
    plt.close()
    
    # pos_img1 = cv2.imread(os.path.join(save_dir, 'pos.jpg'))

def RGB_oxford_com_pose(sat_map, grd_img, gt_u, gt_v, gt_theta,\
        pre_u,pre_v, pre_theta,com_u, com_v, come_theta,save_dir='./result_visualize/'):
    
    idx = 0 
    B, _, A, _ = sat_map.shape
    
    sat_map = sat_map[idx]
    # sat_map=sat_map.view(3, 512, 512)
    sat_map=transforms.ToPILImage()(sat_map)
    sat_map.save(os.path.join(save_dir, 'sat.jpg'))
    
    grd_left_imgs=grd_img[idx]
    # grd_left_imgs = grd_left_imgs.view(3, 256, 1024)
    grd_left_imgs=transforms.ToPILImage()(grd_left_imgs)
    grd_left_imgs.save(os.path.join(save_dir, 'grd.jpg'))

    aerial_img1 = cv2.imread(os.path.join(save_dir, 'sat.jpg'))[:,:,::-1]
    fig, ax = plt.subplots()
    ax.imshow(aerial_img1)

    ax_gt_u, ax_gt_v = A/2-gt_u[idx].data.cpu(), A/2+gt_v[idx].data.cpu()
    ax_pre_u, ax_pre_v = A/2-pre_u[idx].data.cpu(), A/2+pre_v[idx].data.cpu()
    ax_com_u, ax_com_v = A/2-com_u, A/2+com_v
    # init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
    gt = ax.scatter(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), marker='*',color='b', s=200, zorder=2)
    # gt = ax.quiver(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2, width  = 0.013, scale = 13)
    pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), marker='*',color='r', s=200, zorder=2)
    # pre = ax.quiver(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), 1, 1, angles=pre_theta[idx].cpu().item(), color='r', zorder=2, width  = 0.013, scale = 13)
    # pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), s=200, marker='*', facecolor='y', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, s=200, marker='*', facecolor='g', label='Ours', edgecolors='white')
    com = ax.scatter(ax_com_u, ax_com_v, color='g', marker='*', s=200, zorder=2)
    # com = ax.quiver(ax_com_u, ax_com_v, 1, 1, angles=come_theta, color='g', zorder=2, width  = 0.013, scale = 13)
    # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
    # gt = ax.quiver(ax_u.cpu().item(), ax_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2)
    
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, 'pos.jpg'),
                transparent=True, dpi=A, bbox_inches='tight',pad_inches=0)
    plt.close()
    
    # pos_img1 = cv2.imread(os.path.join(save_dir, 'pos.jpg'))
    
def RGB_oxford_pose(sat_map, grd_img, gt_u, gt_v, gt_theta,\
        pre_u,pre_v, pre_theta,com_u, com_v, come_theta,save_dir='./result_visualize/'):
    
    idx = 0 
    B, _, A, _ = sat_map.shape
    
    sat_map = sat_map[idx]
    # sat_map=sat_map.view(3, 512, 512)
    sat_map=transforms.ToPILImage()(sat_map)
    sat_map.save(os.path.join(save_dir, 'sat.jpg'))
    
    grd_left_imgs=grd_img[idx]
    # grd_left_imgs = grd_left_imgs.view(3, 256, 1024)
    grd_left_imgs=transforms.ToPILImage()(grd_left_imgs)
    grd_left_imgs.save(os.path.join(save_dir, 'grd.jpg'))

    aerial_img1 = cv2.imread(os.path.join(save_dir, 'sat.jpg'))[:,:,::-1]
    fig, ax = plt.subplots()
    ax.imshow(aerial_img1)

    ax_gt_u, ax_gt_v = A/2-gt_u[idx].data.cpu(), A/2+gt_v[idx].data.cpu()
    ax_pre_u, ax_pre_v = A/2-pre_u[idx].data.cpu(), A/2+pre_v[idx].data.cpu()
    ax_com_u, ax_com_v = A/2-com_u, A/2+com_v
    # init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
    # gt = ax.scatter(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), marker='*',color='b', s=200, zorder=2)
    # gt = ax.quiver(ax_gt_u.cpu().item(), ax_gt_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2, width  = 0.013, scale = 13)
    pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), marker='*',color='r', s=200, zorder=2)
    # pre = ax.quiver(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), 1, 1, angles=pre_theta[idx].cpu().item(), color='r', zorder=2, width  = 0.013, scale = 13)
    # pre = ax.scatter(ax_pre_u.cpu().item(), ax_pre_v.cpu().item(), s=200, marker='*', facecolor='y', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, s=200, marker='*', facecolor='g', label='Ours', edgecolors='white')
    # com = ax.scatter(ax_com_u, ax_com_v, color='g', marker='*', s=200, zorder=2)
    # com = ax.quiver(ax_com_u, ax_com_v, 1, 1, angles=come_theta, color='g', zorder=2, width  = 0.013, scale = 13)
    # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
    # gt = ax.quiver(ax_u.cpu().item(), ax_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2)
    
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, 'pos_pre.jpg'),
                transparent=True, dpi=138.6, bbox_inches='tight',pad_inches=0)
    plt.close()

def line_feat_point_heng(img1, img2, flow, mask, conf, name):
    span = 30

    _,H,W = mask.shape
                    
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.ones((image_1.shape[0] , image_1.shape[1]+ image_2.shape[1] +span, 3))*255  
    # print(pj1.dtype) 
    
    match_x = []
    match_y = []
    for h in range(H):
        for w in range(W):
            if(mask[0,h,w] and (conf[0,h,w]>0.9)):
                if(random.randint(1,200) == 1):
                    if ((w + flow[0,h,w])<512 and (w + flow[0,h,w])>0) and \
                        (((h + flow[1,h,w].item()+image_1.shape[0]+span)>(image_1.shape[0]+span)) and \
                            ((h + flow[1,h,w].item()+image_1.shape[0]+span)<(image_1.shape[0]+image_2.shape[0]+span))):
                        match_x.append([ w ,w + flow[0,h,w].item()+image_1.shape[1]+span])
                        match_y.append([h, h + flow[1,h,w].item()])

    fig, ax = plt.subplots()
    split[:, :image_1.shape[0],:] = image_1.copy()  
    split[:,(image_1.shape[0]+span):,:] = image_2.copy() 
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)   
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1) 
    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()    
    
def line_feat_point(img1, img2, flow, mask, conf, name):
    span = 30

    _,H,W = mask.shape
                    
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.ones((image_1.shape[0] + image_2.shape[0] +span, image_1.shape[1], 3))*255 
    # print(pj1.dtype)
    
    match_x = []
    match_y = []

    for h in range(H):
        for w in range(W):
            if(mask[0,h,w] and (conf[0,h,w]>0.9)):
                if(random.randint(1,200) == 1):
                    if ((w + flow[0,h,w])<512 and (w + flow[0,h,w])>0) and \
                        (((h + flow[1,h,w].item()+image_1.shape[0]+span)>(image_1.shape[0]+span)) and \
                            ((h + flow[1,h,w].item()+image_1.shape[0]+span)<(image_1.shape[0]+image_2.shape[0]+span))):
                        match_x.append([ w ,w + flow[0,h,w].item()])
                        match_y.append([h, h + flow[1,h,w].item()+image_1.shape[0]+span])

    fig, ax = plt.subplots()
    split[:image_1.shape[0],:,:] = image_1.copy() 
    split[(image_1.shape[0]+span):,:,:] = image_2.copy()  
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split) 
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1)   
    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

def line_img_point(img1, img2, flow, mask, conf, name, uv):
    span = 15
    _,H,W = mask.shape
                    
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], (0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    # image_1 = np.resize(image_1, (128, 512,3))
    grd_h,grd_w,_ = image_1.shape
    di = grd_w/512
    image_1 = cv2.resize(image_1, dsize=(512, int(grd_h/di)), interpolation=cv2.INTER_CUBIC)
    split = np.ones((image_1.shape[0] + image_2.shape[0] +span, image_1.shape[1], 3))*255 
    # print(pj1.dtype) 
    img_array = image_1.astype('uint8')
    img = Image.fromarray(img_array)
    img.save('example.png')
    match_x = []
    match_y = []

    for h in range(H):
        for w in range(W):
            if(mask[0,h,w] and (conf[0,h,w]>0.9)):
                if(random.randint(1,200) == 1):
                    if ((w + flow[0,h,w])<512 and (w + flow[0,h,w])>0) and \
                        (((h + flow[1,h,w]+image_1.shape[0]+span)>(image_1.shape[0]+span)) and \
                            ((h + flow[1,h,w]+image_1.shape[0]+span)<(image_1.shape[0]+image_2.shape[0]+span))):
                        match_x.append([ uv[0,h,w,0].item() ,w + flow[0,h,w]])
                        match_y.append([ uv[0,h,w,1].item() ,h + flow[1,h,w]+image_1.shape[0]+span])

    fig, ax = plt.subplots()
    split[:image_1.shape[0],:,:] = image_1.copy()  
    split[(image_1.shape[0]+span):,:,:] = image_2.copy()  
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1)   

    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)

    plt.close()
    
def RGB_VIGOR_pose(sat_img, grd_img, shift_u, shift_v, thetas, args, loop=0, save_dir='./result_visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level] u
        shift_lons: [B, Niters, Level] v
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt
    idx = 1
    grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
    grd.save(os.path.join(save_dir, 'grd_' + str(idx) + '.jpg'))

    sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
    sat.save(os.path.join(save_dir, 'sat_' + str(idx) + '.jpg'))

    ground_img  = cv2.imread(os.path.join(save_dir, 'grd_' + str(idx) + '.jpg'))[:,:,::-1]
    aerial_img1 = cv2.imread(os.path.join(save_dir, 'sat_' + str(idx) + '.jpg'))[:,:,::-1]

    W, H, A     = ground_img.shape[1], ground_img.shape[0], aerial_img1.shape[0]
    fig  = plt.figure(figsize=[15, 11])
    grid = plt.GridSpec(2, 3, wspace=0.1, hspace=0.05)
    ax1  = plt.subplot(grid[0, :2])
    ax1.imshow(ground_img, extent=(0, W, H, 0), zorder=-10)
    ax1.set_title('Ground View', pad=10, fontsize=24)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    
    ax2  = plt.subplot(grid[0, 2])
    # plot aerial view (positive)
    ax2.imshow(aerial_img1, extent=(0, A, A, 0), zorder=-10)
    ax2.set_title('Satellite View (Positive)', pad=16, fontsize=24)
    ax2.set_xlim(0, A)
    ax2.set_ylim(A, 0)

    # plot rays
    colors = ['springgreen', 'deepskyblue', 'orange', 'magenta'] 
    ax1.vlines(x=0.00*W, ymin=0, ymax=H, color=colors[2], linewidth=3, zorder=10)           # South
    ax1.vlines(x=0.25*W, ymin=0, ymax=H, color=colors[3], linewidth=3, zorder=10)           # West
    ax1.vlines(x=0.50*W, ymin=0, ymax=H, color=colors[0], linewidth=3, zorder=10)           # North
    ax1.vlines(x=0.75*W, ymin=0, ymax=H, color=colors[1], linewidth=3, zorder=10)           # East
    ax1.vlines(x=1.00*W, ymin=0, ymax=H, color=colors[2], linewidth=3, zorder=10)           # South
    ax1.axis('off')

    xc1, yc1 = A/2-shift_u[idx].data.cpu(), A/2+shift_v[idx].data.cpu()                                                # See GitHub of VIGOR for this formula
    ax2.scatter(xc1, yc1, s=150, color="yellow", zorder=20)                                 # Center
    ax2.vlines(x=xc1, ymin=0, ymax=yc1, color=colors[0], linewidth=3, zorder=10)            # North
    ax2.hlines(y=yc1, xmin=xc1, xmax=A, color=colors[1], linewidth=3, zorder=10)            # East
    ax2.vlines(x=xc1, ymin=yc1, ymax=A, color=colors[2], linewidth=3, zorder=10)            # South
    ax2.hlines(y=yc1, xmin=0, xmax=xc1, color=colors[3], linewidth=3, zorder=10)            # West
    ax2.axis('off')

    plt.show()

    plt.savefig('result_visualize/'+str(idx)+'.jpg', transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()