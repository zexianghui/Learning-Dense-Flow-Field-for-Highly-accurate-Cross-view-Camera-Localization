import skimage.io as io
import os.path
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from matplotlib.cm import get_cmap


colors = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # 蓝色
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # 浅蓝色
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # 绿色
    (1.0, 0.4980392156862745, 0.054901960784313725),  # 橙色
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # 红色
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # 紫色
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # 棕色
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # 青色
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # 黄色
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # 灰色
    (0.49019607843137253, 0.6196078431372549, 0.7529411764705882),  # 蓝灰色
    (0.9921568627450981, 0.6823529411764706, 0.4196078431372549),  # 浅橙色
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098),  # 浅紫色
    (0.8705882352941177, 0.796078431372549, 0.8941176470588236),  # 浅粉色
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # 粉色
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # 浅紫色
    (0.9490196078431372, 0.6823529411764706, 0.6039215686274509),  # 浅棕色
    (0.996078431372549, 0.7686274509803922, 0.5803921568627451),  # 浅橙色
    (0.9921568627450981, 0.7137254901960784, 0.803921568627451),  # 浅粉色
    (0.9921568627450981, 0.8549019607843137, 0.9254901960784314)  # 浅粉色
]

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
        # init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        # update = ax.scatter(shift_lons[idx, :-1], shift_lats[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lons[idx, -1], shift_lats[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
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
                    transparent=True, dpi=138.6, bbox_inches='tight', pad_inches = 0)
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

#pca方式完成特征图的可视化
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
        # flatten = []

        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape

        # flatten.append(reshape_normalize(grd_feat))

        # flatten = np.concatenate(flatten[:1], axis=0)

        # if level == 0:
        pca_grd = PCA(n_components=3)
        pca_grd.fit(reshape_normalize(grd_feat))

    # for level in range(len(sat_feat_list)):
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]

        B, C, H, W = grd_feat.shape
        grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat))) + 1) / 2).reshape(B, H, W, 3)
        #gt_s2g_new = ((normalize(pca.transform(reshape_normalize(gt_s2g[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

        for idx in range(B):
            if not os.path.exists(os.path.join(save_dir)):
                os.makedirs(os.path.join(save_dir))

            grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
            grd = grd.resize((W,H))
            grd.save(save_dir + ('feat_' + str(loop * B + idx) + '.jpg'))

    return

#feature_map [C,H,W]
def show_feature_map(feature_map, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
        plt.xticks([])    # 去 x 轴刻度
        plt.yticks([])    # 去 y 轴刻度
        plt.savefig(save_path + str(i) + '.jpg',  bbox_inches='tight',pad_inches=0, quality=100, dpi = 138.6)  # 保存图像到本地 512*512
        plt.close()

#和show_feature_map一起完成对特征图的可视化，输入的sat_feat_list、grd_feat_list是一个列表
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


#结果可视化,绿预测，蓝真实，红初始
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

    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.zeros((image_1.shape[0], image_1.shape[1] + image_2.shape[1], 3))   #横向拼接
    # print(pj1.dtype)   #查看数组元素类型


    fig, ax = plt.subplots()
    split[:,:image_1.shape[1],:] = image_1.copy()   #图片jzg在左
    split[:,image_2.shape[1]:,:] = image_2.copy()   #图片lgz在右
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)   #查看拼接情况
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        if flow_conf != None:
            if flow_conf[0,int(match_y[i][0]), int(match_x[i][0])]>0.5:
                line = ax.plot(match_x[i],match_y[i],linewidth =1,color = 'b', alpha = 1)
            else:
                line = ax.plot(match_x[i],match_y[i],linewidth =1,color = 'g', alpha = 1)
        else:
            line = ax.plot(match_x[i],match_y[i],linewidth =0.6,color = 'b', alpha = 1)    #原点：左上角 y竖轴 x横轴
    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    plt.close()

    # image_1 = io.imread(img1)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    # #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    # image_2 = io.imread(img2)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    # #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    # split = np.ones((image_1.shape[0] + image_2.shape[0] +20, image_1.shape[1], 3))*255   #横向拼接
    # # print(pj1.dtype)   #查看数组元素类型


    # fig, ax = plt.subplots()
    # split[:image_2.shape[0], :, :] = image_1.copy()   #图片jzg在左
    # split[image_2.shape[0]+20:, :, :] = image_2.copy()   #图片lgz在右
    # split=np.array(split,dtype=np.uint8)
    # ax.imshow(split)   #查看拼接情况
    # ax.axis('off')

    # for i in range(len(match_x)):
    #     if flow_conf != None:
    #         if flow_conf[0,int(match_y[i][0]), int(match_x[i][0])]>0.5:
    #             line = ax.plot(match_x[i],match_y[i],linewidth =1,color = 'b', alpha = 1)
    #         else:
    #             line = ax.plot(match_x[i],match_y[i],linewidth =1,color = 'g', alpha = 1)
    #     else:
    #         line = ax.plot(match_x[i],match_y[i],linewidth =1,color = 'b', alpha = 1)    #原点：左上角 y竖轴 x横轴
    # plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    # plt.close()

def line_conf_point(img1, img2, match_x, match_y, flow_conf, name):
    #有mask生成点对 默认img1是512*512
    # match_x = []
    # match_y = []
    # #原图像的匹配点
    # for h in range(H):
    #     for w in range(W):
    #         if(mask[0,h,w]):
    #             if(random.randint(1,1700) == 1):
    #                 match_x.append([w,w+512])
    #                 match_y.append([h,h])
    image_1 = io.imread(img1)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    image_2 = io.imread(img2)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    #plt.savefig('result_visualize/sat_ori1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    split = np.zeros((image_1.shape[0], image_1.shape[1] + image_2.shape[1], 3))   #横向拼接
    # print(pj1.dtype)   #查看数组元素类型


    fig, ax = plt.subplots()
    split[:,:image_1.shape[1],:] = image_1.copy()   #图片jzg在左
    split[:,image_2.shape[1]:,:] = image_2.copy()   #图片lgz在右
    split=np.array(split,dtype=np.uint8)
    ax.imshow(split)   #查看拼接情况
    ax.axis('off')
    #x = [[300,300+512],[350,350+512]]
    #y = [[250,250],[230,230]]
    for i in range(len(match_x)):
        if (int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100))>50:
            line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'b', alpha = 1)    #原点：左上角 y竖轴 x横轴
            if flow_conf != None:
                txt = ax.annotate(str(int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100)), xy=((match_x[i][1], match_y[i][1])), \
                                xytext=(2, 2), textcoords='offset points', weight = 'light', color = 'b')
        else:
            line = ax.plot(match_x[i],match_y[i],linewidth =0.4,color = 'y', alpha = 1)    #原点：左上角 y竖轴 x横轴
            if flow_conf != None:
                txt = ax.annotate(str(int(flow_conf[-1][0,0,int(match_y[i][0]), int(match_x[i][0])]*100)), xy=((match_x[i][1], match_y[i][1])), \
                                xytext=(2, 2), textcoords='offset points', weight = 'light', color = 'y')
    #去白边：pad_inches = 0，去坐标轴 ax.axis('off')
    plt.savefig('result_visualize/'+name, transparent=True,dpi=500, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()

#保存原始图像tensor
def save_img(img, save_path):
    img = transforms.functional.to_pil_image(img, mode='RGB')
    img.save(save_path)

#overly('result_visualize/grd_0.png', 'att_visualize/sat_feat_list/level_2/54.png', '3.png')
# def overly(img_path, heat_path, save_path):
#     scale=1
#     img = Image.open(img_path)
#     H = img.size[0]//scale
#     W = img.size[1]//scale
#     heatmap = Image.open(heat_path)

#     img = img.resize((H, W))
#     heatmap = heatmap.resize((H, W))  # 特征图的大小调整为与原始图像相同
#     print(heatmap.size)

#     img = img.convert('RGBA')
#     heatmap = heatmap.convert('RGBA')
#     im = Image.blend(img, heatmap,alpha=0.1)
#     plt.imshow(im)
#     plt.axis("off")
#     plt.xticks([])    # 去 x 轴刻度
#     plt.yticks([])    # 去 y 轴刻度
#     plt.savefig(save_path+'overly.jpg', bbox_inches='tight',pad_inches=0, quality=95, dpi = 138.6)

    
#     #heatmap = np.uint8(255 * heatmap)                           # 将特征图转换为uint8格式
#     #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
#     #heat_img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)     # 将伪彩色图与原始图片融合
#     #heat_img = heatmap * 0.5 + img 　　　　　　        　　　 # 也可以用这种方式融合
#     """
#     img.paste(heatmap, (0,0), heatmap)
#     plt.imshow(img)
#     plt.axis("off")
#     plt.margins(0,0)
#     plt.savefig(save_path, bbox_inches='tight',pad_inches=0,dpi=300, quality=95)
    
#     """    


def overly(img_path, heat_path, save_path):
    heat_img = cv2.imread(heat_path, flags=1)
    org_img =  cv2.imread(img_path, flags=1) 
    # heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    cv2.imwrite(save_path+'heat_img.jpg', heat_img)    #cv2保存热力图片
    add_img = cv2.addWeighted(org_img, 0.7, heat_img, 0.5, 0)
    cv2.imwrite(save_path+'overly.jpg', add_img )

# def heatmap_2_rgb(self, feature_map, output_path):
#     print("heat....")
#     print("feature_map = ", feature_map.shape)

#     feature_map = feature_map.cpu().numpy().squeeze()
#     C, W, H = feature_map.shape
#     # feature_map = feature_map.reshape(W, H, C)
#     # 对每个通道进行处理
#     # 对每个通道进行归一化
#     feature_maps = feature_map
#     _, height, width = feat


def overly_uvth(img_path, flow_conf, save_path, u, v, theta, mask):
    def transPNG(img):
        datas = img.getdata()
        newData = list()
        for item in datas:
            if item[0] == 0 and item[1] == 0 :
                newData.append((255, 255, 255, 0))
            else:
                newData.append((item[0],item[1],item[2],180))
        img.putdata(newData)
        return img

    def to_zhi(array):
        image = Image.new('RGBA', (512, 512))
        pixels = image.load()
        # 遍历数组并更新像素颜色
        for y in range(512):
            for x in range(512):
                value = array[y][x]
                if mask[0,0,y,x]:
                    if value > 0.5:
                        # 大于0.5的值为绿色
                        pixels[x, y] = (0, 0, 255, 180)  # (R, G, B)
                    else:
                        pixels[x, y] = (0, 255, 0, 180)  # (R, G, B)
                else:
                    pixels[x, y] = (0, 0, 0, 0)
        return image
        
    # 打开图像A.jpg和B.jpg
    # image_a = Image.fromarray(flow_conf.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    flow_conf_array = flow_conf.detach().cpu().numpy()

    # 根据jet色彩映射获取颜色映射函数
    # cmap = get_cmap('jet')
    # 根据热力图A生成RGB颜色映射
    # flow_conf_array = cmap(flow_conf_array)
    image_a = to_zhi(flow_conf_array)
    # flow_conf_array = (flow_conf_array * 255).clip(0, 255).astype(np.uint8)  # 将数据缩放到0-255范围并转换为uint8类型
    # image_a = Image.fromarray((flow_conf_array[:, :, :3] * 255).astype(np.uint8))
    # image_a = Image.fromarray(flow_conf_array)
    image_b = Image.open(img_path)

    # 调整图像A的大小，使其比图像B小
    width, height = image_b.size
    scale = 1
    image_a = image_a.resize((int(width/scale), int(height/scale)))

    # Rotate the heatmap image
    angle = theta.cpu().item()  # Rotation angle in degrees
    rotated_heatmap_image = image_a.rotate(angle, resample=Image.BILINEAR, expand=True)

    # Convert the rotated heatmap image to RGBA mode with transparency
    rotated_heatmap_image_rgba = rotated_heatmap_image.convert('RGBA')
    # rotated_heatmap_image_rgba = to_zhi(flow_conf_array)
    rotated_heatmap_image_rgba.save('result.png',  format="PNG", compression="pngquant", quality=100)
    # Create a new blank image of the same size as the background image
    result_image = Image.new('RGBA', image_b.size)

    # # Paste the rotated heatmap image onto the result image at position (u, v) with transparency
    result_image.paste(rotated_heatmap_image_rgba, (int( u.cpu().item()+255-rotated_heatmap_image_rgba.size[0]/2 ), int(-v.cpu().item()+255-rotated_heatmap_image_rgba.size[1]/2)), mask = rotated_heatmap_image_rgba)


    # 将热力图A以jet方式叠加到图片B的对应位置(u, v)
    # Composite the result image with the background image
    final_image = Image.alpha_composite(image_b.convert('RGBA'), result_image)

    # Save the final image
    final_image.save(save_path,  format="PNG", compression="pngquant", quality=100)

def cut_cat(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    # 计算中间位置
    middle = width // 2

    # 拆分图像为两部分
    left_half = image[:, :middle]
    right_half = image[:, middle:]

    # 保存拆分后的图像
    cv2.imwrite('result_visualize/left_half.jpg', left_half)
    cv2.imwrite('result_visualize/right_half.jpg', right_half)

def draw_point(img1, good, bad, name):
    image = io.imread(img1)   # np.ndarray, [h, w, c], 值域(0, 255), RGB
    #plt.savefig('result_visualize/BEV1.jpg', transparent=True, bbox_inches='tight', pad_inches = 0)
    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(len(bad)):
        gt = ax.scatter(bad[i][0].cpu().item(), bad[i][1].cpu().item(), marker='*',facecolors=colors[i], edgecolors='white', s=200, zorder=2)

    for i in range(len(good)):
        gt = ax.scatter(good[i][0].cpu().item(), good[i][1].cpu().item(), marker='o',facecolors=colors[i], edgecolors='white', s=150, zorder=2)
    ax.axis('off')

    plt.savefig(name,transparent=True, dpi=138.6, bbox_inches='tight', pad_inches = 0)
    plt.close()

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
    # com = ax.scatter(ax_com_u, ax_com_v, color='g', marker='*', s=200, zorder=2)
    # com = ax.quiver(ax_com_u, ax_com_v, 1, 1, angles=come_theta, color='g', zorder=2, width  = 0.013, scale = 13)
    # init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
    # gt = ax.quiver(ax_u.cpu().item(), ax_v.cpu().item(), 1, 1, angles=gt_theta[idx].cpu().item(), color='b', zorder=2)
    
    ax.axis('off')

    plt.savefig(os.path.join(save_dir, 'pos.jpg'),
                transparent=True, dpi=A, bbox_inches='tight',pad_inches=0)
    plt.close()