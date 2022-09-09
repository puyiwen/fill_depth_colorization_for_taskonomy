from posixpath import basename
import cv2
from convert_str2nparray import *

import numpy as np
import matplotlib.pyplot as plt
from fill_depth import fill_depth_colorization
from PIL import Image
import os
import shutil
# cmap = plt.cm.viridis
np.set_printoptions(threshold=np.inf)

def save_image_results(prediction,save_depth_path): #prediction,img_file,save_depth_path
    print(prediction.shape)
    # vmin = prediction.min()
    vmin = 0.0
    vmax = prediction.max()
    # vmin = 0.1
    # vmax = 10.0
    save_to_dir = save_depth_path
    fig = plt.figure(figsize=(5.12, 5.12),dpi=100)
    # fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pred_map = ax.imshow(prediction, vmin=vmin, vmax=vmax, cmap='viridis')
    fig.colorbar(pred_map, ax=ax, shrink=0.8)
    fig.savefig(save_to_dir)
    # print(save_to_dir)
    plt.clf()

if __name__ == "__main__":
    
    rgb_path = '/home/puyiwen/Downloads/taskonomy_rgbs/rgbs'
    sparse_depth_path = '/home/puyiwen/taskonomy_depths/depths/'
    save_path = '/home/puyiwen/taskonomy_part'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rgb_dirs = os.listdir(rgb_path)
    index = 0
    for i in range(len(rgb_dirs)):
        rgb_dir = rgb_dirs[i]
        full_rgb_dir_path = os.path.join(rgb_path,rgb_dir)
        full_depth_dir_path = os.path.join(sparse_depth_path,rgb_dir)
        img_files = os.listdir(full_rgb_dir_path)
        for j in range(len(img_files)):
            img_file = img_files[j]
            basename_splits = img_file.split('_')
            basename_both = basename_splits[0] + '_' + basename_splits[1] + '_' + basename_splits[2] + '_' + basename_splits[3] + '_' +basename_splits[4] +'_'  #point_397_view_1_domain_
            rgb_file = basename_both + 'rgb.png'
            depth_file = basename_both + 'depth_zbuffer.png'
            full_rgb_file = os.path.join(full_rgb_dir_path,rgb_file)
            full_depth_file = os.path.join(full_depth_dir_path,depth_file)

            sparse_depth = Image.open(full_depth_file)
            sparse_depth = np.array(sparse_depth,dtype=np.float32)
            sparse_depth = sparse_depth / (2**16 -1)
            sparse_depth[sparse_depth > 0.9] = 0.0

            sparse_depth = sparse_depth * 128
            # save_image_results(sparse_depth,'result_depth_4_org.png')
            # print(sparse_depth.max())
            # print(sparse_depth.min())
            img = cv2.imread(full_rgb_file)
            
            # print("[Info]: On filling depth..")
            depth_filled = fill_depth_colorization(img/255.0, sparse_depth, alpha=1.0)
            # print(depth_filled.max())
            # print(depth_filled.min())
            # print("[Info]: filling depth complete")
            part_save_path = os.path.join(save_path,rgb_dir)
            if not os.path.exists(part_save_path):
                os.mkdir(part_save_path)
            
            new_img_name = basename_both + 'rgb.jpg'
            new_img_path = os.path.join(part_save_path,new_img_name)
            cv2.imwrite(new_img_path,img)
            # shutil.copy(full_rgb_file,part_save_path)
            depth_filled = depth_filled / 128.0 * (2**16 -1)
            depth_filled = depth_filled.astype(np.uint16)
            depth_filled = Image.fromarray(depth_filled)
            
            depth_save_path = os.path.join(part_save_path,depth_file)
            depth_filled.save(depth_save_path)
            index += 1
            if index % 100 == 0:
                print('%d done'%index)
    # depth_filled = np.array(depth_filled,dtype=np.float64)
    # depth_filled = depth_filled / (2**16 -1) *128.0

    # print(depth_filled.max())
    # print(depth_filled.min())
    

    
    # save_image_results(depth_filled,'result_depth_4.png')
