import cv2
from convert_str2nparray import *

import numpy as np
import matplotlib.pyplot as plt
from fill_depth import fill_depth_colorization
from PIL import Image
cmap = plt.cm.viridis
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
    # fig.colorbar(pred_map, ax=ax, shrink=0.8)
    fig.savefig(save_to_dir)
    # print(save_to_dir)
    plt.clf()

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    # depth_relative = (d_max - depth) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :1] # H, W, C

if __name__ == "__main__":
    rgb_path = '/home/puyiwen/Downloads/taskonomy_rgbs/rgbs/airport/point_31_view_1_domain_rgb.png'
    sparse_depth_path = '/home/puyiwen/taskonomy_depths/depths/airport/point_31_view_1_domain_depth_zbuffer.png'
    sparse_depth = Image.open(sparse_depth_path)
    sparse_depth = np.array(sparse_depth,dtype=np.float32)
    sparse_depth = sparse_depth / (2**16 -1)
    sparse_depth[sparse_depth > 0.9] = 0.0

    sparse_depth = sparse_depth * 128
    save_image_results(sparse_depth,'result_depth_3_org.png')
    print(sparse_depth.max())
    print(sparse_depth.min())
    img = cv2.imread(rgb_path)
    # depth_colored = colored_depthmap(sparse_depth, np.min(sparse_depth), np.max(sparse_depth))[:, :, ::-1]
    # depth_colored = np.array(depth_colored, dtype=np.uint8)

    print("[Info]: On filling depth..")
    depth_filled = fill_depth_colorization(img/255.0, sparse_depth, alpha=1.0)
    print(depth_filled.max())
    print(depth_filled.min())
    print("[Info]: filling depth complete")

    # filled_colored = colored_depthmap(depth_filled, np.min(depth_filled), np.max(depth_filled))[:, :, ::-1]
    # filled_colored = np.array(filled_colored, dtype=np.uint8)
    save_image_results(depth_filled,'result_depth_3.png')
    # cv2.imwrite("result.png", filled_colored)
