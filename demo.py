import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = './logs/log_rs/checkpoint-rs.tar'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def get_and_process_data(data_dir):
    num_point = 20000

    imgs = np.load(os.path.join(data_dir, 'imgs.npz'))
    color = imgs['img'] / 255.0
    depth = imgs['depth']

    height = 256
    width = 256
    fovy = np.pi / 4
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01

    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    print(gg)
    o3d.visualization.draw_geometries([cloud, *grippers])


def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)


if __name__ == '__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
