import numpy as np
import open3d as o3d
import imageio
import os
import argparse
from utils.utils import get_frustum_opencv
from utils.utils import depth_to_pcd


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--scene_name', type=str, default='archiviz')
    args.add_argument('--save_root', type=str, default='eval_data')
    args = args.parse_args()

    root_dir = f"synthetic_data/rendered_data/{args.scene_name}"

    save_dir = f'{args.save_root}/{args.scene_name}'
    os.makedirs(save_dir, exist_ok=True)

    #########################################################
    # Read the original pcd
    #########################################################
    # Generate a group of point clouds
    output_dir = f"{root_dir}/Cam_Main"  # Update this path
    depth_path = f"{output_dir}/depth_exr_0001.exr"
    rgb = imageio.imread(f"{output_dir}/rgb_0001.png")
    intrinsics_main = np.load(f"{output_dir}/intrinsics.npy")
    cam_main_pose = np.load(f"{output_dir}/extrinsics.npy")
    cam_main_pose[0:3, 1:3] *= -1 # convert to opencv
    clip_end = np.load(f"{output_dir}/clip_params.npy")[1] if os.path.exists(f"{output_dir}/clip_params.npy") else 100.0
    mask_outside = imageio.imread(f"{root_dir}/masks/outside_mask.png")
    pcd_main, cam_main_pose, resized_mask_main = depth_to_pcd(depth_path, rgb, intrinsics_main, cam_main_pose, mask=None, clip_end=clip_end)

    output_dir = f"{root_dir}/Cam_Mirror"  # Update this path
    depth_path = f"{output_dir}/depth_exr_0001.exr"
    rgb = imageio.imread(f"{root_dir}/imgs/flipped_image_inside_masked.png")
    intrinsics_mirror = np.load(f"{output_dir}/intrinsics.npy")
    cam_mirror_pose = np.load(f"{output_dir}/extrinsics.npy")
    cam_mirror_pose[0:3, 1:3] *= -1 # convert to opencv
    # cam_mirror_pose[0:3, 1] *= -1
    clip_end = np.load(f"{output_dir}/clip_params.npy")[1] if os.path.exists(f"{output_dir}/clip_params.npy") else 100.0
    mask_inside = imageio.imread(f"{root_dir}/masks/flipped_inside_mask.png")
    pcd_mirror, cam_mirror_pose, resized_mask_mirror = depth_to_pcd(depth_path, rgb, intrinsics_mirror, cam_mirror_pose, mask_inside, clip_end)

    pcd = pcd_main + pcd_mirror

    voxel_size = 0.001
    # remove the duplicated points
    pcd_unified = pcd.voxel_down_sample(voxel_size)

    #########################################################
    # Save and visualize
    #########################################################
    o3d.io.write_point_cloud(f"{save_dir}/point_cloud_gt.ply", pcd_unified)
    frustum_main = get_frustum_opencv(cam_main_pose, colour=(0, 0, 1), scale=0.5)
    frustum_mirror = get_frustum_opencv(cam_mirror_pose, colour=(0, 0, 1), scale=0.5)
    os.makedirs(f"{save_dir}/processed_poses", exist_ok=True)
    o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_main.ply", frustum_main)
    o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_mirror.ply", frustum_mirror)
    np.save(f"{save_dir}/processed_poses/cam_main.npy", cam_main_pose)
    np.save(f"{save_dir}/processed_poses/cam_mirror.npy", cam_mirror_pose)


