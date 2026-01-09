import numpy as np
import open3d as o3d
import imageio
import os
import matplotlib.pyplot as plt
import Imath
import OpenEXR
import cv2
import torch
from PIL import Image
import json

def get_frustum_opencv(c2w=np.eye(4), colour=(0, 0, 1), scale=0.05):
    '''
    Visualize OpenCV camera frustum (OpenCV/COLMAP)
    '''
    points_cam = np.array([
        [0, 0, 0],          # camera center
        [-0.8, -0.5, 1],    # top left
        [0.8, -0.5, 1],     # top right
        [-0.8, 0.5, 1],     # bottom left
        [0.8, 0.5, 1],      # bottom right
    ])  # (5, 3)

    points_cam = points_cam * scale  # (5, 3)
    points_world = c2w[:3, :3] @ points_cam.T + c2w[:3, 3:4]  # (3, 5)
    points_world = points_world.T  # (5, 3)
    points_world = points_world.astype(np.float32)

    lines = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 4],
    ])
    
    # open3d_lines = o3d.geometry.LineSet()
    # open3d_lines.points = o3d.utility.Vector3dVector(points_world)
    # open3d_lines.lines = o3d.utility.Vector2iVector(lines)
    # open3d_lines.colors = o3d.utility.Vector3dVector([colour for _ in range(len(lines))])

    # convert to triangle mesh
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(points_world)
    open3d_mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 1],
        [1, 2, 4],
        [2, 4, 3],
        [4, 3, 1],
        [3, 1, 2],
    ])

    open3d_mesh.vertex_colors = o3d.utility.Vector3dVector([colour for _ in range(len(points_world))])
    open3d_mesh.compute_vertex_normals()
    return open3d_mesh

def get_frustum_opengl(c2w=np.eye(4), colour=(0, 0, 1), scale=0.05):   
    '''
    Visualize OpenGL camera frustum (OpenGL/Blender)
    '''
    points_cam = np.array([
        [0, 0, 0, 1],          # camera center
        [-0.8, 0.5, -1, 1],    # top left
        [0.8, 0.5, -1, 1],     # top right
        [-0.8, -0.5, -1, 1],     # bottom left
        [0.8, -0.5, -1, 1],      # bottom right
    ])  # (5, 4)

    points_cam = points_cam * scale  # (5, 4)
    frustum_points = np.matmul(c2w, points_cam.T).T  # (5, 4)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (5, 3)  remove homogenous coordinate
    # points_world = c2w[:3, :3] @ points_cam.T + c2w[:3, 3:4]  # (3, 5)
    # points_world = points_world.T  # (5, 3)
    points_world = frustum_points.astype(np.float32)

    # lines = np.array([
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [1, 2],
    #     [1, 3],
    #     [2, 4],
    #     [3, 4],
    # ])
    
    # open3d_lines = o3d.geometry.LineSet()
    # open3d_lines.points = o3d.utility.Vector3dVector(points_world)
    # open3d_lines.lines = o3d.utility.Vector2iVector(lines)
    # open3d_lines.colors = o3d.utility.Vector3dVector([colour for _ in range(len(lines))])

    # convert to triangle mesh
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(points_world)
    open3d_mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 1],
        [1, 2, 4],
        [2, 4, 3],
        [4, 3, 1],
        [3, 1, 2],
    ])

    open3d_mesh.vertex_colors = o3d.utility.Vector3dVector([colour for _ in range(len(points_world))])
    open3d_mesh.compute_vertex_normals()
    return open3d_mesh

def depth_to_pcd(depth_path, rgb, intrinsics, extrinsics, mask=None, clip_end=None):
    
    exr_file = OpenEXR.InputFile(depth_path)

    # Get the header to access image size information
    header = exr_file.header()
    # breakpoint()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the pixel type as 32-bit float for full precision
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the depth channel
    # Note: Depth data might be stored in a channel named 'R', 'Z', or something else.
    # Replace 'R' with the appropriate channel name if needed.
    depth_str = exr_file.channel('R', pt)

    # Convert the binary string to a NumPy array and reshape it to the image dimensions
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = np.reshape(depth, (height, width))

    # # Downsample to 288x512
    # target_height, target_width = 288, 512
    # depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    # mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_AREA) if mask is not None else None

    # Process depth and RGB
    depth_data = depth[..., 0] if depth.ndim == 3 else depth  # Handle EXR channels
    rgb_normalized = rgb.astype(np.float32) / 255.0

    # Generate grid of pixel coordinates
    height, width = depth_data.shape
    u = np.arange(width)  # Reverse u coordinates
    v = np.arange(height)  # Reverse v coordinates for Y down
    u, v = np.meshgrid(u, v)

    # Flatten and filter valid depth points
    if clip_end is None:
        clip_end = 100.0  # Default far clip, adjust as needed
    
    # Turn mask to binary mask
    if mask is not None:
        mask = (mask > 0)
        valid_mask = (depth_data > 0) & (depth_data < clip_end) & mask
    else:
        valid_mask = (depth_data > 0) & (depth_data < clip_end)

    # Create masked depth visualization
    plt.figure(figsize=(10, 8))
    
    # Create a copy of depth data and set invalid areas to black (0)
    masked_depth = depth_data.copy()
    masked_depth[~valid_mask] = 0
    
    # Plot with black for masked areas
    depth_map = plt.imshow(masked_depth, cmap='turbo')
    plt.colorbar(depth_map, label='Depth (units)')
    
    # Add min/max depth values to title
    min_depth = np.min(masked_depth[depth_data > 0])  # Exclude 0 values
    max_depth = np.max(masked_depth)
    plt.title(f'Masked Depth Map (min: {min_depth:.2f}, max: {max_depth:.2f})')
    
    # Save the figure
    plt.savefig(f"masked_depth_colormap.png")
    plt.close()
    # breakpoint()

    u_flat = u[valid_mask].flatten()
    v_flat = v[valid_mask].flatten()
    depth_flat = depth_data[valid_mask].flatten()
    colors = rgb_normalized[valid_mask]

    
    # Camera coordinates calculation
    f_x, f_y = intrinsics[0, 0], intrinsics[1, 1]
    c_x, c_y = intrinsics[0, 2], intrinsics[1, 2]

    z_cam = depth_flat  # Convert depth to camera Z coordinate
    x_cam = (u_flat - c_x) * z_cam / f_x
    y_cam = (v_flat - c_y) * z_cam / f_y

    # Transform to world coordinates
    camera_to_world = extrinsics  # Invert world-to-camera to get camera-to-world
    homogeneous = np.vstack((x_cam, y_cam, z_cam, np.ones_like(z_cam)))
    world_coords = (camera_to_world @ homogeneous).T[:, :3]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coords)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, camera_to_world, mask

def transform_pcd(pcd, transform):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    transformed_points = np.copy(points)
    # Apply rotation (upper 3x3 of transform matrix)
    transformed_points = transformed_points @ transform[:3, :3].T
    # Apply translation
    transformed_points = transformed_points + transform[:3, 3]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    if colors is not None:
        transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
    return transformed_pcd

def from_main_view_get_mirror_view_pose(known_pose_transform, another_view_original_pose):
    # Reflect against plane with normal=(1,0,0) and center=(0,0,0)
    # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
    normal = np.array([1.0, 0.0, 0.0])
    I = np.eye(3)
    R_proj = I - 2 * np.outer(normal, normal)
    t_proj = np.zeros(3) # Since center is at origin, n^T c = 0
    
    # Get initial pose
    initial_pose = known_pose_transform @ another_view_original_pose
    
    # Project the pose
    R_reflected = R_proj @ initial_pose[:3,:3]
    R_reflected[:,0] *= -1 # Fix determinant
    R_reflected[:,1] *= -1 # Fix determinant
    t_reflected = t_proj + R_proj @ initial_pose[:3,3]
    
    another_view_coord_cam = np.eye(4)
    another_view_coord_cam[:3,:3] = R_reflected
    another_view_coord_cam[:3,3] = t_reflected

    # Reflect against plane with normal=(0,1,0) and center=(0,0,0)
    normal = np.array([0.0, 1.0, 0.0])
    I = np.eye(3)
    R_proj = I - 2 * np.outer(normal, normal)
    t_proj = np.zeros(3)  # Since center is at origin, n^T c = 0
    
    # Project the pose
    R_reflected = R_proj @ another_view_coord_cam[:3,:3]
    R_reflected[:,1] *= -1  # Fix determinant
    t_reflected = t_proj + R_proj @ another_view_coord_cam[:3,3]
    
    final_pose = np.eye(4)
    final_pose[:3,:3] = R_reflected
    final_pose[:3,3] = t_reflected
    return another_view_coord_cam

def from_mirror_view_get_main_view_pose(known_pose_transform, another_view_original_pose):
    # Reflect against plane with normal=(1,0,0) and center=(0,0,0)
    # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
    normal = np.array([1.0, 0.0, 0.0])
    I = np.eye(3)
    R_proj = I - 2 * np.outer(normal, normal)
    t_proj = np.zeros(3) # Since center is at origin, n^T c = 0
    
    # Get initial pose
    initial_pose = known_pose_transform @ another_view_original_pose
    
    # Project the pose
    R_reflected = R_proj @ initial_pose[:3,:3]
    R_reflected[:,0] *= -1 # Fix determinant
    R_reflected[:,1] *= -1 # Fix determinant
    t_reflected = t_proj + R_proj @ initial_pose[:3,3]
    
    another_view_coord_cam = np.eye(4)
    another_view_coord_cam[:3,:3] = R_reflected
    another_view_coord_cam[:3,3] = t_reflected

    # # Reflect against plane with normal=(0,1,0) and center=(0,0,0)
    # normal = np.array([0.0, 1.0, 0.0])
    # I = np.eye(3)
    # R_proj = I - 2 * np.outer(normal, normal)
    # t_proj = np.zeros(3)  # Since center is at origin, n^T c = 0
    
    # # Project the pose
    # R_reflected = R_proj @ another_view_coord_cam[:3,:3]
    # R_reflected[:,1] *= -1  # Fix determinant
    # t_reflected = t_proj + R_proj @ another_view_coord_cam[:3,3]
    
    # final_pose = np.eye(4)
    # final_pose[:3,:3] = R_reflected
    # final_pose[:3,3] = t_reflected
    return another_view_coord_cam

if __name__ == "__main__":

    # Create output directory
    # scene = "blue_bathroom"
    for scene in ["terrazzo"]: #sorted(os.listdir('mirror_data_update/raw_data/synthetic_w_mirror')):
        root_dir = f"synthetic_data/rendered_data/{scene}"

        save_dir = f'temp/synthetic_1920x1080_opencv_000_dust3rEdge_w_mask/{scene}'
        os.makedirs(save_dir, exist_ok=True)

        #########################################################
        # Read the original pcd (None of the pose is Identity)
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

        o3d.io.write_point_cloud(f"{save_dir}/point_cloud_main.ply", pcd_main)
        o3d.io.write_point_cloud(f"{save_dir}/point_cloud_mirror.ply", pcd_mirror)

        voxel_size = 0.001
        # remove the duplicated points
        pcd_unified = pcd.voxel_down_sample(voxel_size)
        print(pcd_unified)


        # # ########################################################
        # # Move the dust3r chosen edge view to identity pose
        # # ########################################################
        # with open(f"/home/jing/Code/dust3r/experiments_DAM_masks/abl-synthetic_gt_mask/poseloss_ours_reweightloss1/{scene}/pcd_log/final_edge.json", "r") as f:
        #     final_edge = json.load(f)['final_edge']
        # fix_view_idx = 0 
        # fix_view_pose = cam_main_pose if fix_view_idx == 1 else cam_mirror_pose
        # moving_pose = cam_mirror_pose if fix_view_idx == 1 else cam_main_pose

        # transform = np.linalg.inv(fix_view_pose)
        # transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) # rotate 90 clockwise around x axis
        transform = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) # rotate 90 clockwise around x axis and 180 degrees around y axis
        cam_main_pose_transformed = transform @ cam_main_pose
        cam_mirror_pose_transformed = transform @ cam_mirror_pose
        pcd_unified_transformed = transform_pcd(pcd_unified, transform)

        # #########################################################
        # # Resize the pcd to match the scale of the duster prediction
        # #########################################################
        # # TODO: change the path to the duster prediction pcd match the scene
        # duster_pred_pcd = o3d.io.read_point_cloud(f"/home/jing/Code/dust3r/experiments_DAM_masks/abl-synthetic_gt_mask/poseloss_ours_reweightloss1/{scene}/pcd_log/{scene}300/pcd_300.ply")
        # # Get the bounding box of both point clouds
        # gt_pcd_bbox = pcd_unified.get_axis_aligned_bounding_box()
        # duster_bbox = duster_pred_pcd.get_axis_aligned_bounding_box()

        # # Get scale factor by comparing bounding box sizes
        # gt_pcd_size = gt_pcd_bbox.get_extent()
        # duster_size = duster_bbox.get_extent()
        # scale_factor = np.mean(duster_size / gt_pcd_size)

        # # Scale the mirror point cloud around origin
        # pcd_unified.scale(scale_factor, center=(0, 0, 0))
        # cam_main_pose[:3, 3] *= scale_factor
        # cam_mirror_pose[:3, 3] *= scale_factor

        # # Save and visualize
        # o3d.io.write_point_cloud(f"{save_dir}/point_cloud_gt.ply", pcd_unified)
        # # o3d.io.write_point_cloud(f"point_cloud_main.ply", pcd_main)
        # # o3d.io.write_point_cloud(f"point_cloud_mirror.ply", pcd_mirror)
        # unit_frustum_scale = get_frustum_opencv(np.eye(4), colour=(0, 1, 0), scale=0.5)
        # frustum_main = get_frustum_opencv(cam_main_pose, colour=(0, 0, 1), scale=0.5)
        # frustum_mirror = get_frustum_opencv(cam_mirror_pose, colour=(0, 0, 1), scale=0.5)
        # os.makedirs(f"{save_dir}/processed_poses", exist_ok=True)
        # o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_main.ply", frustum_main)
        # o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_mirror.ply", frustum_mirror)
        # # o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_unit.ply", unit_frustum_scale)
        # np.save(f"{save_dir}/processed_poses/cam_main.npy", cam_main_pose)
        # np.save(f"{save_dir}/processed_poses/cam_mirror.npy", cam_mirror_pose)


        # Save and visualize
        o3d.io.write_point_cloud(f"{save_dir}/point_cloud_gt.ply", pcd_unified_transformed)
        # o3d.io.write_point_cloud(f"point_cloud_main.ply", pcd_main)
        # o3d.io.write_point_cloud(f"point_cloud_mirror.ply", pcd_mirror)
        unit_frustum_scale = get_frustum_opencv(np.eye(4), colour=(0, 1, 0), scale=0.05)
        frustum_main = get_frustum_opencv(cam_main_pose_transformed, colour=(0, 0, 1), scale=0.05)
        frustum_mirror = get_frustum_opencv(cam_mirror_pose_transformed, colour=(0, 0, 1), scale=0.05)
        os.makedirs(f"{save_dir}/processed_poses", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_main.ply", frustum_main)
        o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_mirror.ply", frustum_mirror)
        # o3d.io.write_triangle_mesh(f"{save_dir}/processed_poses/frustum_unit.ply", unit_frustum_scale)
        np.save(f"{save_dir}/processed_poses/cam_main.npy", cam_main_pose_transformed)
        np.save(f"{save_dir}/processed_poses/cam_mirror.npy", cam_mirror_pose_transformed)
    
    
        
