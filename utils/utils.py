import numpy as np
import open3d as o3d
import OpenEXR
import Imath
from PIL import Image
from PIL import ImageOps
import trimesh
from scipy.spatial.transform import Rotation
import os
from dust3r.viz import to_numpy, add_scene_cam, CAM_COLORS, OPENGL
import cv2
import json
import torch
import imageio.v2 as imageio

import matplotlib.pyplot as plt
from dust3r.utils.image import rgb

def post_process(scene, outdir, iter, use_mirror_masks=False, preset_poses=None, transparent_cams=False, silent=False):
    os.makedirs(outdir, exist_ok=True)
    focals = scene.get_focal_same().detach().repeat(scene.n_imgs, 1)
    principal_points = scene.get_principal_points().detach()
    shape = scene.view_i['true_shape']
    H, W = int(shape[0][0]), int(shape[0][1])
    focal_dict = {
        'focals': focals.cpu().numpy().tolist(),
        'principal_points': principal_points.cpu().numpy().tolist(),
        'H': H,
        'W': W,
    }
    with open(os.path.join(outdir, f'intr_{iter}.json'), 'w') as f:
        json.dump(focal_dict, f)

    depth = [i.detach() for i in scene.get_depthmaps()]
    # save depth
    np.save(os.path.join(outdir, f'depth_0_{iter}.npy'), depth[0].cpu().numpy())
    np.save(os.path.join(outdir, f'depth_1_{iter}.npy'), depth[1].cpu().numpy())

    confs_save = to_numpy([c for c in scene.im_conf])
    np.save(os.path.join(outdir, f'confs_0_{iter}.npy'), confs_save[0])
    np.save(os.path.join(outdir, f'confs_1_{iter}.npy'), confs_save[1])
    
    c2ws = [cam.detach() for cam in scene.get_im_poses()]
    msk = to_numpy(scene.get_masks())

    # save c2ws
    np.save(os.path.join(outdir, f'c2ws_0_{iter}.npy'), c2ws[0][0].cpu().numpy())
    np.save(os.path.join(outdir, f'c2ws_1_{iter}.npy'), c2ws[1][0].cpu().numpy())


    frustums0 = get_frustum_opencv(c2ws[0][0].cpu().numpy(), colour=(0, 0, 1), scale=0.05)
    frustums1 = get_frustum_opencv(c2ws[1][0].cpu().numpy(), colour=(0, 0, 1), scale=0.05)
    o3d.io.write_triangle_mesh(os.path.join(outdir, f'frustums0_{iter}.ply'), frustums0)
    o3d.io.write_triangle_mesh(os.path.join(outdir, f'frustums1_{iter}.ply'), frustums1)

    # Turn use_mirror_masks to False to fall back to the original duster
    pcds = [i.detach() for i in scene.get_pts3d()]
    pts3d, colors = [], []
    for pts, img, conf_mask, mirror_mask in zip(pcds, scene.imgs, msk, scene.mirror_masks):
        if use_mirror_masks:
            mirror_mask = mirror_mask[0].numpy()
            mask = conf_mask * mirror_mask # conf_mask: binary mask
            if mask.sum() > 0:
                pts_masked = pts.cpu().numpy()[mask]
                img_masked = img[mask]
            else:
                pts_masked = pts.cpu().numpy()[mirror_mask]
                img_masked = img[mirror_mask]
        else:
            mask = conf_mask
            pts_masked = pts.cpu().numpy()
            img_masked = img
        pts3d.append(torch.from_numpy(pts_masked).to(device=scene.device))
        colors.append(torch.from_numpy(img_masked).to(device=scene.device))

    pcd = torch.cat([p.reshape(-1, 3) for p in pts3d], dim=0)
    col = torch.cat([c.reshape(-1, 3) for c in colors], dim=0)
    col = to_numpy(col)
    pts = to_numpy(pcd)

    # save pcd with normals
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    pct = trimesh.PointCloud(pts, colors=col)
    print(pct.vertices.shape)
    pct.vertices_normal = normals

    save_pointcloud_with_normals(pct, msk=None, save_path=os.path.join(f'{outdir}/pcd_{iter}.ply') , mask_pc=False, reduce_pc=False)

    cam_size = 0.05
    to_glb(outdir, scene.imgs, col, pts, focals, c2ws, cam_size, as_pointcloud=True, transparent_cams=transparent_cams, silent=silent)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = plt.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]
    # Save RGB, depth and confidence images
    for i in range(len(rgbimg)):
        # Convert to uint8 format (0-255)
        rgb_img = (rgbimg[i] * 255).astype(np.uint8)
        # breakpoint()
        depth_img = (depths[i] * 255).astype(np.uint8) 
        conf_img = (confs[i] * 255).astype(np.uint8)
        
        # Save images
        imageio.imwrite(os.path.join(outdir, f'rgb_{i:03d}.png'), rgb_img)
        imageio.imwrite(os.path.join(outdir, f'depth_{i:03d}.png'), depth_img)
        imageio.imwrite(os.path.join(outdir, f'conf_{i:03d}.png'), conf_img)

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

def find_mirror_edge_point(msk, conf):
    '''
    Find the edge point of the mirror plane
    '''
    msk = msk.astype(np.uint8)* 255
    kernel = np.ones((3,3), np.uint8)
    msk = cv2.erode(msk, kernel, iterations=2)
    # Use Canny edge detection
    edges = cv2.Canny(msk, 100, 200)

    edge_points = np.where(edges > 0)
    edge_points = np.array(list(zip(edge_points[1], edge_points[0])))  # Convert to (x,y) format
    # Get confidence values for all edge points
    edge_confidences = conf[edge_points[:,1], edge_points[:,0]]

    return edge_points, edge_confidences

def save_pointcloud_with_normals(pct, msk, save_path, mask_pc, reduce_pc):

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pct.vertices
    colors = pct.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))

def to_glb(outdir, imgs, col, pts, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(cams2world) == len(focals)
    pts = to_numpy(pts)
    col = to_numpy(col)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    imgs = to_numpy(imgs)
    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pct = trimesh.PointCloud(pts, colors=col)
        scene.add_geometry(pct)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w[0], camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def center_crop_pil_image(input_image, target_width=1024, target_height=576):
    w, h = input_image.size
    h_ratio = h / target_height
    w_ratio = w / target_width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < target_height:
            h = target_height
        input_image = input_image.resize((target_width, h), Image.ANTIALIAS)
    else:
        w = int(w / h_ratio)
        if w < target_width:
            w = target_width
        input_image = input_image.resize((w, target_height), Image.ANTIALIAS)

    return ImageOps.fit(input_image, (target_width, target_height), Image.BICUBIC)

def apply_mask_to_image(img_path, mask_outside_path, root_dir='results'):
    '''Apply mask to image'''
    img = Image.open(img_path).convert('RGB')
    mask_outside = Image.open(mask_outside_path)

    # Convert images to numpy arrays
    img_inside, img_outside = np.array(img), np.array(img)
    mask_outside_array = np.array(mask_outside)

    # Create boolean masks (True where mask is black)
    mask_flipped_inside_bool = mask_outside_array < 128
    mask_outside_bool = mask_outside_array > 128

    mask_flipped_inside_save = mask_outside_array > 128
    mask_outside_save = mask_outside_array < 128
    flipped_inside_mask = Image.fromarray(mask_flipped_inside_save).transpose(Image.FLIP_LEFT_RIGHT)
    flipped_inside_mask.save(f"{root_dir}/flipped_inside_mask.png")
    outside_mask = Image.fromarray(mask_outside_save) 
    outside_mask.save(f"{root_dir}/outside_mask.png")
    print(f"Masks saved as '{root_dir}/flipped_inside_mask.png' and '{root_dir}/outside_mask.png'")

    # Apply mask by setting pixels to black (0) where mask is True
    img_inside[mask_flipped_inside_bool] = 0
    img_outside[mask_outside_bool] = 0

    # Convert back to PIL Image
    img_inside = Image.fromarray(img_inside)
    img_outside = Image.fromarray(img_outside)
    
    flipped_img = img_inside.transpose(Image.FLIP_LEFT_RIGHT)
    # Convert boolean mask to uint8 (0 or 255) for alpha channel
    alpha_mask = np.fliplr(mask_outside_array > 128).astype(np.uint8) * 255
    flipped_img.putalpha(Image.fromarray(alpha_mask))
    save_path = f"{root_dir}/flipped_image_inside_masked.png"
    flipped_img.save(save_path) 

    alpha_mask = np.array(mask_outside_array < 128).astype(np.uint8) * 255
    img_outside.putalpha(Image.fromarray(alpha_mask))
    save_path = f"{root_dir}/image_outside_masked.png"
    img_outside.save(save_path)
    print(f"Mask has been applied and saved as '{save_path}'")

    return flipped_img, img_outside, flipped_inside_mask, outside_mask

def depth_to_pcd(depth_path, rgb, intrinsics, extrinsics, mask=None, clip_end=None):
    
    exr_file = OpenEXR.InputFile(depth_path)

    # Get the header to access image size information
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Define the pixel type as 32-bit float for full precision
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read the depth channel
    # Note: Depth data might be stored in a channel named 'R', 'Z', or something else.
    depth_str = exr_file.channel('R', pt)

    # Convert the binary string to a NumPy array and reshape it to the image dimensions
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = np.reshape(depth, (height, width))

    # Process depth and RGB
    depth_data = depth[..., 0] if depth.ndim == 3 else depth  
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
    
    # Create a copy of depth data and set invalid areas to black (0)
    masked_depth = depth_data.copy()
    masked_depth[~valid_mask] = 0

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
