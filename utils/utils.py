import numpy as np
import open3d as o3d
import OpenEXR
import Imath

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
