import os
from dataclasses import dataclass
import torch
import open3d as o3d
import torch.nn.functional as F
import argparse
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import _resize_pil_image
from dust3r.viz import to_numpy
import numpy as np
from PIL.ImageOps import exif_transpose
import PIL, json
import torchvision.transforms as tvf
import tqdm
from sklearn.linear_model import RANSACRegressor
try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

from utils.utils import apply_mask_to_image
from utils.mirror import inference_one_image_unlabeled, define_dinov2
from utils.utils import center_crop_pil_image, find_mirror_edge_point, post_process
from reflect3r.optim import minimum_spanning_tree, init_from_pts3d_poseloss, global_alignment_iter_poseloss

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_images(size, square_ok=False, verbose=True, force_size=False, read_dir='results/intermediate'):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    folder_or_list = [f'{read_dir}/flipped_image_inside_masked.png', f'{read_dir}/image_outside_masked.png']
    mask_or_list = [f'{read_dir}/flipped_inside_mask.png', f'{read_dir}/outside_mask.png']
    use_mask = mask_or_list is not None
    
    if isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
            if use_mask:
                print(f'>> Loading a list of {len(mask_or_list)} masks')
        if use_mask:
            root, folder_content, mask_content = '', folder_or_list, mask_or_list 
        else:
            root, folder_content = '', folder_or_list
    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    if use_mask:
        for img_path, mask_path in zip(folder_content, mask_content):
            if not img_path.lower().endswith(supported_images_extensions):
                continue
            img = exif_transpose(PIL.Image.open(os.path.join(root, img_path)))
            img = img.convert('RGB')
            mask = exif_transpose(PIL.Image.open(os.path.join(root, mask_path))).convert('L')
            if force_size:
                pass

            img_ori = img #if force_size else center_crop_pil_image(img)
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
                mask = _resize_pil_image(mask, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
                mask = _resize_pil_image(mask, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
                mask = mask.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
                mask = mask.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
            W2, H2 = img.size
            mask = np.array(mask) 
            mask = mask > 128

            if verbose:
                print(f' - adding {img_path} with resolution {W1}x{H1} --> {W2}x{H2}')
                print(f' - adding {mask_path} with resolution {W1}x{H1} --> {W2}x{H2}')
            imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
                [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)), mask=mask[None][None], img_ori=ImgNorm(img_ori)[None]))
    else: 
        for img_path in folder_content:
            if not img_path.lower().endswith(supported_images_extensions):
                continue
            img = exif_transpose(PIL.Image.open(os.path.join(root, img_path))).convert('RGB')
            if force_size:
                img = center_crop_pil_image(img, target_width=512, target_height=320)
            img_ori = img #if force_size else center_crop_pil_image(img)
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
            W2, H2 = img.size
            if verbose:
                print(f' - adding {img_path} with resolution {W1}x{H1} --> {W2}x{H2}')
            imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
                [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)), img_ori=ImgNorm(img_ori)[None]))

    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs

@dataclass
class Config:
    device = 'cuda:0'
    batch_size = 1
    schedule = 'linear'
    niter = 300
    lr = 0.01
    min_conf_thr = 1.5
    use_mirror_masks = True
    silent = False
    image_size = 512
    sigma = 1
    transparent_cams = False
    use_mirror_masks = True


if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument('--input_image_path', type=str, default='examples/example2.png')
    args = args.parse_args()
    
    config = Config()
    reflect3r = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt').to(config.device)


    intermediate_save_dir = f'results/intermediate'
    os.makedirs(intermediate_save_dir, exist_ok=True)
    outdir = f'results'

    ###################################################
    ##          Detect Mirror
    ###################################################
    image_path = args.input_image_path

    inference_one_image_unlabeled(image_path=image_path, model=define_dinov2().eval().to(config.device), device=config.device, save_path=f'{intermediate_save_dir}/detected_mirror.png')

    mirror_part, outside_part, mask_mirror_part, mask_outside_part  = apply_mask_to_image(img_path=image_path, mask_outside_path=f"{intermediate_save_dir}/detected_mirror.png", root_dir=intermediate_save_dir) 

    images = load_images(size=config.image_size, verbose=not config.silent,force_size = False)


    ###################################################
    ##          Run Reflect3r 
    ###################################################

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, reflect3r, config.device, batch_size=config.batch_size)


    view1, view2, pred1, pred2 = [output[k] for k in 'view1 view2 pred1 pred2'.split()]
    # build the optimizer
    scene = PointCloudOptimizer(view1, view2, pred1, pred2, verbose=not config.silent, min_conf_thr=config.min_conf_thr, allow_pw_adaptors=True).to(config.device)

    pts3d, final_edge, im_focals, im_poses = minimum_spanning_tree(
                                                            intermediate_save_dir, scene.view_i, scene.imshapes, scene.edges, scene.pred_i, scene.pred_j, scene.conf_i, scene.conf_j, scene.im_conf, scene.min_conf_thr, config.device, has_im_poses=scene.has_im_poses, verbose=scene.verbose)
    
    scene.final_edge = final_edge
    init_from_pts3d_poseloss(scene, config.sigma, pts3d, im_focals, im_poses, final_edge[0])

    scene.im_poses.requires_grad_(True)
    scene.pw_adaptors.requires_grad_(False)

    # Start the post-optimization
    params = [p for p in scene.parameters() if p.requires_grad]
    print('Global alignement - optimizing for:')
    print([name for name, value in scene.named_parameters() if value.requires_grad])

    lr_base = config.lr
    lr_min = 1e-6
    niter=300
    schedule='cosine'
    optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.9))

    loss = float('inf')
    loss_history = []
    print('min_conf_thr:', config.min_conf_thr)
    print('focals:', scene.im_focals)
    depth = scene.im_depthmaps.clone().detach().cpu().numpy()
    with tqdm.tqdm(total=niter) as bar:
        while bar.n < bar.total:
            loss, lr, scene, loss_pose = global_alignment_iter_poseloss(scene, config.sigma, bar.n, niter, lr_base, lr_min, optimizer, schedule, final_edge[0], intermediate_save_dir)
            bar.set_postfix_str(f'{lr=:g} loss={loss:g}, loss_pose={loss_pose:g}')
            bar.update()
            loss_history.append(loss)
                
    post_process(scene, intermediate_save_dir, bar.n, use_mirror_masks=config.use_mirror_masks, transparent_cams=config.transparent_cams, silent=config.silent)
    
    ###################################################
    ##          Recover Mirror Plane
    ###################################################

    c2ws = torch.cat([cam.detach() for cam in scene.get_im_poses()], dim=0)
    principal_points = scene.get_principal_points().detach()
    focals = scene.get_focal_same().detach().repeat(scene.n_imgs, 1)
    msk = to_numpy(scene.get_masks())
    shape = images[0]['true_shape']
    H, W = int(shape[0][0]), int(shape[0][1])
    pcds = [i.detach() for i in scene.get_pts3d()]
    pts3d, colors = [], []
    for pts, img, conf_mask, mirror_mask in zip(pcds, scene.imgs, msk, scene.mirror_masks):
        if config.use_mirror_masks:
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
    # depth = [i.detach() for i in scene.get_depthmaps()]

    confs = to_numpy([c for c in scene.im_conf])

    masks = None
    mask_pc = False

    imgs = np.array(scene.imgs)

    
    # Obtain the mirror plane in the prediction
    main_view_idx, mirror_view_idx = 1, 0
    proj_pts3d_main_view = pcds[main_view_idx].clone()
    local_main_view_idx = scene.view_i['idx'].index(main_view_idx)
    main_view_mask = scene.view_i['mask'][local_main_view_idx][0] # Now it is the outside mask
    main_view_mask = scene.main_view_mask
    mirror_pts3d = proj_pts3d_main_view[~main_view_mask] 
    center = torch.mean(mirror_pts3d, dim=0)

    # Run PCA to get normal
    mirror_pts3d_centered = mirror_pts3d - center
    cov = torch.matmul(mirror_pts3d_centered.T, mirror_pts3d_centered) / (mirror_pts3d_centered.shape[0] - 1)
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)  # SVD is differentiable
    normal = Vh[-1]
    # Ensure normal points "up" (positive y)
    # Using abs() and sign() is non-differentiable, so use a differentiable version:
    normal = normal * (normal[1] / torch.sqrt(normal[1]*normal[1]))

    # Save normal vector as json
    normal_dict = {
        'normal': [float(normal[0].detach().cpu()), float(normal[1].detach().cpu()), float(normal[2].detach().cpu())],
    }
    
    with open(os.path.join(intermediate_save_dir, 'mirror_normal.json'), 'w') as f:
        json.dump(normal_dict, f, indent=4)

    
    # Find a point on the mirror plane
    anchor_point, anchor_conf = find_mirror_edge_point(to_numpy(scene.main_view_mask.clone()), confs[main_view_idx])
    # Get all 3D points for edge points
    anchor_point_3ds = proj_pts3d_main_view[anchor_point[:,1], anchor_point[:,0]]  # Swap indices to match (row, col) format
    
    # Find inliers using RANSAC
    # Convert to numpy for RANSAC
    anchor_points_np = anchor_point_3ds.detach().cpu().numpy()
    
    # Fit a line using RANSAC
    ransac = RANSACRegressor(random_state=42)
    X = anchor_points_np[:,[0,1]] # Use x,y coordinates
    y = anchor_points_np[:,2] # Fit to z coordinate
    ransac.fit(X, y)
    
    # Get inlier mask
    inlier_mask = ransac.inlier_mask_
    
    # Keep only inlier points
    anchor_point_3ds = anchor_point_3ds[torch.from_numpy(inlier_mask).to(anchor_point_3ds.device)]
    anchor_conf_inlier = torch.from_numpy(anchor_conf[inlier_mask]).to(anchor_point_3ds.device)

    # Get indices of top 10% confidence points
    num_points = len(anchor_conf_inlier)
    k = int(num_points * 0.1)  # Calculate how many points is 10%
    top_k_indices = torch.argsort(anchor_conf_inlier, descending=True)[:k]
    
    # Get the high confidence points and their confidences
    anchor_point_3ds_high_conf = anchor_point_3ds[top_k_indices]
    anchor_conf_high = anchor_conf_inlier[top_k_indices]


    # Save all anchor point 3D positions
    anchor_points_pcd = o3d.geometry.PointCloud()
    anchor_points_pcd.points = o3d.utility.Vector3dVector(anchor_point_3ds_high_conf.detach().cpu().numpy())
    anchor_points_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(anchor_point_3ds_high_conf), 1)))  # Red points

    anchor_point_3d = anchor_point_3ds_high_conf[torch.argsort(anchor_point_3ds_high_conf[:, 2])[len(anchor_point_3ds_high_conf)//2]]


    # Project ray from camera through anchor point to intersect with plane
    K = torch.eye(3).to(config.device)
    K[0,0] = K[1,1] = focals[0]
    K[0,2] = principal_points[0][0]
    K[1,2] = principal_points[0][1]
    cam_main_view = c2ws[1]
    c2w = cam_main_view
    
    # Get all points in masked area
    y_coords, x_coords = torch.where(~scene.main_view_mask)

    # Get ray directions for all points in camera space
    ray_dirs_cam = torch.stack([
        (x_coords.to(config.device) - K[0,2]) / K[0,0],
        (y_coords.to(config.device) - K[1,2]) / K[1,1],
        torch.ones_like(x_coords.to(config.device))
    ], dim=-1).float()
    
    # Normalize ray directions
    ray_dirs_cam = F.normalize(ray_dirs_cam, dim=-1)
    
    # Transform rays to world space
    ray_dirs_world = (c2w[:3,:3] @ ray_dirs_cam.unsqueeze(-1)).squeeze(-1)
    ray_origin = c2w[:3,3].to(config.device)
    
    # Compute intersection with plane for all rays
    # Plane equation: dot(p - anchor_point_3d, normal) = 0
    # Ray equation: p = ray_origin + t * ray_dir
    # Solve for t: dot(ray_origin + t*ray_dir - anchor_point_3d, normal) = 0
    denoms = torch.einsum('nd,d->n', ray_dirs_world, normal)
    ts = torch.einsum('d,d->', anchor_point_3d - ray_origin, normal) / denoms
    
    # Get 3D points of intersection
    points_3d = ray_origin.unsqueeze(0) + ts.unsqueeze(-1) * ray_dirs_world

    # Create point cloud for intersection points
    intersection_pcd = o3d.geometry.PointCloud()
    intersection_pcd.points = o3d.utility.Vector3dVector(points_3d.detach().cpu().numpy())
    intersection_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (len(points_3d), 1)))  # Black points

    # Create point cloud for original points
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(pcd.detach().cpu().numpy())
    original_pcd.colors = o3d.utility.Vector3dVector(col.detach().cpu().numpy())  # Use original colors

    # Combine point clouds
    combined_pcd = intersection_pcd + original_pcd
    o3d.io.write_point_cloud(f"{outdir}/reconstructed_point_cloud.ply", combined_pcd)


