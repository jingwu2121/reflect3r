import os, cv2
import torch
import sys
from torchvision import transforms
import open3d as o3d
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import glob
# sys.path.append('./extern/dust3r')
from dust3r.utils.image import rgb
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt.optimizer import PointCloudOptimizer
from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores
from dust3r.cloud_opt.base_opt import cosine_schedule, linear_schedule, adjust_learning_rate_by_lr
from dust3r.cloud_opt.init_im_poses import dict_to_sparse_graph, rigid_points_registration, sRT_to_4x4, estimate_focal, fast_pnp
from dust3r.utils.image import _resize_pil_image
from dust3r.utils.geometry import geotrf, inv
from dust3r.viz import to_numpy, add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from PIL import Image, ImageOps
import numpy as np
import trimesh
from PIL.ImageOps import exif_transpose
import PIL, json
from scipy.spatial.transform import Rotation
import torchvision.transforms as tvf
import scipy.sparse as sp
import tqdm
import torch
from sklearn.linear_model import RANSACRegressor
try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

def init_from_pts3d_poseloss(self, sigma, pts3d, im_focals, im_poses, final_edge):
    # set all pairwise poses
    for e, (i, j) in enumerate(self.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        (fix_view_idx, optim_view_idx) = final_edge
        main_view_idx, mirror_view_idx = 1, 0
        # Initialize the main view
        cam2world_fix = im_poses[fix_view_idx]
        depth = geotrf(inv(cam2world_fix), pts3d[fix_view_idx])[..., 2]
        self._set_depthmap(fix_view_idx, depth)

        if im_focals[fix_view_idx] is not None:
            self._set_focal_same(im_focals[fix_view_idx])

        # Get the mirror plane
        proj_pts3d_main_view = pts3d[main_view_idx].clone()
        local_main_view_idx = self.view_i['idx'].index(main_view_idx)
        self.main_view_mask = self.view_i['mask'][local_main_view_idx][0] # Now it is the outside mask
        main_view_mask = self.main_view_mask
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

        # reflection matrix
        I = torch.eye(3, device=normal.device)
        flip_matrix = torch.tensor([[-1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], device=normal.device).to(torch.float)
        
        R = I - 2 * torch.outer(normal, normal) 
        t = 2 * torch.dot(normal, center) * normal
        refl_matrix = torch.cat([R, t.unsqueeze(1)], dim=1)
        refl_matrix_4x4 = torch.cat([refl_matrix, torch.tensor([0, 0, 0, 1], device=normal.device).unsqueeze(0)], dim=0)

        cam2world_optim = refl_matrix_4x4 @ cam2world_fix
        cam2world_optim = cam2world_optim @ flip_matrix # camera to world

        depth = geotrf(inv(cam2world_optim), pts3d[optim_view_idx])[..., 2]

        self._set_depthmap(optim_view_idx, depth)
        self._set_im_pose(self.im_poses, optim_view_idx, cam2world_optim)

    if self.verbose:
        print(' init loss =', float(self(sigma)[0]))


def global_alignment_iter_poseloss(net, sigma, cur_iter, niter, lr_base, lr_min, optimizer, schedule, final_edge=None, intermediate_save_dir=None):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()
    loss, loss_pose = net(sigma, cur_iter, intermediate_save_dir)
    loss.backward()
    optimizer.step()

    return float(loss), lr, net, float(loss_pose)

def minimum_spanning_tree(outdir, view_i, imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr, device, has_im_poses=True, niter_PnP=10, verbose=True):
    n_imgs = len(imshapes)

    # Remove the black area in the mirror view (It is fine to keep the black area in the main view, since we will compute the mirror plane with it)
    mirror_view_idx = 0 
    local_mirror_view_idx = view_i['idx'].index(mirror_view_idx)
    mirror_view_mask = (view_i['mask'][local_mirror_view_idx][0] * 1).to(device)
    conf_i['0_1'] = conf_i['0_1'] * mirror_view_mask
    conf_j['1_0'] = conf_j['1_0'] * mirror_view_mask
    conf_i['0_1'].requires_grad = False
    conf_j['1_0'].requires_grad = False
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f' init edge ({i}*,{j}*) {score=}')
    i_j = edge_str(i, j)
    pts3d[i] = pred_i[i_j].clone()
    pts3d[j] = pred_j[i_j].clone()

    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]
    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_i[i_j])

        if i in done:
            if verbose:
                print(f' init edge ({i},{j}*) {score=}')
            assert j not in done
            # align pred[i] with pts3d[i], and then set j accordingly
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[j] = geotrf(trf, pred_j[i_j])
            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:
            if verbose:
                print(f' init edge ({i}*,{j}) {score=}')
            assert i not in done
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(sparse_graph.values())  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])
        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None
    
    return pts3d, msp_edges, im_focals, im_poses