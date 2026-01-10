# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import trimesh
import open3d as o3d
from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf
from dust3r.utils.device import to_cpu, to_numpy
# from temp import get_frustum_opencv
import roma
from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, get_conf_trf, _fast_depthmap_to_pts3d)

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, preset_poses=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.Parameter(self.rand_pose(self.POSE_DIM))[None]  # camera poses

        self.im_focals = nn.Parameter(torch.FloatTensor(
            [self.focal_break*np.log(max(self.imshapes[0][0], self.imshapes[0][1]))]))  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute aa
        if 'mask' in self.view_i.keys():
            self.mirror_masks_vmirror_vmain = [self.view_i['mask'][1][0], self.view_j['mask'][1][0]] # 288x512
            self.mirror_masks_vmirror_vmain_flat = [self.view_i['mask'][1][0].flatten(), self.view_j['mask'][1][0].flatten()] # 147456
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        if 'mask' in self.view_i.keys():
            if self.str_edges[0] == '0_1':
                self.mirror_masks = [self.view_i['mask'][1], self.view_j['mask'][1]] if 'mask' in self.view_i.keys() else [None, None]
                self.conf_mask = [self.conf_i['0_1'] > self.min_conf_thr, self.conf_j['0_1'] > self.min_conf_thr]
            elif self.str_edges[0] == '1_0':
                self.mirror_masks = [self.view_j['mask'][0], self.view_i['mask'][0]] if 'mask' in self.view_j.keys() else [None, None]
                self.conf_mask = [self.conf_j['1_0'] > self.min_conf_thr, self.conf_i['1_0'] > self.min_conf_thr]

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param
    
    def _set_focal_same(self, focal, force=False):
        param = self.im_focals
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()
    
    def get_focal_same(self):
        return (self.im_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        I = torch.eye(4, device=cam2world.device) 
        if int(self.final_edge[0][0]) == 1:
            c2ws = [cam2world, I[None]]
        elif int(self.final_edge[0][0]) == 0:
            c2ws = [I[None], cam2world]

        return c2ws

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focal_same().repeat(self.n_imgs, 1)
        # focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses() # [2, 4, 4]
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)

        
        ########################
        # project to world frame
        main_view_idx = 1
        main_view_pts = geotrf(im_poses[main_view_idx], rel_ptmaps[main_view_idx][None])
        mirror_view_idx = 0
        mirror_view_pts = geotrf(im_poses[mirror_view_idx], rel_ptmaps[mirror_view_idx][None])

        pts3d = torch.cat([mirror_view_pts, main_view_pts], dim=0)
        return pts3d

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def my_get_pts3d(self):
        res_raw = self.depth_to_pts3d()
        res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res_raw, self.imshapes)]
        return res_raw, res
    
    def _set_pose_optim1pose(self, poses, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose
    
    def forward(self, sigma, cur_iter=None, intermediate_save_dir=None):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)
        proj_pts3d_resized = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(proj_pts3d, self.imshapes)]

        # Get the mirror plane
        main_view_pts3d = proj_pts3d_resized[1] 
        main_view_idx = self.view_i['idx'].index(1)
        main_view_mask = self.view_i['mask'][main_view_idx][0] # Now it is the outside mask
        mirror_pts3d = main_view_pts3d[~main_view_mask] 
        center = torch.mean(mirror_pts3d, dim=0)

        # Run PCA to get normal
        mirror_pts3d_centered = mirror_pts3d - center
        cov = torch.matmul(mirror_pts3d_centered.T, mirror_pts3d_centered) / (mirror_pts3d_centered.shape[0] - 1)
        U, S, Vh = torch.linalg.svd(cov, full_matrices=False)  # SVD is differentiable
        normal = Vh[-1]
        # Ensure normal points "up" (positive y)
        # Using abs() and sign() is non-differentiable, so use a differentiable version:
        normal = normal * (normal[1] / torch.sqrt(normal[1]*normal[1]))

        # Reflect pose0 against mirror plane using torch operations
        pose_fix = torch.eye(4, device=normal.device)
        # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
        I = torch.eye(3, device=normal.device)
        flip_matrix = torch.tensor([[-1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], device=normal.device).to(torch.float)
        
        R = I - 2 * torch.outer(normal, normal) 
        t = 2 * torch.dot(normal, center) * normal
        refl_matrix = torch.cat([R, t.unsqueeze(1)], dim=1)
        refl_matrix_4x4 = torch.cat([refl_matrix, torch.tensor([0, 0, 0, 1], device=normal.device).unsqueeze(0)], dim=0)

        pose_optim_ref = refl_matrix_4x4 @ pose_fix
        pose_optim_ref = pose_optim_ref @ flip_matrix # camera to world
        pose_optim_ref_rot = pose_optim_ref[:3, :3]
        pose_optim_ref_trans = pose_optim_ref[:3, 3]

        pose_optim_ref_xyzw = roma.rotmat_to_unitquat(pose_optim_ref_rot)
        pose_optim_ref_dir = pose_optim_ref_xyzw[:3] / (torch.norm(pose_optim_ref_xyzw[:3]) + 1e-8)
        # o3d.io.write_triangle_mesh('optim_cam_mirror_pre.ply', get_frustum_opencv(pose0_ref.clone().detach().cpu().numpy()))

        pose_optim = self.get_im_poses()[self.final_edge[0][1]][0]
        pose_optim_rot = pose_optim[:3, :3]
        pose_optim_trans = pose_optim[:3, 3]
        pose_optim_xyzw = roma.rotmat_to_unitquat(pose_optim_rot)
        pose_optim_dir = pose_optim_xyzw[:3] / (torch.norm(pose_optim_xyzw[:3]) + 1e-8)

        l_trans = torch.sum((pose_optim_trans - pose_optim_ref_trans) ** 2)
        l_rot = 1 - torch.sum(pose_optim_dir * pose_optim_ref_dir)
        loss_pose = l_trans + l_rot

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        # Check if mirror_view_mask exists
        if not hasattr(self, 'mirror_masks_vmirror_vmain_flat'):
            # If not, use default weights without masking
            li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
            lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        else:
            # print("The loss is using mirror mask!!!!!!!!!!!!!!!")
            mirror_view_mask = self.mirror_masks_vmirror_vmain_flat[0].int().to(self.device) # view 0

            li_1 = self.dist(proj_pts3d[1], aligned_pred_i[0], weight=self._weight_i[0]).sum() / self.total_area_i # X^1,(1,0)
            li_0 = self.dist(proj_pts3d[0], aligned_pred_i[1], weight=self._weight_i[1]*mirror_view_mask).sum() / mirror_view_mask.sum() # X^0,(0,1)
            # li_0 = self.dist(proj_pts3d[0], aligned_pred_i[1], weight=self._weight_i[1]).sum() / self.total_area_i # X^0,(0,1)

            lj_0 = self.dist(proj_pts3d[0], aligned_pred_j[0], weight=self._weight_j[0]*mirror_view_mask).sum() / mirror_view_mask.sum() # X^0,(1,0)
            # lj_0 = self.dist(proj_pts3d[0], aligned_pred_j[0], weight=self._weight_j[0]).sum() / self.total_area_i # X^0,(1,0)
            lj_1 = self.dist(proj_pts3d[1], aligned_pred_j[1], weight=self._weight_j[1]).sum() / self.total_area_i # X^1,(0,1)
            li = li_1 + li_0
            lj = lj_0 + lj_1

        
        return li + lj + sigma * loss_pose, sigma * loss_pose


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img
