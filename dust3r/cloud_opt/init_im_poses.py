# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Initialization functions for global alignment
# --------------------------------------------------------
from functools import cache

import numpy as np
import scipy.sparse as sp
import torch
import cv2
import roma
from tqdm import tqdm
import trimesh
import open3d as o3d
from dust3r.utils.geometry import geotrf, inv, get_med_dist_between_poses
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.viz import to_numpy

from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores, _fast_depthmap_to_pts3d

@torch.no_grad()
def init_from_known_poses(self, niter_PnP=10, min_conf_thr=3):
    device = self.device

    # indices of known poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    assert nkp == self.n_imgs, 'not all poses are known'

    # get all focals
    nkf, _, im_focals = get_known_focals(self)
    breakpoint()
    assert nkf == self.n_imgs
    im_pp = self.get_principal_points()

    best_depthmaps = {}
    # init all pairwise poses
    for e, (i, j) in enumerate(tqdm(self.edges, disable=not self.verbose)):
        i_j = edge_str(i, j)

        # find relative pose for this pair
        P1 = torch.eye(4, device=device)
        msk = self.conf_i[i_j] > min(min_conf_thr, self.conf_i[i_j].min() - 0.1)
        _, P2 = fast_pnp(self.pred_j[i_j], float(im_focals[i].mean()),
                         pp=im_pp[i], msk=msk, device=device, niter_PnP=niter_PnP)

        # align the two predicted camera with the two gt cameras
        s, R, T = align_multiple_poses(torch.stack((P1, P2)), known_poses[[i, j]])
        # normally we have known_poses[i] ~= sRT_to_4x4(s,R,T,device) @ P1
        # and geotrf(sRT_to_4x4(1,R,T,device), s*P2[:3,3])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

        # remember if this is a good depthmap
        score = float(self.conf_i[i_j].mean())
        if score > best_depthmaps.get(i, (0,))[0]:
            best_depthmaps[i] = score, i_j, s

    # init all image poses
    for n in range(self.n_imgs):
        assert known_poses_msk[n]
        _, i_j, scale = best_depthmaps[n]
        depth = self.pred_i[i_j][:, :, 2]
        self._set_depthmap(n, depth * scale)


@torch.no_grad()
def init_minimum_spanning_tree(self, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    pts3d, msp_edges, im_focals, im_poses = minimum_spanning_tree(self.imshapes, self.edges,
                                                          self.pred_i, self.pred_j, self.conf_i, self.conf_j, self.im_conf, self.min_conf_thr,
                                                          device, has_im_poses=self.has_im_poses, verbose=self.verbose,
                                                          **kw)
    return init_from_pts3d(self, pts3d, im_focals, im_poses, msp_edges[0])

def init_from_pts3d_samefocal(self, pts3d, im_focals, im_poses):
    
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
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
        # set the same focal for all images using the first image's focal
        if im_focals[0] is not None:
            self._set_focal_same(im_focals[0])

    if self.verbose:
        print(' init loss =', float(self()))

def my_init_from_pts3d(self, pts3d, im_focals, im_poses, final_edge):
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
        (main_view_idx, mirror_view_idx) = final_edge
        # Initialize the main view
        cam2world_main = im_poses[main_view_idx]
        depth = geotrf(inv(cam2world_main), pts3d[main_view_idx])[..., 2]
        self._set_depthmap(main_view_idx, depth)
        self._set_pose(self.im_poses, main_view_idx, cam2world_main)
        if im_focals[main_view_idx] is not None:
            self._set_focal(main_view_idx, im_focals[main_view_idx])

        # Get the mirror plane
        proj_pts3d_main_view = self.get_pts3d(raw=False)[main_view_idx]
        local_main_view_idx = self.view_i['idx'].index(main_view_idx)
        self.main_view_mask = self.view_i['mask'][local_main_view_idx][0] # Now it is the outside mask
        main_view_mask = self.main_view_mask
        mirror_pts3d = proj_pts3d_main_view[~main_view_mask] 
        center = torch.mean(mirror_pts3d, dim=0)

        # Run PCA to get normal
        mirror_pts3d_centered = mirror_pts3d - center
        cov = torch.matmul(mirror_pts3d_centered.T, mirror_pts3d_centered) / (mirror_pts3d_centered.shape[0] - 1)
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        # The normal is the eigenvector corresponding to smallest eigenvalue
        normal = eigenvecs[:, 0]
        # Ensure normal points "up" (positive y)
        normal = normal * torch.sign(normal[1])

        # reflection matrix
        # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
        I = torch.eye(3, device=normal.device)
        R_reflection = I - 2 * torch.outer(normal, normal) 
        t_reflection = 2 * torch.dot(normal, center) * normal

        # Initialize the mirror view
        cam2world_mirror = im_poses[mirror_view_idx]
        R_reflected = R_reflection @ cam2world_main[:3,:3] # det=-1
        R_reflected[:, 0] *= -1
        t_reflected = t_reflection + R_reflection @ cam2world_main[:3,3]
        cam2world_mirror[:3,:3] = R_reflected # det=1
        cam2world_mirror[:3,3] = t_reflected

        depth = geotrf(inv(cam2world_mirror), pts3d[mirror_view_idx])[..., 2]
        self._set_depthmap(mirror_view_idx, depth)
        self._set_pose(self.im_poses, mirror_view_idx, cam2world_mirror)
        if im_focals[mirror_view_idx] is not None:
            self._set_focal(mirror_view_idx, im_focals[mirror_view_idx])

    if self.verbose:
        print(' init loss =', float(self()))

def my_init_from_pts3d_poseloss(self, sigma, pts3d, im_focals, im_poses, final_edge):
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
        # self._set_pose(self.im_poses, fix_view_idx, cam2world_fix)
        if im_focals[fix_view_idx] is not None:
            self._set_focal_same(im_focals[fix_view_idx])

        # Get the mirror plane
        proj_pts3d_main_view = pts3d[main_view_idx]
        local_main_view_idx = self.view_i['idx'].index(main_view_idx)
        self.main_view_mask = self.view_i['mask'][local_main_view_idx][0] # Now it is the outside mask
        main_view_mask = self.main_view_mask
        mirror_pts3d = proj_pts3d_main_view[~main_view_mask] 
        center = torch.mean(mirror_pts3d, dim=0)

        # Run PCA to get normal
        mirror_pts3d_centered = mirror_pts3d - center
        cov = torch.matmul(mirror_pts3d_centered.T, mirror_pts3d_centered) / (mirror_pts3d_centered.shape[0] - 1)
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        # The normal is the eigenvector corresponding to smallest eigenvalue
        normal = eigenvecs[:, 0]
        # Ensure normal points "up" (positive y)
        normal = normal * torch.sign(normal[1])

        # reflection matrix
        # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
        I = torch.eye(3, device=normal.device)
        R_reflection = I - 2 * torch.outer(normal, normal) 
        t_reflection = 2 * torch.dot(normal, center) * normal

        # Initialize the mirror view
        cam2world_optim = im_poses[optim_view_idx]
        R_reflected = R_reflection @ cam2world_fix[:3,:3] # det=-1
        R_reflected[:, 0] *= -1
        t_reflected = t_reflection + R_reflection @ cam2world_fix[:3,3]
        cam2world_optim[:3,:3] = R_reflected # det=1
        cam2world_optim[:3,3] = t_reflected

        depth = geotrf(inv(cam2world_optim), pts3d[optim_view_idx])[..., 2]
        self._set_depthmap(optim_view_idx, depth)
        # self._set_pose(self.im_poses, optim_view_idx, cam2world_optim)
        self._set_im_pose(self.im_poses, optim_view_idx, cam2world_optim)
        if im_focals[optim_view_idx] is not None:
            self._set_focal_same(im_focals[optim_view_idx])

    if self.verbose:
        print(' init loss =', float(self(sigma)[0]))

def my_init_from_pts3d_poseloss_old(self, sigma, pts3d, im_focals, im_poses, final_edge):
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
        # Initialize the fixed view
        # cam2world_main = im_poses[main_view_idx]
        cam2world_fix = im_poses[fix_view_idx]
        depth = geotrf(inv(cam2world_fix), pts3d[fix_view_idx])[..., 2]
        self._set_depthmap(fix_view_idx, depth)
        if im_focals[fix_view_idx] is not None:
            self._set_focal_same(im_focals[fix_view_idx])
        
        ########################### debug ############################  
        # # Save depth map with turbo colormap
        # import matplotlib.pyplot as plt
        # from matplotlib.cm import turbo
        # import numpy as np
        # from PIL import Image

        # # Get depth values and convert to numpy
        # depth_np = depth.detach().cpu().numpy()
        
        # # Normalize depth values to 0-1 range
        # depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        
        # # Apply turbo colormap
        # depth_colored = (turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        
        # # Save as PNG using PIL
        # depth_img = Image.fromarray(depth_colored)
        # depth_img.save('depth_turbo.png')
        # Create point cloud from pts3d
        # get pointmaps in camera frame
        focals = self.get_focal_same().repeat(self.n_imgs, 1)
        # focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses() # [2, 4, 4]
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        breakpoint()

        main_view_idx = 1
        main_view_pts = geotrf(im_poses[main_view_idx], rel_ptmaps[fix_view_idx][None])

        debug_pcd = o3d.geometry.PointCloud()
        debug_pcd.points = o3d.utility.Vector3dVector(main_view_pts.reshape(-1, 3).detach().cpu().numpy())
        o3d.io.write_point_cloud("temp.ply", debug_pcd)
        breakpoint()
        ########################### debug ############################  

        # Get the mirror plane 
        proj_pts3d_main_view = pts3d[main_view_idx]
        local_main_view_idx = self.view_i['idx'].index(main_view_idx)
        self.main_view_mask = self.view_i['mask'][local_main_view_idx][0] # Now it is the outside mask
        main_view_mask = self.main_view_mask
        mirror_pts3d = proj_pts3d_main_view[~main_view_mask] 
        center = torch.mean(mirror_pts3d, dim=0)
        
        # Run PCA to get normal
        mirror_pts3d_centered = mirror_pts3d - center
        cov = torch.matmul(mirror_pts3d_centered.T, mirror_pts3d_centered) / (mirror_pts3d_centered.shape[0] - 1)
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        # The normal is the eigenvector corresponding to smallest eigenvalue
        normal = eigenvecs[:, 0]
        # Ensure normal points "up" (positive y)
        normal = normal * torch.sign(normal[1])

        ########################### debug ############################
        # # Save mirror points as PLY
        # # Create points array with mirror points, center and normal point
        # normal_pt = center + 0.1 * normal  # Point 0.1 units along normal direction
        # all_pts = torch.cat([mirror_pts3d, center.unsqueeze(0), normal_pt.unsqueeze(0)], dim=0)
        
        # # Create colors array - white for mirror points, red for center, blue for normal point
        # colors = torch.ones((len(all_pts), 3))
        # colors[:-2] = torch.tensor([1.0, 1.0, 1.0])  # White for mirror points
        # colors[-2] = torch.tensor([1.0, 0.0, 0.0])   # Red for center point
        # colors[-1] = torch.tensor([0.0, 0.0, 1.0])   # Blue for normal point
        
        # mirror_pcd = o3d.geometry.PointCloud()
        # mirror_pcd.points = o3d.utility.Vector3dVector(mirror_pts3d.reshape(-1, 3).detach().cpu().numpy())
        # # mirror_pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        # o3d.io.write_point_cloud("main_view_idx.ply", mirror_pcd)
        ########################### debug ############################

        # reflection matrix
        # R = I - 2nn^T, t = 2(n^T c)n where n is normal and c is center
        I = torch.eye(3, device=normal.device)
        R_reflection = I - 2 * torch.outer(normal, normal) 
        t_reflection = 2 * torch.dot(normal, center) * normal

        # Initialize the optim view
        cam2world_optim = im_poses[optim_view_idx]
        R_reflected = R_reflection @ cam2world_fix[:3,:3] # det=-1
        R_reflected[:, 0] *= -1
        t_reflected = t_reflection + R_reflection @ cam2world_fix[:3,3]
        cam2world_optim[:3,:3] = R_reflected # det=1
        cam2world_optim[:3,3] = t_reflected

        ########################### debug ############################
        # o3d.io.write_triangle_mesh("frustum_fix.ply", get_frustum_opencv(cam2world_fix.detach().cpu().numpy()))
        # o3d.io.write_triangle_mesh("frustum_optim.ply", get_frustum_opencv(cam2world_optim.detach().cpu().numpy()))
        ########################### debug ############################

        depth = geotrf(inv(cam2world_optim), pts3d[optim_view_idx])[..., 2]
        # depth = geotrf(inv(im_poses[optim_view_idx]), pts3d[optim_view_idx])[..., 2]

        ########################### debug ############################
        # breakpoint()
        # ########################### debug ############################
        self._set_depthmap(optim_view_idx, depth)

        ########################### debug ############################
        # mirror_pcd = o3d.geometry.PointCloud()
        # mirror_pcd.points = o3d.utility.Vector3dVector(self.get_pts3d(raw=True)[optim_view_idx].detach().cpu().numpy())
        # # mirror_pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        # o3d.io.write_point_cloud("optim_view_idx.ply", mirror_pcd)
        # breakpoint()
        ########################### debug ############################

        self._set_im_pose(self.im_poses, optim_view_idx, cam2world_optim)


        ########################### debug ############################
        # focals = self.get_focal_same().repeat(self.n_imgs, 1)
        # # focals = self.get_focals()
        # pp = self.get_principal_points()
        # im_poses = self.get_im_poses() # [2, 4, 4]
        # depth = self.get_depthmaps(raw=True)

        # # get pointmaps in camera frame
        # rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)

        # main_view_idx = 1
        # main_view_pts = geotrf(im_poses[main_view_idx], rel_ptmaps[main_view_idx][None])
        # mirror_view_idx = 0
        # mirror_view_pts = geotrf(im_poses[mirror_view_idx], rel_ptmaps[mirror_view_idx][None])        
        ########################### debug ############################
    if self.verbose:
        print(' init loss =', float(self(sigma)[0]))

def init_from_pts3d(self, pts3d, im_focals, im_poses, final_edge=None):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)

    # set all pairwise poses
    # pred = [self.pred_i, self.pred_j]
    for e, (i, j) in enumerate(self.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
        # s, R, T = rigid_points_registration(pred[final_edge[1-e]][i_j], pts3d[0], conf=self.conf_i[i_j])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                self._set_focal(i, im_focals[i])

    if self.verbose:
        print(' init loss =', float(self()))


def minimum_spanning_tree(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,
                          device, has_im_poses=True, niter_PnP=10, verbose=True):
    n_imgs = len(imshapes)
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    # todo = sorted(zip(-msp.data, msp.col, msp.row))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f' init edge ({i}*,{j}*) {score=}')
    i_j = edge_str(i, j)
    pts3d[i] = pred_i[i_j].clone()
    pts3d[j] = pred_j[i_j].clone()

    
    # for view_idx, pts in enumerate([pts3d[i], pts3d[j]]):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy().reshape(-1, 3))
    #     o3d.io.write_point_cloud(f"debug/initial_pts3d_view{view_idx}.ply", pcd)
    # breakpoint()
    
    # # Create point clouds for both views
    # colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    # edges_names = ['1_0', '1_0', '0_1', '0_1']
    # all_pcds = [pred_i['1_0'], pred_j['1_0'], pred_i['0_1'], pred_j['0_1']]
    # c = 0
    # for idx, (pcd, color) in enumerate(zip(all_pcds, colors)):
    #     pcd_i = o3d.geometry.PointCloud()
    #     pcd_i.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3).cpu().numpy())
    #     red_color = np.array(color)[None].repeat(len(pcd.reshape(-1, 3)), axis=0)
    #     pcd_i.colors = o3d.utility.Vector3dVector(red_color)
    #     o3d.io.write_point_cloud(f"debug/pts3d_{edges_names[c]}_view{c}.ply", pcd_i)
    #     c += 1

    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])
        im_focals[j] = estimate_focal(pred_i[i_j])

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
                # breakpoint()
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
    
    # ###################### debug ######################
    # temp_pcd = o3d.geometry.PointCloud()
    # temp_pcd.points = o3d.utility.Vector3dVector(pts3d[0].reshape(-1, 3).cpu().numpy())
    # o3d.io.write_point_cloud("temp_mirror.ply", temp_pcd)
    # o3d.io.write_triangle_mesh("frustum_mirror.ply", get_frustum_opencv(im_poses[0].detach().cpu().numpy()))
    # temp_pcd = o3d.geometry.PointCloud()
    # temp_pcd.points = o3d.utility.Vector3dVector(pts3d[1].reshape(-1, 3).cpu().numpy())
    # o3d.io.write_point_cloud("temp_main.ply", temp_pcd)
    # o3d.io.write_triangle_mesh("frustum_main.ply", get_frustum_opencv(im_poses[1].detach().cpu().numpy()))
    # breakpoint()
    # ###################### debug ######################
    return pts3d, msp_edges, im_focals, im_poses


def dict_to_sparse_graph(dic):
    n_imgs = max(max(e) for e in dic) + 1
    res = sp.dok_array((n_imgs, n_imgs))
    for edge, value in dic.items():
        res[edge] = value
    return res


def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)


def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W/2, H/2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode='weiszfeld').ravel()
    return float(focal)


@cache
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S/2, S*3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W/2, H/2)
    else:
        pp = to_numpy(pp)

    best = 0,
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world


def get_known_poses(self):
    if self.has_im_poses:
        known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
        known_poses = self.get_im_poses()
        return known_poses_msk.sum(), known_poses_msk, known_poses
    else:
        return 0, None, None


def get_known_focals(self):
    if self.has_im_poses:
        known_focal_msk = self.get_known_focal_mask()
        known_focals = self.get_focals()
        return known_focal_msk.sum(), known_focal_msk, known_focals
    else:
        return 0, None, None


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    return s, R, T
