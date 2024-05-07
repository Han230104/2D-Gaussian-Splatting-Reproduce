import os
from os import makedirs
import sys 
sys.path.insert(0, sys.path[0]+"/../")
from argparse import ArgumentParser
from tqdm import tqdm
import random
import math
import torch
from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c

        
def tsdf_fusion(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "tsdf")

    makedirs(render_path, exist_ok=True)
    o3d_device = o3d.core.Device("CUDA:0")
    
    voxel_size = 0.004
    trunc = 0.012
    depth_scale = 1.0
    depth_max = 6.0

    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=8,
            block_count=50000,
            device=o3d_device)
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            render_pkg = render(view, gaussians, pipeline, background)
            
            depth = render_pkg["depth_map"][None]
            rgb = render_pkg["render"]
            
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width=view.image_width, 
                    height=view.image_height, 
                    cx = view.image_width/2,
                    cy = view.image_height/2,
                    fx = view.image_width / (2 * math.tan(view.FoVx / 2.)),
                    fy = view.image_height / (2 * math.tan(view.FoVy / 2.)))
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            
            o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_color = o3d_color.to(o3d_device)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, depth_scale, depth_max)
            # Activate them in the underlying hash map (may have been inserted)
            vbg.hashmap().activate(frustum_block_coords)

            # Find buf indices in the underlying engine
            buf_indices, _ = vbg.hashmap().find(frustum_block_coords)
            o3d.core.cuda.synchronize()

            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            o3d.core.cuda.synchronize()

            extrinsic_dev = extrinsic.to(o3d_device, o3c.float32)
            xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3, 3:]

            intrinsic_dev = intrinsic.to(o3d_device, o3c.float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3c.int64)
            v = (uvd[1] / d).round().to(o3c.int64)
            o3d.core.cuda.synchronize()

            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < o3d_depth.columns) & (v < o3d_depth.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]
            depth_readings = o3d_depth.as_tensor()[v_proj, u_proj, 0].to(o3c.float32) / depth_scale
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) \
            & (depth_readings < depth_max) \
            & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            o3d.core.cuda.synchronize()

            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            tsdf[valid_voxel_indices] = (tsdf[valid_voxel_indices] * w + sdf[mask_inlier].reshape(w.shape)) / (wp)
            
            color_readings = o3d_color.as_tensor()[v_proj, u_proj].to(o3c.float32)
            color = vbg.attribute('color').reshape((-1, 3))
            color[valid_voxel_indices] = (color[valid_voxel_indices] * w + color_readings[mask_inlier]) / (wp)
            
            weight[valid_voxel_indices] = wp
            o3d.core.cuda.synchronize()

        mesh = vbg.extract_triangle_mesh().to_legacy()
        
        # write mesh
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf.ply", mesh)
            
            
def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
        train_cameras = scene.getTrainCameras()
    
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cams = train_cameras
        tsdf_fusion(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args))
