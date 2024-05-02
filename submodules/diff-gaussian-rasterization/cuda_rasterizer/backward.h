/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float3* T0,
		const float3* T1,
		const float3* T3,
		const float3* normals,
		const float* normal_map,
		const float* opacity,
		const float* colors,
		const float* final_Ts,
		const float* final_As,
		const float* final_Ds,
		const float* final_D_2s,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_ddistortion_maps,
		const float* dL_dnormal_maps,
		const float* dL_ddepth_maps,
		float3* dL_dmean2D,
		float3* dL_dT0,
		float3* dL_dT1,
		float3* dL_dT3,
		float* dL_dopacity,
		float* dL_dcolors,
		float3* dL_dnormals
		);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* view,
		const float* proj,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dnormal,
		float3* dL_dmean2D,
		float3* dL_dT0s,
		float3* dL_dT1s,
		float3* dL_dT3s,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif