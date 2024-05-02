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
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}


__device__ void compute_radii_center(const float3& p_orig, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* projmatrix, const int W, const int H, float& radii, float2& p_pixel_xy, float3& rT0, float3& rT1, float3& rT3, float3 &zaxis)
{
		const float far_n = FAR_PLANE;
		const float near_n = NEAR_PLANE;
        // Create scaling matrix
	    glm::mat3 S = glm::mat3(1.0f);
	    S[0][0] = mod * scale.x;
	    S[1][1] = mod * scale.y;
	    S[2][2] = mod * scale.z;

	    // Normalize quaternion to get valid rotation
	    glm::vec4 q = rot;
	    float r = q.x;
	    float x = q.y;
	    float y = q.z;
	    float z = q.w;

	    // Compute rotation matrix from quaternion
	    glm::mat3 R = glm::mat3(
		    1.f - 2.f * (y * y + z * z), 	2.f * (x * y - r * z), 			2.f * (x * z + r * y),
		    2.f * (x * y + r * z), 			1.f - 2.f * (x * x + z * z), 	2.f * (y * z - r * x),
		    2.f * (x * z - r * y), 			2.f * (y * z + r * x), 			1.f - 2.f * (x * x + y * y)
	    );

		zaxis = {2.f * (x * z + r * y), 
				 2.f * (y * z - r * x), 
				 1.f - 2.f * (x * x + y * y)};

	    glm::mat3 L = S * R;

        // center of Gaussians in the camera coordinate
        glm::vec4 p_world = glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1);

		glm::mat4x2 M = glm::mat4x2(
			L[0][0], L[0][1], 
            L[1][0], L[1][1], 
			L[2][0], L[2][1], 
			0.0f   , 0.0f    
		);

		glm::mat4 World2Ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
            projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
            projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
            projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
		);
		glm::mat4 Ndc2Pixel = glm::mat4(
			W/2,	0,		0,				W/2,
			0,		H/2,	0,				H/2,
			0,		0,		far_n-near_n,	near_n,
			0,		0,		0,				1 
		);
		glm::mat4 P = World2Ndc * Ndc2Pixel;

		glm::vec4 p_pixel = p_world * P;
        glm::mat4x2 uv_view = M * P;
        glm::mat4x3 T = glm::mat4x3(
            uv_view[0][0], uv_view[0][1], p_pixel.x,
            uv_view[1][0], uv_view[1][1], p_pixel.y,
			uv_view[2][0], uv_view[2][1], p_pixel.z,
			uv_view[3][0], uv_view[3][1], p_pixel.w  
        );

        float3 T0 = {T[0][0], T[0][1], T[0][2]};
        float3 T1 = {T[1][0], T[1][1], T[1][2]};
        float3 T2 = {T[2][0], T[2][1], T[2][2]};
        float3 T3 = {T[3][0], T[3][1], T[3][2]};

		// Compute AABB
    	// Homogneous plane is very useful for both ray-splat intersection and bounding box computation
    	// we know how to compute u,v given x,y homogeneous plane already; computing AABB is done by a reverse process.
    	// i.e compute the x, y s.t. |h_u^4| = 1 and |h_v^4|=1 (distance of gaussian center to plane in the uv space)
        float3 temp_point = {1.0f, 1.0f, -1.0f};
        float distance = sumf3(T3 * T3 * temp_point);
		
        float3 f = {
			1 / (distance + 0.0000001f), 
			1 / (distance + 0.0000001f), 
			-1 / (distance + 0.0000001f)
		};
        float2 point_image = {
            sumf3(f * T0 * T3),
            sumf3(f * T1 * T3)
        };  
		
        float2 temp = {
            sumf3(f * T0 * T0),
            sumf3(f * T1 * T1)
		};
        float2 half_extend = point_image * point_image - temp;
        float2 radii2 = sqrtf2(maxf2(1e-4, half_extend));
		
		radii = 2.0f * ceil(max(radii2.x, radii2.y));
		p_pixel_xy = {point_image.x, point_image.y};
		rT0 = T0;
		rT1 = T1;
		rT3 = T3;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* rgb,
	float3* T0,
	float3* T1,
	float3* T3,
	float3* normals,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 p_view;
	// Perform near culling, quit if outside.
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_world = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float radius;
	float2 point_image;
	float3 rT0, rT1, rT3;
	float3 zaxis;
	compute_radii_center(p_world, scales[idx], scale_modifier, rotations[idx], projmatrix, W, H, radius, point_image, rT0, rT1, rT3, zaxis);

	float3 normal = transformVec4x3(zaxis, viewmatrix);
	if (sumf3(normal * p_view) > 0.f)
		normal = {-normal.x, -normal.y, -normal.z};
	
	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = radius;
	points_xy_image[idx] = point_image;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	T0[idx] = rT0;
	T1[idx] = rT1;
	T3[idx] = rT3;
	normals[idx] = normal;
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ opacity,
	const float3* __restrict__ T0,
	const float3* __restrict__ T1,
	const float3* __restrict__ T3,
	const float3* __restrict__ normals,
	float* __restrict__ final_T,
	float* __restrict__ final_A,
	float* __restrict__ final_D,
	float* __restrict__ final_D_2,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_distortion_map,
	float* __restrict__ out_depth_map,
	float* __restrict__ out_normal_map,
	float* __restrict__ out_alpha_map)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	
	const bool inside = pix.x < W && pix.y < H;  // Check if this thread is associated with a valid pixel or outside.
	bool done = !inside;  // Done threads can help with fetching, but don't rasterize

	// Load start/end range of IDs to process in bit sorted list.
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ float3 collected_T0[BLOCK_SIZE];
	__shared__ float3 collected_T1[BLOCK_SIZE];
	__shared__ float3 collected_T3[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	const float far_n = FAR_PLANE;
	const float near_n = NEAR_PLANE;
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t max_contributor = -1;
	float C[CHANNELS + 1] = { 0 };
	float render_normal[3] = { 0 };
	float A = 0.0f;
	float D = 0.0f;
	float D_2 = 0.0f;
	float distortion_loss = 0.0f;
	float depth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_opacity[block.thread_rank()] = opacity[coll_id];
			collected_T0[block.thread_rank()] = T0[coll_id];
			collected_T1[block.thread_rank()] = T1[coll_id];
			collected_T3[block.thread_rank()] = T3[coll_id];
			collected_normal[block.thread_rank()] = normals[coll_id];
		}
		block.sync();
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			const float2 xy = collected_xy[j];
			const float3 T0 = collected_T0[j];
			const float3 T1 = collected_T1[j];
			const float3 T3 = collected_T3[j];
			const float3 normal = collected_normal[j];

			float3 k = pixf.x * T3 - T0;
			float3 l = pixf.y * T3 - T1;
			float3 inter_p = cross(k, l);
			float2 s = {inter_p.x / inter_p.z, inter_p.y / inter_p.z};
			
			float depth_cam = T3.x * s.x + T3.y * s.y + T3.z;
			if (depth_cam <= near_n)
				continue;
			const float m = far_n / (far_n - near_n) * (1 - near_n / depth_cam);
			

			float dist3d = s.x * s.x + s.y * s.y;
			float filtersze_2 = 2.0f;
			float dist2d = filtersze_2 * ((pixf.x - xy.x) * (pixf.x - xy.x) + (pixf.y - xy.y) * (pixf.y - xy.y));
			float dist = min(dist3d, dist2d);
			float power = -0.5f * dist;
			const float opa = collected_opacity[j];
			
			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			float test_T = T * (1 - alpha);
			if (test_T > 0.5f)
			{
				depth = depth_cam;
				max_contributor = contributor;
			}
			
			if (test_T < 0.0001f)
			{	
				done = true;
				continue;	
			}

			const float w = alpha * T;
			distortion_loss += w * (m * m * A + D_2 - 2 * m * D);
			
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			C[CHANNELS] += w;
			
			render_normal[0] += w * normal.x;
			render_normal[1] += w * normal.y;
			render_normal[2] += w * normal.z;

			T = test_T;
			last_contributor = contributor;  // Keep track of last range entry to update this pixel.
			A += w;
			D += w * m;
			D_2 += w * m * m;
		}
	}

	// All threads that treat valid pixel write out their final rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		final_A[pix_id] = A;
		final_D[pix_id] = D;
		final_D_2[pix_id] = D_2;
		n_contrib[pix_id] = last_contributor;
		n_contrib[pix_id + H * W] = max_contributor;

		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		out_alpha_map[pix_id] = C[CHANNELS];
		out_distortion_map[pix_id] =  distortion_loss /  (A * A + 1e-7);
		out_depth_map[pix_id] = depth;

		out_normal_map[pix_id] = render_normal[0];
		out_normal_map[H * W + pix_id] = render_normal[1];
		out_normal_map[2 * H * W + pix_id] = render_normal[2];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* opacity,
	const float3* T0,
	const float3* T1,
	const float3* T3,
	const float3* normals,
	float* final_T,
	float* final_A,
	float* final_D,
	float* final_D_2,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_distortion_map,
	float* out_depth_map,
	float* out_normal_map,
	float* out_alpha_map)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		opacity,
		T0,
		T1,
		T3,
		normals,
		final_T,
		final_A,
		final_D,
		final_D_2,
		n_contrib,
		bg_color,
		out_color,
		out_distortion_map,
		out_depth_map,
		out_normal_map,
		out_alpha_map);
}


void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* rgb,
	float3* T0,
	float3* T1,
	float3* T3,
	float3* normals,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		rgb,
		T0,
		T1,
		T3,
		normals,
		grid,
		tiles_touched,
		prefiltered);
}