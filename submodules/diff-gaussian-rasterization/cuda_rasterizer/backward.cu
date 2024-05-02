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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


__device__ void compute_radii_center(int idx, const float3& p_orig, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* projmatrix, const int W, const int H, const float3* dL_dmean2D, const float3 dL_dnormal,
glm::vec3* dL_dmeans, float3* dL_dT0s, float3* dL_dT1s, float3* dL_dT3s, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	const float far_n = FAR_PLANE;
	const float near_n = NEAR_PLANE;
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
	    1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
	    2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
	    2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 L = S * R;

    // center of Gaussians in the ndc coordinate
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

    const float3 T0 = {T[0][0], T[0][1], T[0][2]};
    const float3 T1 = {T[1][0], T[1][1], T[1][2]};
    const float3 T3 = {T[3][0], T[3][1], T[3][2]};

	const float3 temp_point = {1.0f, 1.0f, -1.0f};
    float distance = sumf3(T3 * T3 * temp_point);

	const float f = 1 / (distance + 0.0000001f);

	if(dL_dmean2D[idx].x != 0 || dL_dmean2D[idx].y != 0)
	{
		const float dpx_dT00 = f * T3.x;
		const float dpx_dT01 = f * T3.y;
		const float dpx_dT02 = -f * T3.z;
		const float dpy_dT10 = f * T3.x;
		const float dpy_dT11 = f * T3.y;
		const float dpy_dT12 = -f * T3.z;
		const float dpx_dT30 = T0.x * (f - 2 * f * f * T3.x * T3.x);
		const float dpx_dT31 = T0.y * (f - 2 * f * f * T3.y * T3.y);
		const float dpx_dT32 = -T0.z * (f + 2 * f * f * T3.z * T3.z);
		const float dpy_dT30 = T1.x * (f - 2 * f * f * T3.x * T3.x);
		const float dpy_dT31 = T1.y * (f - 2 * f * f * T3.y * T3.y);
		const float dpy_dT32 = -T1.z * (f + 2 * f * f * T3.z * T3.z);

		dL_dT0s[idx].x += dL_dmean2D[idx].x * dpx_dT00;
		dL_dT0s[idx].y += dL_dmean2D[idx].x * dpx_dT01;
		dL_dT0s[idx].z += dL_dmean2D[idx].x * dpx_dT02;
		dL_dT1s[idx].x += dL_dmean2D[idx].y * dpy_dT10;
		dL_dT1s[idx].y += dL_dmean2D[idx].y * dpy_dT11;
		dL_dT1s[idx].z += dL_dmean2D[idx].y * dpy_dT12;
		dL_dT3s[idx].x += dL_dmean2D[idx].x * dpx_dT30 + dL_dmean2D[idx].y * dpy_dT30;
		dL_dT3s[idx].y += dL_dmean2D[idx].x * dpx_dT31 + dL_dmean2D[idx].y * dpy_dT31;
		dL_dT3s[idx].z += dL_dmean2D[idx].x * dpx_dT32 + dL_dmean2D[idx].y * dpy_dT32;
	}
	
	dL_dmeans[idx].x += dL_dT0s[idx].z * P[0][0] + dL_dT1s[idx].z * P[1][0] + dL_dT3s[idx].z * P[3][0];
	dL_dmeans[idx].y += dL_dT0s[idx].z * P[0][1] + dL_dT1s[idx].z * P[1][1] + dL_dT3s[idx].z * P[3][1];
	dL_dmeans[idx].z += dL_dT0s[idx].z * P[0][2] + dL_dT1s[idx].z * P[1][2] + dL_dT3s[idx].z * P[3][2];

	const float dL_dM00 = dL_dT0s[idx].x * P[0][0] + dL_dT1s[idx].x * P[1][0] + dL_dT3s[idx].x * P[3][0];
	const float dL_dM10 = dL_dT0s[idx].x * P[0][1] + dL_dT1s[idx].x * P[1][1] + dL_dT3s[idx].x * P[3][1];
	const float dL_dM20 = dL_dT0s[idx].x * P[0][2] + dL_dT1s[idx].x * P[1][2] + dL_dT3s[idx].x * P[3][2];

	const float dL_dM01 = dL_dT0s[idx].y * P[0][0] + dL_dT1s[idx].y * P[1][0] + dL_dT3s[idx].y * P[3][0];
	const float dL_dM11 = dL_dT0s[idx].y * P[0][1] + dL_dT1s[idx].y * P[1][1] + dL_dT3s[idx].y * P[3][1];
	const float dL_dM21 = dL_dT0s[idx].y * P[0][2] + dL_dT1s[idx].y * P[1][2] + dL_dT3s[idx].y * P[3][2];

	glm::mat3x2 dL_dM = glm::mat3x2{
		dL_dM00, dL_dM01,
		dL_dM10, dL_dM11,
		dL_dM20, dL_dM21
	};

	glm::mat3 Rt = glm::transpose(R);
	glm::mat2x3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (- dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2]) - 4 * x * (dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (- dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2]) - 4 * y * (dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	const float dLcon_dx = dL_dnormal.x * 2.f * z - dL_dnormal.y * 2.f * r - dL_dnormal.z * 4.f * x;
	const float dLcon_dy = dL_dnormal.x * 2.f * r + dL_dnormal.y * 2.f * z - dL_dnormal.z * 4.f * y;
	const float dLcon_dz = dL_dnormal.x * 2.f * x + dL_dnormal.y * 2.f * y;
	const float dLcon_dr = dL_dnormal.x * 2.f * y - dL_dnormal.y * 2.f * x;
	dL_dq.x += dLcon_dr;
	dL_dq.y += dLcon_dx;
	dL_dq.z += dLcon_dy;
	dL_dq.w += dLcon_dz;

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w};

} 


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const glm::vec3* campos,
	const float3* dL_dnormals,
	float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float3* dL_dT0s,
	float3* dL_dT1s,
	float3* dL_dT3s,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float3 p_world = { means[idx].x, means[idx].y, means[idx].z };
	const float3 p_view = transformPoint4x3(p_world, viewmatrix);
	glm::vec4 q = rotations[idx];
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;
	float3 zaxis = {2.f * (x * z + r * y), 
					2.f * (y * z - r * x), 
					1.f - 2.f * (x * x + y * y)};
	float3 normal = transformVec4x3(zaxis, viewmatrix);
	float dL_dnx = 0.f;
	float dL_dny = 0.f;
	float dL_dnz = 0.f;
	if (sumf3(normal * p_view) > 0.f)
	{
		dL_dnx = -dL_dnormals[idx].x;
		dL_dny = -dL_dnormals[idx].y;
		dL_dnz = -dL_dnormals[idx].z;
	}
	else{
		dL_dnx = dL_dnormals[idx].x;
		dL_dny = dL_dnormals[idx].y;
		dL_dnz = dL_dnormals[idx].z;
	}
	
	const float3 dL_dnormal = {
		dL_dnx * viewmatrix[0] + dL_dny * viewmatrix[1] + dL_dnz * viewmatrix[2],
		dL_dnx * viewmatrix[4] + dL_dny * viewmatrix[5] + dL_dnz * viewmatrix[6],
		dL_dnx * viewmatrix[8] + dL_dny * viewmatrix[9] + dL_dnz * viewmatrix[10],
	};

	compute_radii_center(idx, means[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, W, H, dL_dmean2D, dL_dnormal, dL_dmeans, dL_dT0s, dL_dT1s, dL_dT3s, dL_dscale, dL_drot);
	
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
	
	dL_dmean2D[idx].x = 4.0f * dL_dmeans[idx].x;
	dL_dmean2D[idx].y = 4.0f * dL_dmeans[idx].y;
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float3* __restrict__ T0,
	const float3* __restrict__ T1,
	const float3* __restrict__ T3,
	const float3* __restrict__ normals,
	const float* __restrict__ normal_map,
	const float* __restrict__ opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const float* __restrict__ final_As,
	const float* __restrict__ final_Ds,
	const float* __restrict__ final_D_2s,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_ddistortion_maps,
	const float* __restrict__ dL_dnormal_maps,
	const float* __restrict__ dL_ddepth_maps,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dT0,
	float3* __restrict__ dL_dT1,
	float3* __restrict__ dL_dT3,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_dnormals
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // locate the block's gaussian num range
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // one round process block_size's gaussian, how many rounds ?
	int toDo = range.y - range.x; // this block's gaussian number

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ float3 collected_T0[BLOCK_SIZE];
	__shared__ float3 collected_T1[BLOCK_SIZE];
	__shared__ float3 collected_T3[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the product of all (1 - alpha) factors. 
	const float far_n = FAR_PLANE;
	const float near_n = NEAR_PLANE;
	const float T_final = inside ? final_Ts[pix_id] : 0;
	const float A = inside ? final_As[pix_id] : 0;
	const float D = inside ? final_Ds[pix_id] : 0;
	const float D_2 = inside ? final_D_2s[pix_id] : 0;
	
	float T = T_final;
	
	// We start from the back. The ID of the last contributing Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int max_contributor = inside ? n_contrib[pix_id + H * W] : 0;

	float accum_rec[C + 1] = { 0 };
	float accum_normal_rec[3] = {0};

	float dL_dpixel[C];
	float dL_ddistortion_map = 0;
	float dL_dnormal_map[3];
	float dL_dmax_depth = 0;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_ddistortion_map = dL_ddistortion_maps[pix_id] / (A * A + 1e-7);
		for (int i = 0; i < 3; i++)
	 		dL_dnormal_map[i] = dL_dnormal_maps[i * H * W + pix_id];
		dL_dmax_depth = dL_ddepth_maps[pix_id];
	}
		
	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_normal[3] = { 0 };

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_opacity[block.thread_rank()] = opacity[coll_id];
			collected_T0[block.thread_rank()] = T0[coll_id];
			collected_T1[block.thread_rank()] = T1[coll_id];
			collected_T3[block.thread_rank()] = T3[coll_id];
			collected_normal[block.thread_rank()] = normals[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
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

			T = T / (1.f - alpha);
			const float w = alpha * T;
			const float dchannel_dcolor = w;

			/*
			dLdis_dw_k	=	m_k^2 * [sum(w) - w_k] + [sum(w * m^2) - w_k * m_k^2] - 2 * mk * [sum(w * m) - w_k * m_k]
			dLdis_dm_k	= 	2 * w_k * m_k * [sum(w) - w_k] - 2 * w_k * [sum(w * m) - w_k * m_k]		*/
			const float dLdis_dw = dL_ddistortion_map * (m * m * A + D_2 - 2 * m * D);
			const float dLdis_dm = dL_ddistortion_map * 2 * w * (m * A - D);

			// Propagate gradients to per-Gaussian colors and keep gradients w.r.t. alpha (blending factor for a Gaussian/pixel pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				dL_dalpha += (c - accum_rec[ch]) * dL_dpixel[ch];
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dpixel[ch]); 
			}

			// accum_rec[C] = last_alpha + (1.f - last_alpha) * accum_rec[C];
			// dL_dalpha += (1 - accum_rec[C]) * dLdis_dw;

			float normal_tmp[3] = {normal.x, normal.y, normal.z};
			for (int ch = 0; ch < 3; ++ch) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal_tmp[ch];

				dL_dalpha += (normal_tmp[ch] - accum_normal_rec[ch]) * dL_dnormal_map[ch];
				atomicAdd(&dL_dnormals[global_id].x, w * dL_dnormal_map[ch]);	
			}

			dL_dalpha *= T;
			last_alpha = alpha;
			// Account for fact that alpha also influences how much of the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = opa * dL_dalpha;
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);  // Update gradients w.r.t. opacity of the Gaussian	

			float dLdis_dcam = dLdis_dm * (far_n * near_n) / ((far_n - near_n) * depth_cam * depth_cam);
			if (contributor == max_contributor-1) 
				dLdis_dcam += dL_dmax_depth;
			
			float dL_du = dLdis_dcam * T3.x;
			float dL_dv = dLdis_dcam * T3.y;
			float dL_dT30 = dLdis_dcam * s.x;
			float dL_dT31 = dLdis_dcam * s.y;
			float dL_dT32 = dLdis_dcam;

			if(dist2d < dist3d){
				const float dL_dpx = dL_dG * G * 2 * (pixf.x - xy.x);
				const float dL_dpy = dL_dG * G * 2 * (pixf.y - xy.y);
				atomicAdd(&dL_dmean2D[global_id].x, dL_dpx);			
				atomicAdd(&dL_dmean2D[global_id].y, dL_dpy);
			}
			else{
				dL_du += dL_dG * G * -s.x;
				dL_dv += dL_dG * G * -s.y;
			}
			
			const float dL_diterx = dL_du / inter_p.z;
			const float dL_ditery = dL_dv / inter_p.z;
			const float dL_diterz = -(dL_du * inter_p.x + dL_dv * inter_p.y) / (inter_p.z * inter_p.z);
			
			const float dL_dkx = dL_diterz * l.y - dL_ditery * l.z;
			const float dL_dky = dL_diterx * l.z - dL_diterz * l.x;
			const float dL_dkz = dL_ditery * l.x - dL_diterx * l.y;

			const float dL_dlx = dL_ditery * k.z - dL_diterz * k.y;
			const float dL_dly = dL_diterz * k.x - dL_diterx * k.z;
			const float dL_dlz = dL_diterx * k.y - dL_ditery * k.x;
			
			dL_dT30 += dL_dkx * pixf.x + dL_dlx * pixf.y;
			dL_dT31 += dL_dky * pixf.x + dL_dly * pixf.y;
			dL_dT32 += dL_dkz * pixf.x + dL_dlz * pixf.y;

			const float dL_dT00 = -dL_dkx;
			const float dL_dT01 = -dL_dky;
			const float dL_dT02 = -dL_dkz;
			const float dL_dT10 = -dL_dlx;
			const float dL_dT11 = -dL_dly;
			const float dL_dT12 = -dL_dlz;

			atomicAdd(&dL_dT0[global_id].x, dL_dT00);			
			atomicAdd(&dL_dT0[global_id].y, dL_dT01);		
			atomicAdd(&dL_dT0[global_id].z, dL_dT02);		
			atomicAdd(&dL_dT1[global_id].x, dL_dT10);		
			atomicAdd(&dL_dT1[global_id].y, dL_dT11);	
			atomicAdd(&dL_dT1[global_id].z, dL_dT12);	
			atomicAdd(&dL_dT3[global_id].x, dL_dT30);	
			atomicAdd(&dL_dT3[global_id].y, dL_dT31);	
			atomicAdd(&dL_dT3[global_id].z, dL_dT32);

		}
	}
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dnormal,
	float3* dL_dmean2D,
	float3* dL_dT0s,
	float3* dL_dT1s,
	float3* dL_dT3s,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		W, H,
		campos,
		(float3*)dL_dnormal,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dT0s,
		dL_dT1s,
		dL_dT3s,
		dL_dcolor,
		dL_dsh,
		dL_dscale,
		dL_drot);
}


void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		T0,
		T1,
		T3,
		normals,
		normal_map,
		opacity,
		colors,
		final_Ts,
		final_As,
		final_Ds,
		final_D_2s,
		n_contrib,
		dL_dpixels,
		dL_ddistortion_maps,
		dL_dnormal_maps,
		dL_ddepth_maps,
		dL_dmean2D,
		dL_dT0,
		dL_dT1,
		dL_dT3,
		dL_dopacity,
		dL_dcolors,
		dL_dnormals
		);
}
