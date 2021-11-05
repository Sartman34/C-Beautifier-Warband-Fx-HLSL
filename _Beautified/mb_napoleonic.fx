///////////////////////////////////////////////////////////////////////////////////
//
// Mount&Blade Warband Shaders
// You can add edit main shaders and lighting system with this file.
// You cannot change fx_configuration.h file since it holds application dependent
// configration parameters. Sorry its not well documented.
// Please send your feedbacks to our forums.
//
// All rights reserved.
// www.taleworlds.com
//
//
///////////////////////////////////////////////////////////////////////////////////
// compile_fx.bat:
// ------------------------------
// @echo off
// fxc /D PS_2_X=ps_2_a /T fx_2_0 /Fo mb_2a.fxo mb.fx
// fxc /D PS_2_X=ps_2_b /T fx_2_0 /Fo mb_2b.fxo mb.fx
// pause>nul
///////////////////////////////////////////////////////////////////////////////////

#if !defined(PS_2_X)
	#error "define high quality shader profile: PS_2_X ( ps_2_b or ps_2_a )"
#endif
#include "fx_configuration.h"
#define NUM_LIGHTS 10
#define NUM_SIMUL_LIGHTS 4
#define NUM_WORLD_MATRICES 32
#define PCF_NONE 0
#define PCF_DEFAULT 1
#define PCF_NVIDIA 2
#define VERTEX_LIGHTING_SCALER 1.0h
#define VERTEX_LIGHTING_SPECULAR_SCALER 1.0h
#define USE_PRECOMPILED_SHADER_LISTS 
#define GIVE_ERROR_HERE {\
	for(int i = 0; i < 1000; i++){\
		Output.RGBColor *= Output.RGBColor;\
	}\
}
#define GIVE_ERROR_HERE_VS {\
	for(int i = 0; i < 1000; i++){\
		Out.Pos *= Out.Pos;\
	}\
}
#ifdef NO_GAMMA_CORRECTIONS
	#define INPUT_TEX_GAMMA(col_rgb) (col_rgb) = (col_rgb)
	#define INPUT_OUTPUT_GAMMA(col_rgb) (col_rgb) = (col_rgb)
	#define OUTPUT_GAMMA(col_rgb) (col_rgb) = (col_rgb)
#else
	#define INPUT_TEX_GAMMA(col_rgb) (col_rgb) = pow((col_rgb), input_gamma.x)
	#define INPUT_OUTPUT_GAMMA(col_rgb) (col_rgb) = pow((col_rgb), output_gamma.x)
	#define OUTPUT_GAMMA(col_rgb) (col_rgb) = pow((col_rgb), output_gamma_inv.x)
#endif
#ifdef DONT_INIT_OUTPUTS
	#pragma warning(disable: 4000)
	#define INITIALIZE_OUTPUT(structure, var) structure var;
#else
	#define INITIALIZE_OUTPUT(structure, var) structure var = (structure)0;
#endif
#pragma warning(disable: 3571)
#define OUTPUT_STRUCTURES 
#define FUNCTIONS 
#define PER_MESH_CONSTANTS 
#define PER_FRAME_CONSTANTS 
#define PER_SCENE_CONSTANTS 
#define APPLICATION_CONSTANTS 
#define MISC_SHADERS 
#define UI_SHADERS 
#define SHADOW_RELATED_SHADERS 
#define WATER_SHADERS 
#define SKYBOX_SHADERS 
#define HAIR_SHADERS 
#define FACE_SHADERS 
#define FLORA_SHADERS 
#define MAP_SHADERS 
#define SOFT_PARTICLE_SHADERS 
#define STANDART_SHADERS 
#define STANDART_RELATED_SHADER 
#define OCEAN_SHADERS 
#ifdef USE_NEW_TREE_SYSTEM
	#define NEWTREE_SHADERS 
#endif
#ifdef CLOTH_SHADERS
	texture PositionTexture;
	sampler PositionSampler: register(s0) = sampler_state{
		Texture =  < PositionTexture > ;
		AddressU = CLAMP;
		AddressV = CLAMP;
		MinFilter = POINT;
		MagFilter = POINT;
	};
	texture PrevPositionTexture;
	sampler PrevPositionSampler: register(s1) = sampler_state{
		Texture =  < PrevPositionTexture > ;
		AddressU = CLAMP;
		AddressV = CLAMP;
		MinFilter = POINT;
		MagFilter = POINT;
	};
	texture NormalTexture;
	sampler NormalSampler: register(s2) = sampler_state{
		Texture =  < NormalTexture > ;
		AddressU = CLAMP;
		AddressV = CLAMP;
		MinFilter = LINEAR;
		MagFilter = LINEAR;
	};
#endif
#ifdef PER_MESH_CONSTANTS
	cbuffer SkinningMatrixConstants{
		float4x4 matWorldArray[NUM_WORLD_MATRICES]: WORLDMATRIXARRAY;
	}
	cbuffer MeshConstants{
		float4x4 matWorld;
		float4 vMaterialColor = float4(255.f / 255.f, 230.f / 255.f, 200.f / 255.f, 1.0f);
		float4 vMaterialColor2;
		float spec_coef = 1.0f;
		float4 vSpecularColor = float4(5, 5, 5, 5);
		float fMaterialPower = 16.f;
	}
#endif
#ifdef PER_FRAME_CONSTANTS
	cbuffer FrameConstants{
		float time_var = 1.0f;
	}
	cbuffer ViewConstants{
		float4x4 matWorldViewProj;
		float4x4 matViewProj;
		float4 vCameraPos;
		float4x4 matSunViewProj;
	}
	cbuffer WaterViewConstants{
		float4x4 matWaterViewProj;
		float4x4 matWaterWorldViewProj;
		float4 texture_offset = {
			0, 0, 0, 0
		};
	}
#endif
#ifdef PER_SCENE_CONSTANTS
	cbuffer SceneConstants{
		float vFloraWindStrength = 0.14f;
		float vWaterWindStrength = 0.14f;
		float4 vAmbientColor = float4(64.f / 255.f, 64.f / 255.f, 64.f / 255.f, 1.0f);
		float4 vGroundAmbientColor = float4(84.f / 255.f, 44.f / 255.f, 54.f / 255.f, 1.0f);
		float3 vSkyLightDir;
		float4 vSkyLightColor;
		float3 vSunDir;
		float4 vSunColor;
		float fFogDensity = 0.05f;
		float reflection_factor;
	};
#endif
#ifdef APPLICATION_CONSTANTS
	cbuffer ApplicationConstants{
		static const float input_gamma = 2.2f;
		bool use_depth_effects = false;
		float fShadowMapSize = 4096;
		float fShadowMapNextPixel = 1.0f / 4096;
		float4 output_gamma = float4(2.2f, 2.2f, 2.2f, 2.2f);
		float4 output_gamma_inv = float4(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f);
		float far_clip_Inv;
		float4 vDepthRT_HalfPixel_ViewportSizeInv;
		static const float map_normal_detail_factor = 1.4f;
		static const float uv_2_scale = 1.237;
		static const float fShadowBias = 0.00002f;
		#ifdef USE_NEW_TREE_SYSTEM
			float flora_detail = 40.0f;
			#define flora_detail_fade (flora_detail * FLORA_DETAIL_FADE_MUL)
			#define flora_detail_fade_inv (flora_detail - flora_detail_fade)
			#define flora_detail_clip (max(0, flora_detail_fade - 20.0f))
		#endif
		float4 debug_vector = {
			0, 0, 0, 1
		};
	};
	cbuffer UnusedConstants{
		int iLightPointCount;
		int iLightIndices[NUM_SIMUL_LIGHTS] = {
			0, 1, 2, 3
		};
		float3 vLightPosDir[NUM_LIGHTS];
		float4 vLightDiffuse[NUM_LIGHTS];
		float4 vPointLightColor;
		float4x4 matView;
		float4x4 matWorldView;
		bool bUseMotionBlur = false;
		float4x4 matMotionBlur;
		#ifdef USE_LIGHTING_PASS
			int light_count_this_pass = 1;
			float4 g_vPointLightPosXYZ_InvRadius[MAX_LIGHTS_PER_PASS];
			float4 g_vPointLightColor[MAX_LIGHTS_PER_PASS];
		#endif
	}
#endif
#if defined(USE_SHARED_DIFFUSE_MAP) || !defined(USE_DEVICE_TEXTURE_ASSIGN)
	texture diffuse_texture;
#endif
#ifndef USE_DEVICE_TEXTURE_ASSIGN
	texture diffuse_texture_2;
	texture specular_texture;
	texture normal_texture;
	texture env_texture;
	texture shadowmap_texture;
	texture cubic_texture;
	texture depth_texture;
	texture screen_texture;
	#ifdef USE_REGISTERED_SAMPLERS
		sampler ReflectionTextureSampler: register(fx_ReflectionTextureSampler_RegisterS) = sampler_state{
			Texture = env_texture;
		};
		sampler EnvTextureSampler: register(fx_EnvTextureSampler_RegisterS) = sampler_state{
			Texture = env_texture;
		};
		sampler Diffuse2Sampler: register(fx_Diffuse2Sampler_RegisterS) = sampler_state{
			Texture = diffuse_texture_2;
		};
		sampler NormalTextureSampler: register(fx_NormalTextureSampler_RegisterS) = sampler_state{
			Texture = normal_texture;
		};
		sampler SpecularTextureSampler: register(fx_SpecularTextureSampler_RegisterS) = sampler_state{
			Texture = specular_texture;
		};
		sampler DepthTextureSampler: register(fx_DepthTextureSampler_RegisterS) = sampler_state{
			Texture = depth_texture;
		};
		sampler CubicTextureSampler: register(fx_CubicTextureSampler_RegisterS) = sampler_state{
			Texture = cubic_texture;
		};
		sampler ShadowmapTextureSampler: register(fx_ShadowmapTextureSampler_RegisterS) = sampler_state{
			Texture = shadowmap_texture;
		};
		sampler ScreenTextureSampler: register(fx_ScreenTextureSampler_RegisterS) = sampler_state{
			Texture = screen_texture;
		};
		sampler MeshTextureSampler: register(fx_MeshTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler ClampedTextureSampler: register(fx_ClampedTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler FontTextureSampler: register(fx_FontTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler CharacterShadowTextureSampler: register(fx_CharacterShadowTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler MeshTextureSamplerNoFilter: register(fx_MeshTextureSamplerNoFilter_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler DiffuseTextureSamplerNoWrap: register(fx_DiffuseTextureSamplerNoWrap_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler GrassTextureSampler: register(fx_GrassTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
	#else
		sampler ReflectionTextureSampler = sampler_state{
			Texture = env_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler EnvTextureSampler = sampler_state{
			Texture = env_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler Diffuse2Sampler = sampler_state{
			Texture = diffuse_texture_2;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler NormalTextureSampler = sampler_state{
			Texture = normal_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler SpecularTextureSampler = sampler_state{
			Texture = specular_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler DepthTextureSampler = sampler_state{
			Texture = depth_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler CubicTextureSampler = sampler_state{
			Texture = cubic_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler ShadowmapTextureSampler = sampler_state{
			Texture = shadowmap_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = NONE;
			MagFilter = NONE;
		};
		sampler ScreenTextureSampler = sampler_state{
			Texture = screen_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler MeshTextureSampler = sampler_state{
			Texture = diffuse_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler ClampedTextureSampler = sampler_state{
			Texture = diffuse_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler FontTextureSampler = sampler_state{
			Texture = diffuse_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler CharacterShadowTextureSampler = sampler_state{
			Texture = diffuse_texture;
			AddressU = BORDER;
			AddressV = BORDER;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler MeshTextureSamplerNoFilter = sampler_state{
			Texture = diffuse_texture;
			AddressU = WRAP;
			AddressV = WRAP;
			MinFilter = NONE;
			MagFilter = NONE;
		};
		sampler DiffuseTextureSamplerNoWrap = sampler_state{
			Texture = diffuse_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
		sampler GrassTextureSampler = sampler_state{
			Texture = diffuse_texture;
			AddressU = CLAMP;
			AddressV = CLAMP;
			MinFilter = LINEAR;
			MagFilter = LINEAR;
		};
	#endif
#else
	sampler ReflectionTextureSampler: register(fx_ReflectionTextureSampler_RegisterS);
	sampler EnvTextureSampler: register(fx_EnvTextureSampler_RegisterS);
	sampler Diffuse2Sampler: register(fx_Diffuse2Sampler_RegisterS);
	sampler NormalTextureSampler: register(fx_NormalTextureSampler_RegisterS);
	sampler SpecularTextureSampler: register(fx_SpecularTextureSampler_RegisterS);
	sampler DepthTextureSampler: register(fx_DepthTextureSampler_RegisterS);
	sampler CubicTextureSampler: register(fx_CubicTextureSampler_RegisterS);
	sampler ShadowmapTextureSampler: register(fx_ShadowmapTextureSampler_RegisterS);
	sampler ScreenTextureSampler: register(fx_ScreenTextureSampler_RegisterS);
	#ifdef USE_SHARED_DIFFUSE_MAP
		sampler MeshTextureSampler: register(fx_MeshTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler ClampedTextureSampler: register(fx_ClampedTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler FontTextureSampler: register(fx_FontTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler CharacterShadowTextureSampler: register(fx_CharacterShadowTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler MeshTextureSamplerNoFilter: register(fx_MeshTextureSamplerNoFilter_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler DiffuseTextureSamplerNoWrap: register(fx_DiffuseTextureSamplerNoWrap_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
		sampler GrassTextureSampler: register(fx_GrassTextureSampler_RegisterS) = sampler_state{
			Texture = diffuse_texture;
		};
	#else
		sampler MeshTextureSampler: register(fx_MeshTextureSampler_RegisterS);
		sampler ClampedTextureSampler: register(fx_ClampedTextureSampler_RegisterS);
		sampler FontTextureSampler: register(fx_FontTextureSampler_RegisterS);
		sampler CharacterShadowTextureSampler: register(fx_CharacterShadowTextureSampler_RegisterS);
		sampler MeshTextureSamplerNoFilter: register(fx_MeshTextureSamplerNoFilter_RegisterS);
		sampler DiffuseTextureSamplerNoWrap: register(fx_DiffuseTextureSamplerNoWrap_RegisterS);
		sampler GrassTextureSampler: register(fx_GrassTextureSampler_RegisterS);
	#endif
#endif
#ifdef OUTPUT_STRUCTURES
	struct PS_OUTPUT{
		half4 RGBColor: COLOR;
	};
#endif
#ifdef FUNCTIONS
	half GetSunAmount(uniform const int PcfMode, half3 ShadowTexCoord, half2 ShadowTexelPos){
		half sun_amount;
		if(PcfMode == PCF_NVIDIA){
			sun_amount = tex2Dproj(ShadowmapTextureSampler, half4(ShadowTexCoord, 1.0h)).r;
		}
		else {
			half2 lerps = frac(ShadowTexelPos);
			half sourcevals[4];
			sourcevals[0] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord.xy).r < ShadowTexCoord.z) ? 0.0h: 1.0h;
			sourcevals[1] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord.xy + half2(fShadowMapNextPixel, 0)).r < ShadowTexCoord.z) ? 0.0h: 1.0h;
			sourcevals[2] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord.xy + half2(0, fShadowMapNextPixel)).r < ShadowTexCoord.z) ? 0.0h: 1.0h;
			sourcevals[3] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord.xy + half2(fShadowMapNextPixel, fShadowMapNextPixel)).r < ShadowTexCoord.z) ? 0.0h: 1.0h;
			sun_amount = lerp(lerp(sourcevals[0], sourcevals[1], lerps.x), lerp(sourcevals[2], sourcevals[3], lerps.x), lerps.y);
		}
		return sun_amount;
	}
	half GetSunAmountNvidia(half3 ShadowTexCoord){
		return tex2Dproj(ShadowmapTextureSampler, half4(ShadowTexCoord, 1.0h)).r;
	}
	float get_fog_amount(float d){
		return 1.0f / exp2(d * fFogDensity);
	}
	float get_fog_amount_new(float d, float wz){
		return get_fog_amount(d);
	}
	static const float2 specularShift = float2(0.138 - 0.5, 0.254 - 0.5);
	static const float2 specularExp = float2(256.0, 32.0) * 0.7;
	static const float3 specularColor0 = float3(0.9, 1.0, 1.0) * 0.898 * 0.99;
	static const float3 specularColor1 = float3(1.0, 0.9, 1.0) * 0.74 * 0.99;
	float HairSingleSpecularTerm(half3 T, half3 H, float exponent){
		float dotTH = dot(T, H);
		float sinTH = sqrt(1.0 - dotTH * dotTH);
		return pow(sinTH, exponent);
	}
	half3 ShiftTangent(float3 T, half3 N, float shiftAmount){
		return normalize(T + shiftAmount * N);
	}
	half3 calculate_hair_specular(half3 normal, float3 tangent, half3 lightVec, half3 viewVec, float2 tc){
		half shiftTex = tex2D(Diffuse2Sampler, tc).a;
		half3 T1 = ShiftTangent(tangent, normal, specularShift.x + shiftTex);
		half3 T2 = ShiftTangent(tangent, normal, specularShift.y + shiftTex);
		half3 H = normalize(lightVec + viewVec);
		half3 specular = vSunColor.xyz * specularColor0 * HairSingleSpecularTerm(T1, H, specularExp.x);
		half3 specular2 = vSunColor.xyz * specularColor1 * HairSingleSpecularTerm(T2, H, specularExp.y);
		half specularMask = tex2D(Diffuse2Sampler, tc * 10.0f).a;
		specular2 *= specularMask;
		half specularAttenuation = saturate(1.75 * dot(normal, lightVec) + 0.25);
		specular = (specular + specular2) * specularAttenuation;
		return specular;
	}
	half HairDiffuseTerm(half3 N, half3 L){
		return saturate(0.75 * dot(N, L) + 0.25);
	}
	half face_NdotL(half3 n, half3 l){
		half wNdotL = dot(n.xyz, l.xyz);
		return saturate(max(0.2h * (wNdotL + 0.9h), wNdotL));
	}
	half4 calculate_point_lights_diffuse(const float3 vWorldPos, const half3 vWorldN, const bool face_like_NdotL){
		half4 total = 0;
		[loop]for(int j = 0; j < iLightPointCount; j++){
			int i = iLightIndices[j];
			half3 point_to_light = vLightPosDir[i] - vWorldPos;
			half LD = dot(point_to_light, point_to_light);
			half wNdotL = dot(vWorldN, point_to_light);
			half fAtten = VERTEX_LIGHTING_SCALER / LD;
			if(face_like_NdotL){
				total += max(0.2h * (wNdotL + 0.9h), wNdotL) * vLightDiffuse[i] * fAtten;
			}
			else {
				total += saturate(wNdotL) * vLightDiffuse[i] * fAtten;
			}
		}
		return total;
	}
	half3 calculate_point_lights_diffuse_ex_1(const float3 vWorldPos, const half3 vWorldN, const bool face_like_NdotL){
		half3 total = 0;
		[loop]for(int j = 0; j < iLightPointCount; j++){
			int i = iLightIndices[j];
			half3 point_to_light = vLightPosDir[i] - vWorldPos;
			half LD = dot(point_to_light, point_to_light);
			half wNdotL = dot(vWorldN, point_to_light);
			half fAtten = VERTEX_LIGHTING_SCALER / LD;
			if(face_like_NdotL){
				total += max(0.2h * (wNdotL + 0.9h), wNdotL) * vLightDiffuse[i].rgb * fAtten;
			}
			else {
				total += saturate(wNdotL) * vLightDiffuse[i].rgb * fAtten;
			}
		}
		return total;
	}
	half3 get_ambientTerm(uint ambientTermType, half3 normal, half3 DirToSky, half sun_amount){
		half3 ambientTerm;
		if(ambientTermType == 0){
			ambientTerm = vAmbientColor.rgb;
		}
		else if(ambientTermType == 1){
			half3 g_vGroundColorTEMP = vGroundAmbientColor.rgb * sun_amount;
			half lerpFactor = (dot(normal, DirToSky) + 1.0h) * 0.5h;
			ambientTerm = lerp(g_vGroundColorTEMP, vAmbientColor.rgb, lerpFactor);
		}
		return ambientTerm;
	}
	float4x4 build_instance_frame_matrix(float3 vInstanceData0, float3 vInstanceData1, float3 vInstanceData2, float3 vInstanceData3){
		const float3 position = vInstanceData0.xyz;
		float3 frame_s = vInstanceData1;
		float3 frame_f = vInstanceData2;
		float3 frame_u = vInstanceData3;
		float4x4 matWorldOfInstance = {
			float4(frame_s.x, frame_f.x, frame_u.x, position.x), float4(frame_s.y, frame_f.y, frame_u.y, position.y), float4(frame_s.z, frame_f.z, frame_u.z, position.z), float4(0.0f, 0.0f, 0.0f, 1.0f)
		};
		return matWorldOfInstance;
	}
	float4 skinning_deform(float4 vPosition, float4 vBlendWeights, float4 vBlendIndices){
		return mul(matWorldArray[vBlendIndices.x], vPosition) * vBlendWeights.x + mul(matWorldArray[vBlendIndices.y], vPosition) * vBlendWeights.y + mul(matWorldArray[vBlendIndices.z], vPosition) * vBlendWeights.z + mul(matWorldArray[vBlendIndices.w], vPosition) * vBlendWeights.w;
	}
	#define DEFINE_TECHNIQUES(tech_name, vs_name, ps_name) \
		technique tech_name{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_name(PCF_NONE);\
				PixelShader = compile ps_2_0 ps_name(PCF_NONE);\
			}\
		}\
		technique tech_name##_SHDW{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_name(PCF_DEFAULT);\
				PixelShader = compile ps_2_0 ps_name(PCF_DEFAULT);\
			}\
		}\
		technique tech_name##_SHDWNVIDIA{\
			pass P0{\
				VertexShader = compile vs_2_a vs_name(PCF_NVIDIA);\
				PixelShader = compile ps_2_a ps_name(PCF_NVIDIA);\
			}\
		}
	#define DEFINE_TECHNIQUES_HIGH(tech_name, vs_name, ps_name) \
		technique tech_name{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_name(PCF_NONE);\
				PixelShader = compile PS_2_X ps_name(PCF_NONE);\
			}\
		}\
		technique tech_name##_SHDW{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_name(PCF_DEFAULT);\
				PixelShader = compile PS_2_X ps_name(PCF_DEFAULT);\
			}\
		}\
		technique tech_name##_SHDWNVIDIA{\
			pass P0{\
				VertexShader = compile vs_2_a vs_name(PCF_NVIDIA);\
				PixelShader = compile ps_2_a ps_name(PCF_NVIDIA);\
			}\
		}
#endif
#ifdef USE_LIGHTING_PASS
	struct VS_OUTPUT_LIGTING{
		float4 Pos: POSITION;
		half4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 WorldPos: TEXCOORD1;
		float3 ViewDir: TEXCOORD2;
		half3 WorldNormal: TEXCOORD3;
		#ifdef USE_WORLD_SPACE_LIGHTING
			half3 WorldTangent: TEXCOORD4;
			half3 WorldBinormal: TEXCOORD5;
		#else
			half3 LightDir1: TEXCOORD4;
			half3 LightDir2: TEXCOORD5;
			half3 LightDir3: TEXCOORD6;
		#endif
	};
	VS_OUTPUT_LIGTING vs_main_standart_light(uniform const int use_bumpmap, uniform const int use_skinning, uniform const int use_specularfactor, float4 vPosition: POSITION, float2 tc: TEXCOORD0, float3 vNormal: NORMAL, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_LIGTING, Out);
		float4 vObjectPos;
		half3 vObjectN, vObjectT, vObjectB;
		#ifdef USE_WORLD_SPACE_LIGHTING
			vObjectT = half3(1, 0, 0);
			vObjectB = half3(0, 1, 0);
		#endif
		if(use_skinning){
			vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
			vObjectN = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vNormal) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vNormal) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vNormal) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vNormal) * vBlendWeights.w);
			if(use_bumpmap){
				vObjectT = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vTangent) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vTangent) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vTangent) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vTangent) * vBlendWeights.w);
				vObjectB = (cross(vObjectN, vObjectT));
				bool left_handed = (dot(cross(vNormal, vTangent), vBinormal) < 0.0h);
				if(left_handed){
					vObjectB = -vObjectB;
				}
			}
		}
		else{
			vObjectPos = vPosition;
			vObjectN = vNormal;
			if(use_bumpmap){
				vObjectT = vTangent;
				vObjectB = vBinormal;
			}
		}
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		Out.Tex0 = tc;
		float4 vWorldPos = (float4)mul(matWorld, vObjectPos);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vObjectN));
		half3 vWorldB = normalize(mul((float3x3)matWorld, vObjectB));
		half3 vWorldT = normalize(mul((float3x3)matWorld, vObjectT));
		if(use_bumpmap){
			half3x3 TBNMatrix = half3x3(vWorldT, vWorldB, vWorldN);
			if(use_specularfactor){
				Out.ViewDir = (mul(TBNMatrix, normalize(vCameraPos.xyz - vWorldPos.xyz)));
			}
		}
		else {
			if(use_specularfactor){
				Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			}
		}
		Out.WorldPos = vWorldPos;
		Out.VertexColor = vVertexColor;
		Out.WorldNormal = vWorldN;
		#ifdef USE_WORLD_SPACE_LIGHTING
			Out.WorldTangent = vWorldT;
			Out.WorldBinormal = vWorldB;
		#else
			{
				Out.LightDir1 = g_vPointLightPosXYZ_InvRadius[0].a * (g_vPointLightPosXYZ_InvRadius[0].xyz - vWorldPos);
				Out.LightDir2 = g_vPointLightPosXYZ_InvRadius[1].a * (g_vPointLightPosXYZ_InvRadius[1].xyz - vWorldPos);
				Out.LightDir3 = g_vPointLightPosXYZ_InvRadius[2].a * (g_vPointLightPosXYZ_InvRadius[2].xyz - vWorldPos);
				if(use_bumpmap){
					half3x3 TBNMatrix = half3x3(vWorldT, vWorldB, vWorldN);
					Out.LightDir1 = mul(TBNMatrix, Out.LightDir1);
					Out.LightDir2 = mul(TBNMatrix, Out.LightDir2);
					Out.LightDir3 = mul(TBNMatrix, Out.LightDir3);
				}
			}
		#endif
		return Out;
	}
	float calculate_cubic_shadow(uniform const int use_shadows, float3 point_to_light){
		float total_shadow = 0;
		static const float invSize = 1.0f / 512;
		if(use_shadows == 3){
			float3 samples[6] = {
				invSize, 0, 0, -invSize, 0, 0, 0, invSize, 0, 0, -invSize, 0, 0, 0, invSize, 0, 0, -invSize
			};
			float light_len = length(point_to_light);
			static const float sample_radius = 0.314f;
			for(int i = 0; i < 6; i++){
				float3 tc = point_to_light + samples[i] * sample_radius;
				float shadow_len = texCUBE(CubicTextureSampler, tc).x;
				total_shadow += float(light_len < shadow_len);
			}
			total_shadow /= 6;
		}
		else if(use_shadows == 2){
			float3 samples[4] = {
				invSize, 0, 0, -invSize, 0, 0, 0, invSize, 0, 0, -invSize, 0
			};
			float light_len = length(point_to_light);
			static const float sample_radius = 0.314f;
			for(int i = 0; i < 4; i++){
				float3 tc = point_to_light + samples[i] * sample_radius;
				float shadow_len = texCUBE(CubicTextureSampler, tc).x;
				total_shadow += float(light_len < shadow_len);
			}
			total_shadow *= 0.2f;
		}
		else if(use_shadows == 1){
			float shadow_len = texCUBE(CubicTextureSampler, point_to_light).x;
			float light_len = length(point_to_light);
			total_shadow = float(light_len < shadow_len);
		}
		else {
			total_shadow = 1;
		}
		return total_shadow;
	}
	half4 calculate_light_factor(half4 light_color, float3 point_to_light, float3 point_to_light_TBN, half3 normal, float3 viewdir, half4 tex_col, half4 specColor, uniform const int use_specularfactor){
		half4 final_color;
		half lVec_d2 = dot(point_to_light, point_to_light);
		half lightAtten = saturate(1.0 - lVec_d2);
		static const int use_shadows = 0;
		float total_shadow = calculate_cubic_shadow(use_shadows, point_to_light);
		final_color.rgb = total_shadow * light_color;
		final_color.rgb *= tex_col.rgb;
		half diffuseTerm = saturate(dot(point_to_light_TBN, normal));
		final_color.rgb *= diffuseTerm;
		if(use_specularfactor){
			half4 light_specColor = specColor * light_color;
			half3 vHalf = normalize(viewdir + point_to_light_TBN);
			half3 specularTerm = light_specColor.rgb * pow(saturate(dot(vHalf, normal)), fMaterialPower);
			final_color.rgb += specularTerm;
		}
		final_color.a = lightAtten;
		return final_color;
	}
	PS_OUTPUT ps_main_standart_light(VS_OUTPUT_LIGTING In, uniform const int light_count, uniform const int use_dxt5, uniform const int use_bumpmap, uniform const int use_specularfactor, uniform const int use_specularmap){
		PS_OUTPUT Output;
		float3 world_pos = In.WorldPos;
		float3 viewdir = In.ViewDir;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		half3 normal;
		if(use_bumpmap){
			if(use_dxt5){
				normal.xy = (2.0h * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0h);
				normal.z = sqrt(1.0h - dot(normal.xy, normal.xy));
			}
			else {
				normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0h);
			}
		}
		else {
			if(use_dxt5){
				GIVE_ERROR_HERE;
			}
			normal = In.WorldNormal;
		}
		half4 specColor;
		if(use_specularfactor){
			specColor = 0.1 * spec_coef * vSpecularColor;
			if(use_specularmap){
				half spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0).rgb, 0.33);
				specColor *= spec_tex_factor;
			}
			else {
				specColor *= tex_col.a;
			}
		}
		else if(use_specularmap){
			GIVE_ERROR_HERE;
		}
		#ifdef USE_WORLD_SPACE_LIGHTING
			half3x3 TBNMatrix = half3x3(In.WorldTangent, In.WorldBinormal, In.WorldNormal);
		#endif
		Output.RGBColor.rgb = 0.0h;
		Output.RGBColor.a = 1.0h;
		for(int il = 0; il < light_count; il++){
			if(g_vPointLightPosXYZ_InvRadius[il].a == 0.0f)break;
			float3 point_to_light = g_vPointLightPosXYZ_InvRadius[il].a * (g_vPointLightPosXYZ_InvRadius[il].xyz - world_pos);
			#ifdef USE_WORLD_SPACE_LIGHTING
				float3 point_to_light_TBN = mul(TBNMatrix, point_to_light);
			#else
				float3 point_to_light_TBN = (il == 0) ? In.LightDir1.xyz: ((il == 1) ? In.LightDir2.xyz: In.LightDir3.xyz);
			#endif
			half4 light_color = g_vPointLightColor[il];
			float4 cur_light_factor = calculate_light_factor(light_color, point_to_light, point_to_light_TBN, normal, viewdir, tex_col, specColor, use_specularfactor);
			half3 cur_light = cur_light_factor.rgb;
			half atten = cur_light_factor.a;
			Output.RGBColor.rgb += cur_light * atten * 0.02;
		}
		if(false){
			Output.RGBColor.rgb *= tex_col.a;
		}
		return Output;
	}
	#ifdef USE_PRECOMPILED_SHADER_LISTS
		VertexShader light_vertex_shaders[] = {
			compile vs_2_0 vs_main_standart_light(0, 0, 0), compile vs_2_0 vs_main_standart_light(0, 0, 1), compile vs_2_0 vs_main_standart_light(0, 1, 0), compile vs_2_0 vs_main_standart_light(0, 1, 1), compile vs_2_0 vs_main_standart_light(1, 0, 0), compile vs_2_0 vs_main_standart_light(1, 0, 1), compile vs_2_0 vs_main_standart_light(1, 1, 0), compile vs_2_0 vs_main_standart_light(1, 1, 1)
		};
		PixelShader one_light_pixel_shaders[] = {
			compile ps_2_0 ps_main_standart_light(1, 0, 0, 0, 0), NULL, compile ps_2_0 ps_main_standart_light(1, 0, 0, 1, 0), compile ps_2_0 ps_main_standart_light(1, 0, 0, 1, 1), compile ps_2_0 ps_main_standart_light(1, 0, 1, 0, 0), NULL, compile ps_2_0 ps_main_standart_light(1, 0, 1, 1, 0), compile ps_2_0 ps_main_standart_light(1, 0, 1, 1, 1), compile ps_2_0 ps_main_standart_light(1, 1, 0, 0, 0), NULL, compile ps_2_0 ps_main_standart_light(1, 1, 0, 1, 0), compile ps_2_0 ps_main_standart_light(1, 1, 0, 1, 1), compile ps_2_0 ps_main_standart_light(1, 1, 1, 0, 0), NULL, compile ps_2_0 ps_main_standart_light(1, 1, 1, 1, 0), compile ps_2_0 ps_main_standart_light(1, 1, 1, 1, 1)
		};
		PixelShader two_light_pixel_shaders[] = {
			compile PS_2_X ps_main_standart_light(2, 0, 0, 0, 0), NULL, compile PS_2_X ps_main_standart_light(2, 0, 0, 1, 0), compile PS_2_X ps_main_standart_light(2, 0, 0, 1, 1), compile PS_2_X ps_main_standart_light(2, 0, 1, 0, 0), NULL, compile PS_2_X ps_main_standart_light(2, 0, 1, 1, 0), compile PS_2_X ps_main_standart_light(2, 0, 1, 1, 1), compile PS_2_X ps_main_standart_light(2, 1, 0, 0, 0), NULL, compile PS_2_X ps_main_standart_light(2, 1, 0, 1, 0), compile PS_2_X ps_main_standart_light(2, 1, 0, 1, 1), compile PS_2_X ps_main_standart_light(2, 1, 1, 0, 0), NULL, compile PS_2_X ps_main_standart_light(2, 1, 1, 1, 0), compile PS_2_X ps_main_standart_light(2, 1, 1, 1, 1)
		};
		#define DEFINE_LIGHTING_TECHNIQUE(tech_name, use_dxt5, use_bumpmap, use_skinning, use_specularfactor, use_specularmap) \
			technique tech_name##_LIGHT1{\
				pass P0{\
					VertexShader = light_vertex_shaders[(4 * use_bumpmap) + (2 * use_skinning) + use_specularfactor];\
					PixelShader = one_light_pixel_shaders[(8 * use_dxt5) + (4 * use_bumpmap) + (2 * use_specularfactor) + use_specularmap];\
				}\
			}\
			technique tech_name##_LIGHT2{\
				pass P0{\
					VertexShader = light_vertex_shaders[(4 * use_bumpmap) + (2 * use_skinning) + use_specularfactor];\
					PixelShader = two_light_pixel_shaders[(8 * use_dxt5) + (4 * use_bumpmap) + (2 * use_specularfactor) + use_specularmap];\
				}\
			}
	#else
		#define DEFINE_LIGHTING_TECHNIQUE(tech_name, use_dxt5, use_bumpmap, use_skinning, use_specularfactor, use_specularmap) \
			technique tech_name##_LIGHT1{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_light(use_bumpmap, use_skinning, use_specularfactor);\
					PixelShader = compile ps_2_0 ps_main_standart_light(1, use_dxt5, use_bumpmap, use_specularfactor, use_specularmap);\
				}\
			}\
			technique tech_name##_LIGHT2{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_light(use_bumpmap, use_skinning, use_specularfactor);\
					PixelShader = compile PS_2_X ps_main_standart_light(2, use_dxt5, use_bumpmap, use_specularfactor, use_specularmap);\
				}\
			}
	#endif
#else
	#define DEFINE_LIGHTING_TECHNIQUE(tech_name, use_dxt5, use_bumpmap, use_skinning, use_specularfactor, use_specularmap) 
#endif
#ifdef MISC_SHADERS
	struct VS_OUTPUT_FONT{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	VS_OUTPUT_FONT vs_font(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_OUTPUT_FONT Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VertexShader vs_font_compiled_2_0 = compile vs_2_0 vs_font();
	struct VS_OUTPUT_NOTEXTURE{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float Fog: FOG;
	};
	VS_OUTPUT_NOTEXTURE vs_main_notexture(float4 vPosition: POSITION, float4 vColor: COLOR){
		VS_OUTPUT_NOTEXTURE Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_notexture(VS_OUTPUT_NOTEXTURE In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		return Output;
	}
	technique notexture{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_notexture();
			PixelShader = compile ps_2_0 ps_main_notexture();
		}
	}
	struct VS_OUTPUT_CLEAR_FLOATING_POINT_BUFFER{
		float4 Pos: POSITION;
	};
	VS_OUTPUT_CLEAR_FLOATING_POINT_BUFFER vs_clear_floating_point_buffer(float4 vPosition: POSITION){
		VS_OUTPUT_CLEAR_FLOATING_POINT_BUFFER Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		return Out;
	}
	PS_OUTPUT ps_clear_floating_point_buffer(){
		PS_OUTPUT Out;
		Out.RGBColor = half4(0.0h, 0.0h, 0.0h, 0.0h);
		return Out;
	}
	technique clear_floating_point_buffer{
		pass P0{
			VertexShader = compile vs_2_0 vs_clear_floating_point_buffer();
			PixelShader = compile ps_2_0 ps_clear_floating_point_buffer();
		}
	}
	struct VS_OUTPUT_FONT_X{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	VS_OUTPUT_FONT_X vs_main_no_shadow(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		VS_OUTPUT_FONT_X Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor + vLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSunDir)) * vSunColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_no_shadow(VS_OUTPUT_FONT_X In){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = In.Color * tex_col;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_main_no_shadow_no_alpha(VS_OUTPUT_FONT_X In){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSamplerNoFilter, In.Tex0);
		tex_col.a = 1.0h;
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = In.Color * tex_col;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_simple_no_filtering(VS_OUTPUT_FONT_X In){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSamplerNoFilter, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = In.Color * tex_col;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_no_shading(VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor *= tex2D(MeshTextureSampler, In.Tex0);
		return Output;
	}
	PS_OUTPUT ps_no_shading_no_alpha(VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor *= tex2D(MeshTextureSamplerNoFilter, In.Tex0);
		Output.RGBColor.a = 1.0h;
		return Output;
	}
	float CalcPennonAnimation(float3 Offset){
		const float curtime = (time_var * 9);
		return float(sin(curtime + Offset.z + (Offset.y - Offset.x)) * (Offset.x * 0.33));
	}
	float CalcPennonVerticalAnimation(float3 Offset){
		const float curtime = (time_var * 9);
		return float(sin(curtime + Offset.z + (Offset.z - Offset.x)));
	}
	float CalcFlagAnimation(float3 Offset){
		const float curtime = (time_var * 4);
		return float(sin(curtime + Offset.z + (Offset.y - Offset.x)) * (Offset.x * 0.15));
	}
	float CalcCTFPennonAnimation(float3 Offset){
		const float curtime = (time_var * 8);
		return float(sin(curtime + Offset.z + (Offset.y - Offset.x)) * (Offset.y * 0.3));
	}
	float3 CalcSailAnimation(float3 Offset){
		const float grandwave = sin(time_var * 0.2f);
		const float curtime = (time_var * 2.5);
		float xmul = (Offset.x * (vFloraWindStrength * 0.08f));
		float value = (sin(curtime + Offset.z + (Offset.y - Offset.x)) * xmul);
		value *= grandwave;
		return float3(value, value, value);
	}
	technique diffuse_no_shadow{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_no_shadow();
			PixelShader = compile ps_2_0 ps_main_no_shadow();
		}
	}
	technique diffuse_no_shadow_no_alpha{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_no_shadow();
			PixelShader = compile ps_2_0 ps_main_no_shadow_no_alpha();
		}
	}
	technique simple_shading{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_main_no_shadow();
		}
	}
	technique simple_shading_no_filter{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_simple_no_filtering();
		}
	}
	technique no_shading{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_no_shading();
		}
	}
	technique no_shading_no_alpha{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_no_shading_no_alpha();
		}
	}
#endif
#ifdef UI_SHADERS
	PS_OUTPUT ps_font_uniform_color(VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor.a *= tex2D(FontTextureSampler, In.Tex0).a;
		return Output;
	}
	PS_OUTPUT ps_font_background(VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		Output.RGBColor.a = 1.0h;
		Output.RGBColor.rgb = tex2D(FontTextureSampler, In.Tex0).rgb + In.Color.rgb;
		return Output;
	}
	PS_OUTPUT ps_font_outline(VS_OUTPUT_FONT In){
		half4 sample = tex2D(FontTextureSampler, In.Tex0);
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor.a = (1.0h - sample.r) + sample.a;
		Output.RGBColor.rgb *= sample.a + 0.05h;
		Output.RGBColor = saturate(Output.RGBColor);
		return Output;
	}
	technique font_uniform_color{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_font_uniform_color();
		}
	}
	technique font_background{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_font_background();
		}
	}
	technique font_outline{
		pass P0{
			VertexShader = vs_font_compiled_2_0;
			PixelShader = compile ps_2_0 ps_font_outline();
		}
	}
#endif
#ifdef SHADOW_RELATED_SHADERS
	struct VS_OUTPUT_SHADOWMAP{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
		float Depth: TEXCOORD1;
	};
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap_skin(float4 vPosition: POSITION, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		VS_OUTPUT_SHADOWMAP Out;
		float4 vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		Out.Tex0 = tc;
		Out.Depth = Out.Pos.z / Out.Pos.w;
		return Out;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0){
		VS_OUTPUT_SHADOWMAP Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Depth = Out.Pos.z / Out.Pos.w;
		float3 vScreenNormal = mul((float3x3)matWorldViewProj, vNormal);
		Out.Depth -= vScreenNormal.z * (fShadowBias);
		return Out;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap_biased(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0){
		VS_OUTPUT_SHADOWMAP Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Depth = Out.Pos.z / Out.Pos.w;
		float3 vScreenNormal = mul((float3x3)matWorldViewProj, vNormal);
		Out.Depth -= vScreenNormal.z * (fShadowBias);
		Out.Pos.z += 0.0025f;
		return Out;
	}
	PS_OUTPUT ps_main_shadowmap(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		Output.RGBColor.a = tex2D(MeshTextureSampler, In.Tex0).a;
		Output.RGBColor.a -= 0.5h;
		clip(Output.RGBColor.a);
		Output.RGBColor.rgb = In.Depth;
		return Output;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap_light(uniform const bool skinning, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		#ifdef USE_LIGHTING_PASS
			VS_OUTPUT_SHADOWMAP Out;
			float4 vObjectPos;
			if(skinning){
				vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
			}
			else{
				vObjectPos = vPosition;
			}
			float far_clip = g_vPointLightPosXYZ_InvRadius[0].a;
			Out.Pos = mul(matWorldViewProj, vObjectPos);
			Out.Tex0 = tc;
			float4 vWorldPos = mul(matWorld, vPosition);
			float3 lightVec = far_clip * (g_vPointLightPosXYZ_InvRadius[0].xyz - vWorldPos);
			Out.Depth = length(lightVec);
			float local_fShadowBias = fShadowBias * 50;
			float3 vScreenNormal = mul((float3x3)matWorldViewProj, vNormal);
			Out.Depth -= vScreenNormal.z * local_fShadowBias;
		#else
			INITIALIZE_OUTPUT(VS_OUTPUT_SHADOWMAP, Out);
		#endif
		return Out;
	}
	PS_OUTPUT ps_main_shadowmap_light(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		#ifdef USE_LIGHTING_PASS
			Output.RGBColor = In.Depth;
		#else
			Output.RGBColor = half4(1, 0, 0, 1);
		#endif
		return Output;
	}
	PS_OUTPUT ps_render_character_shadow(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		Output.RGBColor = 1.0h;
		return Output;
	}
	VertexShader vs_main_shadowmap_compiled = compile vs_2_0 vs_main_shadowmap();
	VertexShader vs_main_shadowmap_skin_compiled = compile vs_2_0 vs_main_shadowmap_skin();
	PixelShader ps_main_shadowmap_compiled = compile ps_2_0 ps_main_shadowmap();
	PixelShader ps_main_shadowmap_light_compiled = compile ps_2_0 ps_main_shadowmap_light();
	PixelShader ps_render_character_shadow_compiled = compile ps_2_0 ps_render_character_shadow();
	technique renderdepth{
		pass P0{
			VertexShader = vs_main_shadowmap_compiled;
			PixelShader = ps_main_shadowmap_compiled;
		}
	}
	technique renderdepth_biased{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_shadowmap_biased();
			PixelShader = ps_main_shadowmap_compiled;
		}
	}
	technique renderdepthwithskin{
		pass P0{
			VertexShader = vs_main_shadowmap_skin_compiled;
			PixelShader = ps_main_shadowmap_compiled;
		}
	}
	technique renderdepth_light{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_shadowmap_light(false);
			PixelShader = ps_main_shadowmap_light_compiled;
		}
	}
	technique renderdepthwithskin_light{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_shadowmap_light(true);
			PixelShader = ps_main_shadowmap_light_compiled;
		}
	}
	technique render_character_shadow{
		pass P0{
			VertexShader = vs_main_shadowmap_compiled;
			PixelShader = ps_render_character_shadow_compiled;
		}
	}
	technique render_character_shadow_with_skin{
		pass P0{
			VertexShader = vs_main_shadowmap_skin_compiled;
			PixelShader = ps_render_character_shadow_compiled;
		}
	}
	float blurred_read_alpha(float2 texCoord){
		float sample_start = tex2D(CharacterShadowTextureSampler, texCoord).r;
		static const int SAMPLE_COUNT = 4;
		static const float2 offsets[SAMPLE_COUNT] = {
			 - 1, 1, 1, 1, 0, 2, 0, 3, 
		};
		float blur_amount = saturate(1.0f - texCoord.y);
		blur_amount *= blur_amount;
		float sampleDist = (6.0f / 256.0f) * blur_amount;
		float sample = sample_start;
		for(int i = 0; i < SAMPLE_COUNT; i++){
			float2 sample_pos = texCoord + sampleDist * offsets[i];
			float sample_here = tex2D(CharacterShadowTextureSampler, sample_pos).a;
			sample += sample_here;
		}
		sample /= SAMPLE_COUNT + 1;
		return sample;
	}
	struct VS_OUTPUT_CHARACTER_SHADOW{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float3 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_CHARACTER_SHADOW vs_character_shadow(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR){
		INITIALIZE_OUTPUT(VS_OUTPUT_CHARACTER_SHADOW, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		if(PcfMode != PCF_NONE){
			float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
			float wNdotSun = max( - 0.0001, dot(vWorldN, -vSunDir));
			Out.SunLight = (wNdotSun) * vSunColor;
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_character_shadow(uniform const int PcfMode, VS_OUTPUT_CHARACTER_SHADOW In){
		PS_OUTPUT Output;
		if(PcfMode == PCF_NONE){
			Output.RGBColor.a = blurred_read_alpha(In.Tex0) * In.Color.a;
		}
		else {
			half sun_amount = 0.05h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor.a = saturate(blurred_read_alpha(In.Tex0) * In.Color.a * sun_amount);
		}
		Output.RGBColor.rgb = In.Color.rgb;
		return Output;
	}
	DEFINE_TECHNIQUES(character_shadow, vs_character_shadow, ps_character_shadow)
	PS_OUTPUT ps_character_shadow_new(uniform const int PcfMode, VS_OUTPUT_CHARACTER_SHADOW In){
		PS_OUTPUT Output;
		if(PcfMode == PCF_NONE){
			Output.RGBColor.a = tex2D(CharacterShadowTextureSampler, In.Tex0).r * In.Color.a;
		}
		else {
			half sun_amount = 0.05h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor.a = saturate(tex2D(CharacterShadowTextureSampler, In.Tex0).r * In.Color.a * sun_amount);
		}
		Output.RGBColor.rgb = In.Color.rgb;
		return Output;
	}
	DEFINE_TECHNIQUES(character_shadow_new, vs_character_shadow, ps_character_shadow_new)
#endif
#ifdef WATER_SHADERS
	struct VS_OUTPUT_WATER{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
		float4 PosWater: TEXCOORD1;
		half3 CameraDir: TEXCOORD2;
		half4 LightDir_Alpha: TEXCOORD3;
		half4 LightDif: TEXCOORD4;
		half4 projCoord: TEXCOORD5;
		half Depth: TEXCOORD6;
		float Fog: FOG;
	};
	VS_OUTPUT_WATER vs_main_water(uniform const bool mud_factor, float4 vPosition: POSITION, float3 vNormal: NORMAL, float4 vColor: COLOR, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL){
		VS_OUTPUT_WATER Out = (VS_OUTPUT_WATER)0;
		if(!mud_factor){
			vPosition.z += ((sin((time_var * 2.2) + ((vPosition.z + (vPosition.y - vPosition.x)) * 0.5))) * vWaterWindStrength);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc + texture_offset.xy;
		Out.PosWater = mul(matWaterWorldViewProj, vPosition);
		half3 vWorldPos = (half3)mul(matWorld, vPosition);
		half3 point_to_camera_normal = normalize(vCameraPos.xyz - vWorldPos.xyz);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.CameraDir = mul(TBNMatrix, point_to_camera_normal);
		Out.LightDir_Alpha.xyz = mul(TBNMatrix, -vSunDir);
		Out.LightDif = vSunColor * vColor;
		Out.LightDir_Alpha.a = vColor.a;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		if(use_depth_effects){
			Out.projCoord.xy = (half2(Out.Pos.x, -Out.Pos.y) + Out.Pos.w) / 2;
			Out.projCoord.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * Out.Pos.w);
			Out.projCoord.zw = Out.Pos.zw;
			Out.Depth = Out.Pos.z * far_clip_Inv;
		}
		return Out;
	}
	PS_OUTPUT ps_main_water(VS_OUTPUT_WATER In, uniform const bool use_high, uniform const bool apply_depth, uniform const bool mud_factor){
		PS_OUTPUT Output;
		half3 normal;
		if(!apply_depth){
			normal = half3(0, 0, 1);
		}
		else {
			normal.xy = (2.0h * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0h);
			normal.z = sqrt(1.0h - dot(normal.xy, normal.xy));
		}
		half NdotL = saturate(dot(normal, In.LightDir_Alpha.xyz));
		half3 vView = normalize(In.CameraDir);
		half4 tex;
		if(apply_depth){
			tex = tex2D(ReflectionTextureSampler, (0.25h * normal.xy) + half2(0.5h + 0.5h * (In.PosWater.x / In.PosWater.w), 0.5h - 0.5h * (In.PosWater.y / In.PosWater.w)));
		}
		else {
			tex = tex2D(EnvTextureSampler, (vView - normal).yx * 3.4h);
		}
		INPUT_OUTPUT_GAMMA(tex.rgb);
		Output.RGBColor = 0.01h * NdotL * In.LightDif;
		if(mud_factor){
			Output.RGBColor *= 0.125h;
		}
		float satval = (saturate(dot(vView, normal)));
		float fresnel = 1 - satval;
		float origfreesnel = fresnel;
		fresnel = 0.0204f + 0.9796 * (fresnel * fresnel * fresnel * fresnel * fresnel);
		if(!apply_depth){
			fresnel = min(fresnel, 0.01f);
		}
		if(mud_factor){
			Output.RGBColor.rgb += lerp(tex.rgb * half3(0.105, 0.175, 0.160) * fresnel, tex.rgb, fresnel);
		}
		else {
			Output.RGBColor.rgb += (tex.rgb * fresnel);
		}
		Output.RGBColor.a = 1.0h - 0.3h * In.CameraDir.z;
		half vertex_alpha = In.LightDir_Alpha.a;
		Output.RGBColor.a *= vertex_alpha;
		if(mud_factor){
			Output.RGBColor.a = 1.0h;
		}
		half3 cWaterColor;
		if(apply_depth){
			const half3 g_cDownWaterColor = mud_factor ? half3(4.5h / 255.0h, 8.0h / 255.0h, 6.0h / 255.0h): half3(1.0h / 255.0h, 4.0h / 255.0h, 6.0h / 255.0h);
			const half3 g_cUpWaterColor = mud_factor ? half3(5.0h / 255.0h, 7.0h / 255.0h, 7.0h / 255.0h): half3(1.0h / 255.0h, 5.0h / 255.0h, 10.0h / 255.0h);
			cWaterColor = lerp(g_cUpWaterColor, g_cDownWaterColor, satval);
		}
		else {
			cWaterColor = In.LightDif.xyz;
		}
		float fog_fresnel_factor = saturate(dot(In.CameraDir, normal));
		fog_fresnel_factor *= fog_fresnel_factor;
		fog_fresnel_factor *= fog_fresnel_factor;
		if(!apply_depth){
			fog_fresnel_factor *= 0.1f;
			fog_fresnel_factor += 0.05f;
		}
		Output.RGBColor.rgb += cWaterColor * fog_fresnel_factor;
		if(mud_factor){
			Output.RGBColor.rgb += half3(0.022h, 0.02h, 0.005h) * origfreesnel;
		}
		if(apply_depth && use_depth_effects){
			half depth = tex2Dproj(DepthTextureSampler, In.projCoord).r;
			half alpha_factor;
			if((depth + 0.0005) < In.Depth){
				alpha_factor = 1;
			}
			else{
				alpha_factor = saturate((depth - In.Depth) * 2048);
			}
			Output.RGBColor.w *= alpha_factor;
			Output.RGBColor.w += saturate((depth - In.Depth) * 32);
			static const bool use_refraction = true;
			if(use_refraction && use_high){
				half4 coord_start = In.projCoord;
				half4 coord_disto = coord_start;
				coord_disto.xy += (normal.xy * saturate(Output.RGBColor.w) * 0.1h);
				half depth_here = tex2D(DepthTextureSampler, coord_disto.xy).r;
				half4 refraction;
				if(depth_here < depth)refraction = tex2Dproj(ScreenTextureSampler, coord_disto);
				else refraction = tex2Dproj(ScreenTextureSampler, coord_start);
				INPUT_OUTPUT_GAMMA(refraction.rgb);
				Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, refraction.rgb, saturate(1.0h - Output.RGBColor.w) * 0.55h);
				if(Output.RGBColor.a > 0.1h){
					Output.RGBColor.a *= 1.75h;
				}
				if(mud_factor){
					Output.RGBColor.a *= 1.25h;
				}
			}
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = saturate(Output.RGBColor.a);
		if(!apply_depth){
			Output.RGBColor.a = 1.0f;
		}
		return Output;
	}
	technique watermap{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water(false);
			PixelShader = compile ps_2_0 ps_main_water(false, true, false);
		}
	}
	technique watermap_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water(false);
			PixelShader = compile PS_2_X ps_main_water(true, true, false);
		}
	}
	technique watermap_mud{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water(true);
			PixelShader = compile ps_2_0 ps_main_water(false, true, true);
		}
	}
	technique watermap_mud_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water(true);
			PixelShader = compile PS_2_X ps_main_water(true, true, true);
		}
	}
#endif
#ifdef SKYBOX_SHADERS
	PS_OUTPUT ps_skybox_shading(VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor *= tex2D(MeshTextureSampler, In.Tex0);
		return Output;
	}
	PS_OUTPUT ps_skybox_shading_new(uniform bool use_hdr, VS_OUTPUT_FONT In){
		PS_OUTPUT Output;
		if(use_hdr){
			Output.RGBColor = In.Color;
			Output.RGBColor *= tex2D(Diffuse2Sampler, In.Tex0);
			half2 scaleBias = vSpecularColor.rg;
			half exFactor16 = tex2D(EnvTextureSampler, In.Tex0).r;
			Output.RGBColor.rgb *= exp2(exFactor16 * scaleBias.r + scaleBias.g);
		}
		else{
			Output.RGBColor = In.Color;
			half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
			INPUT_TEX_GAMMA(tex_col.rgb);
			Output.RGBColor *= tex_col;
		}
		Output.RGBColor.a = 1.0h;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		if(In.Color.a == 0.0h){
			Output.RGBColor.rgb = saturate(Output.RGBColor.rgb);
		}
		return Output;
	}
	VS_OUTPUT_FONT vs_skybox(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_OUTPUT_FONT Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Pos.z = Out.Pos.w;
		float3 P = vPosition.xyz;
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		P.z *= 0.2f;
		float d = length(P);
		Out.Fog = get_fog_amount(d);
		float4 vWorldPos = mul(matWorld, vPosition);
		Out.Color.a = (vWorldPos.z < -10.0f) ? 0.0h: 1.0h;
		return Out;
	}
	VertexShader vs_skybox_compiled = compile vs_2_0 vs_skybox();
	technique skybox{
		pass P0{
			VertexShader = vs_skybox_compiled;
			PixelShader = compile ps_2_0 ps_skybox_shading();
		}
	}
	technique skybox_new{
		pass P0{
			VertexShader = vs_skybox_compiled;
			PixelShader = compile ps_2_0 ps_skybox_shading_new(false);
		}
	}
	technique skybox_new_HDR{
		pass P0{
			VertexShader = vs_skybox_compiled;
			PixelShader = compile ps_2_0 ps_skybox_shading_new(true);
		}
	}
#endif
#ifdef STANDART_RELATED_SHADER
	struct VS_OUTPUT{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT vs_main(uniform const int PcfMode, uniform const bool UseSecondLight, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		if(UseSecondLight){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half4 vMaterialColorComb = vMaterialColor * vColor;
		Out.Color = (vMaterialColorComb * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColorComb;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT vs_main_Instanced(uniform const int PcfMode, uniform const bool UseSecondLight, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		Out.Pos = mul(matWorldOfInstance, float4(vPosition.xyz, 1.0f));
		Out.Pos = mul(matViewProj, Out.Pos);
		float4 vWorldPos = (float4)mul(matWorldOfInstance, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorldOfInstance, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		if(UseSecondLight){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half4 vMaterialColorComb = vMaterialColor * vColor;
		Out.Color = (vMaterialColorComb * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColorComb;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main(VS_OUTPUT In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		half sun_amount = 1.0h;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VertexShader vs_main_compiled_PCF_NONE_true = compile vs_2_0 vs_main(PCF_NONE, true);
	VertexShader vs_main_compiled_PCF_DEFAULT_true = compile vs_2_0 vs_main(PCF_DEFAULT, true);
	VertexShader vs_main_compiled_PCF_NVIDIA_true = compile vs_2_a vs_main(PCF_NVIDIA, true);
	VertexShader vs_main_compiled_PCF_NONE_false = compile vs_2_0 vs_main(PCF_NONE, false);
	VertexShader vs_main_compiled_PCF_DEFAULT_false = compile vs_2_0 vs_main(PCF_DEFAULT, false);
	VertexShader vs_main_compiled_PCF_NVIDIA_false = compile vs_2_a vs_main(PCF_NVIDIA, false);
	PixelShader ps_main_compiled_PCF_NONE = compile ps_2_0 ps_main(PCF_NONE);
	PixelShader ps_main_compiled_PCF_DEFAULT = compile ps_2_0 ps_main(PCF_DEFAULT);
	PixelShader ps_main_compiled_PCF_NVIDIA = compile ps_2_a ps_main(PCF_NVIDIA);
	technique diffuse{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NONE_true;
			PixelShader = ps_main_compiled_PCF_NONE;
		}
	}
	technique diffuse_SHDW{
		pass P0{
			VertexShader = vs_main_compiled_PCF_DEFAULT_true;
			PixelShader = ps_main_compiled_PCF_DEFAULT;
		}
	}
	technique diffuse_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NVIDIA_true;
			PixelShader = ps_main_compiled_PCF_NVIDIA;
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(diffuse, 0, 0, 0, 0, 0)
	technique diffuse_dynamic{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NONE_false;
			PixelShader = ps_main_compiled_PCF_NONE;
		}
	}
	technique diffuse_dynamic_SHDW{
		pass P0{
			VertexShader = vs_main_compiled_PCF_DEFAULT_false;
			PixelShader = ps_main_compiled_PCF_DEFAULT;
		}
	}
	technique diffuse_dynamic_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NVIDIA_false;
			PixelShader = ps_main_compiled_PCF_NVIDIA;
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(diffuse_dynamic, 0, 0, 0, 0, 0)
	technique diffuse_dynamic_Instanced{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_Instanced(PCF_NONE, false);
			PixelShader = ps_main_compiled_PCF_NONE;
		}
	}
	technique diffuse_dynamic_Instanced_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_Instanced(PCF_DEFAULT, false);
			PixelShader = ps_main_compiled_PCF_DEFAULT;
		}
	}
	technique diffuse_dynamic_Instanced_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_Instanced(PCF_NVIDIA, false);
			PixelShader = ps_main_compiled_PCF_NVIDIA;
		}
	}
	technique envmap_metal{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NONE_true;
			PixelShader = ps_main_compiled_PCF_NONE;
		}
	}
	technique envmap_metal_SHDW{
		pass P0{
			VertexShader = vs_main_compiled_PCF_DEFAULT_true;
			PixelShader = ps_main_compiled_PCF_DEFAULT;
		}
	}
	technique envmap_metal_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_compiled_PCF_NVIDIA_true;
			PixelShader = ps_main_compiled_PCF_NVIDIA;
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(envmap_metal, 0, 0, 0, 0, 0)
	struct VS_OUTPUT_BUMP{
		float4 Pos: POSITION;
		half4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLightDir: TEXCOORD1;
		half3 SkyLightDir: TEXCOORD2;
		half2 ViewDir: TEXCOORD3;
		half3 ShadowTexCoord: TEXCOORD4;
		half2 ShadowTexelPos: TEXCOORD5;
		float Fog: FOG;
	};
	VS_OUTPUT_BUMP vs_main_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, uniform const bool use_parallaxmapping = false){
		INITIALIZE_OUTPUT(VS_OUTPUT_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.SunLightDir.xyz = mul(TBNMatrix, -vSunDir);
		Out.SkyLightDir = mul(TBNMatrix, half3(0, 0, 1));
		Out.VertexColor = vVertexColor;
		half3 vViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		float fresnel = 1 - (saturate(dot(vViewDir, vWorldN)));
		fresnel *= fresnel + 0.1h;
		Out.SunLightDir.w = fresnel;
		if(use_parallaxmapping){
			vViewDir = normalize(mul(TBNMatrix, vViewDir));
			Out.ViewDir = vViewDir.xy;
		}
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_bump(VS_OUTPUT_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half3 normal;
		normal.xy = (2.0h * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0h);
		normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		half3 total_light = vAmbientColor.rgb;
		if(PcfMode != PCF_NONE){
			half sun_amount = 0.03h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			total_light += ((saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount))) * vSunColor.rgb;
		}
		else {
			total_light += saturate(dot(In.SunLightDir.xyz, normal.xyz)) * vSunColor.rgb;
		}
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0h;
		Output.RGBColor *= vMaterialColor;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_main_bump_simple(VS_OUTPUT_BUMP In, uniform const int PcfMode, uniform const bool use_parallaxmapping = false){
		PS_OUTPUT Output;
		if(use_parallaxmapping){
			float factor = (0.01f * vSpecularColor.x);
			float BIAS = (factor * -0.5f);
			float SCALE = factor;
			float4 Normal = tex2D(NormalTextureSampler, In.Tex0);
			float h = Normal.a * SCALE + BIAS;
			In.Tex0.xy += h * Normal.z * In.ViewDir.xy;
		}
		half3 normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0h);
		half sun_amount = 1.0h;
		if(PcfMode != PCF_NONE){
			if(PcfMode == PCF_NVIDIA)sun_amount = saturate(0.15h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos));
			else sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		half3 total_light = vAmbientColor.rgb;
		total_light += ((saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount * sun_amount))) * vSunColor.rgb;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0h;
		Output.RGBColor *= vMaterialColor;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		Output.RGBColor.rgb *= max(0.6h, In.SunLightDir.w);
		Output.RGBColor.rgb = OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_main_bump_simple_multitex(VS_OUTPUT_BUMP In, uniform const int PcfMode, uniform const bool use_parallaxmapping = false){
		PS_OUTPUT Output;
		if(use_parallaxmapping){
			float factor = (0.01f * vSpecularColor.x);
			float BIAS = (factor * -0.5f);
			float SCALE = factor;
			float4 Normal = tex2D(NormalTextureSampler, In.Tex0);
			float h = Normal.a * SCALE + BIAS;
			In.Tex0.xy += h * Normal.z * In.ViewDir.xy;
		}
		half4 multi_tex_col = tex2D(MeshTextureSampler, In.Tex0);
		half4 tex_col2 = tex2D(Diffuse2Sampler, In.Tex0 * uv_2_scale);
		half inv_alpha = (1.0h - In.VertexColor.a);
		multi_tex_col.rgb *= inv_alpha;
		multi_tex_col.rgb += tex_col2.rgb * In.VertexColor.a;
		INPUT_TEX_GAMMA(multi_tex_col.rgb);
		half3 normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0h);
		half sun_amount = 1.0h;
		if(PcfMode != PCF_NONE){
			if(PcfMode == PCF_NVIDIA)sun_amount = saturate(0.15h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos));
			else sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		half3 total_light = vAmbientColor.rgb;
		total_light += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount)) * vSunColor.rgb;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0h;
		Output.RGBColor *= multi_tex_col;
		Output.RGBColor.rgb *= In.VertexColor.rgb;
		Output.RGBColor.rgb *= max(0.6h, In.SunLightDir.w);
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VertexShader vs_main_bump_compiled_PCF_NONE = compile vs_2_0 vs_main_bump(PCF_NONE);
	VertexShader vs_main_bump_compiled_PCF_DEFAULT = compile vs_2_0 vs_main_bump(PCF_DEFAULT);
	VertexShader vs_main_bump_compiled_PCF_NVIDIA = compile vs_2_a vs_main_bump(PCF_NVIDIA);
	technique bumpmap{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NONE;
			PixelShader = compile ps_2_0 ps_main_bump(PCF_NONE);
		}
	}
	technique bumpmap_SHDW{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_DEFAULT;
			PixelShader = compile ps_2_0 ps_main_bump(PCF_DEFAULT);
		}
	}
	technique bumpmap_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NVIDIA;
			PixelShader = compile ps_2_a ps_main_bump(PCF_NVIDIA);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(bumpmap, 1, 1, 0, 0, 0)
	technique dot3{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NONE;
			PixelShader = compile ps_2_0 ps_main_bump_simple(PCF_NONE);
		}
	}
	technique dot3_SHDW{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_DEFAULT;
			PixelShader = compile ps_2_0 ps_main_bump_simple(PCF_DEFAULT);
		}
	}
	technique dot3_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NVIDIA;
			PixelShader = compile ps_2_a ps_main_bump_simple(PCF_NVIDIA);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3, 0, 1, 0, 0, 0)
	technique dot3_high{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NONE;
			PixelShader = compile PS_2_X ps_main_bump_simple(PCF_NONE);
		}
	}
	technique dot3_high_SHDW{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_DEFAULT;
			PixelShader = compile PS_2_X ps_main_bump_simple(PCF_DEFAULT);
		}
	}
	technique dot3_high_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NVIDIA;
			PixelShader = compile ps_2_a ps_main_bump_simple(PCF_NVIDIA);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3_high, 0, 1, 0, 0, 0)
	technique dot3_parallax{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump(PCF_NONE, true);
			PixelShader = compile PS_2_X ps_main_bump_simple(PCF_NONE, true);
		}
	}
	technique dot3_parallax_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump(PCF_DEFAULT, true);
			PixelShader = compile PS_2_X ps_main_bump_simple(PCF_DEFAULT, true);
		}
	}
	technique dot3_parallax_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_bump(PCF_NVIDIA, true);
			PixelShader = compile ps_2_a ps_main_bump_simple(PCF_NVIDIA, true);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3_parallax, 0, 1, 0, 0, 0)
	technique dot3_multitex{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NONE;
			PixelShader = compile ps_2_0 ps_main_bump_simple_multitex(PCF_NONE);
		}
	}
	technique dot3_multitex_SHDW{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_DEFAULT;
			PixelShader = compile ps_2_0 ps_main_bump_simple_multitex(PCF_DEFAULT);
		}
	}
	technique dot3_multitex_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NVIDIA;
			PixelShader = compile ps_2_a ps_main_bump_simple_multitex(PCF_NVIDIA);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3_multitex, 0, 1, 0, 0, 0)
	technique dot3_multitex_high{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NONE;
			PixelShader = compile PS_2_X ps_main_bump_simple_multitex(PCF_NONE);
		}
	}
	technique dot3_multitex_high_SHDW{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_DEFAULT;
			PixelShader = compile PS_2_X ps_main_bump_simple_multitex(PCF_DEFAULT);
		}
	}
	technique dot3_multitex_high_SHDWNVIDIA{
		pass P0{
			VertexShader = vs_main_bump_compiled_PCF_NVIDIA;
			PixelShader = compile ps_2_a ps_main_bump_simple_multitex(PCF_NVIDIA);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3_multitex_high, 0, 1, 0, 0, 0)
	technique dot3_multitex_parallax{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump(PCF_NONE, true);
			PixelShader = compile PS_2_X ps_main_bump_simple_multitex(PCF_NONE, true);
		}
	}
	technique dot3_multitex_parallax_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump(PCF_DEFAULT, true);
			PixelShader = compile PS_2_X ps_main_bump_simple_multitex(PCF_DEFAULT, true);
		}
	}
	technique dot3_multitex_parallax_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_bump(PCF_NVIDIA, true);
			PixelShader = compile ps_2_a ps_main_bump_simple_multitex(PCF_NVIDIA, true);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(dot3_multitex_parallax, 0, 1, 0, 0, 0)
	struct VS_OUTPUT_ENVMAP_SPECULAR{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float4 Tex0: TEXCOORD0;
		half3 vSpecular: TEXCOORD1;
		half4 SunLight: TEXCOORD2;
		half3 ShadowTexCoord: TEXCOORD3;
		half2 ShadowTexelPos: TEXCOORD4;
		float Fog: FOG;
	};
	VS_OUTPUT_ENVMAP_SPECULAR vs_envmap_specular(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_ENVMAP_SPECULAR, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		if(false){
			float4 vWorldPos1 = mul(matMotionBlur, vPosition);
			float3 delta_vector = vWorldPos1.xyz - vWorldPos.xyz;
			float maxMoveLength = length(delta_vector);
			float3 moveDirection = delta_vector / maxMoveLength;
			if(maxMoveLength > 0.25f){
				maxMoveLength = 0.25f;
				vWorldPos1.xyz = vWorldPos.xyz + delta_vector * maxMoveLength;
			}
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.12f) ? 1: 0;
			float y_factor = saturate(vPosition.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float extra_alpha = 0.1f;
			float start_alpha = (1.0f + extra_alpha);
			float end_alpha = start_alpha - 1.8f;
			float alpha = saturate(lerp(start_alpha, end_alpha, delta_coefficient_smooth));
			vColor.a = saturate(0.5f - vPosition.y) + alpha + 0.25;
			Out.Pos = mul(matViewProj, vWorldPos);
		}
		else {
			Out.Pos = mul(matWorldViewProj, vPosition);
		}
		Out.Tex0.xy = tc;
		half3 relative_cam_pos = normalize(vCameraPos.xyz - vWorldPos.xyz);
		float2 envpos;
		half3 tempvec = relative_cam_pos - vWorldN;
		half3 vHalf = normalize(relative_cam_pos - vSunDir);
		float3 fSpecular = spec_coef * vSunColor.rgb * vSpecularColor.rgb * pow(saturate(dot(vHalf, vWorldN)), fMaterialPower);
		Out.vSpecular = fSpecular;
		Out.vSpecular *= vColor.rgb;
		envpos.x = (tempvec.y);
		envpos.y = tempvec.z;
		envpos += 1.0f;
		Out.Tex0.zw = envpos;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		half wNdotSun = max( - 0.0001f, dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		Out.SunLight.a = vColor.a;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_ENVMAP_SPECULAR vs_envmap_specular_Instanced(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_ENVMAP_SPECULAR, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = mul(matWorldOfInstance, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorldOfInstance, vNormal));
		if(false){
			float4 vWorldPos1;
			float3 moveDirection;
			if(true){
				const float blur_len = 0.2f;
				moveDirection = -normalize(float3(matWorldOfInstance[0][0], matWorldOfInstance[1][0], matWorldOfInstance[2][0]));
				moveDirection.y -= blur_len * 0.285;
				vWorldPos1 = vWorldPos + float4(moveDirection, 0) * blur_len;
			}
			else {
				vWorldPos1 = mul(matMotionBlur, vPosition);
				moveDirection = normalize(vWorldPos1.xyz - vWorldPos.xyz);
			}
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.12f) ? 1: 0;
			float y_factor = saturate(vPosition.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float extra_alpha = 0.1f;
			float start_alpha = (1.0f + extra_alpha);
			float end_alpha = start_alpha - 1.8f;
			float alpha = saturate(lerp(start_alpha, end_alpha, delta_coefficient_smooth));
			vColor.a = saturate(0.5f - vPosition.y) + alpha + 0.25;
		}
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0.xy = tc;
		half3 relative_cam_pos = normalize(vCameraPos.xyz - vWorldPos.xyz);
		float2 envpos;
		half3 tempvec = relative_cam_pos - vWorldN;
		half3 vHalf = normalize(relative_cam_pos - vSunDir);
		float3 fSpecular = spec_coef * vSunColor.rgb * vSpecularColor.rgb * pow(saturate(dot(vHalf, vWorldN)), fMaterialPower);
		Out.vSpecular = fSpecular;
		Out.vSpecular *= vColor.rgb;
		envpos.x = (tempvec.y);
		envpos.y = tempvec.z;
		envpos += 1.0f;
		Out.Tex0.zw = envpos;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		half wNdotSun = max( - 0.0001h, dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		Out.SunLight.a = vColor.a;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_envmap_specular(VS_OUTPUT_ENVMAP_SPECULAR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 texColor = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(texColor.rgb);
		half3 specTexture = tex2D(SpecularTextureSampler, In.Tex0.xy).rgb;
		half3 fSpecular = specTexture * In.vSpecular.rgb;
		half3 envColor = tex2D(EnvTextureSampler, In.Tex0.zw).rgb;
		if((PcfMode != PCF_NONE)){
			half sun_amount = 0.1h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			half4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular) * sun_amount;
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight.rgb * sun_amount + 0.3h) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		else {
			half4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular);
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight.rgb + 0.3h) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = 1.0h;
		if(false)Output.RGBColor.a = In.SunLight.a;
		return Output;
	}
	PS_OUTPUT ps_envmap_specular_singlespec(VS_OUTPUT_ENVMAP_SPECULAR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half2 spectex_Col = tex2D(SpecularTextureSampler, In.Tex0.xy).ag;
		half specTexture = dot(spectex_Col, spectex_Col) * 0.5;
		half3 fSpecular = specTexture * In.vSpecular.rgb;
		half4 texColor = saturate((saturate(In.Color + 0.5h) * specTexture) * 2.0h + 0.25h);
		half3 envColor = tex2D(EnvTextureSampler, In.Tex0.zw).rgb;
		if((PcfMode != PCF_NONE)){
			half sun_amount = 0.1h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			half4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular) * sun_amount;
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight.rgb * sun_amount + 0.3h) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		else {
			half4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular);
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight.rgb + 0.3h) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = 1.0h;
		return Output;
	}
	DEFINE_TECHNIQUES(watermap_for_objects, vs_envmap_specular, ps_envmap_specular_singlespec)
	struct VS_OUTPUT_BUMP_DYNAMIC{
		float4 Pos: POSITION;
		float4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	VS_OUTPUT_BUMP_DYNAMIC vs_main_bump_interior(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_BUMP_DYNAMIC, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.VertexColor = vVertexColor;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_bump_interior(VS_OUTPUT_BUMP_DYNAMIC In){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		Output.RGBColor = float4(total_light.rgb, 1.0);
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		Output.RGBColor.rgb = saturate(OUTPUT_GAMMA(Output.RGBColor.rgb));
		Output.RGBColor.a = In.VertexColor.a;
		return Output;
	}
	technique bumpmap_interior{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump_interior();
			PixelShader = compile ps_2_0 ps_main_bump_interior();
		}
	}
	struct VS_OUTPUT_BUMP_DYNAMIC_NEW{
		float4 Pos: POSITION;
		float4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 ViewDir: TEXCOORD1;
		float Fog: FOG;
	};
	VS_OUTPUT_BUMP_DYNAMIC_NEW vs_main_bump_interior_new(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_BUMP_DYNAMIC_NEW, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.VertexColor = vVertexColor;
		float3 viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.ViewDir = mul(TBNMatrix, viewdir);
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_bump_interior_new(VS_OUTPUT_BUMP_DYNAMIC_NEW In, uniform const bool use_specularmap){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		Output.RGBColor = float4(total_light.rgb, 1.0);
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		if(use_specularmap){
			float4 fSpecular = 0;
			float4 specColor = 0.1 * spec_coef * vSpecularColor;
			float spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0).rgb, 0.33);
			specColor *= spec_tex_factor;
			Output.RGBColor += specColor * fSpecular;
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor = saturate(Output.RGBColor);
		Output.RGBColor.a = In.VertexColor.a;
		return Output;
	}
	technique bumpmap_interior_new{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump_interior_new();
			PixelShader = compile ps_2_0 ps_main_bump_interior_new(false);
		}
	}
	technique bumpmap_interior_new_specmap{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump_interior_new();
			PixelShader = compile ps_2_0 ps_main_bump_interior_new(true);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(bumpmap_interior, 1, 1, 0, 0, 0)
#endif
#ifdef STANDART_SHADERS
	struct VS_OUTPUT_STANDART{
		float4 Pos: POSITION;
		half4 VertexColor: COLOR0;
		#ifdef INCLUDE_VERTEX_LIGHTING
			half3 VertexLighting: COLOR1;
		#endif
		float4 Tex0: TEXCOORD0;
		half3 SunLightDir: TEXCOORD1;
		half3 SkyLightDir: TEXCOORD2;
		half3 ViewDir: TEXCOORD3;
		half3 ShadowTexCoord: TEXCOORD4;
		half2 ShadowTexelPos: TEXCOORD6;
		float Fog: FOG;
	};
	VS_OUTPUT_STANDART vs_main_standart(uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_skinning, uniform const int flagwave_type, uniform const bool use_envmap, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		half3 vObjectN, vObjectT, vObjectB;
		if(use_skinning){
			vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
			vObjectN = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vNormal) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vNormal) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vNormal) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vNormal) * vBlendWeights.w);
			if(use_bumpmap){
				vObjectT = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vTangent) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vTangent) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vTangent) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vTangent) * vBlendWeights.w);
				vObjectB = cross(vObjectN, vObjectT);
				float3 crossy = cross(vNormal, vTangent);
				float dotty = dot(crossy, vBinormal);
				if(dotty < 0.0f){
					vObjectB = -vObjectB;
				}
			}
		}
		else {
			if((flagwave_type > 0) && (flagwave_type != 3 || ((tc.y >= 0.07 && tc.y <= 0.93) || ((tc.y <= 0.07 || tc.y >= 0.93) && (tc.x >= 0.07 && tc.x <= 0.93))))){
				float4 orgPos = vPosition;
				half sideval = 0;
				float4 Position1;
				float4 Position2;
				float4 nextPos = orgPos;
				for(int p = 1; p < 3; p++){
					if(p == 2){
						nextPos.x = orgPos.x;
						nextPos.y -= 0.05f;
					}
					else nextPos.x += 0.05f;
					if(p == 1){
						Position1 = nextPos;
					}
					if(p == 2){
						Position2 = nextPos;
					}
				}
				if(flagwave_type == 1){
					sideval = vNormal.z;
					vPosition.z += CalcPennonAnimation(vPosition.xyz);
					Position1.z += CalcPennonAnimation(Position1.xyz);
					Position2.z += CalcPennonAnimation(Position2.xyz);
				}
				else if(flagwave_type == 2){
					sideval = -vNormal.y;
					vPosition.y += CalcFlagAnimation(vPosition.xyz);
					Position1.y += CalcFlagAnimation(Position1.xyz);
					Position2.y += CalcFlagAnimation(Position2.xyz);
				}
				else if(flagwave_type == 3){
					sideval = vNormal.z;
					vPosition.xyz += CalcSailAnimation(vPosition.xyz);
					Position1.xyz += CalcSailAnimation(Position1.xyz);
					Position2.xyz += CalcSailAnimation(Position2.xyz);
				}
				else if(flagwave_type == 4){
					sideval = -vNormal.x;
					vPosition.x += CalcCTFPennonAnimation(vPosition.xyz);
					Position1.x += CalcCTFPennonAnimation(Position1.xyz);
					Position2.x += CalcCTFPennonAnimation(Position2.xyz);
				}
				else if(flagwave_type == 5){
					vPosition.x += (CalcPennonVerticalAnimation(vPosition.xyz) * (tc.x * 0.6));
				}
				if(flagwave_type != 5){
					vNormal = cross(Position1.xyz - vPosition.xyz, Position2.xyz - vPosition.xyz);
				}
				if(flagwave_type == 2){
					vNormal.y += 1;
				}
				else if(flagwave_type == 4){
					vNormal.x += 1;
				}
				if(sideval > 0)vNormal = -vNormal;
			}
			vObjectPos = vPosition;
			vObjectN = vNormal;
			if(use_bumpmap){
				vObjectT = vTangent;
				vObjectB = vBinormal;
			}
		}
		float4 vWorldPos = mul(matWorld, vObjectPos);
		half3 vWorldN = normalize(mul((half3x3)matWorld, vObjectN));
		const bool use_motion_blur = false;
		if(use_motion_blur){
			float4 vWorldPos1 = mul(matMotionBlur, vObjectPos);
			half3 moveDirection = normalize(vWorldPos1.xyz - vWorldPos.xyz);
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.1f) ? 1: 0;
			float y_factor = saturate(vObjectPos.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float start_alpha = 1.1f;
			float end_alpha = start_alpha - 1.8f;
			float alpha = saturate(lerp(start_alpha, end_alpha, delta_coefficient_smooth));
			vVertexColor.a = saturate(0.5f - vObjectPos.y) + alpha + 0.25;
		}
		if(use_motion_blur){
			Out.Pos = mul(matViewProj, vWorldPos);
		}
		else {
			Out.Pos = mul(matWorldViewProj, vObjectPos);
		}
		Out.Tex0.xy = tc;
		half3 viewdir;
		if(use_bumpmap){
			half3 vWorld_binormal = normalize(mul((half3x3)matWorld, vObjectB));
			half3 vWorld_tangent = normalize(mul((half3x3)matWorld, vObjectT));
			half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, half3(0, 0, 1));
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse_ex_1(vWorldPos.xyz, vWorldN, false).rgb;
			#endif
			viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir.xyz = mul(TBNMatrix, viewdir);
		}
		else{
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos.xyz, vWorldN, false).rgb;
			#endif
			viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir.xyz = viewdir;
			Out.SunLightDir = vWorldN;
		}
		Out.VertexColor.a *= vMaterialColor.a;
		if(use_envmap){
			half2 envpos;
			half3 tempvec = viewdir - vWorldN;
			envpos.x = (tempvec.y);
			envpos.y = tempvec.z;
			envpos += 1.0h;
			Out.Tex0.zw = envpos;
		}
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			if(PcfMode != PCF_NVIDIA){
				Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
			}
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_STANDART vs_main_standart_Instanced(uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_skinning, uniform const int flagwave_type, uniform const bool use_envmap, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		half3 vObjectN, vObjectT, vObjectB;
		if(use_skinning){
			GIVE_ERROR_HERE_VS;
		}
		else{
			float4 orgPos = vPosition;
			if((flagwave_type > 0) && (flagwave_type != 3 || ((tc.y >= 0.07 && tc.y <= 0.93) || ((tc.y <= 0.07 || tc.y >= 0.93) && (tc.x >= 0.07 && tc.x <= 0.93))))){
				half sideval = 0;
				float4 Position1;
				float4 Position2;
				float4 nextPos = orgPos;
				for(int p = 1; p < 3; p++){
					if(p == 2){
						nextPos.x = orgPos.x;
						nextPos.y -= .05;
					}
					else nextPos.x += .05;
					if(p == 1){
						Position1 = nextPos;
					}
					if(p == 2){
						Position2 = nextPos;
					}
				}
				if(flagwave_type == 1){
					sideval = vNormal.z;
					vPosition.z += CalcPennonAnimation(vPosition.xyz);
					Position1.z += CalcPennonAnimation(Position1.xyz);
					Position2.z += CalcPennonAnimation(Position2.xyz);
				}
				else if(flagwave_type == 2){
					sideval = -vNormal.y;
					vPosition.y += CalcFlagAnimation(vPosition.xyz);
					Position1.y += CalcFlagAnimation(Position1.xyz);
					Position2.y += CalcFlagAnimation(Position2.xyz);
				}
				else if(flagwave_type == 3){
					sideval = vNormal.z;
					vPosition.xyz += CalcSailAnimation(vPosition.xyz);
					Position1.xyz += CalcSailAnimation(Position1.xyz);
					Position2.xyz += CalcSailAnimation(Position2.xyz);
				}
				else if(flagwave_type == 4){
					sideval = -vNormal.x;
					vPosition.x += CalcCTFPennonAnimation(vPosition.xyz);
					Position1.x += CalcCTFPennonAnimation(Position1.xyz);
					Position2.x += CalcCTFPennonAnimation(Position2.xyz);
				}
				else if(flagwave_type == 5){
					vPosition.x += (CalcPennonVerticalAnimation(vPosition.xyz) * (tc.x * 0.6));
				}
				if(flagwave_type != 5){
					vNormal = cross(Position1.xyz - vPosition.xyz, Position2.xyz - vPosition.xyz);
				}
				if(flagwave_type == 2){
					vNormal.y += 1;
				}
				else if(flagwave_type == 4){
					vNormal.x += 1;
				}
				if(sideval > 0)vNormal = -vNormal;
			}
			vObjectPos = vPosition;
			vObjectN = vNormal;
			if(use_bumpmap){
				vObjectT = vTangent;
				vObjectB = vBinormal;
			}
		}
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = mul(matWorldOfInstance, vObjectPos);
		half3 vWorldN = normalize(mul((half3x3)matWorldOfInstance, vObjectN));
		const bool use_motion_blur = false;
		if(use_motion_blur){
			float4 vWorldPos1;
			float3 moveDirection;
			if(true){
				const float blur_len = 0.2f;
				moveDirection = -normalize(float3(matWorldOfInstance[0][0], matWorldOfInstance[1][0], matWorldOfInstance[2][0]));
				moveDirection.y -= blur_len * 0.285;
				vWorldPos1 = vWorldPos + float4(moveDirection, 0) * blur_len;
			}
			else {
				vWorldPos1 = mul(matMotionBlur, vObjectPos);
				moveDirection = normalize(vWorldPos1.xyz - vWorldPos.xyz);
			}
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.1f) ? 1: 0;
			float y_factor = saturate(vObjectPos.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float start_alpha = 1.1f;
			float end_alpha = start_alpha - 1.8f;
			float alpha = saturate(lerp(start_alpha, end_alpha, delta_coefficient_smooth));
			vVertexColor.a = saturate(0.5f - vObjectPos.y) + alpha + 0.25;
		}
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0.xy = tc;
		half3 viewdir;
		if(use_bumpmap){
			half3 vWorld_binormal = normalize(mul((half3x3)matWorldOfInstance, vObjectB));
			half3 vWorld_tangent = normalize(mul((half3x3)matWorldOfInstance, vObjectT));
			half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, half3(0, 0, 1));
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse_ex_1(vWorldPos.xyz, vWorldN, false).rgb;
			#endif
			viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir.xyz = mul(TBNMatrix, viewdir);
		}
		else{
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos.xyz, vWorldN, false).rgb;
			#endif
			viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir.xyz = viewdir;
			Out.SunLightDir = vWorldN;
		}
		Out.VertexColor.a *= vMaterialColor.a;
		if(use_envmap){
			half2 envpos;
			half3 tempvec = viewdir - vWorldN;
			envpos.x = (tempvec.y);
			envpos.y = tempvec.z;
			envpos += 1.0h;
			Out.Tex0.zw = envpos;
		}
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			if(PcfMode != PCF_NVIDIA){
				Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
			}
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_standart(VS_OUTPUT_STANDART In, uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_specularfactor, uniform const bool use_specularmap, uniform const bool ps2x, uniform const bool use_aniso, uniform const bool terrain_color_ambient = true, uniform const bool use_envmap = false, uniform const bool use_coloredspecmap = false, uniform const int parallaxmapping_type = 0){
		PS_OUTPUT Output;
		if(parallaxmapping_type > 0){
			float factor = (0.01f * vSpecularColor.x);
			float BIAS = (factor * -0.5f);
			float SCALE = factor;
			float3 View = normalize(In.ViewDir);
			if(parallaxmapping_type == 1){
				float4 Normal = tex2D(NormalTextureSampler, In.Tex0);
				float h = Normal.a * SCALE + BIAS;
				In.Tex0.xy += h * Normal.z * View.xy;
			}
			if(parallaxmapping_type == 2){
				const int ITERS = 3;
				float3 uvh = float3(In.Tex0.xy, 0.0f);
				for(int i = 0; i < ITERS; i++){
					float4 Normal = tex2D(NormalTextureSampler, uvh.xy);
					float h = Normal.a * SCALE + BIAS;
					uvh += (h - uvh.z) * Normal.z * View;
				}
				In.Tex0.xy = uvh.xy;
			}
		}
		half3 normal = use_bumpmap ? (2.0h * tex2D(NormalTextureSampler, In.Tex0.xy).xyz - 1.0h): In.SunLightDir.rgb;
		half sun_amount = (PcfMode != PCF_NONE) ? (((PcfMode == PCF_NVIDIA) || ps2x) ? ((PcfMode == PCF_NVIDIA) ? GetSunAmountNvidia(In.ShadowTexCoord): GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos)): GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos)): 1.0h;
		half4 sum_ammount_color_mul = vSunColor * sun_amount;
		const uint ambientTermType = (terrain_color_ambient && (ps2x || !use_specularfactor)) ? 1: 0;
		const half3 DirToSky = use_bumpmap ? In.SkyLightDir: half3(0.0h, 0.0h, 1.0h);
		half3 total_light = get_ambientTerm(ambientTermType, normal, DirToSky, sun_amount);
		half3 aniso_specular;
		if(use_aniso){
			aniso_specular = calculate_hair_specular(normal, float3(0, 1, 0), ((use_bumpmap) ? In.SunLightDir.xyz: -vSunDir), In.ViewDir.xyz, In.Tex0.xy);
		}
		if(use_bumpmap){
			if(use_aniso){
				total_light.rgb += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) + aniso_specular) * sum_ammount_color_mul.rgb;
			}
			else {
				total_light.rgb += (saturate(dot(In.SunLightDir.xyz, normal.xyz))) * sum_ammount_color_mul.rgb;
			}
			if(ps2x || !use_specularfactor){
				total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
			}
			#ifdef INCLUDE_VERTEX_LIGHTING
				if(ps2x || !use_specularfactor || (PcfMode == PCF_NONE)){
					total_light.rgb += In.VertexLighting;
				}
			#endif
		}
		else {
			if(use_aniso){
				total_light.rgb += (saturate(dot( - vSunDir, normal.xyz)) + aniso_specular) * sum_ammount_color_mul.rgb;
			}
			else {
				total_light.rgb += (saturate(dot( - vSunDir, normal.xyz))) * sum_ammount_color_mul.rgb;
			}
			if(ambientTermType != 1 && !ps2x){
				total_light += saturate(dot( - vSkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
			}
			#ifdef INCLUDE_VERTEX_LIGHTING
				total_light.rgb += In.VertexLighting;
			#endif
		}
		Output.RGBColor.rgb = (PcfMode != PCF_NONE) ? total_light.rgb: min(total_light.rgb, 2.0h);
		Output.RGBColor.rgb *= vMaterialColor.rgb;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor.rgb *= tex_col.rgb;
		Output.RGBColor.rgb *= In.VertexColor.rgb;
		if(use_specularfactor){
			half4 fSpecular = 0;
			half4 specColor = 0.1 * spec_coef * vSpecularColor;
			if(use_specularmap){
				half spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0.xy).rgb, 0.33);
				if(use_coloredspecmap){
					half4 exponential = (specColor * 16.5);
					specColor = (Output.RGBColor * spec_tex_factor) * exponential;
				}
				else {
					specColor *= spec_tex_factor;
				}
			}
			else {
				specColor *= tex_col.a;
			}
			half4 sun_specColor = specColor * sum_ammount_color_mul;
			half3 vHalf = normalize(In.ViewDir.xyz + ((use_bumpmap) ? In.SunLightDir: -vSunDir));
			fSpecular = sun_specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower);
			if(use_envmap){
				half3 envColor = tex2D(EnvTextureSampler, In.Tex0.zw).rgb;
				if(use_coloredspecmap){
					fSpecular.rgb += (specColor.rgb * envColor) * 0.039;
				}
				else {
					fSpecular.rgb += (specColor.rgb * envColor) * 0.035;
				}
			}
			if(PcfMode != PCF_DEFAULT){
				fSpecular *= In.VertexColor;
			}
			if(use_bumpmap){
				if(ps2x || (PcfMode == PCF_NONE)){
				}
			}
			Output.RGBColor += fSpecular;
		}
		else if(use_specularmap){
			GIVE_ERROR_HERE;
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = In.VertexColor.a;
		if((!use_specularfactor) || use_specularmap){
			Output.RGBColor.a *= tex_col.a;
		}
		return Output;
	}
	PS_OUTPUT ps_main_standart_old_good(VS_OUTPUT_STANDART In, uniform const int PcfMode, uniform const bool use_specularmap, uniform const bool use_aniso){
		PS_OUTPUT Output;
		half sun_amount = 1.0h;
		if(PcfMode != PCF_NONE){
			sun_amount = 0.03h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		half3 normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0.xy).xyz - 1.0h);
		const half3 DirToSky = In.SkyLightDir;
		half3 total_light = get_ambientTerm(1, normal, DirToSky, sun_amount);
		half4 specColor = vSunColor * (vSpecularColor * 0.1);
		if(use_specularmap){
			half spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0.xy).rgb, 0.33);
			specColor *= spec_tex_factor;
		}
		half3 vHalf = normalize(In.ViewDir.xyz + In.SunLightDir);
		half4 fSpecular = specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower);
		if(use_aniso){
			half3 tangent_ = half3(0, 1, 0);
			fSpecular.rgb += calculate_hair_specular(normal, tangent_, In.SunLightDir, In.ViewDir.xyz, In.Tex0.xy);
		}
		else{
			fSpecular.rgb *= spec_coef;
		}
		total_light += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) + fSpecular.rgb) * sun_amount * vSunColor.rgb;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor.rgb;
		#ifdef INCLUDE_VERTEX_LIGHTING
			total_light.rgb += In.VertexLighting;
		#endif
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0h;
		Output.RGBColor *= vMaterialColor;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = In.VertexColor.a * tex_col.a;
		return Output;
	}
	#ifdef USE_PRECOMPILED_SHADER_LISTS
		#define DEFINE_STANDART_TECHNIQUE(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor, flagwave_type, use_coloredspecmap) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, terraincolor, false, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, terraincolor, false, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, false, use_coloredspecmap);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_HIGH(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor, flagwave_type, use_envmap, use_coloredspecmap, parallaxmapping_type) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor, flagwave_type, use_envmap, use_coloredspecmap) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor, use_envmap, use_coloredspecmap);\
				}\
			}
		#define DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, flagwave_type, use_envmap, use_coloredspecmap, parallaxmapping_type) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}
	#else
		#define DEFINE_STANDART_TECHNIQUE(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, flagwave_type, use_coloredspecmap) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, true, false, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, true, false, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, false);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, false, use_coloredspecmap);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_HIGH(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, flagwave_type, use_envmap, use_coloredspecmap, parallaxmapping_type) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, flagwave_type, use_envmap, use_coloredspecmap) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, false, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap);\
				}\
			}
		#define DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, flagwave_type, use_envmap, use_coloredspecmap, parallaxmapping_type) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_a vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, use_skinning, flagwave_type, use_envmap);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, true, use_envmap, use_coloredspecmap, parallaxmapping_type);\
				}\
			}
	#endif
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospecmap, true, false, true, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_specmap, true, false, true, true, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_specmap_colorspec, true, false, true, true, false, true, 0, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_nospecmap, true, true, true, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_specmap, true, true, true, true, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_specmap_colorspec, true, true, true, true, false, true, 0, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_nospecmap_high, true, true, true, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_specmap_high, true, true, true, true, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_specmap_colorspec_high, true, true, true, true, false, true, 0, false, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospecmap_high, true, false, true, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_specmap_high, true, false, true, true, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_specmap_colorspec_high, true, false, true, true, false, true, 0, false, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse, false, false, true, true, false, true, 0, true, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse_colorspec, false, false, true, true, false, true, 0, true, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse_skin_bump, true, true, true, true, false, true, 0, true, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse_skin_bump_colorspec, true, true, true, true, false, true, 0, true, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse_noskin_bump, true, false, true, true, false, true, 0, true, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(envmap_specular_diffuse_noskin_bump_colorspec, true, false, true, true, false, true, 0, true, true, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospecmap, false, false, true, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospecmap_noterraincolor, false, false, true, false, false, false, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_specmap, false, false, true, true, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_specmap_colorspec, false, false, true, true, false, true, 0, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_nospecmap, false, true, true, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_specmap, false, true, true, true, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_specmap_colorspec, false, true, true, true, false, true, 0, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospec, false, false, false, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospec_noterraincolor, false, false, false, false, false, false, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospec, true, false, false, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospec_noterraincolor, true, false, false, false, false, false, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_nospec, false, true, false, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_nospec_noterraincolor, false, true, false, false, false, false, 0, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_nospec, true, true, false, false, false, true, 0, false)
	DEFINE_STANDART_TECHNIQUE(pennon_shader_nobump, false, false, false, false, false, false, 1, false)
	DEFINE_STANDART_TECHNIQUE(flag_shader_nobump, false, false, false, false, false, false, 2, false)
	DEFINE_STANDART_TECHNIQUE(sail_shader_nobump, false, false, false, false, false, false, 3, false)
	DEFINE_STANDART_TECHNIQUE(ctf_pennon_shader_nobump, false, false, false, false, false, false, 4, false)
	DEFINE_STANDART_TECHNIQUE(pennon_vertical_shader_nobump, false, false, false, false, false, false, 1, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospec_high, true, false, false, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bumpparallaxfloor_nospec_high, true, false, false, false, false, true, 0, false, false, 1)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bumpparallax_nospec_high, true, false, false, false, false, true, 0, false, false, 2)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_nobump_nospec_high, false, false, false, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospec_high_noterraincolor, true, false, false, false, false, false, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bumpparallaxfloor_nospec_high_noterraincolor, true, false, false, false, false, false, 0, false, false, 1)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bumpparallax_nospec_high_noterraincolor, true, false, false, false, false, false, 0, false, false, 2)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_nobump_nospec_high_noterraincolor, false, false, false, false, false, false, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_nospec_high, true, true, false, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_nobump_nospec_high, false, true, false, false, false, true, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_nobump_nospec_high_noterraincolor, false, true, false, false, false, false, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(pennon_shader, true, false, false, false, false, false, 1, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(flag_shader, true, false, false, false, false, false, 2, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(sail_shader, true, false, false, false, false, false, 3, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(ctf_pennon_shader, true, false, false, false, false, false, 4, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(pennon_vertical_shader, true, false, false, false, false, false, 5, false, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_specmap_Instanced, false, false, true, true, false, true, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_specmap_Instanced_colorspec, false, false, true, true, false, true, 0, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_specmap_Instanced, true, false, true, true, false, true, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_specmap_Instanced_colorspec, true, false, true, true, false, true, 0, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospecmap_Instanced, false, false, false, false, false, true, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospecmap_noterraincolor_Instanced, false, false, false, false, false, false, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_nospec_high_Instanced, true, false, false, false, false, true, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_nospec_high_noterraincolor_Instanced, true, false, false, false, false, false, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospec_high_Instanced, false, false, false, false, false, true, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospec_high_noterraincolor_Instanced, false, false, false, false, false, false, 0, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bump_specmap_high_Instanced, true, false, true, true, false, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bump_specmap_high_Instanced_colorspec, true, false, true, true, false, 0, false, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bump_nospecmap_high_Instanced, true, false, false, false, false, 0, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bumpparallaxfloor_nospecmap_high_Instanced, true, false, false, false, false, 0, false, false, 1)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bumpparallax_nospecmap_high_Instanced, true, false, false, false, false, 0, false, false, 2)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(envmap_specular_diffuse_Instanced, false, false, true, true, false, 0, true, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(envmap_specular_diffuse_Instanced_colorspec, false, false, true, true, false, 0, true, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(envmap_specular_diffuse_Instanced_bump, true, false, true, true, false, 0, true, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(envmap_specular_diffuse_Instanced_bump_colorspec, true, false, true, true, false, 0, true, true, false)
	technique standart_skin_bump_nospecmap_high_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, true, true, false, false);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_NONE, false, true);
		}
	}
	technique standart_skin_bump_nospecmap_high_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, true, true, false, false);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_DEFAULT, false, true);
		}
	}
	technique standart_skin_bump_nospecmap_high_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, true, true, false, false);
			PixelShader = compile ps_2_a ps_main_standart_old_good(PCF_NVIDIA, false, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, true, true, false, false);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_NONE, true, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, true, true, false, false);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_DEFAULT, true, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, true, true, false, false);
			PixelShader = compile ps_2_a ps_main_standart_old_good(PCF_NVIDIA, true, true);
		}
	}
#endif
#ifdef HAIR_SHADERS
	struct VS_OUTPUT_SIMPLE_HAIR{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_SIMPLE_HAIR vs_hair(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_SIMPLE_HAIR, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = vColor * diffuse_light;
		half wNdotSun = dot(vWorldN, -vSunDir);
		Out.SunLight = max(0.2f * (wNdotSun + 0.9f), wNdotSun) * vSunColor * vColor;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_hair(VS_OUTPUT_SIMPLE_HAIR In, uniform const int PcfMode, uniform const bool no_blend = false){
		PS_OUTPUT Output;
		half4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		half4 tex2_col;
		if(!no_blend){
			tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		}
		half4 final_col;
		INPUT_TEX_GAMMA(tex1_col.rgb);
		final_col = tex1_col * vMaterialColor;
		if(!no_blend){
			half alpha = saturate(((2.0h * vMaterialColor2.a) + tex2_col.a) - 1.9h);
			final_col.rgb *= (1.0h - alpha);
			final_col.rgb += tex2_col.rgb * alpha;
		}
		half4 total_light = In.Color;
		if((PcfMode != PCF_NONE)){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			total_light.rgb += In.SunLight.rgb * sun_amount;
		}
		else {
			total_light.rgb += In.SunLight.rgb;
		}
		Output.RGBColor = final_col * total_light;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(hair_shader, vs_hair, ps_hair)
	technique hair_shader_noblend{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair(PCF_NONE);
			PixelShader = compile PS_2_X ps_hair(PCF_NONE, true);
		}
	}
	technique hair_shader_noblend_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair(PCF_DEFAULT);
			PixelShader = compile PS_2_X ps_hair(PCF_DEFAULT, true);
		}
	}
	technique hair_shader_noblend_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_hair(PCF_NVIDIA);
			PixelShader = compile ps_2_a ps_hair(PCF_NVIDIA, true);
		}
	}
	struct VS_INPUT_HAIR{
		float4 vPosition: POSITION;
		float3 vNormal: NORMAL;
		float3 vTangent: BINORMAL;
		float2 tc: TEXCOORD0;
		float4 vColor: COLOR0;
	};
	struct VS_OUTPUT_HAIR{
		float4 Pos: POSITION;
		half4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 VertexLighting: TEXCOORD1;
		half3 normal: TEXCOORD2;
		half3 tangent: TEXCOORD3;
		half3 viewVec: TEXCOORD4;
		half3 ShadowTexCoord: TEXCOORD5;
		half2 ShadowTexelPos: TEXCOORD6;
		float Fog: FOG;
	};
	VS_OUTPUT_HAIR vs_hair_aniso(uniform const int PcfMode, VS_INPUT_HAIR In){
		INITIALIZE_OUTPUT(VS_OUTPUT_HAIR, Out);
		Out.Pos = mul(matWorldViewProj, In.vPosition);
		float4 vWorldPos = (float4)mul(matWorld, In.vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, In.vNormal));
		Out.Tex0 = In.tc;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.VertexLighting = saturate(In.vColor * diffuse_light);
		Out.VertexColor = In.vColor;
		if(true){
			half3 Pview = (vCameraPos.xyz - vWorldPos.xyz);
			Out.normal = vWorldN;
			Out.tangent = normalize(mul((float3x3)matWorld, In.vTangent));
			Out.viewVec = normalize(Pview);
		}
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_hair_aniso(VS_OUTPUT_HAIR In, uniform const int PcfMode, uniform const bool no_blend = false){
		PS_OUTPUT Output;
		half3 lightDir = -vSunDir;
		half3 hairBaseColor = vMaterialColor.rgb;
		half3 diffuse = hairBaseColor * vSunColor.rgb * In.VertexColor.rgb * HairDiffuseTerm(In.normal, lightDir);
		half4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex1_col.rgb);
		half4 tex2_col;
		half alpha;
		if(!no_blend){
			tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
			alpha = saturate(((2.0h * vMaterialColor2.a) + tex2_col.a) - 1.9h);
		}
		half4 final_col = tex1_col;
		final_col.rgb *= hairBaseColor;
		if(!no_blend){
			final_col.rgb *= (1.0h - alpha);
			final_col.rgb += tex2_col.rgb * alpha;
		}
		half sun_amount = 1.0h;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		half3 specular = calculate_hair_specular(In.normal, In.tangent, lightDir, In.viewVec, In.Tex0);
		half3 total_light = vAmbientColor.rgb;
		total_light.rgb += (((diffuse + specular) * sun_amount));
		total_light.rgb += In.VertexLighting.rgb;
		Output.RGBColor.rgb = total_light.rgb * final_col.rgb;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = tex1_col.a * vMaterialColor.a;
		Output.RGBColor = saturate(Output.RGBColor);
		return Output;
	}
	technique hair_shader_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_NONE);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_NONE, false);
		}
	}
	technique hair_shader_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_DEFAULT);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_DEFAULT, false);
		}
	}
	technique hair_shader_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_hair_aniso(PCF_NVIDIA);
			PixelShader = compile ps_2_a ps_hair_aniso(PCF_NVIDIA, false);
		}
	}
	technique hair_shader_aniso_noblend{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_NONE);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_NONE, true);
		}
	}
	technique hair_shader_aniso_noblend_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_DEFAULT);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_DEFAULT, true);
		}
	}
	technique hair_shader_aniso_noblend_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_hair_aniso(PCF_NVIDIA);
			PixelShader = compile ps_2_a ps_hair_aniso(PCF_NVIDIA, true);
		}
	}
#endif
#ifdef FACE_SHADERS
	struct VS_OUTPUT_SIMPLE_FACE{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_SIMPLE_FACE vs_face(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_SIMPLE_FACE, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half4 vMaterialColorCombination = vMaterialColor * vColor;
		Out.Color = vMaterialColorCombination * diffuse_light;
		half wNdotSun = dot(vWorldN, -vSunDir);
		Out.SunLight = max(0.2h * (wNdotSun + 0.9h), wNdotSun) * vSunColor * vMaterialColorCombination;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_face(VS_OUTPUT_SIMPLE_FACE In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		half4 tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		half4 tex_col;
		tex_col = tex2_col * In.Color.a + tex1_col * (1.0h - In.Color.a);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if((PcfMode != PCF_NONE)){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		}
		else {
			Output.RGBColor = tex_col * (In.Color + In.SunLight);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = vMaterialColor.a;
		return Output;
	}
	DEFINE_TECHNIQUES(face_shader, vs_face, ps_face)
	DEFINE_LIGHTING_TECHNIQUE(face_shader, 0, 0, 0, 0, 0)
	struct VS_INPUT_FACE{
		float4 Position: POSITION;
		float2 TC: TEXCOORD0;
		float4 VertexColor: COLOR0;
		float3 Normal: NORMAL;
		float3 Tangent: TANGENT;
		float3 Binormal: BINORMAL;
	};
	struct VS_OUTPUT_FACE{
		float4 Pos: POSITION;
		half4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 WorldPos: TEXCOORD1;
		half3 ViewVec: TEXCOORD2;
		half3 SunLightDir: TEXCOORD3;
		half4 ShadowTexCoord: TEXCOORD4;
		half2 ShadowTexelPos: TEXCOORD5;
		#ifdef INCLUDE_VERTEX_LIGHTING
			half3 VertexLighting: TEXCOORD6;
		#endif
		float Fog: FOG;
	};
	VS_OUTPUT_STANDART vs_main_standart_face_mod(uniform const int PcfMode, uniform const bool use_bumpmap, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		half3 vObjectN, vObjectT, vObjectB;
		vObjectPos = vPosition;
		vObjectN = vNormal;
		if(use_bumpmap){
			vObjectT = vTangent;
			vObjectB = vBinormal;
		}
		float4 vWorldPos = mul(matWorld, vObjectPos);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0.xy = tc;
		half3 vWorldN = normalize(mul((float3x3)matWorld, vObjectN));
		half3x3 TBNMatrix;
		if(use_bumpmap){
			half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vObjectB));
			half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vObjectT));
			TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		}
		if(use_bumpmap){
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, -vSkyLightDir);
		}
		else{
			Out.SunLightDir = vWorldN;
		}
		Out.VertexColor = vVertexColor;
		if(use_bumpmap){
			Out.ViewDir.xyz = mul(TBNMatrix, normalize(vCameraPos.xyz - vWorldPos.xyz));
		}
		else{
			Out.ViewDir.xyz = normalize(vCameraPos.xyz - vWorldPos.xyz);
		}
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			if(PcfMode != PCF_NVIDIA){
				Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
			}
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_standart_face_mod(VS_OUTPUT_STANDART In, uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_ps2a){
		PS_OUTPUT Output;
		half3 total_light = vAmbientColor.rgb;
		half3 normal;
		if(use_bumpmap){
			half3 tex1_norm, tex2_norm;
			tex1_norm = tex2D(NormalTextureSampler, In.Tex0.xy).rgb;
			if(use_ps2a){
				tex2_norm = tex2D(SpecularTextureSampler, In.Tex0.xy).rgb;
				normal = lerp(tex1_norm, tex2_norm, In.VertexColor.a);
				normal = 2.0h * normal - 1.0h;
				normal = normalize(normal);
			}
			else{
				normal = (2.0h * tex1_norm - 1.0h);
			}
		}
		else{
			normal = In.SunLightDir.xyz;
		}
		half sun_amount = 1.0;
		if(PcfMode != PCF_NONE){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		if(use_bumpmap){
			total_light += face_NdotL(In.SunLightDir.xyz, normal.xyz) * sun_amount * vSunColor.rgb;
			if(use_ps2a){
				total_light += face_NdotL(In.SkyLightDir.xyz, normal.xyz) * vSkyLightColor.rgb;
			}
		}
		else {
			total_light += face_NdotL( - vSunDir, normal.xyz) * sun_amount * vSunColor.rgb;
			if(use_ps2a){
				total_light += face_NdotL( - vSkyLightDir, normal.xyz) * vSkyLightColor.rgb;
			}
		}
		if(PcfMode != PCF_NONE)Output.RGBColor.rgb = total_light.rgb;
		else Output.RGBColor.rgb = min(total_light.rgb, 2.0f);
		half3 tex1_col = tex2D(MeshTextureSampler, In.Tex0.xy).rgb;
		half3 tex2_col = tex2D(Diffuse2Sampler, In.Tex0.xy).rgb;
		half3 tex_col = lerp(tex1_col, tex2_col, In.VertexColor.a);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor.rgb *= tex_col;
		Output.RGBColor.rgb *= (In.VertexColor.rgb * vMaterialColor.rgb);
		if(use_ps2a){
			half3 fSpecular = 0;
			half3 specColor = vSpecularColor.rgb * vSunColor.rgb;
			if(false){
				specColor *= tex2D(SpecularTextureSampler, In.Tex0.xy).rgb;
			}
			half3 vHalf = normalize(In.ViewDir.xyz + In.SunLightDir.xyz);
			fSpecular = specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower) * sun_amount;
			half fresnel = saturate(1.0h - dot(In.ViewDir.xyz, normal));
			Output.RGBColor.rgb += fSpecular * fresnel;
		}
		Output.RGBColor.rgb = saturate(OUTPUT_GAMMA(Output.RGBColor.rgb));
		Output.RGBColor.a = vMaterialColor.a;
		return Output;
	}
	technique face_shader_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_NONE, true);
			PixelShader = compile ps_2_0 ps_main_standart_face_mod(PCF_NONE, true, false);
		}
	}
	technique face_shader_high_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_DEFAULT, true);
			PixelShader = compile ps_2_0 ps_main_standart_face_mod(PCF_DEFAULT, true, false);
		}
	}
	technique face_shader_high_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart_face_mod(PCF_NVIDIA, true);
			PixelShader = compile ps_2_a ps_main_standart_face_mod(PCF_NVIDIA, true, false);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(face_shader_high, 0, 1, 0, 0, 0)
	technique faceshader_high_specular{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_NONE, true);
			PixelShader = compile PS_2_X ps_main_standart_face_mod(PCF_NONE, true, true);
		}
	}
	technique faceshader_high_specular_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_DEFAULT, true);
			PixelShader = compile PS_2_X ps_main_standart_face_mod(PCF_DEFAULT, true, true);
		}
	}
	technique faceshader_high_specular_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart_face_mod(PCF_NVIDIA, true);
			PixelShader = compile ps_2_a ps_main_standart_face_mod(PCF_NVIDIA, true, true);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(faceshader_high_specular, 0, 1, 0, 0, 0)
	technique faceshader_simple{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_NONE, false);
			PixelShader = compile ps_2_0 ps_main_standart_face_mod(PCF_NONE, false, false);
		}
	}
	technique faceshader_simple_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart_face_mod(PCF_DEFAULT, false);
			PixelShader = compile ps_2_0 ps_main_standart_face_mod(PCF_DEFAULT, false, false);
		}
	}
	technique faceshader_simple_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart_face_mod(PCF_NVIDIA, false);
			PixelShader = compile ps_2_a ps_main_standart_face_mod(PCF_NVIDIA, false, false);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(faceshader_high_specular, 0, 1, 0, 0, 0)
	VS_OUTPUT vs_main_skin(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES, uniform const int PcfMode){
		INITIALIZE_OUTPUT(VS_OUTPUT, Out);
		float4 vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
		half3 vObjectN = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vNormal) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vNormal) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vNormal) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vNormal) * vBlendWeights.w);
		float4 vWorldPos = mul(matWorld, vObjectPos);
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		half3 vWorldN = normalize(mul((half3x3)matWorld, vObjectN));
		Out.Tex0 = tc;
		Out.Color = vAmbientColor;
		Out.Color += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half4 stuffcolors = vMaterialColor * vColor;
		Out.Color *= stuffcolors;
		Out.Color = min(1, Out.Color);
		half wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = wNdotSun * vSunColor * stuffcolors;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	technique skin_diffuse{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_skin(PCF_NONE);
			PixelShader = ps_main_compiled_PCF_NONE;
		}
	}
	technique skin_diffuse_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_skin(PCF_DEFAULT);
			PixelShader = ps_main_compiled_PCF_DEFAULT;
		}
	}
	technique skin_diffuse_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_skin(PCF_NVIDIA);
			PixelShader = ps_main_compiled_PCF_NVIDIA;
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(skin_diffuse, 0, 0, 1, 0, 0)
#endif
#ifdef FLORA_SHADERS
	struct VS_OUTPUT_FLORA{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	struct VS_OUTPUT_FLORA_NO_SHADOW{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	float3 CalcFloraVertexAnimation(float2 Offset){
		const float displacementamp = 0.033f * -40.0f;
		const float displacementfreq = 0.33f * 3.6f;
		const float displacementscale = 2.5f * (vFloraWindStrength * 0.129f);
		const float grandwave = sin(time_var * 0.2f);
		float3 wind = 0;
		wind.x += sin(Offset.x * displacementamp + time_var * displacementfreq) * displacementscale;
		wind.y += cos(Offset.y * (displacementamp * 0.5f) + time_var * displacementfreq) * (displacementscale * 0.5f);
		wind *= grandwave;
		wind.x -= displacementscale;
		wind.z += (wind.x * 0.5f);
		wind.y -= (displacementscale * 0.5f);
		return wind;
	}
	float3 CalcFloraPineVertexAnimation(float3 Offset){
		const float displacementamp = 0.033f * -40.0f;
		const float displacementfreq = 0.33f * 3.6f;
		const float displacementscale = 2.5f * (vFloraWindStrength * 0.129f);
		const float grandwave = sin(time_var * 0.2f);
		float3 wind = 0;
		wind.x += sin((Offset.z - (Offset.x * 2.0f)) * displacementamp + time_var * displacementfreq) * displacementscale;
		wind.y += cos(Offset.y * (displacementamp * 0.5f) + time_var * displacementfreq) * (displacementscale * 0.5f);
		wind *= grandwave;
		wind.x -= displacementscale;
		wind.z += (wind.x * 0.5f);
		wind.y -= (displacementscale * 0.5f);
		return wind;
	}
	VS_OUTPUT_FLORA vs_plume(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_plume_no_shadow(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_plume_skinned(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		float4 vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		float4 vWorldPos = (float4)mul(matWorld, vObjectPos);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_plume_skinned_no_shadow(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		float4 vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vObjectPos).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_plume_move(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.x >= 0.18 && tc.x <= 0.25) && (tc.y >= 0.18 && tc.y <= 0.25)){
		}
		else {
			vPosition.x += (sin((time_var * 2.3) + vPosition.x + (vPosition.y - vPosition.x)) * 0.015) * noise(time_var);
			vPosition.y += (sin((time_var * 2.3) + vPosition.y + (vPosition.y - vPosition.x)) * 0.015) * noise(time_var);
			vPosition.z += (cos((time_var * 2.3) + vPosition.z + (vPosition.y - vPosition.x)) * 0.015) * noise(time_var);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_plume_move_no_shadow(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		vPosition.z += ((sin((time_var * 2.2) + ((vPosition.z + (vPosition.y - vPosition.x)) * 0.1))) * 0.14);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_flowermove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.x <= 0.41) || (tc.y >= 0.55) || (tc.x >= 0.65) || (tc.x <= 0.5 && tc.y <= 0.2)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_flowermove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.x <= 0.41) || (tc.y >= 0.55) || (tc.x >= 0.65) || (tc.x <= 0.5 && tc.y <= 0.2)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_sprucemove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y >= 0.60) || (tc.y >= 0.1 && tc.y <= 0.45) || (tc.y >= 0.45 && tc.x <= 0.85)){
			vPosition.xyz += CalcFloraPineVertexAnimation(vPosition.xyz);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_sprucemove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y >= 0.60) || (tc.y >= 0.1 && tc.y <= 0.45) || (tc.y >= 0.45 && tc.x <= 0.85)){
			vPosition.xyz += CalcFloraPineVertexAnimation(vPosition.xyz);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_oimleavesmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y <= 0.1) || (tc.x >= 0.65) || (tc.y >= 0.3 && tc.y <= 0.5) || (tc.y >= 0.65 && tc.y <= 0.8) || ((tc.y >= 0.5 && tc.y <= 0.65) && ((tc.x >= 0.1 && tc.x <= 0.25) || (tc.x >= 0.5))) || ((tc.y >= 0.8) && (tc.x >= 0.15 && tc.x <= 0.43))){
			vPosition.xyz += CalcFloraPineVertexAnimation(vPosition.xyz);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_oimleavesmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y <= 0.1) || (tc.x >= 0.65) || (tc.y >= 0.3 && tc.y <= 0.5) || (tc.y >= 0.65 && tc.y <= 0.8) || ((tc.y >= 0.5 && tc.y <= 0.65) && ((tc.x >= 0.1 && tc.x <= 0.25) || (tc.x >= 0.5))) || ((tc.y >= 0.8) && (tc.x >= 0.15 && tc.x <= 0.43))){
			vPosition.xyz += CalcFloraPineVertexAnimation(vPosition.xyz);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_topmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if(tc.y <= 0.25){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_topmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if(tc.y <= 0.25){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_topleftmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y <= 0.25) || (tc.x <= 0.25)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_topleftmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y <= 0.25) || (tc.x <= 0.25)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_toprightmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y <= 0.25) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_toprightmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y <= 0.25) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_topleftrightmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y <= 0.25) || (tc.x <= 0.25) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_topleftrightmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y <= 0.25) || (tc.x <= 0.25) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_bottommove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if(tc.y >= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_bottommove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if(tc.y >= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_bottomrightmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y >= 0.75) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_bottomrightmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y >= 0.75) || (tc.x >= 0.75)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_bottomleftmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y >= 0.75) || (tc.x <= 0.25)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_bottomleftmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y >= 0.75) || (tc.x <= 0.25)){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_leftmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if(tc.x <= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_leftmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if(tc.x <= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_rightmove(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if(tc.x >= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow_rightmove(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if(tc.x >= 0.75){
			vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_Instanced(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = (float4)mul(matWorldOfInstance, vPosition);
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0 = tc;
		const half4 combicolor1 = (vAmbientColor + vSunColor * 0.06h);
		Out.Color = vColor * combicolor1;
		Out.Color.a *= vMaterialColor.a;
		const half4 combicolor2 = (vSunColor * 0.34h) * vMaterialColor;
		Out.SunLight = vColor * combicolor2;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_flora(VS_OUTPUT_FLORA In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.05h);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if(PcfMode != PCF_NONE){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		}
		else {
			Output.RGBColor = tex_col * ((In.Color + In.SunLight));
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_flora_no_shadow(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		vPosition.xyz += CalcFloraVertexAnimation(vPosition.xy);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_flora_no_shadow(VS_OUTPUT_FLORA_NO_SHADOW In){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.05f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = tex_col * In.Color;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VS_OUTPUT_FLORA vs_grass(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		if((tc.y <= 0.25) || (tc.y >= 0.5 && tc.y <= 0.75)){
			const float displacementfreq = 0.33 * 3.6;
			const float displacementscale = 5.0 * (vFloraWindStrength * 0.093);
			const float displacementamp = 0.033 * -14;
			const float grandwave = sin(time_var * 0.2);
			float3 wind = 0;
			wind.x += sin(vPosition.x * displacementamp + time_var * displacementfreq) * displacementscale;
			wind.y += cos(vPosition.y * (displacementamp * 0.5) + time_var * displacementfreq) * (displacementscale * 0.5);
			wind.xy *= grandwave;
			wind.x += 0.20f;
			wind.z -= (wind.x);
			wind.y += 0.10f;
			vPosition.xyz += wind;
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vAmbientColor;
		if(PcfMode != PCF_NONE){
			const half4 precalcsun = (vSunColor * 0.55h) * vMaterialColor;
			Out.SunLight = vColor * precalcsun;
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		else {
			const half4 precalcsun2 = vSunColor * 0.5h;
			Out.SunLight = vColor * precalcsun2;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		Out.Color.a = min(1.0h, (1.0h - (d / 50.0h)) * 2.0h);
		return Out;
	}
	PS_OUTPUT ps_grass(VS_OUTPUT_FLORA In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(GrassTextureSampler, In.Tex0);
		clip(tex_col.a - 0.1h);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if((PcfMode != PCF_NONE)){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		}
		else {
			Output.RGBColor = tex_col * ((In.Color + In.SunLight));
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VS_OUTPUT_FLORA_NO_SHADOW vs_grass_no_shadow(float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA_NO_SHADOW, Out);
		if((tc.y <= 0.25) || (tc.y >= 0.5 && tc.y <= 0.75)){
			const float displacementfreq = 0.33 * 3.6;
			const float displacementscale = 5.0 * (vFloraWindStrength * 0.093);
			const float displacementamp = 0.033 * -14;
			const float grandwave = sin(time_var * 0.2);
			float3 wind = 0;
			wind.x += sin(vPosition.x * displacementamp + time_var * displacementfreq) * displacementscale;
			wind.y += cos(vPosition.y * (displacementamp * 0.5) + time_var * displacementfreq) * (displacementscale * 0.5);
			wind.xy *= grandwave;
			wind.x += 0.20f;
			wind.z -= (wind.x);
			wind.y += 0.10f;
			vPosition.xyz += wind;
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		Out.Color.a = min(1.0h, (1.0h - (d / 50.0h)) * 2.0h);
		return Out;
	}
	PS_OUTPUT ps_grass_no_shadow(VS_OUTPUT_FLORA_NO_SHADOW In){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.1f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = tex_col * In.Color;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(plume, vs_plume, ps_flora)
	technique plume_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_plume_no_shadow();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(plume_skinned, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(plume_skinned, vs_plume_skinned, ps_flora)
	technique plume_skinned_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_plume_skinned_no_shadow();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(plume_skinned, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(plume_move, vs_plume_move, ps_flora)
	technique plume_move_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_plume_move_no_shadow();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(plume_move, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_sprucemove, vs_flora_sprucemove, ps_flora)
	technique flora_sprucemove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_sprucemove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_sprucemove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_flowermove, vs_flora_flowermove, ps_flora)
	technique flora_flowermove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_flowermove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_flowermove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_oimleavesmove, vs_flora_oimleavesmove, ps_flora)
	technique flora_oimleavesmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_oimleavesmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_oimleavesmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_topmove, vs_flora_topmove, ps_flora)
	technique flora_topmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_topmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_topmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_topleftmove, vs_flora_topleftmove, ps_flora)
	technique flora_topleftmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_topleftmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_topleftmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_toprightmove, vs_flora_toprightmove, ps_flora)
	technique flora_toprightmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_toprightmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_toprightmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_topleftrightmove, vs_flora_topleftrightmove, ps_flora)
	technique flora_topleftrightmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_topleftrightmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_topleftrightmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_bottommove, vs_flora_bottommove, ps_flora)
	technique flora_bottommove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_bottommove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_bottommove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_bottomleftmove, vs_flora_bottomleftmove, ps_flora)
	technique flora_bottomleftmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_bottomleftmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_bottomleftmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_bottomrightmove, vs_flora_bottomrightmove, ps_flora)
	technique flora_bottomrightmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_bottomrightmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_bottomrightmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_rightmove, vs_flora_rightmove, ps_flora)
	technique flora_rightmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_rightmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_rightmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_leftmove, vs_flora_leftmove, ps_flora)
	technique flora_leftmove_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow_leftmove();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora_leftmove, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora, vs_flora, ps_flora)
	technique flora_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_flora_no_shadow();
			PixelShader = compile ps_2_0 ps_flora_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(flora, 0, 0, 0, 0, 0)
	DEFINE_TECHNIQUES(flora_Instanced, vs_flora_Instanced, ps_flora)
	technique grass_no_shadow{
		pass P0{
			VertexShader = compile vs_2_0 vs_grass_no_shadow();
			PixelShader = compile ps_2_0 ps_grass_no_shadow();
		}
	}
	DEFINE_TECHNIQUES(grass, vs_grass, ps_grass)
	technique grass_PRESHADED{
		pass P0{
			VertexShader = compile vs_2_0 vs_grass_no_shadow();
			PixelShader = compile ps_2_0 ps_grass_no_shadow();
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(grass, 0, 0, 0, 0, 0)
#endif
#ifdef MAP_SHADERS
	struct VS_OUTPUT_MAP{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
		half3 ViewDir: TEXCOORD4;
		half3 WorldNormal: TEXCOORD5;
	};
	VS_OUTPUT_MAP vs_main_map(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		half wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_map(VS_OUTPUT_MAP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		half sun_amount = 1;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		{
			float fresnel = 1 - (saturate(dot(normalize(In.ViewDir), normalize(In.WorldNormal))));
			fresnel *= fresnel;
			Output.RGBColor.rgb *= max(0.6f, fresnel + 0.1f);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(diffuse_map, vs_main_map, ps_main_map)
	struct VS_OUTPUT_MAP_BUMP{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half3 ShadowTexCoord: TEXCOORD1;
		half2 ShadowTexelPos: TEXCOORD2;
		float Fog: FOG;
		half3 SunLightDir: TEXCOORD3;
		half3 SkyLightDir: TEXCOORD4;
		half3 ViewDir: TEXCOORD5;
		half3 WorldNormal: TEXCOORD6;
	};
	VS_OUTPUT_MAP_BUMP vs_main_map_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_map_bump(VS_OUTPUT_MAP_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		half3 normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0 * map_normal_detail_factor).rgb - 1.0h);
		half4 In_SunLight = saturate(dot(normal, In.SunLightDir)) * vSunColor * vMaterialColor;
		half sun_amount = 1;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		Output.RGBColor = tex_col * ((In.Color + In_SunLight * sun_amount));
		{
			float fresnel = 1 - (saturate(dot(normalize(In.ViewDir), normalize(In.WorldNormal))));
			fresnel *= fresnel;
			Output.RGBColor.rgb *= max(0.6, fresnel + 0.1);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(diffuse_map_bump, vs_main_map_bump, ps_main_map_bump)
	struct VS_OUTPUT_MAP_MOUNTAIN{
		float4 Pos: POSITION;
		float Fog: FOG;
		half4 Color: COLOR0;
		float3 Tex0: TEXCOORD0;
		half4 SunLight: TEXCOORD1;
		half3 ShadowTexCoord: TEXCOORD2;
		half2 ShadowTexelPos: TEXCOORD3;
		half3 ViewDir: TEXCOORD6;
		half3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_MAP_MOUNTAIN vs_map_mountain(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_MOUNTAIN, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0.xy = tc;
		Out.Tex0.z = (0.7f * (vWorldPos.z - 1.5f));
		float4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		half wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_map_mountain(VS_OUTPUT_MAP_MOUNTAIN In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(tex_col.rgb);
		tex_col.rgb += saturate(In.Tex0.z * (tex_col.a) - 1.5h);
		tex_col.a = 1.0h;
		if((PcfMode != PCF_NONE)){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = saturate(tex_col) * ((In.Color + In.SunLight * sun_amount));
		}
		else {
			Output.RGBColor = saturate(tex_col) * (In.Color + In.SunLight);
		}
		{
			float fresnel = 1 - (saturate(dot(In.ViewDir, In.WorldNormal)));
			Output.RGBColor.rgb *= max(0.6f, fresnel + 0.1f);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(map_mountain, vs_map_mountain, ps_map_mountain)
	struct VS_OUTPUT_MAP_MOUNTAIN_BUMP{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float3 Tex0: TEXCOORD0;
		half3 ShadowTexCoord: TEXCOORD1;
		half2 ShadowTexelPos: TEXCOORD2;
		float Fog: FOG;
		half3 SunLightDir: TEXCOORD3;
		half3 SkyLightDir: TEXCOORD4;
		half3 ViewDir: TEXCOORD5;
		half3 WorldNormal: TEXCOORD6;
	};
	VS_OUTPUT_MAP_MOUNTAIN_BUMP vs_map_mountain_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_MOUNTAIN_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.Tex0.xy = tc;
		Out.Tex0.z = (0.7f * (vWorldPos.z - 1.5f));
		float4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_map_mountain_bump(VS_OUTPUT_MAP_MOUNTAIN_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		half4 sample_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(sample_col.rgb);
		half4 tex_col = sample_col;
		tex_col.rgb += saturate(In.Tex0.z * (sample_col.a) - 1.5h);
		tex_col.a = 1.0h;
		half3 normal = (2.0h * tex2D(NormalTextureSampler, In.Tex0.xy * map_normal_detail_factor).rgb - 1.0h);
		half4 In_SunLight = saturate(dot(normal, In.SunLightDir)) * vSunColor;
		if((PcfMode != PCF_NONE)){
			half sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = saturate(tex_col) * ((In.Color + In_SunLight * sun_amount));
		}
		else {
			Output.RGBColor = saturate(tex_col) * (In.Color + In_SunLight);
		}
		{
			float fresnel = 1 - (saturate(dot(In.ViewDir, In.WorldNormal)));
			Output.RGBColor.rgb *= max(0.6, fresnel + 0.1);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(map_mountain_bump, vs_map_mountain_bump, ps_map_mountain_bump)
	struct VS_OUTPUT_MAP_WATER{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		half3 LightDir: TEXCOORD1;
		half3 CameraDir: TEXCOORD2;
		half4 PosWater: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_MAP_WATER vs_map_water(uniform const bool reflections, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_WATER, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc + texture_offset.xy;
		half4 diffuse_light = vAmbientColor + vLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half wNdotSun = max( - 0.0001h, dot(vWorldN, -vSunDir));
		diffuse_light += (wNdotSun) * vSunColor;
		Out.Color = (vMaterialColor * vColor) * diffuse_light;
		if(reflections){
			float4 water_pos = mul(matWaterViewProj, vWorldPos);
			Out.PosWater.xy = (float2(water_pos.x, -water_pos.y) + water_pos.w) / 2;
			Out.PosWater.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * water_pos.w);
			Out.PosWater.zw = water_pos.zw;
		}
		{
			half3 vWorldN = half3(0, 0, 1);
			half3 vWorld_tangent = half3(1, 0, 0);
			half3 vWorld_binormal = half3(0, 1, 0);
			half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			half3 point_to_camera_normal = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.CameraDir = mul(TBNMatrix, -point_to_camera_normal);
			Out.LightDir = mul(TBNMatrix, -vSunDir);
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_map_water(uniform const bool reflections, VS_OUTPUT_MAP_WATER In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		half4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		half3 normal;
		normal.xy = (2.0h * tex2D(NormalTextureSampler, In.Tex0 * 8).ag - 1.0h);
		normal.z = sqrt(1.0h - dot(normal.xy, normal.xy));
		half NdotL = saturate(dot(normal, In.LightDir));
		half3 vView = normalize(In.CameraDir);
		float fresnel = 1 - (saturate(dot(vView, normal)));
		fresnel = 0.0204f + 0.9796 * (fresnel * fresnel * fresnel * fresnel * fresnel);
		Output.RGBColor.rgb += fresnel * In.Color.rgb;
		if(reflections){
			In.PosWater.xy += 0.35f * normal.xy;
			half4 tex = tex2Dproj(ReflectionTextureSampler, In.PosWater);
			INPUT_OUTPUT_GAMMA(tex.rgb);
			tex.rgb = min(tex.rgb, 4.0h);
			Output.RGBColor.rgb *= NdotL * lerp(tex_col.rgb, tex.rgb, reflection_factor);
		}
		else {
			Output.RGBColor.rgb *= tex_col.rgb;
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = In.Color.a * tex_col.a;
		return Output;
	}
	technique map_water{
		pass P0{
			VertexShader = compile vs_2_0 vs_map_water(false);
			PixelShader = compile ps_2_0 ps_map_water(false);
		}
	}
	technique map_water_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_map_water(true);
			PixelShader = compile ps_2_0 ps_map_water(true);
		}
	}
#endif
#ifdef SOFT_PARTICLE_SHADERS
	struct VS_DEPTHED_FLARE{
		float4 Pos: POSITION;
		half4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
		float4 projCoord: TEXCOORD1;
		float Depth: TEXCOORD2;
	};
	VS_DEPTHED_FLARE vs_main_depthed_flare(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_DEPTHED_FLARE Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Color = vColor * vMaterialColor;
		Out.Tex0 = tc;
		if(use_depth_effects){
			Out.projCoord.xy = (float2(Out.Pos.x, -Out.Pos.y) + Out.Pos.w) / 2;
			Out.projCoord.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * Out.Pos.w);
			Out.projCoord.zw = Out.Pos.zw;
			Out.Depth = Out.Pos.z * far_clip_Inv;
		}
		float3 vWorldPos = mul(matWorld, vPosition).xyz;
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_depthed_flare(VS_DEPTHED_FLARE In, uniform const bool sun_like, uniform const bool blend_adding){
		PS_OUTPUT Output;
		half4 OutputColor = In.Color;
		OutputColor *= tex2D(MeshTextureSampler, In.Tex0);
		clip(OutputColor.a - 0.01h);
		if(!blend_adding){
			OUTPUT_GAMMA(OutputColor.rgb);
		}
		if(use_depth_effects){
			float depth = tex2Dproj(DepthTextureSampler, In.projCoord).r;
			float alpha_factor;
			if(sun_like){
				alpha_factor = depth;
				float densit = (fFogDensity + 0.001f);
				float fog_factor = 1.001f - (10.f * densit);
				alpha_factor *= fog_factor;
			}
			else{
				alpha_factor = saturate((depth - In.Depth) * 4096);
			}
			if(blend_adding){
				OutputColor *= alpha_factor;
			}
			else{
				OutputColor.a *= alpha_factor;
			}
		}
		Output.RGBColor = OutputColor;
		return Output;
	}
	VertexShader vs_main_depthed_flare_compiled = compile vs_2_0 vs_main_depthed_flare();
	technique soft_sunflare{
		pass P0{
			VertexShader = vs_main_depthed_flare_compiled;
			PixelShader = compile ps_2_0 ps_main_depthed_flare(true, true);
		}
	}
	technique soft_particle_add{
		pass P0{
			VertexShader = vs_main_depthed_flare_compiled;
			PixelShader = compile ps_2_0 ps_main_depthed_flare(false, true);
		}
	}
	technique soft_particle_modulate{
		pass P0{
			VertexShader = vs_main_depthed_flare_compiled;
			PixelShader = compile ps_2_0 ps_main_depthed_flare(false, false);
		}
	}
#endif
#ifdef SPEEDTREE_SHADERS
	#ifndef SPEEDTREE_NUM_WIND_MATRICES
		#define SPEEDTREE_NUM_WIND_MATRICES 6
	#endif
	#ifndef SPEEDTREE_MAX_NUM_LEAF_ANGLES
		#define SPEEDTREE_MAX_NUM_LEAF_ANGLES 8
	#endif
	#ifndef SPEEDTREE_NUM_360_IMAGES
		#define SPEEDTREE_NUM_360_IMAGES 64
	#endif
	#include "SpeedTrees5/Shaders/SpeedTree.fx"
#endif
#ifdef OCEAN_SHADERS
	struct VS_OUTPUT_OCEAN{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
		float3 LightDir: TEXCOORD1;
		float4 LightDif: TEXCOORD2;
		float3 CameraDir: TEXCOORD3;
		float4 PosWater: TEXCOORD4;
		float Fog: FOG;
	};
	inline float get_wave_height_temp(const float pos[2], const float coef, const float freq1, const float freq2, const float time){
		return coef * sin((pos[0] + pos[1]) * freq1 + time) * cos((pos[0] - pos[1]) * freq2 + (time + 4));
	}
	VS_OUTPUT_OCEAN vs_main_ocean(float4 vPosition: POSITION, float2 tc: TEXCOORD0){
		VS_OUTPUT_OCEAN Out = (VS_OUTPUT_OCEAN)0;
		float4 vWorldPos = mul(matWorld, vPosition);
		float3 viewVec = vCameraPos.xyz - vWorldPos.xyz;
		float wave_distance_factor = (1.0f - saturate(length(viewVec) * 0.01));
		float pos_vector[2] = {
			vWorldPos.x, vWorldPos.y
		};
		vWorldPos.z += get_wave_height_temp(pos_vector, debug_vector.z, debug_vector.x, debug_vector.y, time_var) * wave_distance_factor;
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.PosWater = mul(matWaterViewProj, vWorldPos);
		float3 vNormal;
		if(wave_distance_factor > 0.0f){
			float3 near_wave_heights[2];
			near_wave_heights[0].xy = vWorldPos.xy + float2(0.1f, 0.0f);
			near_wave_heights[1].xy = vWorldPos.xy + float2(0.0f, 1.0f);
			float pos_vector0[2] = {
				near_wave_heights[0].x, near_wave_heights[0].y
			};
			near_wave_heights[0].z = get_wave_height_temp(pos_vector0, debug_vector.z, debug_vector.x, debug_vector.y, time_var);
			float pos_vector1[2] = {
				near_wave_heights[1].x, near_wave_heights[1].y
			};
			near_wave_heights[1].z = get_wave_height_temp(pos_vector1, debug_vector.z, debug_vector.x, debug_vector.y, time_var);
			float3 v0 = normalize(near_wave_heights[0] - vWorldPos.xyz);
			float3 v1 = normalize(near_wave_heights[1] - vWorldPos.xyz);
			vNormal = cross(v0, v1);
		}
		else {
			vNormal = float3(0, 0, 1);
		}
		float3 vWorldN = vNormal;
		float3 vWorld_tangent = float3(1, 0, 0);
		float3 vWorld_binormal = normalize(cross(vWorld_tangent, vNormal));
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		float3 point_to_camera_normal = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.CameraDir = mul(TBNMatrix, point_to_camera_normal);
		Out.Tex0 = vWorldPos.xy;
		Out.LightDir = 0;
		Out.LightDif = vAmbientColor;
		Out.LightDir += mul(TBNMatrix, -vSunDir);
		Out.LightDif += vSunColor;
		Out.LightDir = normalize(Out.LightDir);
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_main_ocean(VS_OUTPUT_OCEAN In){
		PS_OUTPUT Output;
		const float texture_factor = 1.0f;
		float3 normal;
		normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0 * texture_factor).ag - 1.0f);
		normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		static const float detail_factor = 16 * texture_factor;
		float3 detail_normal;
		detail_normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0 * detail_factor).ag - 1.0f);
		detail_normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		float NdotL = saturate(dot(normal, In.LightDir));
		float4 tex = tex2D(ReflectionTextureSampler, 0.5f * normal.xy + float2(0.5f + 0.5f * (In.PosWater.x / In.PosWater.w), 0.5f - 0.5f * (In.PosWater.y / In.PosWater.w)));
		INPUT_OUTPUT_GAMMA(tex.rgb);
		Output.RGBColor = 0.01f * NdotL * In.LightDif;
		float3 vView = normalize(In.CameraDir);
		float fresnel = 1 - (saturate(dot(vView, normal)));
		fresnel = 0.0204f + 0.9796 * (fresnel * fresnel * fresnel * fresnel * fresnel);
		Output.RGBColor.rgb += (tex.rgb * fresnel);
		Output.RGBColor.w = 1.0f - 0.3f * In.CameraDir.z;
		float3 cWaterColor = 2 * float3(20.0f / 255.0f, 45.0f / 255.0f, 100.0f / 255.0f) * vSunColor.rgb;
		float fog_fresnel_factor = saturate(dot(In.CameraDir, normal));
		fog_fresnel_factor *= fog_fresnel_factor;
		fog_fresnel_factor *= fog_fresnel_factor;
		Output.RGBColor.rgb += cWaterColor * fog_fresnel_factor;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = 1;
		return Output;
	}
	technique simple_ocean{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_ocean();
			PixelShader = compile ps_2_0 ps_main_ocean();
		}
	}
#endif
#ifdef CLOTH_SHADERS
	float4 cloth_size = float4(16.0f, 16.0f, 1.0f / 16.0f, 1.0f);
	float4 cloth_winddir_effect = float4(0, 0, 1, 10.0f);
	float4x4 cloth_world_mat;
	float timestep = 0.01f;
	#define CLOTH_SPACE_SCALE (1.0f)
	#define CLOTH_SPACE_SCALE_INV (1.0f / CLOTH_SPACE_SCALE)
	bool is_static(float2 uv){
		return uv.y < cloth_size.z;
		return(uv.y < cloth_size.z) && ((uv.x > (1.0f - (cloth_size.z * 5))) || (uv.x < (cloth_size.z * 5)) || ((uv.x > 0.4) && (uv.x < 0.6)));
		return(uv.y < cloth_size.z) && ((uv.x > (1.0f - cloth_size.z)) || (uv.x < cloth_size.z));
	}
	bool is_seam(float2 uv){
		return uv.x < (cloth_size.z) || uv.x > (1.0f - cloth_size.z) || uv.y < (cloth_size.z) || uv.y > (1.0f - cloth_size.z);
	}
	float4 default_cloth_pos(float2 uv){
		static const float3 local_position = float3( - 0.5f, 0.225f, -0.18f);
		float3 identity_size = float3((uv.x + local_position.x), ( - uv.y + local_position.y), local_position.z);
		float size_x = cloth_size.x * cloth_size.z * 0.5f;
		float size_y = cloth_size.y * cloth_size.z;
		float4 default_obj_space_pos = float4(size_x * identity_size.x, size_y * identity_size.y, identity_size.z, 1.0f);
		float uv_fac = identity_size.x;
		float torsion2 = 0.45f;
		default_obj_space_pos.z += (uv_fac * uv_fac) * torsion2;
		return float4(mul(cloth_world_mat, default_obj_space_pos).xyz * CLOTH_SPACE_SCALE, 1.0f);
	}
	struct VS_OUTPUT_CLOTH1{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
		float3 WorldNormal: TEXCOORD1;
		float3 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_CLOTH1 vs_cloth_render1(uniform const int PcfMode, float2 tc: TEXCOORD0){
		VS_OUTPUT_CLOTH1 Out = (VS_OUTPUT_CLOTH1)0;
		float4 coord = float4(tc, 0, 0);
		float4 vWorldPos = tex2Dlod(PositionSampler, coord);
		vWorldPos.xyz *= CLOTH_SPACE_SCALE_INV;
		half3 world_normal = tex2Dlod(NormalSampler, coord).rgb * 2.0f - 1.0f;
		Out.WorldNormal = world_normal;
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0 = tc;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		float d = distance(vCameraPos.xyz, vWorldPos.xyz);
		Out.Fog = get_fog_amount(d);
		return Out;
	}
	PS_OUTPUT ps_cloth_render1(uniform const int PcfMode, VS_OUTPUT_CLOTH1 In){
		PS_OUTPUT Output;
		if(is_static(In.Tex0)){
			Output.RGBColor = half4(1, 0, 0, 1);
		}
		else {
			Output.RGBColor = 1.0h - tex2D(ClampedTextureSampler, In.Tex0);
		}
		half sun_amount = 1.0h;
		if(PcfMode != PCF_NONE){
			sun_amount = 0.08h + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		half3 world_normal = normalize(In.WorldNormal);
		float two_sided_light = sun_amount * max(saturate(dot(vSunDir, world_normal)), saturate(dot( - vSunDir, world_normal)));
		half3 lighting = vSunColor * two_sided_light;
		Output.RGBColor.rgb *= lighting;
		Output.RGBColor *= vMaterialColor;
		half3 detail = 1.0h - tex2D(NormalTextureSampler, In.Tex0 * 10.0h);
		Output.RGBColor.rgb *= (0.5h + 0.5h * detail);
		return Output;
	}
	technique cloth_render1{
		pass P0{
			VertexShader = compile vs_3_0 vs_cloth_render1(PCF_NONE);
			PixelShader = compile ps_3_0 ps_cloth_render1(PCF_NONE);
		}
	}
	technique cloth_render1_SHDW{
		pass P0{
			VertexShader = compile vs_3_0 vs_cloth_render1(PCF_DEFAULT);
			PixelShader = compile ps_3_0 ps_cloth_render1(PCF_DEFAULT);
		}
	}
	technique cloth_render1_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_3_0 vs_cloth_render1(PCF_NVIDIA);
			PixelShader = compile ps_3_0 ps_cloth_render1(PCF_NVIDIA);
		}
	}
	struct VS_OUTPUT_CLOTH_SIM{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
	};
	VS_OUTPUT_CLOTH_SIM cloth_simulation_vs(float4 pos: POSITION, uniform const bool use_halfpixel_correction){
		VS_OUTPUT_CLOTH_SIM Out = (VS_OUTPUT_CLOTH_SIM)0;
		Out.Pos = pos;
		Out.Tex0 = (float2(pos.x, -pos.y) * 0.5f + 0.5f);
		if(use_halfpixel_correction){
		}
		else {
			Out.Tex0 += (0.5f / cloth_size.xy);
		}
		return Out;
	}
	float4 global_movement;
	float4 collision_sphere;
	float4 collision_plane;
	bool reset_cloth = false;
	#define MAX_CLOTH_ELLIPSOID 16
	int num_ellipsoid = 1;
	float4x4 collision_ellipsoid_matrices[MAX_CLOTH_ELLIPSOID];
	float4x4 collision_ellipsoid_inv_matrices[MAX_CLOTH_ELLIPSOID];
	float3 distance_constraint(float3 position, float3 center, float targetDistance, float responsiveness){
		float3 delta = position - center;
		float delta_len = length(delta);
		float distance = delta_len;
		return(targetDistance - distance) * delta * (responsiveness / distance);
	}
	void floor_constraint(inout float3 x, float level){
		if(x.z < level){
			x.z = level;
		}
	}
	void plane_constraint(inout float3 position, float3 normal, float dist){
		float distance = dot(position, normal) - dist;
		distance -= 0.05f;
		if(distance < 0){
			position -= distance * normal;
		}
	}
	void sphere_constraint(inout float3 x, float3 center, float r){
		float3 delta = x - center;
		float dist = length(delta);
		if(dist < r){
			x = center + delta * (r / dist);
		}
	}
	float3 ellipsoid_constraint(float3 position, float4x4 transform, float4x4 transformInv){
		float3 position0 = mul(transformInv, float4(position, 1));
		float3 center = 0;
		float radius = 1.0f;
		float3 delta0 = position0 - center;
		float distance = length(delta0);
		if(distance < radius){
			delta0.z = 0.0f;
			delta0 = (radius - distance) * (delta0 / distance);
			float3 delta = mul(transform, float4(delta0, 0));
			return delta;
		}
		else return 0;
	}
	void do_collisions(inout float3 pos){
		for(int i_e = 0; i_e < num_ellipsoid; i_e++){
			pos += ellipsoid_constraint(pos, collision_ellipsoid_matrices[i_e], collision_ellipsoid_inv_matrices[i_e]);
		}
		plane_constraint(pos, collision_plane.xyz, collision_plane.w);
	}
	float4 cloth_inner_dynamics_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		float stiffness = 0.25f;
		float static_stiffness = 0.75f;
		float2 uv = IN.Tex0;
		if(reset_cloth)return default_cloth_pos(uv);
		if(is_static(uv)){
			return default_cloth_pos(uv);
		}
		else {
			float3 cur_pos = tex2D(PositionSampler, uv).rgb;
			float3 dx = float3(0.0f, 0.0f, 0.0f);
			float2 cloth_wh = float2(cloth_size.x * cloth_size.z, cloth_size.y * cloth_size.z);
			float constraintDist = cloth_size.z * CLOTH_SPACE_SCALE;
			float pixel_size = 1.0f / cloth_size.x;
			float inv_pixel_size = 1.0f - pixel_size;
			if(true){
				float2 neighbors_uv[] = {
					float2(pixel_size, 0.0), float2( - pixel_size, 0.0), float2(0.0, pixel_size), float2(0.0, -pixel_size)
				};
				if(uv.x < inv_pixel_size){
					float3 x1 = tex2D(PositionSampler, uv + neighbors_uv[0]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[0]) ? static_stiffness: stiffness;
					dx = distance_constraint(cur_pos, x1, constraintDist, real_stiffness);
				}
				if(uv.x > pixel_size){
					float3 x2 = tex2D(PositionSampler, uv + neighbors_uv[1]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[1]) ? static_stiffness: stiffness;
					dx = dx + distance_constraint(cur_pos, x2, constraintDist, real_stiffness);
				}
				if(uv.y < inv_pixel_size){
					float3 x3 = tex2D(PositionSampler, uv + neighbors_uv[2]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[2]) ? static_stiffness: stiffness;
					dx = dx + distance_constraint(cur_pos, x3, constraintDist, real_stiffness);
				}
				if(uv.y > pixel_size){
					float3 x4 = tex2D(PositionSampler, uv + neighbors_uv[3]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[3]) ? static_stiffness: stiffness;
					dx = dx + distance_constraint(cur_pos, x4, constraintDist, real_stiffness);
				}
			}
			if(true){
				float2 neighbors_uv[] = {
					float2(pixel_size, pixel_size), float2( - pixel_size, pixel_size), float2(pixel_size, -pixel_size), float2( - pixel_size, -pixel_size)
				};
				float3 dxs = float3(0.0f, 0.0f, 0.0f);
				float cross_constraintDist = constraintDist * sqrt(2.0f);
				float shear_stiffness = 0.45f * 0.3f;
				if((uv.x < inv_pixel_size) && (uv.y < inv_pixel_size)){
					float3 xs1 = tex2D(PositionSampler, uv + neighbors_uv[0]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[0]) ? static_stiffness: stiffness;
					dxs = distance_constraint(cur_pos, xs1, cross_constraintDist, shear_stiffness * real_stiffness);
				}
				if((uv.x > pixel_size) && (uv.y < inv_pixel_size)){
					float3 xs2 = tex2D(PositionSampler, uv + neighbors_uv[1]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[1]) ? static_stiffness: stiffness;
					dxs = dxs + distance_constraint(cur_pos, xs2, cross_constraintDist, shear_stiffness * real_stiffness);
				}
				if((uv.x < inv_pixel_size) && (uv.y > pixel_size)){
					float3 xs3 = tex2D(PositionSampler, uv + neighbors_uv[2]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[2]) ? static_stiffness: stiffness;
					dxs = dxs + distance_constraint(cur_pos, xs3, cross_constraintDist, shear_stiffness * real_stiffness);
				}
				if((uv.x > pixel_size) && (uv.y > pixel_size)){
					float3 xs4 = tex2D(PositionSampler, uv + neighbors_uv[3]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[3]) ? static_stiffness: stiffness;
					dxs = dxs + distance_constraint(cur_pos, xs4, cross_constraintDist, shear_stiffness * real_stiffness);
				}
				dx += dxs;
			}
			if(true){
				float constraintDist2 = 2.0f * constraintDist;
				float2 neighbors_uv[] = {
					float2(pixel_size, 0.0f), float2( - pixel_size, 0.0f), float2(0.0f, pixel_size), float2(0.0f, -pixel_size)
				};
				float pixel_size2 = pixel_size * 2;
				float inv_pixel_size2 = 1.0f - pixel_size2;
				float3 dxb = float3(0.0f, 0.0f, 0.0f);
				if(uv.x < inv_pixel_size2){
					float3 x1 = tex2D(PositionSampler, uv + neighbors_uv[0]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[0]) ? static_stiffness: stiffness;
					dxb = distance_constraint(cur_pos, x1, constraintDist2, 0.035 * real_stiffness);
				}
				if(uv.x > pixel_size2){
					float3 x2 = tex2D(PositionSampler, uv + neighbors_uv[1]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[1]) ? static_stiffness: stiffness;
					dxb += distance_constraint(cur_pos, x2, constraintDist2, 0.035 * real_stiffness);
				}
				if(uv.y < inv_pixel_size2){
					float3 x3 = tex2D(PositionSampler, uv + neighbors_uv[2]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[2]) ? static_stiffness: stiffness;
					dxb += distance_constraint(cur_pos, x3, constraintDist2, 0.035 * real_stiffness);
				}
				if(uv.y > pixel_size2){
					float3 x4 = tex2D(PositionSampler, uv + neighbors_uv[3]).rgb;
					float real_stiffness = is_static(uv + neighbors_uv[3]) ? static_stiffness: stiffness;
					dxb += distance_constraint(cur_pos, x4, constraintDist2, 0.035 * real_stiffness);
				}
				dx += dxb;
			}
			float3 old_pos = cur_pos;
			cur_pos += dx;
			do_collisions(cur_pos);
			dx = cur_pos - old_pos;
			float len_diff = length(dx);
			if(len_diff > 0.15f * CLOTH_SPACE_SCALE){
				cur_pos = old_pos + ((dx) / len_diff) * (0.15f * CLOTH_SPACE_SCALE);
			}
			return float4(cur_pos, 1.0f);
		}
	}
	technique cloth_inner_dynamics{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_3_0 cloth_inner_dynamics_ps();
		}
	}
	float4 cloth_collisions_shader_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		if(is_static(IN.Tex0)){
			return default_cloth_pos(IN.Tex0);
		}
		else {
			float3 cur_pos = tex2D(PositionSampler, IN.Tex0).rgb;
			return float4(cur_pos, 1.0f);
		}
	}
	technique cloth_collisions_shader{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_3_0 cloth_collisions_shader_ps();
		}
	}
	float3 Integrate(float3 x, float3 oldx, float3 a, float timestep2, float damping){
		return damping * (x - oldx) + a * timestep2;
	}
	float4 cloth_forces_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		if(is_static(IN.Tex0)){
			return default_cloth_pos(IN.Tex0);
		}
		else {
			float3 cur_pos = tex2D(PositionSampler, IN.Tex0).rgb;
			float3 old_pos = tex2D(PrevPositionSampler, IN.Tex0).rgb;
			const float damping = 0.98f;
			float3 total_acc = float3(0, 0, -9.8f);
			float3 world_normal_here = normalize(tex2D(NormalSampler, IN.Tex0).rgb * 2.0f - 1.0f);
			float wind_force = cloth_winddir_effect.w;
			float3 winddir = cloth_winddir_effect.xyz;
			float wind_effect = dot(world_normal_here, float3(winddir));
			total_acc += winddir * wind_force * (wind_effect * wind_effect);
			float3 delta_pos = Integrate(cur_pos * CLOTH_SPACE_SCALE_INV, old_pos * CLOTH_SPACE_SCALE_INV, total_acc, timestep * timestep, damping);
			cur_pos += delta_pos * CLOTH_SPACE_SCALE;
			float len_diff = length(cur_pos - old_pos);
			if(len_diff > 0.05){
				cur_pos = old_pos + ((cur_pos - old_pos) / len_diff) * 0.05;
			}
			return float4(cur_pos, 1.0f);
		}
	}
	technique cloth_forces{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_2_0 cloth_forces_ps();
		}
	}
	float4 cloth_init_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		return default_cloth_pos(IN.Tex0);
	}
	technique cloth_init{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_2_0 cloth_init_ps();
		}
	}
	float4 cloth_generate_normalmap_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		float3 pos_here = tex2D(PositionSampler, IN.Tex0).rgb;
		bool inverted = false;
		float3 pos_near, pos_up;
		if(IN.Tex0.x > (1.0f - cloth_size.z)){
			pos_near = tex2D(PositionSampler, IN.Tex0 - float2(cloth_size.z, 0.0f)).rgb;
			if(IN.Tex0.y > (1.0f - cloth_size.z)){
				pos_up = tex2D(PositionSampler, IN.Tex0 - float2(0.0f, cloth_size.z)).rgb;
			}
			else {
				pos_up = tex2D(PositionSampler, IN.Tex0 + float2(0.0f, cloth_size.z)).rgb;
				inverted = true;
			}
		}
		else if(IN.Tex0.y > (1.0f - cloth_size.z)){
			pos_near = tex2D(PositionSampler, IN.Tex0 + float2(cloth_size.z, 0.0f)).rgb;
			pos_up = tex2D(PositionSampler, IN.Tex0 - float2(0.0f, cloth_size.z)).rgb;
			inverted = true;
		}
		else {
			pos_near = tex2D(PositionSampler, IN.Tex0 + float2(cloth_size.z, 0.0f)).rgb;
			pos_up = tex2D(PositionSampler, IN.Tex0 + float2(0.0f, cloth_size.z)).rgb;
		}
		float3 dir1 = (pos_near - pos_here);
		float3 dir2 = (pos_here - pos_up);
		float3 cross_result = cross(dir1, dir2);
		float3 normal_here = normalize(cross_result);
		if(inverted)normal_here = -normal_here;
		normal_here = (normal_here * 0.5f) + 0.5f;
		return float4(normal_here, 1.0f);
	}
	technique cloth_generate_normalmap{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_2_0 cloth_generate_normalmap_ps();
		}
	}
	float4 cloth_normalmap_blur_ps(VS_OUTPUT_CLOTH_SIM IN): COLOR0{
		float2 texCoord = IN.Tex0;
		float3 sample_start = tex2D(PositionSampler, texCoord).rgb;
		static const int SAMPLE_COUNT = 8;
		static const float2 offsets[SAMPLE_COUNT] = {
			 - 1, -1, 0, -1, 1, -1, -1, 0, 1, 0, -1, 1, 0, 1, 1, 1, 
		};
		float sampleDist = cloth_size.z * 1.74f;
		float3 sample = sample_start;
		for(int i = 0; i < SAMPLE_COUNT; i++){
			float2 sample_pos = texCoord + sampleDist * offsets[i];
			float3 sample_here;
			sample_here = tex2D(PositionSampler, sample_pos).rgb;
			sample += sample_here;
		}
		sample /= SAMPLE_COUNT;
		return float4(sample.rgb, 1);
	}
	technique cloth_normalmap_blur{
		pass P0{
			VertexShader = compile vs_2_0 cloth_simulation_vs(false);
			PixelShader = compile ps_2_0 cloth_normalmap_blur_ps();
		}
	}
	struct VS_OUTPUT_COLLISION_DEPTHMAP{
		float4 Pos: POSITION;
		float Depth: TEXCOORD1;
	};
	VS_OUTPUT_COLLISION_DEPTHMAP vs_main_collision_depthmap(float4 vPosition: POSITION, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		VS_OUTPUT_COLLISION_DEPTHMAP Out;
		float4 vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
		vObjectPos.z += debug_vector.z;
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		Out.Depth = Out.Pos.z;
		return Out;
	}
	PS_OUTPUT cloth_depthmap_shader_ps(VS_OUTPUT_COLLISION_DEPTHMAP In){
		PS_OUTPUT Output;
		Output.RGBColor.rgb = In.Depth;
		Output.RGBColor.rgb = debug_vector.y + In.Depth;
		Output.RGBColor.a = 1.0f;
		return Output;
	}
	technique cloth_depthmap_shader{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_collision_depthmap();
			PixelShader = compile ps_2_0 cloth_depthmap_shader_ps();
		}
	}
#endif
#ifdef NEWTREE_SHADERS
	VS_OUTPUT_FLORA vs_flora_billboards(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 view_vec = (vCameraPos.xyz - vWorldPos.xyz);
		float dist_to_vertex = length(view_vec);
		half alpha_val = saturate(0.5f + ((dist_to_vertex - flora_detail_fade) / flora_detail_fade_inv));
		Out.Pos = mul(matWorldViewProj, vPosition);
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		half4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		half4 stuffcolor = vMaterialColor * vColor;
		Out.Color = (stuffcolor * diffuse_light);
		Out.Color.a *= alpha_val;
		half wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * stuffcolor;
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.Fog = get_fog_amount(dist_to_vertex);
		return Out;
	}
	DEFINE_TECHNIQUES(tree_billboards_flora, vs_flora_billboards, ps_flora)
	VS_OUTPUT_BUMP vs_main_bump_billboards(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vPointLightDir: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_BUMP, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 view_vec = (vCameraPos.xyz - vWorldPos.xyz);
		float dist_to_vertex = length(view_vec);
		if(dist_to_vertex < flora_detail_clip){
			Out.Pos = float4(0, 0, -1, 1);
			return Out;
		}
		float alpha_val = saturate(0.5f + ((dist_to_vertex - flora_detail_fade) / flora_detail_fade_inv));
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		half3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		half3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		half3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		half3x3 TBNMatrix = half3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		if(PcfMode != PCF_NONE){
			half4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexelPos = Out.ShadowTexCoord.xy * fShadowMapSize;
		}
		Out.SunLightDir.xyz = mul(TBNMatrix, -vSunDir);
		Out.SkyLightDir = mul(TBNMatrix, -vSkyLightDir);
		Out.VertexColor = vVertexColor;
		Out.VertexColor.a *= alpha_val;
		half3 vViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		float fresnel = 1 - (saturate(dot(vViewDir, vWorldN)));
		fresnel *= fresnel + 0.1h;
		Out.SunLightDir.w = fresnel;
		Out.Fog = get_fog_amount(dist_to_vertex);
		return Out;
	}
	DEFINE_TECHNIQUES(tree_billboards_dot3_alpha, vs_main_bump_billboards, ps_main_bump_simple)
#endif