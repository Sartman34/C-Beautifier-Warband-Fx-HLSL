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
#define INCLUDE_VERTEX_LIGHTING 
#define VERTEX_LIGHTING_SCALER 1.0f
#define VERTEX_LIGHTING_SPECULAR_SCALER 1.0f
#define USE_PRECOMPILED_SHADER_LISTS 
#define GIVE_ERROR_HERE {for(int i=0;i<1000;i++){Output.RGBColor*=Output.RGBColor;}}
#define GIVE_ERROR_HERE_VS {for(int i=0;i<1000;i++){Out.Pos*=Out.Pos;}}
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
#ifdef PER_MESH_CONSTANTS
	float4x4 matWorldViewProj;
	float4x4 matWorldView;
	float4x4 matWorld;
	float4x4 matWaterWorldViewProj;
	float4x4 matWorldArray[NUM_WORLD_MATRICES]: WORLDMATRIXARRAY;
	float4 vMaterialColor = float4(255.f / 255.f, 230.f / 255.f, 200.f / 255.f, 1.0f);
	float4 vMaterialColor2;
	float fMaterialPower = 16.f;
	float4 vSpecularColor = float4(5, 5, 5, 5);
	float4 texture_offset = {
		0, 0, 0, 0
	};
	int iLightPointCount;
	int iLightIndices[NUM_SIMUL_LIGHTS] = {
		0, 1, 2, 3
	};
	bool bUseMotionBlur = false;
	float4x4 matMotionBlur;
#endif
#ifdef PER_FRAME_CONSTANTS
	float time_var = 0.0f;
	float4x4 matWaterViewProj;
#endif
#ifdef PER_SCENE_CONSTANTS
	float fFogDensity = 0.05f;
	float3 vSkyLightDir;
	float4 vSkyLightColor;
	float3 vSunDir;
	float4 vSunColor;
	float4 vAmbientColor = float4(64.f / 255.f, 64.f / 255.f, 64.f / 255.f, 1.0f);
	float4 vGroundAmbientColor = float4(84.f / 255.f, 44.f / 255.f, 54.f / 255.f, 1.0f);
	float4 vCameraPos;
	float4x4 matSunViewProj;
	float4x4 matView;
	float4x4 matViewProj;
	float3 vLightPosDir[NUM_LIGHTS];
	float4 vLightDiffuse[NUM_LIGHTS];
	float4 vPointLightColor;
	float reflection_factor;
#endif
#ifdef APPLICATION_CONSTANTS
	bool use_depth_effects = false;
	float far_clip_Inv;
	float4 vDepthRT_HalfPixel_ViewportSizeInv;
	float fShadowMapNextPixel = 1.0f / 4096;
	float fShadowMapSize = 4096;
	static const float input_gamma = 2.2f;
	float4 output_gamma = float4(2.2f, 2.2f, 2.2f, 2.2f);
	float4 output_gamma_inv = float4(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f);
	float4 debug_vector = {
		0, 0, 0, 1
	};
	float spec_coef = 1.0f;
	static const float map_normal_detail_factor = 1.4f;
	static const float uv_2_scale = 1.237;
	static const float fShadowBias = 0.00002f;
	#ifdef USE_NEW_TREE_SYSTEM
		float flora_detail = 40.0f;
		#define flora_detail_fade (flora_detail*FLORA_DETAIL_FADE_MUL)
		#define flora_detail_fade_inv (flora_detail-flora_detail_fade)
		#define flora_detail_clip (max(0,flora_detail_fade-20.0f))
	#endif
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
	sampler DepthTextureSampler: register(fx_CubicTextureSampler_RegisterS);
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
		float4 RGBColor: COLOR;
	};
#endif
#ifdef FUNCTIONS
	float GetSunAmount(uniform const int PcfMode, float4 ShadowTexCoord, float2 ShadowTexelPos){
		float sun_amount;
		if(PcfMode == PCF_NVIDIA){
			sun_amount = tex2Dproj(ShadowmapTextureSampler, ShadowTexCoord).r;
		}
		else {
			float2 lerps = frac(ShadowTexelPos);
			float sourcevals[4];
			sourcevals[0] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord).r < ShadowTexCoord.z) ? 0.0f: 1.0f;
			sourcevals[1] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord + float2(fShadowMapNextPixel, 0)).r < ShadowTexCoord.z) ? 0.0f: 1.0f;
			sourcevals[2] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord + float2(0, fShadowMapNextPixel)).r < ShadowTexCoord.z) ? 0.0f: 1.0f;
			sourcevals[3] = (tex2D(ShadowmapTextureSampler, ShadowTexCoord + float2(fShadowMapNextPixel, fShadowMapNextPixel)).r < ShadowTexCoord.z) ? 0.0f: 1.0f;
			sun_amount = lerp(lerp(sourcevals[0], sourcevals[1], lerps.x), lerp(sourcevals[2], sourcevals[3], lerps.x), lerps.y);
		}
		return sun_amount;
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
	float HairSingleSpecularTerm(float3 T, float3 H, float exponent){
		float dotTH = dot(T, H);
		float sinTH = sqrt(1.0 - dotTH * dotTH);
		return pow(sinTH, exponent);
	}
	float3 ShiftTangent(float3 T, float3 N, float shiftAmount){
		return normalize(T + shiftAmount * N);
	}
	float3 calculate_hair_specular(float3 normal, float3 tangent, float3 lightVec, float3 viewVec, float2 tc){
		float shiftTex = tex2D(Diffuse2Sampler, tc).a;
		float3 T1 = ShiftTangent(tangent, normal, specularShift.x + shiftTex);
		float3 T2 = ShiftTangent(tangent, normal, specularShift.y + shiftTex);
		float3 H = normalize(lightVec + viewVec);
		float3 specular = vSunColor * specularColor0 * HairSingleSpecularTerm(T1, H, specularExp.x);
		float3 specular2 = vSunColor * specularColor1 * HairSingleSpecularTerm(T2, H, specularExp.y);
		float specularMask = tex2D(Diffuse2Sampler, tc * 10.0f).a;
		specular2 *= specularMask;
		float specularAttenuation = saturate(1.75 * dot(normal, lightVec) + 0.25);
		specular = (specular + specular2) * specularAttenuation;
		return specular;
	}
	float HairDiffuseTerm(float3 N, float3 L){
		return saturate(0.75 * dot(N, L) + 0.25);
	}
	float face_NdotL(float3 n, float3 l){
		float wNdotL = dot(n.xyz, l.xyz);
		return saturate(max(0.2f * (wNdotL + 0.9f), wNdotL));
	}
	float4 calculate_point_lights_diffuse(const float3 vWorldPos, const float3 vWorldN, const bool face_like_NdotL, const bool exclude_0){
		const int exclude_index = 0;
		float4 total = 0;
		for(int j = 0;
		j < iLightPointCount;
		j++){
			if(!exclude_0 || j != exclude_index){
				int i = iLightIndices[j];
				float3 point_to_light = vLightPosDir[i] - vWorldPos;
				float LD = dot(point_to_light, point_to_light);
				float3 L = normalize(point_to_light);
				float wNdotL = dot(vWorldN, L);
				float fAtten = VERTEX_LIGHTING_SCALER / LD;
				if(face_like_NdotL){
					total += max(0.2f * (wNdotL + 0.9f), wNdotL) * vLightDiffuse[i] * fAtten;
				}
				else{
					total += saturate(wNdotL) * vLightDiffuse[i] * fAtten;
				}
			}
		}
		return total;
	}
	float4 calculate_point_lights_specular(const float3 vWorldPos, const float3 vWorldN, const float3 vWorldView, const bool exclude_0){
		float4 total = 0;
		for(int i = 0;
		i < iLightPointCount;
		i++){
			{
				float3 point_to_light = vLightPosDir[i] - vWorldPos;
				float LD = dot(point_to_light, point_to_light);
				float3 L = normalize(point_to_light);
				float fAtten = VERTEX_LIGHTING_SPECULAR_SCALER / LD;
				float3 vHalf = normalize(vWorldView + L);
				total += fAtten * vLightDiffuse[i] * pow(saturate(dot(vHalf, vWorldN)), fMaterialPower);
			}
		}
		return total;
	}
	float4 get_ambientTerm(int ambientTermType, float3 normal, float3 DirToSky, float sun_amount){
		float4 ambientTerm;
		if(ambientTermType == 0){
			ambientTerm = vAmbientColor;
		}
		else if(ambientTermType == 1){
			float4 g_vGroundColorTEMP = vGroundAmbientColor * sun_amount;
			float4 g_vSkyColorTEMP = vAmbientColor;
			float lerpFactor = (dot(normal, DirToSky) + 1.0f) * 0.5f;
			float4 hemiColor = lerp(g_vGroundColorTEMP, g_vSkyColorTEMP, lerpFactor);
			ambientTerm = hemiColor;
		}
		else {
			float4 cubeColor = texCUBE(CubicTextureSampler, normal);
			ambientTerm = vAmbientColor * cubeColor;
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
				VertexShader = compile vs_2_0 vs_name(PCF_NVIDIA);\
				PixelShader = compile ps_2_a ps_name(PCF_NVIDIA);\
			}\
		}
#endif
#define DEFINE_LIGHTING_TECHNIQUE(tech_name, use_dxt5, use_bumpmap, use_skinning, use_specularfactor, use_specularmap) 
#ifdef MISC_SHADERS
	struct VS_OUTPUT_FONT{
		float4 Pos: POSITION;
		float Fog: FOG;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
	};
	VS_OUTPUT_FONT vs_font(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_OUTPUT_FONT Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float d = length(P);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
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
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
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
		Out.RGBColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
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
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	VS_OUTPUT_FONT_X vs_main_no_shadow(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		VS_OUTPUT_FONT_X Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor + vLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSunDir)) * vSunColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_no_shadow(VS_OUTPUT_FONT_X In){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = In.Color * tex_col;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_simple_no_filtering(VS_OUTPUT_FONT_X In){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSamplerNoFilter, In.Tex0);
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
		Output.RGBColor.a = 1.0f;
		return Output;
	}
	technique diffuse_no_shadow{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_no_shadow();
			PixelShader = compile ps_2_0 ps_main_no_shadow();
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
		Output.RGBColor.a = 1.0f;
		Output.RGBColor.rgb = tex2D(FontTextureSampler, In.Tex0).rgb + In.Color.rgb;
		return Output;
	}
	PS_OUTPUT ps_font_outline(VS_OUTPUT_FONT In){
		float4 sample = tex2D(FontTextureSampler, In.Tex0);
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor.a = (1.0f - sample.r) + sample.a;
		Output.RGBColor.rgb *= sample.a + 0.05f;
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
		Out.Depth = Out.Pos.z / Out.Pos.w;
		Out.Tex0 = tc;
		return Out;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0){
		VS_OUTPUT_SHADOWMAP Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Depth = Out.Pos.z / Out.Pos.w;
		if(1){
			float3 vScreenNormal = mul((float3x3)matWorldViewProj, vNormal);
			Out.Depth -= vScreenNormal.z * (fShadowBias);
		}
		Out.Tex0 = tc;
		return Out;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap_biased(float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0){
		VS_OUTPUT_SHADOWMAP Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Depth = Out.Pos.z / Out.Pos.w;
		if(1){
			float3 vScreenNormal = mul((float3x3)matWorldViewProj, vNormal);
			Out.Depth -= vScreenNormal.z * (fShadowBias);
			Out.Pos.z += (0.0025f);
		}
		Out.Tex0 = tc;
		return Out;
	}
	PS_OUTPUT ps_main_shadowmap(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		Output.RGBColor.a = tex2D(MeshTextureSampler, In.Tex0).a;
		Output.RGBColor.a -= 0.5f;
		clip(Output.RGBColor.a);
		Output.RGBColor.rgb = In.Depth;
		return Output;
	}
	VS_OUTPUT_SHADOWMAP vs_main_shadowmap_light(uniform const bool skinning, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_SHADOWMAP, Out);
		return Out;
	}
	PS_OUTPUT ps_main_shadowmap_light(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		Output.RGBColor = float4(1, 0, 0, 1);
		return Output;
	}
	PS_OUTPUT ps_render_character_shadow(VS_OUTPUT_SHADOWMAP In){
		PS_OUTPUT Output;
		Output.RGBColor = 1.0f;
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
		float3 sample_start = tex2D(CharacterShadowTextureSampler, texCoord).rgb;
		static const int SAMPLE_COUNT = 4;
		static const float2 offsets[SAMPLE_COUNT] = {
			 - 1, 1, 1, 1, 0, 2, 0, 3, 
		};
		float blur_amount = saturate(1.0f - texCoord.y);
		blur_amount *= blur_amount;
		float sampleDist = (6.0f / 256.0f) * blur_amount;
		float sample = sample_start;
		for(int i = 0;
		i < SAMPLE_COUNT;
		i++){
			float2 sample_pos = texCoord + sampleDist * offsets[i];
			float sample_here = tex2D(CharacterShadowTextureSampler, sample_pos).a;
			sample += sample_here;
		}
		sample /= SAMPLE_COUNT + 1;
		return sample;
	}
	struct VS_OUTPUT_CHARACTER_SHADOW{
		float4 Pos: POSITION;
		float Fog: FOG;
		float2 Tex0: TEXCOORD0;
		float4 Color: COLOR0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
	};
	VS_OUTPUT_CHARACTER_SHADOW vs_character_shadow(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR){
		INITIALIZE_OUTPUT(VS_OUTPUT_CHARACTER_SHADOW, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		if(PcfMode != PCF_NONE){
			float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
			float wNdotSun = max( - 0.0001, dot(vWorldN, -vSunDir));
			Out.SunLight = (wNdotSun) * vSunColor;
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_character_shadow(uniform const int PcfMode, VS_OUTPUT_CHARACTER_SHADOW In){
		PS_OUTPUT Output;
		if(PcfMode == PCF_NONE){
			Output.RGBColor.a = blurred_read_alpha(In.Tex0) * In.Color.a;
		}
		else {
			float sun_amount = 0.05f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
			float sun_amount = 0.05f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
		float4 LightDir_Alpha: TEXCOORD1;
		float4 LightDif: TEXCOORD2;
		float3 CameraDir: TEXCOORD3;
		float4 PosWater: TEXCOORD4;
		float Fog: FOG;
		float4 projCoord: TEXCOORD5;
		float Depth: TEXCOORD6;
	};
	VS_OUTPUT_WATER vs_main_water(float4 vPosition: POSITION, float3 vNormal: NORMAL, float4 vColor: COLOR, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL){
		VS_OUTPUT_WATER Out = (VS_OUTPUT_WATER)0;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.PosWater = mul(matWaterWorldViewProj, vPosition);
		float3 vWorldPos = (float3)mul(matWorld, vPosition);
		float3 point_to_camera_normal = normalize(vCameraPos - vWorldPos);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3 P = mul(matWorldView, vPosition);
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.CameraDir = mul(TBNMatrix, point_to_camera_normal);
		Out.Tex0 = tc + texture_offset.xy;
		Out.LightDif = 0;
		float totalLightPower = 0;
		Out.LightDir_Alpha.xyz = mul(TBNMatrix, -vSunDir);
		Out.LightDif += vSunColor * vColor;
		totalLightPower += length(vSunColor.xyz);
		Out.LightDir_Alpha.a = vColor.a;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		if(use_depth_effects){
			Out.projCoord.xy = (float2(Out.Pos.x, -Out.Pos.y) + Out.Pos.w) / 2;
			Out.projCoord.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * Out.Pos.w);
			Out.projCoord.zw = Out.Pos.zw;
			Out.Depth = Out.Pos.z * far_clip_Inv;
		}
		return Out;
	}
	PS_OUTPUT ps_main_water(VS_OUTPUT_WATER In, uniform const bool use_high, uniform const bool apply_depth, uniform const bool mud_factor){
		PS_OUTPUT Output;
		const bool rgb_normalmap = false;
		float3 normal;
		if(rgb_normalmap){
			normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0f);
		}
		else {
			normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0f);
			normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		}
		if(!apply_depth){
			normal = float3(0, 0, 1);
		}
		float NdotL = saturate(dot(normal, In.LightDir_Alpha.xyz));
		float3 vView = normalize(In.CameraDir);
		float4 tex;
		if(apply_depth){
			tex = tex2D(ReflectionTextureSampler, (0.25f * normal.xy) + float2(0.5f + 0.5f * (In.PosWater.x / In.PosWater.w), 0.5f - 0.5f * (In.PosWater.y / In.PosWater.w)));
		}
		else {
			tex = tex2D(EnvTextureSampler, (vView - normal).yx * 3.4f);
		}
		INPUT_OUTPUT_GAMMA(tex.rgb);
		Output.RGBColor = 0.01f * NdotL * In.LightDif;
		if(mud_factor){
			Output.RGBColor *= 0.125f;
		}
		float fresnel = 1 - (saturate(dot(vView, normal)));
		fresnel = 0.0204f + 0.9796 * (fresnel * fresnel * fresnel * fresnel * fresnel);
		if(!apply_depth){
			fresnel = min(fresnel, 0.01f);
		}
		if(mud_factor){
			Output.RGBColor.rgb += lerp(tex.rgb * float3(0.105, 0.175, 0.160) * fresnel, tex.rgb, fresnel);
		}
		else {
			Output.RGBColor.rgb += (tex.rgb * fresnel);
		}
		Output.RGBColor.a = 1.0f - 0.3f * In.CameraDir.z;
		float vertex_alpha = In.LightDir_Alpha.a;
		Output.RGBColor.a *= vertex_alpha;
		if(mud_factor){
			Output.RGBColor.a = 1.0f;
		}
		const float3 g_cDownWaterColor = mud_factor ? float3(4.5f / 255.0f, 8.0f / 255.0f, 6.0f / 255.0f): float3(1.0f / 255.0f, 4.0f / 255.0f, 6.0f / 255.0f);
		const float3 g_cUpWaterColor = mud_factor ? float3(5.0f / 255.0f, 7.0f / 255.0f, 7.0f / 255.0f): float3(1.0f / 255.0f, 5.0f / 255.0f, 10.0f / 255.0f);
		float3 cWaterColor = lerp(g_cUpWaterColor, g_cDownWaterColor, saturate(dot(vView, normal)));
		if(!apply_depth){
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
			Output.RGBColor.rgb += float3(0.022f, 0.02f, 0.005f) * (1.0f - saturate(dot(vView, normal)));
		}
		if(apply_depth && use_depth_effects){
			float depth = tex2Dproj(DepthTextureSampler, In.projCoord).r;
			float alpha_factor;
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
				float4 coord_start = In.projCoord;
				float4 coord_disto = coord_start;
				coord_disto.xy += (normal.xy * saturate(Output.RGBColor.w) * 0.1f);
				float depth_here = tex2D(DepthTextureSampler, coord_disto).r;
				float4 refraction;
				if(depth_here < depth)refraction = tex2Dproj(ScreenTextureSampler, coord_disto);
				else refraction = tex2Dproj(ScreenTextureSampler, coord_start);
				INPUT_OUTPUT_GAMMA(refraction.rgb);
				Output.RGBColor.rgb = lerp(Output.RGBColor.rgb, refraction.rgb, saturate(1.0f - Output.RGBColor.w) * 0.55f);
				if(Output.RGBColor.a > 0.1f){
					Output.RGBColor.a *= 1.75f;
				}
				if(mud_factor){
					Output.RGBColor.a *= 1.25f;
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
			VertexShader = compile vs_2_0 vs_main_water();
			PixelShader = compile ps_2_0 ps_main_water(false, true, false);
		}
	}
	technique watermap_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water();
			PixelShader = compile PS_2_X ps_main_water(true, true, false);
		}
	}
	technique watermap_mud{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water();
			PixelShader = compile ps_2_0 ps_main_water(false, true, true);
		}
	}
	technique watermap_mud_high{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_water();
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
			float2 scaleBias = float2(vSpecularColor.x, vSpecularColor.y);
			float exFactor16 = tex2D(EnvTextureSampler, In.Tex0).r;
			float exFactor8 = floor(exFactor16 * 255) / 255;
			Output.RGBColor.rgb *= exp2(exFactor16 * scaleBias.x + scaleBias.y);
		}
		else{
			Output.RGBColor = In.Color;
			float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
			INPUT_TEX_GAMMA(tex_col.rgb);
			Output.RGBColor *= tex_col;
		}
		Output.RGBColor.a = 1;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		if(In.Color.a == 0.0f){
			Output.RGBColor.rgb = saturate(Output.RGBColor.rgb);
		}
		return Output;
	}
	VS_OUTPUT_FONT vs_skybox(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_OUTPUT_FONT Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Pos.z = Out.Pos.w;
		float3 P = vPosition;
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		P.z *= 0.2f;
		float d = length(P);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		Out.Color.a = (vWorldPos.z < -10.0f) ? 0.0f: 1.0f;
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
		float Fog: FOG;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
	};
	VS_OUTPUT vs_main(uniform const int PcfMode, uniform const bool UseSecondLight, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		if(UseSecondLight){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	VS_OUTPUT vs_main_Instanced(uniform const int PcfMode, uniform const bool UseSecondLight, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		Out.Pos = mul(matWorldOfInstance, float4(vPosition.xyz, 1.0f));
		Out.Pos = mul(matViewProj, Out.Pos);
		float4 vWorldPos = (float4)mul(matWorldOfInstance, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorldOfInstance, vNormal));
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		if(UseSecondLight){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float4 P = mul(matView, vWorldPos);
		float d = length(P.xyz);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main(VS_OUTPUT In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		float sun_amount = 1.0f;
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
			VertexShader = compile vs_2_0 vs_main_Instanced(PCF_NVIDIA, false);
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
		float4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 SunLightDir: TEXCOORD1;
		float3 SkyLightDir: TEXCOORD2;
		float4 PointLightDir: TEXCOORD3;
		float4 ShadowTexCoord: TEXCOORD4;
		float2 ShadowTexelPos: TEXCOORD5;
		float Fog: FOG;
		float3 ViewDir: TEXCOORD6;
		float3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_BUMP vs_main_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vPointLightDir: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3 P = mul(matWorldView, vPosition);
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.SunLightDir = mul(TBNMatrix, -vSunDir);
		Out.SkyLightDir = mul(TBNMatrix, -vSkyLightDir);
		#ifdef USE_LIGHTING_PASS
			Out.PointLightDir = vWorldPos;
		#else
			Out.PointLightDir.rgb = 2.0f * vPointLightDir.rgb - 1.0f;
			Out.PointLightDir.a = vPointLightDir.a;
		#endif
		Out.VertexColor = vVertexColor;
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_bump(VS_OUTPUT_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		float3 normal;
		normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0f);
		normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		if(PcfMode != PCF_NONE){
			float sun_amount = 0.03f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			total_light += ((saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount))) * vSunColor;
		}
		else {
			total_light += saturate(dot(In.SunLightDir.xyz, normal.xyz)) * vSunColor;
		}
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			total_light += saturate(dot(In.PointLightDir.xyz, normal.xyz)) * vPointLightColor;
		#endif
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0f;
		Output.RGBColor *= vMaterialColor;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_main_bump_simple(VS_OUTPUT_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		float3 normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0f);
		float sun_amount = 1.0f;
		if(PcfMode != PCF_NONE){
			if(PcfMode == PCF_NVIDIA)sun_amount = saturate(0.15f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos));
			else sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		total_light += ((saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount * sun_amount))) * vSunColor;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			total_light += saturate(dot(In.PointLightDir.xyz, normal.xyz)) * vPointLightColor;
		#endif
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0f;
		Output.RGBColor *= vMaterialColor;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		float fresnel = 1 - (saturate(dot(In.ViewDir, In.WorldNormal)));
		Output.RGBColor.rgb *= max(0.6, fresnel * fresnel + 0.1);
		Output.RGBColor.rgb = OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	PS_OUTPUT ps_main_bump_simple_multitex(VS_OUTPUT_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		float4 tex_col2 = tex2D(Diffuse2Sampler, In.Tex0 * uv_2_scale);
		float4 multi_tex_col = tex_col;
		float inv_alpha = (1.0f - In.VertexColor.a);
		multi_tex_col.rgb *= inv_alpha;
		multi_tex_col.rgb += tex_col2.rgb * In.VertexColor.a;
		INPUT_TEX_GAMMA(multi_tex_col.rgb);
		float3 normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0f);
		float sun_amount = 1.0f;
		if(PcfMode != PCF_NONE){
			if(PcfMode == PCF_NVIDIA)sun_amount = saturate(0.15f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos));
			else sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		total_light += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) * (sun_amount)) * vSunColor;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			total_light += saturate(dot(In.PointLightDir.xyz, normal.xyz)) * vPointLightColor;
		#endif
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0f;
		Output.RGBColor *= multi_tex_col;
		Output.RGBColor.rgb *= In.VertexColor.rgb;
		Output.RGBColor.a *= In.PointLightDir.a;
		float fresnel = 1 - (saturate(dot(normalize(In.ViewDir), normalize(In.WorldNormal))));
		Output.RGBColor.rgb *= max(0.6, fresnel * fresnel + 0.1);
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
	struct VS_OUTPUT_ENVMAP_SPECULAR{
		float4 Pos: POSITION;
		float Fog: FOG;
		float4 Color: COLOR0;
		float4 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float3 vSpecular: TEXCOORD4;
	};
	VS_OUTPUT_ENVMAP_SPECULAR vs_envmap_specular(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_ENVMAP_SPECULAR, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		if(bUseMotionBlur){
			float4 vWorldPos1 = mul(matMotionBlur, vPosition);
			float3 delta_vector = vWorldPos1 - vWorldPos;
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
		float3 relative_cam_pos = normalize(vCameraPos - vWorldPos);
		float2 envpos;
		float3 tempvec = relative_cam_pos - vWorldN;
		float3 vHalf = normalize(relative_cam_pos - vSunDir);
		float3 fSpecular = spec_coef * vSunColor * vSpecularColor * pow(saturate(dot(vHalf, vWorldN)), fMaterialPower);
		Out.vSpecular = fSpecular;
		Out.vSpecular *= vColor.rgb;
		envpos.x = (tempvec.y);
		envpos.y = tempvec.z;
		envpos += 1.0f;
		Out.Tex0.zw = envpos;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = max( - 0.0001f, dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		Out.SunLight.a = vColor.a;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	VS_OUTPUT_ENVMAP_SPECULAR vs_envmap_specular_Instanced(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_ENVMAP_SPECULAR, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = mul(matWorldOfInstance, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorldOfInstance, vNormal));
		if(bUseMotionBlur){
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
				moveDirection = normalize(vWorldPos1 - vWorldPos);
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
		float3 relative_cam_pos = normalize(vCameraPos - vWorldPos);
		float2 envpos;
		float3 tempvec = relative_cam_pos - vWorldN;
		float3 vHalf = normalize(relative_cam_pos - vSunDir);
		float3 fSpecular = spec_coef * vSunColor * vSpecularColor * pow(saturate(dot(vHalf, vWorldN)), fMaterialPower);
		Out.vSpecular = fSpecular;
		Out.vSpecular *= vColor.rgb;
		envpos.x = (tempvec.y);
		envpos.y = tempvec.z;
		envpos += 1.0f;
		Out.Tex0.zw = envpos;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = max( - 0.0001f, dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		Out.SunLight.a = vColor.a;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matView, vWorldPos);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_envmap_specular(VS_OUTPUT_ENVMAP_SPECULAR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 texColor = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(texColor.rgb);
		float3 specTexture = tex2D(SpecularTextureSampler, In.Tex0.xy).rgb;
		float3 fSpecular = specTexture * In.vSpecular.rgb;
		float3 envColor = tex2D(EnvTextureSampler, In.Tex0.zw).rgb;
		if((PcfMode != PCF_NONE)){
			float sun_amount = 0.1f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			float4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular) * sun_amount;
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight * sun_amount + 0.3f) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		else {
			float4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular);
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight + 0.3f) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = 1.0f;
		if(bUseMotionBlur)Output.RGBColor.a = In.SunLight.a;
		return Output;
	}
	PS_OUTPUT ps_envmap_specular_singlespec(VS_OUTPUT_ENVMAP_SPECULAR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float2 spectex_Col = tex2D(SpecularTextureSampler, In.Tex0.xy).ag;
		float specTexture = dot(spectex_Col, spectex_Col) * 0.5;
		float3 fSpecular = specTexture * In.vSpecular.rgb;
		float4 texColor = saturate((saturate(In.Color + 0.5f) * specTexture) * 2.0f + 0.25f);
		float3 envColor = tex2D(EnvTextureSampler, In.Tex0.zw).rgb;
		if((PcfMode != PCF_NONE)){
			float sun_amount = 0.1f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			float4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular) * sun_amount;
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight * sun_amount + 0.3f) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		else {
			float4 vcol = In.Color;
			vcol.rgb += (In.SunLight.rgb + fSpecular);
			Output.RGBColor = (texColor * vcol);
			Output.RGBColor.rgb += (In.SunLight + 0.3f) * (In.Color.rgb * envColor.rgb * specTexture);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = 1.0f;
		return Output;
	}
	DEFINE_TECHNIQUES(envmap_specular_diffuse, vs_envmap_specular, ps_envmap_specular)
	DEFINE_TECHNIQUES(envmap_specular_diffuse_Instanced, vs_envmap_specular_Instanced, ps_envmap_specular)
	DEFINE_TECHNIQUES(watermap_for_objects, vs_envmap_specular, ps_envmap_specular_singlespec)
	struct VS_OUTPUT_BUMP_DYNAMIC{
		float4 Pos: POSITION;
		float4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		#ifndef USE_LIGHTING_PASS
			float3 vec_to_light_0: TEXCOORD1;
			float3 vec_to_light_1: TEXCOORD2;
			float3 vec_to_light_2: TEXCOORD3;
		#endif
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
		#ifndef USE_LIGHTING_PASS
			float3 point_to_light = vLightPosDir[iLightIndices[0]] - vWorldPos.xyz;
			Out.vec_to_light_0.xyz = mul(TBNMatrix, point_to_light);
			point_to_light = vLightPosDir[iLightIndices[1]] - vWorldPos.xyz;
			Out.vec_to_light_1.xyz = mul(TBNMatrix, point_to_light);
			point_to_light = vLightPosDir[iLightIndices[2]] - vWorldPos.xyz;
			Out.vec_to_light_2.xyz = mul(TBNMatrix, point_to_light);
		#endif
		Out.VertexColor = vVertexColor;
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_bump_interior(VS_OUTPUT_BUMP_DYNAMIC In){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		#ifndef USE_LIGHTING_PASS
			float3 normal;
			normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0f);
			normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
			float LD = dot(In.vec_to_light_0.xyz, In.vec_to_light_0.xyz);
			float3 L = normalize(In.vec_to_light_0.xyz);
			float wNdotL = dot(normal, L);
			total_light += saturate(wNdotL) * vLightDiffuse[iLightIndices[0]] / (LD);
			LD = dot(In.vec_to_light_1.xyz, In.vec_to_light_1.xyz);
			L = normalize(In.vec_to_light_1.xyz);
			wNdotL = dot(normal, L);
			total_light += saturate(wNdotL) * vLightDiffuse[iLightIndices[1]] / (LD);
			LD = dot(In.vec_to_light_2.xyz, In.vec_to_light_2.xyz);
			L = normalize(In.vec_to_light_2.xyz);
			wNdotL = dot(normal, L);
			total_light += saturate(wNdotL) * vLightDiffuse[iLightIndices[2]] / (LD);
		#endif
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
		#ifndef USE_LIGHTING_PASS
			float3 vec_to_light_0: TEXCOORD1;
			float3 vec_to_light_1: TEXCOORD2;
			float3 vec_to_light_2: TEXCOORD3;
		#endif
		float3 ViewDir: TEXCOORD4;
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
		#ifndef USE_LIGHTING_PASS
			float3 point_to_light = vLightPosDir[iLightIndices[0]] - vWorldPos.xyz;
			Out.vec_to_light_0.xyz = mul(TBNMatrix, point_to_light);
			point_to_light = vLightPosDir[iLightIndices[1]] - vWorldPos.xyz;
			Out.vec_to_light_1.xyz = mul(TBNMatrix, point_to_light);
			point_to_light = vLightPosDir[iLightIndices[2]] - vWorldPos.xyz;
			Out.vec_to_light_2.xyz = mul(TBNMatrix, point_to_light);
		#endif
		Out.VertexColor = vVertexColor;
		float3 viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.ViewDir = mul(TBNMatrix, viewdir);
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_bump_interior_new(VS_OUTPUT_BUMP_DYNAMIC_NEW In, uniform const bool use_specularmap, uniform const bool greenbump = false){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		#ifndef USE_LIGHTING_PASS
			float3 normal;
			if(greenbump){
				normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0f);
				normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
			}
			else {
				normal = 2.0f * tex2D(NormalTextureSampler, In.Tex0).rgb - 1.0f;
			}
			float LD_0 = saturate(1.0f / dot(In.vec_to_light_0.xyz, In.vec_to_light_0.xyz));
			float3 L_0 = normalize(In.vec_to_light_0.xyz);
			float wNdotL_0 = dot(normal, L_0);
			total_light += saturate(wNdotL_0) * vLightDiffuse[iLightIndices[0]] * (LD_0);
			float LD_1 = saturate(1.0f / dot(In.vec_to_light_1.xyz, In.vec_to_light_1.xyz));
			float3 L_1 = normalize(In.vec_to_light_1.xyz);
			float wNdotL_1 = dot(normal, L_1);
			total_light += saturate(wNdotL_1) * vLightDiffuse[iLightIndices[1]] * (LD_1);
			float LD_2 = saturate(1.0f / dot(In.vec_to_light_2.xyz, In.vec_to_light_2.xyz));
			float3 L_2 = normalize(In.vec_to_light_2.xyz);
			float wNdotL_2 = dot(normal, L_2);
			total_light += saturate(wNdotL_2) * vLightDiffuse[iLightIndices[2]] * (LD_2);
		#endif
		Output.RGBColor = float4(total_light.rgb, 1.0);
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		if(use_specularmap){
			float4 fSpecular = 0;
			float4 light0_specColor = vLightDiffuse[iLightIndices[0]] * LD_0;
			float3 vHalf_0 = normalize(In.ViewDir + L_0);
			fSpecular = light0_specColor * pow(saturate(dot(vHalf_0, normal)), fMaterialPower);
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
	technique bumpmap_interior_new_greenbump{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_bump_interior_new();
			PixelShader = compile ps_2_0 ps_main_bump_interior_new(false, true);
		}
	}
	DEFINE_LIGHTING_TECHNIQUE(bumpmap_interior, 1, 1, 0, 0, 0)
#endif
#ifdef STANDART_SHADERS
	struct VS_OUTPUT_STANDART{
		float4 Pos: POSITION;
		float Fog: FOG;
		float4 VertexColor: COLOR0;
		#ifdef INCLUDE_VERTEX_LIGHTING
			float3 VertexLighting: COLOR1;
		#endif
		float2 Tex0: TEXCOORD0;
		float3 SunLightDir: TEXCOORD1;
		float3 SkyLightDir: TEXCOORD2;
		#ifndef USE_LIGHTING_PASS
			float4 PointLightDir: TEXCOORD3;
		#endif
		float4 ShadowTexCoord: TEXCOORD4;
		float2 ShadowTexelPos: TEXCOORD5;
		float3 ViewDir: TEXCOORD6;
	};
	VS_OUTPUT_STANDART vs_main_standart(uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_skinning, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		float3 vObjectN, vObjectT, vObjectB;
		if(use_skinning){
			vObjectPos = skinning_deform(vPosition, vBlendWeights, vBlendIndices);
			vObjectN = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vNormal) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vNormal) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vNormal) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vNormal) * vBlendWeights.w);
			if(use_bumpmap){
				vObjectT = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vTangent) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vTangent) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vTangent) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vTangent) * vBlendWeights.w);
				vObjectB = (cross(vObjectN, vObjectT));
				bool left_handed = (dot(cross(vNormal, vTangent), vBinormal) < 0.0f);
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
		float4 vWorldPos = mul(matWorld, vObjectPos);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vObjectN));
		const bool use_motion_blur = bUseMotionBlur && (!use_skinning);
		if(use_motion_blur){
			#ifdef STATIC_MOVEDIR
				const float blur_len = 0.25f;
				float3 moveDirection = -normalize(float3(matWorld[0][0], matWorld[1][0], matWorld[2][0]));
				moveDirection.y -= blur_len * 0.285;
				float4 vWorldPos1 = vWorldPos + float4(moveDirection, 0) * blur_len;
			#else
				float4 vWorldPos1 = mul(matMotionBlur, vObjectPos);
				float3 moveDirection = normalize(vWorldPos1 - vWorldPos);
			#endif
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.1f) ? 1: 0;
			float y_factor = saturate(vObjectPos.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float extra_alpha = 0.1f;
			float start_alpha = (1.0f + extra_alpha);
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
		Out.Tex0 = tc;
		if(use_bumpmap){
			float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vObjectB));
			float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vObjectT));
			float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, float3(0, 0, 1));
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos, vWorldN, false, true);
			#endif
			#ifndef USE_LIGHTING_PASS
				const int effective_light_index = iLightIndices[0];
				float3 point_to_light = vLightPosDir[effective_light_index] - vWorldPos.xyz;
				Out.PointLightDir.xyz = mul(TBNMatrix, normalize(point_to_light));
				float LD = dot(point_to_light, point_to_light);
				Out.PointLightDir.a = saturate(1.0f / LD);
			#endif
			float3 viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir = mul(TBNMatrix, viewdir);
			#ifndef USE_LIGHTING_PASS
				if(PcfMode == PCF_NONE){
					Out.ShadowTexCoord = calculate_point_lights_specular(vWorldPos, vWorldN, viewdir, true);
				}
			#endif
		}
		else{
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
			#endif
			Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.SunLightDir = vWorldN;
			#ifndef USE_LIGHTING_PASS
				Out.SkyLightDir = calculate_point_lights_specular(vWorldPos, vWorldN, Out.ViewDir, false);
			#endif
		}
		Out.VertexColor.a *= vMaterialColor.a;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matWorldView, vObjectPos);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	VS_OUTPUT_STANDART vs_main_standart_Instanced(uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_skinning, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		float3 vObjectN, vObjectT, vObjectB;
		if(use_skinning){
			GIVE_ERROR_HERE_VS;
		}
		else{
			vObjectPos = vPosition;
			vObjectN = vNormal;
			if(use_bumpmap){
				vObjectT = vTangent;
				vObjectB = vBinormal;
			}
		}
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = mul(matWorldOfInstance, vObjectPos);
		float3 vWorldN = normalize(mul((float3x3)matWorldOfInstance, vObjectN));
		const bool use_motion_blur = bUseMotionBlur && (!use_skinning);
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
				moveDirection = normalize(vWorldPos1 - vWorldPos);
			}
			float delta_coefficient_sharp = (dot(vWorldN, moveDirection) > 0.1f) ? 1: 0;
			float y_factor = saturate(vObjectPos.y + 0.15);
			vWorldPos = lerp(vWorldPos, vWorldPos1, delta_coefficient_sharp * y_factor);
			float delta_coefficient_smooth = saturate(dot(vWorldN, moveDirection) + 0.5f);
			float extra_alpha = 0.1f;
			float start_alpha = (1.0f + extra_alpha);
			float end_alpha = start_alpha - 1.8f;
			float alpha = saturate(lerp(start_alpha, end_alpha, delta_coefficient_smooth));
			vVertexColor.a = saturate(0.5f - vObjectPos.y) + alpha + 0.25;
		}
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0 = tc;
		if(use_bumpmap){
			float3 vWorld_binormal = normalize(mul((float3x3)matWorldOfInstance, vObjectB));
			float3 vWorld_tangent = normalize(mul((float3x3)matWorldOfInstance, vObjectT));
			float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, float3(0, 0, 1));
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos, vWorldN, false, true);
			#endif
			#ifndef USE_LIGHTING_PASS
				const int effective_light_index = iLightIndices[0];
				float3 point_to_light = vLightPosDir[effective_light_index] - vWorldPos.xyz;
				Out.PointLightDir.xyz = mul(TBNMatrix, normalize(point_to_light));
				float LD = dot(point_to_light, point_to_light);
				Out.PointLightDir.a = saturate(1.0f / LD);
			#endif
			float3 viewdir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.ViewDir = mul(TBNMatrix, viewdir);
			#ifndef USE_LIGHTING_PASS
				if(PcfMode == PCF_NONE){
					Out.ShadowTexCoord = calculate_point_lights_specular(vWorldPos, vWorldN, viewdir, true);
				}
			#endif
		}
		else{
			Out.VertexColor = vVertexColor;
			#ifdef INCLUDE_VERTEX_LIGHTING
				Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
			#endif
			Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.SunLightDir = vWorldN;
			#ifndef USE_LIGHTING_PASS
				Out.SkyLightDir = calculate_point_lights_specular(vWorldPos, vWorldN, Out.ViewDir, false);
			#endif
		}
		Out.VertexColor.a *= vMaterialColor.a;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matView, vWorldPos);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_standart(VS_OUTPUT_STANDART In, uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_specularfactor, uniform const bool use_specularmap, uniform const bool ps2x, uniform const bool use_aniso, uniform const bool terrain_color_ambient = true, uniform const bool greenbump = false){
		PS_OUTPUT Output;
		float3 normal;
		if(use_bumpmap){
			if(greenbump){
				normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0).ag - 1.0f);
				normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
			}
			else {
				normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0) - 1.0f);
			}
		}
		else {
			normal = In.SunLightDir;
		}
		float sun_amount = 1;
		if(PcfMode != PCF_NONE){
			if((PcfMode == PCF_NVIDIA) || ps2x)sun_amount = 0.05f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			else sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		const int ambientTermType = (terrain_color_ambient && (ps2x || !use_specularfactor)) ? 1: 0;
		const float3 DirToSky = use_bumpmap ? In.SkyLightDir: float3(0.0f, 0.0f, 1.0f);
		float4 total_light = get_ambientTerm(ambientTermType, normal, DirToSky, sun_amount);
		float3 aniso_specular = 0;
		if(use_aniso){
			if(!ps2x){
				GIVE_ERROR_HERE;
			}
			float3 direction = float3(0, 1, 0);
			aniso_specular = calculate_hair_specular(normal, direction, ((use_bumpmap) ? In.SunLightDir: -vSunDir), In.ViewDir, In.Tex0);
		}
		if(use_bumpmap){
			total_light.rgb += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) + aniso_specular) * sun_amount * vSunColor;
			if(ps2x || !use_specularfactor){
				total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
			}
			#ifdef INCLUDE_VERTEX_LIGHTING
				if(ps2x || !use_specularfactor || (PcfMode == PCF_NONE)){
					total_light.rgb += In.VertexLighting;
				}
			#endif
			#ifndef USE_LIGHTING_PASS
				float light_atten = In.PointLightDir.a;
				const int effective_light_index = iLightIndices[0];
				total_light += saturate(dot(In.PointLightDir.xyz, normal.xyz) * vLightDiffuse[effective_light_index] * light_atten);
			#endif
		}
		else{
			total_light.rgb += (saturate(dot( - vSunDir, normal.xyz)) + aniso_specular) * sun_amount * vSunColor;
			if(ambientTermType != 1 && !ps2x){
				total_light += saturate(dot( - vSkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
			}
			#ifdef INCLUDE_VERTEX_LIGHTING
				total_light.rgb += In.VertexLighting;
			#endif
		}
		if(PcfMode != PCF_NONE)Output.RGBColor.rgb = total_light.rgb;
		else Output.RGBColor.rgb = min(total_light.rgb, 2.0f);
		Output.RGBColor.rgb *= vMaterialColor.rgb;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor.rgb *= tex_col.rgb;
		Output.RGBColor.rgb *= In.VertexColor.rgb;
		if(use_specularfactor){
			float4 fSpecular = 0;
			float4 specColor = 0.1 * spec_coef * vSpecularColor;
			if(use_specularmap){
				float spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0).rgb, 0.33);
				specColor *= spec_tex_factor;
			}
			else {
				specColor *= tex_col.a;
			}
			float4 sun_specColor = specColor * vSunColor * sun_amount;
			float3 vHalf = normalize(In.ViewDir + ((use_bumpmap) ? In.SunLightDir: -vSunDir));
			fSpecular = sun_specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower);
			if(PcfMode != PCF_DEFAULT){
				fSpecular *= In.VertexColor;
			}
			if(use_bumpmap){
				if(PcfMode == PCF_NONE){
					fSpecular.rgb += specColor * In.ShadowTexCoord.rgb;
				}
				if(ps2x || (PcfMode == PCF_NONE)){
					#ifndef USE_LIGHTING_PASS
						float light_atten = In.PointLightDir.a;
						const int effective_light_index = iLightIndices[0];
						float4 light_specColor = specColor * vLightDiffuse[effective_light_index] * (light_atten * 0.5);
						vHalf = normalize(In.ViewDir + In.PointLightDir);
						fSpecular += light_specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower);
					#endif
				}
			}
			else {
				fSpecular.rgb += specColor * In.SkyLightDir * 0.1;
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
		float sun_amount = 1;
		if(PcfMode != PCF_NONE){
			sun_amount = 0.03f + GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		float3 normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0) - 1.0f);
		static const int ambientTermType = 1;
		float3 DirToSky = In.SkyLightDir;
		float4 total_light = get_ambientTerm(ambientTermType, normal, DirToSky, sun_amount);
		float4 specColor = vSunColor * (vSpecularColor * 0.1);
		if(use_specularmap){
			float spec_tex_factor = dot(tex2D(SpecularTextureSampler, In.Tex0).rgb, 0.33);
			specColor *= spec_tex_factor;
		}
		float3 vHalf = normalize(In.ViewDir + In.SunLightDir);
		float4 fSpecular = specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower);
		if(use_aniso){
			float3 tangent_ = float3(0, 1, 0);
			fSpecular.rgb += calculate_hair_specular(normal, tangent_, In.SunLightDir, In.ViewDir, In.Tex0);
		}
		else{
			fSpecular.rgb *= spec_coef;
		}
		total_light += (saturate(dot(In.SunLightDir.xyz, normal.xyz)) + fSpecular) * sun_amount * vSunColor;
		total_light += saturate(dot(In.SkyLightDir.xyz, normal.xyz)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			float light_atten = In.PointLightDir.a;
			const int effective_light_index = iLightIndices[0];
			total_light += saturate(dot(In.PointLightDir.xyz, normal.xyz)) * vLightDiffuse[effective_light_index] * light_atten;
		#endif
		#ifdef INCLUDE_VERTEX_LIGHTING
			total_light.rgb += In.VertexLighting;
		#endif
		Output.RGBColor.rgb = total_light.rgb;
		Output.RGBColor.a = 1.0f;
		Output.RGBColor *= vMaterialColor;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor *= In.VertexColor;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = In.VertexColor.a * tex_col.a;
		return Output;
	}
	#ifdef USE_PRECOMPILED_SHADER_LISTS
		VertexShader standart_vs_noshadow[] = {
			compile vs_2_0 vs_main_standart(PCF_NONE, 0, 0), compile vs_2_0 vs_main_standart(PCF_NONE, 0, 1), compile vs_2_0 vs_main_standart(PCF_NONE, 1, 0), compile vs_2_0 vs_main_standart(PCF_NONE, 1, 1)
		};
		VertexShader standart_vs_default[] = {
			compile vs_2_0 vs_main_standart(PCF_DEFAULT, 0, 0), compile vs_2_0 vs_main_standart(PCF_DEFAULT, 0, 1), compile vs_2_0 vs_main_standart(PCF_DEFAULT, 1, 0), compile vs_2_0 vs_main_standart(PCF_DEFAULT, 1, 1)
		};
		VertexShader standart_vs_nvidia[] = {
			compile vs_2_0 vs_main_standart(PCF_NVIDIA, 0, 0), compile vs_2_0 vs_main_standart(PCF_NVIDIA, 0, 1), compile vs_2_0 vs_main_standart(PCF_NVIDIA, 1, 0), compile vs_2_0 vs_main_standart(PCF_NVIDIA, 1, 1)
		};
		#define DEFINE_STANDART_TECHNIQUE(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor) \
			technique tech_name{\
				pass P0{\
					VertexShader = standart_vs_noshadow[(2 * use_bumpmap) + use_skinning];\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = standart_vs_default[(2 * use_bumpmap) + use_skinning];\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = standart_vs_nvidia[(2 * use_bumpmap) + use_skinning];\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_HIGH(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso, terraincolor) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, false);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, false);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, false);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso, terraincolor);\
				}\
			}
		#define DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}
	#else
		#define DEFINE_STANDART_TECHNIQUE(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_0 ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, false, use_aniso);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_HIGH(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart(PCF_NVIDIA, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, use_bumpmap, use_skinning, use_specularfactor, use_specularmap)
		#define DEFINE_STANDART_TECHNIQUE_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, false);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, false);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, false);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}
		#define DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(tech_name, use_bumpmap, use_skinning, use_specularfactor, use_specularmap, use_aniso) \
			technique tech_name{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NONE, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDW{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_DEFAULT, use_bumpmap, use_skinning);\
					PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}\
			technique tech_name##_SHDWNVIDIA{\
				pass P0{\
					VertexShader = compile vs_2_0 vs_main_standart_Instanced(PCF_NVIDIA, use_bumpmap, use_skinning);\
					PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, use_bumpmap, use_specularfactor, use_specularmap, true, use_aniso);\
				}\
			}
	#endif
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospecmap, true, false, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_specmap, true, false, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_nospecmap, true, true, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_specmap, true, true, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_nospecmap_high, true, true, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_specmap_high, true, true, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospecmap_high, true, false, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_specmap_high, true, false, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospecmap, false, false, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_specmap, false, false, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_nospecmap, false, true, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_specmap, false, true, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospec, false, false, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_nobump_nospec_noterraincolor, false, false, false, false, false, false)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospec, true, false, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_noskin_bump_nospec_noterraincolor, true, false, false, false, false, false)
	DEFINE_STANDART_TECHNIQUE(standart_skin_nobump_nospec, false, true, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE(standart_skin_bump_nospec, true, true, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospec_high, true, false, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_noskin_bump_nospec_high_noterraincolor, true, false, false, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH(standart_skin_bump_nospec_high, true, true, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospec_Instanced, false, false, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospec_noterraincolor_Instanced, false, false, false, false, false, false)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_nospecmap_Instanced, true, false, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_specmap_Instanced, false, false, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_specmap_Instanced, true, false, true, true, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_nobump_nospecmap_Instanced, false, false, true, false, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_nospec_high_Instanced, true, false, false, false, false, true)
	DEFINE_STANDART_TECHNIQUE_INSTANCED(standart_noskin_bump_nospec_high_noterraincolor_Instanced, true, false, false, false, false, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bump_specmap_high_Instanced, true, false, true, true, false)
	DEFINE_STANDART_TECHNIQUE_HIGH_INSTANCED(standart_noskin_bump_nospecmap_high_Instanced, true, false, true, false, false)
	#define DEFINE_STANDART_TECHNIQUE_GREENBUMP(tech_name, high, terraincolor) \
		technique tech_name{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, true, false);\
				PixelShader = compile PS_2_X ps_main_standart(PCF_NONE, true, false, false, high, false, terraincolor, true);\
			}\
		}\
		technique tech_name##_SHDW{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, true, false);\
				PixelShader = compile PS_2_X ps_main_standart(PCF_DEFAULT, true, false, false, high, false, terraincolor, true);\
			}\
		}\
		technique tech_name##_SHDWNVIDIA{\
			pass P0{\
				VertexShader = compile vs_2_0 vs_main_standart(PCF_NVIDIA, true, false);\
				PixelShader = compile ps_2_a ps_main_standart(PCF_NVIDIA, true, false, false, high, false, terraincolor, true);\
			}\
		}\
		DEFINE_LIGHTING_TECHNIQUE(tech_name, 0, true, false, false, false)
	DEFINE_STANDART_TECHNIQUE_GREENBUMP(standart_noskin_bump_nospec_noterraincolor_greenbump, false, false)
	DEFINE_STANDART_TECHNIQUE_GREENBUMP(standart_noskin_bump_nospec_high_noterraincolor_greenbump, true, false)
	technique standart_skin_bump_nospecmap_high_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, true, true);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_NONE, false, true);
		}
	}
	technique standart_skin_bump_nospecmap_high_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, true, true);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_DEFAULT, false, true);
		}
	}
	technique standart_skin_bump_nospecmap_high_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, true, true);
			PixelShader = compile ps_2_a ps_main_standart_old_good(PCF_NVIDIA, false, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_NONE, true, true);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_NONE, true, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_main_standart(PCF_DEFAULT, true, true);
			PixelShader = compile PS_2_X ps_main_standart_old_good(PCF_DEFAULT, true, true);
		}
	}
	technique standart_skin_bump_specmap_high_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_main_standart(PCF_NVIDIA, true, true);
			PixelShader = compile ps_2_a ps_main_standart_old_good(PCF_NVIDIA, true, true);
		}
	}
#endif
#ifdef HAIR_SHADERS
	struct VS_OUTPUT_SIMPLE_HAIR{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_SIMPLE_HAIR vs_hair(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_SIMPLE_HAIR, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, true, false);
		#endif
		Out.Color = vColor * diffuse_light;
		float wNdotSun = dot(vWorldN, -vSunDir);
		Out.SunLight = max(0.2f * (wNdotSun + 0.9f), wNdotSun) * vSunColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_hair(VS_OUTPUT_SIMPLE_HAIR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		float4 tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		float4 final_col;
		INPUT_TEX_GAMMA(tex1_col.rgb);
		final_col = tex1_col * vMaterialColor;
		float alpha = saturate(((2.0f * vMaterialColor2.a) + tex2_col.a) - 1.9f);
		final_col.rgb *= (1.0f - alpha);
		final_col.rgb += tex2_col.rgb * alpha;
		float4 total_light = In.Color;
		if((PcfMode != PCF_NONE)){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
	struct VS_INPUT_HAIR{
		float4 vPosition: POSITION;
		float3 vNormal: NORMAL;
		float3 vTangent: BINORMAL;
		float2 tc: TEXCOORD0;
		float4 vColor: COLOR0;
	};
	struct VS_OUTPUT_HAIR{
		float4 Pos: POSITION;
		float2 Tex0: TEXCOORD0;
		float4 VertexLighting: TEXCOORD1;
		float3 viewVec: TEXCOORD2;
		float3 normal: TEXCOORD3;
		float3 tangent: TEXCOORD4;
		float4 VertexColor: COLOR0;
		float4 ShadowTexCoord: TEXCOORD6;
		float2 ShadowTexelPos: TEXCOORD7;
		float Fog: FOG;
	};
	VS_OUTPUT_HAIR vs_hair_aniso(uniform const int PcfMode, VS_INPUT_HAIR In){
		INITIALIZE_OUTPUT(VS_OUTPUT_HAIR, Out);
		Out.Pos = mul(matWorldViewProj, In.vPosition);
		float4 vWorldPos = (float4)mul(matWorld, In.vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, In.vNormal));
		float3 P = mul(matWorldView, In.vPosition);
		Out.Tex0 = In.tc;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, true, false);
		#endif
		Out.VertexLighting = saturate(In.vColor * diffuse_light);
		Out.VertexColor = In.vColor;
		if(true){
			float3 Pview = vCameraPos - vWorldPos;
			Out.normal = normalize(mul(matWorld, In.vNormal));
			Out.tangent = normalize(mul(matWorld, In.vTangent));
			Out.viewVec = normalize(Pview);
		}
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_hair_aniso(VS_OUTPUT_HAIR In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float3 lightDir = -vSunDir;
		float3 hairBaseColor = vMaterialColor.rgb;
		float3 diffuse = hairBaseColor * vSunColor.rgb * In.VertexColor.rgb * HairDiffuseTerm(In.normal, lightDir);
		float4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex1_col.rgb);
		float4 tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		float alpha = saturate(((2.0f * vMaterialColor2.a) + tex2_col.a) - 1.9f);
		float4 final_col = tex1_col;
		final_col.rgb *= hairBaseColor;
		final_col.rgb *= (1.0f - alpha);
		final_col.rgb += tex2_col.rgb * alpha;
		float sun_amount = 1;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		float3 specular = calculate_hair_specular(In.normal, In.tangent, lightDir, In.viewVec, In.Tex0);
		float4 total_light = vAmbientColor;
		total_light.rgb += (((diffuse + specular) * sun_amount));
		total_light.rgb += In.VertexLighting.rgb;
		Output.RGBColor.rgb = total_light * final_col.rgb;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		Output.RGBColor.a = tex1_col.a * vMaterialColor.a;
		Output.RGBColor = saturate(Output.RGBColor);
		return Output;
	}
	technique hair_shader_aniso{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_NONE);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_NONE);
		}
	}
	technique hair_shader_aniso_SHDW{
		pass P0{
			VertexShader = compile vs_2_0 vs_hair_aniso(PCF_DEFAULT);
			PixelShader = compile PS_2_X ps_hair_aniso(PCF_DEFAULT);
		}
	}
	technique hair_shader_aniso_SHDWNVIDIA{
		pass P0{
			VertexShader = compile vs_2_a vs_hair_aniso(PCF_NVIDIA);
			PixelShader = compile ps_2_a ps_hair_aniso(PCF_NVIDIA);
		}
	}
#endif
#ifdef FACE_SHADERS
	struct VS_OUTPUT_SIMPLE_FACE{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
	};
	VS_OUTPUT_SIMPLE_FACE vs_face(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_SIMPLE_FACE, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, true, false);
		#endif
		Out.Color = vMaterialColor * vColor * diffuse_light;
		float wNdotSun = dot(vWorldN, -vSunDir);
		Out.SunLight = max(0.2f * (wNdotSun + 0.9f), wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_face(VS_OUTPUT_SIMPLE_FACE In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		float4 tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		float4 tex_col;
		tex_col = tex2_col * In.Color.a + tex1_col * (1.0f - In.Color.a);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if((PcfMode != PCF_NONE)){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
		float Fog: FOG;
		float4 VertexColor: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 WorldPos: TEXCOORD1;
		float3 ViewVec: TEXCOORD2;
		float3 SunLightDir: TEXCOORD3;
		float4 PointLightDir: TEXCOORD4;
		float4 ShadowTexCoord: TEXCOORD5;
		float2 ShadowTexelPos: TEXCOORD6;
		#ifdef INCLUDE_VERTEX_LIGHTING
			float3 VertexLighting: TEXCOORD7;
		#endif
	};
	VS_OUTPUT_STANDART vs_main_standart_face_mod(uniform const int PcfMode, uniform const bool use_bumpmap, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float4 vVertexColor: COLOR0, float4 vBlendWeights: BLENDWEIGHT, float4 vBlendIndices: BLENDINDICES){
		INITIALIZE_OUTPUT(VS_OUTPUT_STANDART, Out);
		float4 vObjectPos;
		float3 vObjectN, vObjectT, vObjectB;
		vObjectPos = vPosition;
		vObjectN = vNormal;
		if(use_bumpmap){
			vObjectT = vTangent;
			vObjectB = vBinormal;
		}
		float4 vWorldPos = mul(matWorld, vObjectPos);
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		float3 vWorldN = normalize(mul((float3x3)matWorld, vObjectN));
		float3x3 TBNMatrix;
		if(use_bumpmap){
			float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vObjectB));
			float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vObjectT));
			TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		}
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		if(use_bumpmap){
			Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
			Out.SkyLightDir = mul(TBNMatrix, -vSkyLightDir);
		}
		else{
			Out.SunLightDir = vWorldN;
		}
		Out.VertexColor = vVertexColor;
		#ifdef INCLUDE_VERTEX_LIGHTING
			Out.VertexLighting = calculate_point_lights_diffuse(vWorldPos, vWorldN, true, true);
		#endif
		#ifndef USE_LIGHTING_PASS
			const int effective_light_index = iLightIndices[0];
			float3 point_to_light = vLightPosDir[effective_light_index] - vWorldPos.xyz;
			float LD = dot(point_to_light, point_to_light);
			Out.PointLightDir.a = saturate(1.0f / LD);
			if(use_bumpmap){
				Out.PointLightDir.xyz = mul(TBNMatrix, normalize(point_to_light));
			}
			else{
				Out.PointLightDir.xyz = normalize(point_to_light);
			}
		#endif
		if(use_bumpmap){
			Out.ViewDir = mul(TBNMatrix, normalize(vCameraPos.xyz - vWorldPos.xyz));
		}
		else{
			Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		}
		float3 P = mul(matWorldView, vObjectPos);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_standart_face_mod(VS_OUTPUT_STANDART In, uniform const int PcfMode, uniform const bool use_bumpmap, uniform const bool use_ps2a){
		PS_OUTPUT Output;
		float4 total_light = vAmbientColor;
		float3 normal;
		if(use_bumpmap){
			float3 tex1_norm, tex2_norm;
			tex1_norm = tex2D(NormalTextureSampler, In.Tex0);
			if(use_ps2a){
				tex2_norm = tex2D(SpecularTextureSampler, In.Tex0);
				normal = lerp(tex1_norm, tex2_norm, In.VertexColor.a);
				normal = 2.0f * normal - 1.0f;
				normal = normalize(normal);
			}
			else{
				normal = (2 * tex1_norm - 1);
			}
		}
		else{
			normal = In.SunLightDir.xyz;
		}
		float sun_amount = 1;
		if(PcfMode != PCF_NONE){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		if(use_bumpmap){
			total_light += face_NdotL(In.SunLightDir.xyz, normal.xyz) * sun_amount * vSunColor;
			if(use_ps2a){
				total_light += face_NdotL(In.SkyLightDir.xyz, normal.xyz) * vSkyLightColor;
			}
		}
		else {
			total_light += face_NdotL( - vSunDir, normal.xyz) * sun_amount * vSunColor;
			if(use_ps2a){
				total_light += face_NdotL( - vSkyLightDir, normal.xyz) * vSkyLightColor;
			}
		}
		float3 point_lighting = 0;
		#ifndef USE_LIGHTING_PASS
			float light_atten = In.PointLightDir.a * 0.9f;
			const int effective_light_index = iLightIndices[0];
			point_lighting += light_atten * face_NdotL(In.PointLightDir.xyz, normal.xyz) * vLightDiffuse[effective_light_index];
		#endif
		#ifdef INCLUDE_VERTEX_LIGHTING
			if(use_ps2a){
				point_lighting += In.VertexLighting;
			}
		#endif
		total_light.rgb += point_lighting;
		if(PcfMode != PCF_NONE)Output.RGBColor.rgb = total_light.rgb;
		else Output.RGBColor.rgb = min(total_light.rgb, 2.0f);
		float4 tex1_col = tex2D(MeshTextureSampler, In.Tex0);
		float4 tex2_col = tex2D(Diffuse2Sampler, In.Tex0);
		float4 tex_col = lerp(tex1_col, tex2_col, In.VertexColor.a);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor *= tex_col;
		Output.RGBColor.rgb *= (In.VertexColor.rgb * vMaterialColor.rgb);
		if(use_ps2a){
			float fSpecular = 0;
			float4 specColor = vSpecularColor * vSunColor;
			if(false){
				specColor *= tex2D(SpecularTextureSampler, In.Tex0);
			}
			float3 vHalf = normalize(In.ViewDir + In.SunLightDir);
			fSpecular = specColor * pow(saturate(dot(vHalf, normal)), fMaterialPower) * sun_amount;
			float fresnel = saturate(1.0f - dot(In.ViewDir, normal));
			Output.RGBColor += fresnel * fSpecular;
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
		float3 vObjectN = normalize(mul((float3x3)matWorldArray[vBlendIndices.x], vNormal) * vBlendWeights.x + mul((float3x3)matWorldArray[vBlendIndices.y], vNormal) * vBlendWeights.y + mul((float3x3)matWorldArray[vBlendIndices.z], vNormal) * vBlendWeights.z + mul((float3x3)matWorldArray[vBlendIndices.w], vNormal) * vBlendWeights.w);
		float4 vWorldPos = mul(matWorld, vObjectPos);
		Out.Pos = mul(matWorldViewProj, vObjectPos);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vObjectN));
		float3 P = mul(matView, vWorldPos);
		Out.Tex0 = tc;
		Out.Color = vAmbientColor;
		Out.Color += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			Out.Color += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color *= vMaterialColor * vColor;
		Out.Color = min(1, Out.Color);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = wNdotSun * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
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
		float Fog: FOG;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
	};
	struct VS_OUTPUT_FLORA_NO_SHADOW{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
	};
	VS_OUTPUT_FLORA vs_flora(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * (vAmbientColor + vSunColor * 0.06f);
		Out.Color.a *= vMaterialColor.a;
		Out.SunLight = (vSunColor * 0.34f) * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	VS_OUTPUT_FLORA vs_flora_Instanced(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0, float3 vInstanceData0: TEXCOORD1, float3 vInstanceData1: TEXCOORD2, float3 vInstanceData2: TEXCOORD3, float3 vInstanceData3: TEXCOORD4){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		float4x4 matWorldOfInstance = build_instance_frame_matrix(vInstanceData0, vInstanceData1, vInstanceData2, vInstanceData3);
		float4 vWorldPos = (float4)mul(matWorldOfInstance, vPosition);
		Out.Pos = mul(matViewProj, vWorldPos);
		Out.Tex0 = tc;
		Out.Color = vColor * (vAmbientColor + vSunColor * 0.06f);
		Out.Color.a *= vMaterialColor.a;
		Out.SunLight = (vSunColor * 0.34f) * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matView, vWorldPos);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_flora(VS_OUTPUT_FLORA In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.05f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if(PcfMode != PCF_NONE){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_flora_no_shadow(VS_OUTPUT_FLORA_NO_SHADOW In){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.05f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = tex_col * In.Color;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	VS_OUTPUT_FLORA vs_grass(uniform const int PcfMode, float4 vPosition: POSITION, float4 vColor: COLOR0, float2 tc: TEXCOORD0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vAmbientColor;
		if(PcfMode != PCF_NONE){
			Out.SunLight = (vSunColor * 0.55f) * vMaterialColor * vColor;
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		else {
			Out.SunLight = vSunColor * 0.5f * vColor;
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		Out.Color.a = min(1.0f, (1.0f - (d / 50.0f)) * 2.0f);
		return Out;
	}
	PS_OUTPUT ps_grass(VS_OUTPUT_FLORA In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(GrassTextureSampler, In.Tex0);
		clip(tex_col.a - 0.1f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		if((PcfMode != PCF_NONE)){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		Out.Color.a = min(1.0f, (1.0f - (d / 50.0f)) * 2.0f);
		return Out;
	}
	PS_OUTPUT ps_grass_no_shadow(VS_OUTPUT_FLORA_NO_SHADOW In){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		clip(tex_col.a - 0.1f);
		INPUT_TEX_GAMMA(tex_col.rgb);
		Output.RGBColor = tex_col * In.Color;
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
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
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
		float3 ViewDir: TEXCOORD6;
		float3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_MAP vs_main_map(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos - vWorldPos);
		Out.WorldNormal = vWorldN;
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_map(VS_OUTPUT_MAP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		float sun_amount = 1;
		if((PcfMode != PCF_NONE)){
			sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
		}
		Output.RGBColor = tex_col * ((In.Color + In.SunLight * sun_amount));
		{
			float fresnel = 1 - (saturate(dot(normalize(In.ViewDir), normalize(In.WorldNormal))));
			fresnel *= fresnel;
			Output.RGBColor.rgb *= max(0.6, fresnel + 0.1);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(diffuse_map, vs_main_map, ps_main_map)
	struct VS_OUTPUT_MAP_BUMP{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
		float3 SunLightDir: TEXCOORD4;
		float3 SkyLightDir: TEXCOORD5;
		float3 ViewDir: TEXCOORD6;
		float3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_MAP_BUMP vs_main_map_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		#ifndef USE_LIGHTING_PASS
			diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		#endif
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		Out.SunLightDir = normalize(mul(TBNMatrix, -vSunDir));
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos - vWorldPos);
		Out.WorldNormal = vWorldN;
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_map_bump(VS_OUTPUT_MAP_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		float3 normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0 * map_normal_detail_factor).rgb - 1.0f);
		float4 In_SunLight = saturate(dot(normal, In.SunLightDir)) * vSunColor * vMaterialColor;
		float sun_amount = 1;
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
		float4 Color: COLOR0;
		float3 Tex0: TEXCOORD0;
		float4 SunLight: TEXCOORD1;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float3 ViewDir: TEXCOORD6;
		float3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_MAP_MOUNTAIN vs_map_mountain(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_MOUNTAIN, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0.xy = tc;
		Out.Tex0.z = (0.7f * (vWorldPos.z - 1.5f));
		float4 diffuse_light = vAmbientColor;
		if(true){
			diffuse_light += vLightColor;
		}
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos - vWorldPos);
		Out.WorldNormal = vWorldN;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_map_mountain(VS_OUTPUT_MAP_MOUNTAIN In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(tex_col.rgb);
		tex_col.rgb += saturate(In.Tex0.z * (tex_col.a) - 1.5f);
		tex_col.a = 1.0f;
		if((PcfMode != PCF_NONE)){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
			Output.RGBColor = saturate(tex_col) * ((In.Color + In.SunLight * sun_amount));
		}
		else {
			Output.RGBColor = saturate(tex_col) * (In.Color + In.SunLight);
		}
		{
			float fresnel = 1 - (saturate(dot(In.ViewDir, In.WorldNormal)));
			Output.RGBColor.rgb *= max(0.6, fresnel + 0.1);
		}
		OUTPUT_GAMMA(Output.RGBColor.rgb);
		return Output;
	}
	DEFINE_TECHNIQUES(map_mountain, vs_map_mountain, ps_map_mountain)
	struct VS_OUTPUT_MAP_MOUNTAIN_BUMP{
		float4 Pos: POSITION;
		float4 Color: COLOR0;
		float3 Tex0: TEXCOORD0;
		float4 ShadowTexCoord: TEXCOORD2;
		float2 ShadowTexelPos: TEXCOORD3;
		float Fog: FOG;
		float3 SunLightDir: TEXCOORD4;
		float3 SkyLightDir: TEXCOORD5;
		float3 ViewDir: TEXCOORD6;
		float3 WorldNormal: TEXCOORD7;
	};
	VS_OUTPUT_MAP_MOUNTAIN_BUMP vs_map_mountain_bump(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float3 vTangent: TANGENT, float3 vBinormal: BINORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_MOUNTAIN_BUMP, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		float3 P = mul(matWorldView, vPosition);
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
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.ViewDir = normalize(vCameraPos - vWorldPos);
		Out.WorldNormal = vWorldN;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_map_mountain_bump(VS_OUTPUT_MAP_MOUNTAIN_BUMP In, uniform const int PcfMode){
		PS_OUTPUT Output;
		float4 sample_col = tex2D(MeshTextureSampler, In.Tex0.xy);
		INPUT_TEX_GAMMA(sample_col.rgb);
		float4 tex_col = sample_col;
		tex_col.rgb += saturate(In.Tex0.z * (sample_col.a) - 1.5f);
		tex_col.a = 1.0f;
		float3 normal = (2.0f * tex2D(NormalTextureSampler, In.Tex0 * map_normal_detail_factor).rgb - 1.0f);
		float4 In_SunLight = saturate(dot(normal, In.SunLightDir)) * vSunColor;
		if((PcfMode != PCF_NONE)){
			float sun_amount = GetSunAmount(PcfMode, In.ShadowTexCoord, In.ShadowTexelPos);
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
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float3 LightDir: TEXCOORD1;
		float3 CameraDir: TEXCOORD3;
		float4 PosWater: TEXCOORD4;
		float Fog: FOG;
	};
	VS_OUTPUT_MAP_WATER vs_map_water(uniform const bool reflections, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0, float4 vLightColor: COLOR1){
		INITIALIZE_OUTPUT(VS_OUTPUT_MAP_WATER, Out);
		Out.Pos = mul(matWorldViewProj, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 P = mul(matWorldView, vPosition);
		Out.Tex0 = tc + texture_offset.xy;
		float4 diffuse_light = vAmbientColor + vLightColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		float wNdotSun = max( - 0.0001f, dot(vWorldN, -vSunDir));
		diffuse_light += (wNdotSun) * vSunColor;
		Out.Color = (vMaterialColor * vColor) * diffuse_light;
		if(reflections){
			float4 water_pos = mul(matWaterViewProj, vWorldPos);
			Out.PosWater.xy = (float2(water_pos.x, -water_pos.y) + water_pos.w) / 2;
			Out.PosWater.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * water_pos.w);
			Out.PosWater.zw = water_pos.zw;
		}
		{
			float3 vWorldN = float3(0, 0, 1);
			float3 vWorld_tangent = float3(1, 0, 0);
			float3 vWorld_binormal = float3(0, 1, 0);
			float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
			float3 point_to_camera_normal = normalize(vCameraPos.xyz - vWorldPos.xyz);
			Out.CameraDir = mul(TBNMatrix, -point_to_camera_normal);
			Out.LightDir = mul(TBNMatrix, -vSunDir);
		}
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_map_water(uniform const bool reflections, VS_OUTPUT_MAP_WATER In){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		float4 tex_col = tex2D(MeshTextureSampler, In.Tex0);
		INPUT_TEX_GAMMA(tex_col.rgb);
		float3 normal;
		normal.xy = (2.0f * tex2D(NormalTextureSampler, In.Tex0 * 8).ag - 1.0f);
		normal.z = sqrt(1.0f - dot(normal.xy, normal.xy));
		float NdotL = saturate(dot(normal, In.LightDir));
		float3 vView = normalize(In.CameraDir);
		float fresnel = 1 - (saturate(dot(vView, normal)));
		fresnel = 0.0204f + 0.9796 * (fresnel * fresnel * fresnel * fresnel * fresnel);
		Output.RGBColor.rgb += fresnel * In.Color.rgb;
		if(reflections){
			In.PosWater.xy += 0.35f * normal.xy;
			float4 tex = tex2Dproj(ReflectionTextureSampler, In.PosWater);
			INPUT_OUTPUT_GAMMA(tex.rgb);
			tex.rgb = min(tex.rgb, 4.0f);
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
		float4 Color: COLOR0;
		float2 Tex0: TEXCOORD0;
		float Fog: FOG;
		float4 projCoord: TEXCOORD1;
		float Depth: TEXCOORD2;
	};
	VS_DEPTHED_FLARE vs_main_depthed_flare(float4 vPosition: POSITION, float4 vColor: COLOR, float2 tc: TEXCOORD0){
		VS_DEPTHED_FLARE Out;
		Out.Pos = mul(matWorldViewProj, vPosition);
		Out.Tex0 = tc;
		Out.Color = vColor * vMaterialColor;
		if(use_depth_effects){
			Out.projCoord.xy = (float2(Out.Pos.x, -Out.Pos.y) + Out.Pos.w) / 2;
			Out.projCoord.xy += (vDepthRT_HalfPixel_ViewportSizeInv.xy * Out.Pos.w);
			Out.projCoord.zw = Out.Pos.zw;
			Out.Depth = Out.Pos.z * far_clip_Inv;
		}
		float3 P = mul(matWorldView, vPosition);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	PS_OUTPUT ps_main_depthed_flare(VS_DEPTHED_FLARE In, uniform const bool sun_like, uniform const bool blend_adding){
		PS_OUTPUT Output;
		Output.RGBColor = In.Color;
		Output.RGBColor *= tex2D(MeshTextureSampler, In.Tex0);
		if(!blend_adding){
			OUTPUT_GAMMA(Output.RGBColor.rgb);
		}
		if(use_depth_effects){
			float depth = tex2Dproj(DepthTextureSampler, In.projCoord).r;
			float alpha_factor;
			if(sun_like){
				float my_depth = 0;
				alpha_factor = depth;
				float fog_factor = 1.001f - (10.f * (fFogDensity + 0.001f));
				alpha_factor *= fog_factor;
			}
			else{
				alpha_factor = saturate((depth - In.Depth) * 4096);
			}
			if(blend_adding){
				Output.RGBColor *= alpha_factor;
			}
			else{
				Output.RGBColor.a *= alpha_factor;
			}
		}
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
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
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
		float3 cWaterColor = 2 * float3(20.0f / 255.0f, 45.0f / 255.0f, 100.0f / 255.0f) * vSunColor;
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
#ifdef NEWTREE_SHADERS
	VS_OUTPUT_FLORA vs_flora_billboards(uniform const int PcfMode, float4 vPosition: POSITION, float3 vNormal: NORMAL, float2 tc: TEXCOORD0, float4 vColor: COLOR0){
		INITIALIZE_OUTPUT(VS_OUTPUT_FLORA, Out);
		float4 vWorldPos = (float4)mul(matWorld, vPosition);
		float3 view_vec = (vCameraPos.xyz - vWorldPos.xyz);
		float dist_to_vertex = length(view_vec);
		float alpha_val = saturate(0.5f + ((dist_to_vertex - flora_detail_fade) / flora_detail_fade_inv));
		Out.Pos = mul(matWorldViewProj, vPosition);
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		Out.Tex0 = tc;
		float4 diffuse_light = vAmbientColor;
		diffuse_light += saturate(dot(vWorldN, -vSkyLightDir)) * vSkyLightColor;
		diffuse_light += calculate_point_lights_diffuse(vWorldPos, vWorldN, false, false);
		Out.Color = (vMaterialColor * vColor * diffuse_light);
		Out.Color.a *= alpha_val;
		float wNdotSun = saturate(dot(vWorldN, -vSunDir));
		Out.SunLight = (wNdotSun) * vSunColor * vMaterialColor * vColor;
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		float3 P = mul(matWorldView, vPosition);
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
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
		float3 vWorldN = normalize(mul((float3x3)matWorld, vNormal));
		float3 vWorld_binormal = normalize(mul((float3x3)matWorld, vBinormal));
		float3 vWorld_tangent = normalize(mul((float3x3)matWorld, vTangent));
		float3 P = mul(matWorldView, vPosition);
		float3x3 TBNMatrix = float3x3(vWorld_tangent, vWorld_binormal, vWorldN);
		if(PcfMode != PCF_NONE){
			float4 ShadowPos = mul(matSunViewProj, vWorldPos);
			Out.ShadowTexCoord = ShadowPos;
			Out.ShadowTexCoord.z /= ShadowPos.w;
			Out.ShadowTexCoord.w = 1.0f;
			Out.ShadowTexelPos = Out.ShadowTexCoord * fShadowMapSize;
		}
		Out.SunLightDir = mul(TBNMatrix, -vSunDir);
		Out.SkyLightDir = mul(TBNMatrix, -vSkyLightDir);
		#ifdef USE_LIGHTING_PASS
			Out.PointLightDir = vWorldPos;
		#else
			Out.PointLightDir.rgb = 2.0f * vPointLightDir.rgb - 1.0f;
			Out.PointLightDir.a = vPointLightDir.a;
		#endif
		Out.VertexColor = vVertexColor;
		Out.VertexColor.a *= alpha_val;
		Out.ViewDir = normalize(vCameraPos.xyz - vWorldPos.xyz);
		Out.WorldNormal = vWorldN;
		float d = length(P);
		Out.Fog = get_fog_amount_new(d, vWorldPos.z);
		return Out;
	}
	DEFINE_TECHNIQUES(tree_billboards_dot3_alpha, vs_main_bump_billboards, ps_main_bump_simple)
#endif