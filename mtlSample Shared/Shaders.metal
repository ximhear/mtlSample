//
//  Shaders.metal
//  mtlSample Shared
//
//  Created by gzonelee on 2022/02/21.
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 texCoord [[attribute(2)]];
//    float3 tangent [[attribute(3)]];
//    float3 bitangent [[attribute(4)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    ColorInOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               texture2d<half> normalMap     [[ texture(TextureIndexNormal) ]],
                               texture2d<float> roughMap     [[ texture(TextureIndexRough) ]],
                               texture2d<float> metalMap     [[ texture(TextureIndexMetalic) ]],
                               texture2d<float> occulusionMap     [[ texture(TextureIndexOcculusion) ]]
                               )
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
    half4 normalSample   = normalMap.sample(colorSampler, in.texCoord.xy);
    float3 rough   = float3(1) - roughMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 metalic   = float3(1) - metalMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 occulusion = float3(1) - occulusionMap.sample(colorSampler, in.texCoord.xy).rrr;

    return float4(colorSample) * float4(normalSample);
}
