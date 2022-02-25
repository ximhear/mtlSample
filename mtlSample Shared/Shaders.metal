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
    float3 tangent [[attribute(3)]];
    float3 bitangent [[attribute(4)]];
} Vertex;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
    float3 worldBitangent;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]])
{
    float4 position = float4(in.position, 1.0);
//    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
//    out.texCoord = in.texCoord;
   
    float dist = -distance(float2(0, 0), float2(position.x, position.z)) / 10;
    matrix_float4x4 r = matrix_float4x4(
                                        float4(cos(dist), 0, -sin(dist), 0),
                                        float4(0, 1, 0, 0),
                                        float4(sin(dist), 0, cos(dist), 0),
                                        float4(0, 0, 0, 1)
                                        );
//    matrix_float4x4 modelMatrix = uniforms.modelMatrix * r;
    matrix_float4x4 modelMatrix = uniforms.modelMatrix;
    matrix_float3x3 normalMatrix = matrix_float3x3(
                                                   float3(modelMatrix.columns[0].xyz),
                                                   float3(modelMatrix.columns[1].xyz),
                                                   float3(modelMatrix.columns[2].xyz)
                                                   );
//    matrix_float3x3 normalMatrix = uniforms.normalMatrix;
    ColorInOut out {
        .position = uniforms.projectionMatrix * uniforms.viewMatrix * modelMatrix * position,
        .texCoord = in.texCoord,
        .worldPosition = (modelMatrix * position).xyz,
        .worldNormal = normalMatrix * in.normal,
        .worldTangent = normalMatrix * in.tangent,
        .worldBitangent = normalMatrix * in.bitangent
  };
  return out;
}

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               texture2d<half> colorMap     [[ texture(TextureIndexColor) ]],
                               texture2d<float> normalMap     [[ texture(TextureIndexNormal) ]],
                               texture2d<float> roughMap     [[ texture(TextureIndexRough) ]],
                               texture2d<float> metalMap     [[ texture(TextureIndexMetalic) ]],
                               texture2d<float> occulusionMap     [[ texture(TextureIndexOcculusion) ]]
                               )
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);
    float3 normalValue   = normalMap.sample(colorSampler, in.texCoord.xy).rgb;
    normalValue = normalValue * 2 - 1;
    normalValue = normalize(normalValue);
    float3 normalDirection = float3x3(in.worldTangent,
                                      in.worldBitangent,
                                      in.worldNormal) * normalValue;
    normalDirection = normalize(normalDirection);
    
    float3 lightPosition = float3(-1, 0, 1);
    float3 lightDirection = normalize(-lightPosition);
    float diffuseIntensity = saturate(-dot(lightDirection, normalDirection));
    
    float3 rough   = float3(1) - roughMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 metalic   = float3(1) - metalMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 occulusion = float3(1) - occulusionMap.sample(colorSampler, in.texCoord.xy).rrr;

    return float4(colorSample) * diffuseIntensity;
}
