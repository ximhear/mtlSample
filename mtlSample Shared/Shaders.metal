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

constant float PI = 3.1415926535897932384626433832795;

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
    float3 spotlightPosition;
    float3 coneDirection;
} ColorInOut;

vertex ColorInOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               constant MeshUniforms & meshUniforms [[ buffer(BufferIndexMeshUniforms) ]])
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
    matrix_float4x4 modelMatrix = meshUniforms.modelMatrix;
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
        .worldBitangent = normalMatrix * in.bitangent,
        .spotlightPosition = normalMatrix * float3( 30, 10, 30),
        .coneDirection = normalMatrix * float3(-1, 0, -1)
  };
  return out;
}

struct Textures {
    texture2d<float> colorMap;
    texture2d<float> normalMap;
    texture2d<float> roughMap;
    texture2d<float> metalMap;
    texture2d<float> occulusionMap;
};

fragment float4 fragmentShader(ColorInOut in [[stage_in]],
                               constant Uniforms & uniforms [[ buffer(BufferIndexUniforms) ]],
                               constant Textures& textures [[ buffer(BufferIndexTextures)]]
                               )
{
    constexpr sampler colorSampler(mip_filter::linear,
                                   mag_filter::linear,
                                   min_filter::linear);

    float3 baseColor = textures.colorMap.sample(colorSampler, in.texCoord.xy).xyz;
    float3 normalValue   = textures.normalMap.sample(colorSampler, in.texCoord.xy).rgb;
    normalValue = normalValue * 2 - 1;
    normalValue = normalize(normalValue);
    float3 normalDirection = float3x3(in.worldTangent,
                                      in.worldBitangent,
                                      in.worldNormal) * normalValue;
    normalDirection = normalize(normalDirection);
    
    float3 lightPosition = float3(-1, 0, 1);
    float3 lightDirection = normalize(-lightPosition);
    float diffuseIntensity = saturate(-dot(lightDirection, normalDirection));
    if (diffuseIntensity < 0.1) {
        diffuseIntensity = 0.1;
    }
    
    float3 spotlightPosition = float3( 30, 10, 40);
    float3 lightColor = float3(1, 0, 0);
    float coneAngle = PI * 9 / 180;
    float3 coneDirection = float3(-1, 0, -1);
    float3 lightAttenuation = float3(1.0, 0.5, 0);
    float coneAttenuation = 32;
    
    float3 rough   = float3(1) - textures.roughMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 metalic   = float3(1) - textures.metalMap.sample(colorSampler, in.texCoord.xy).rrr;
    float3 occulusion = float3(1) - textures.occulusionMap.sample(colorSampler, in.texCoord.xy).rrr;

    float3 color = 0;
    {
        float d = distance(spotlightPosition, in.worldPosition);
        float3 lightDirection = normalize(in.worldPosition - spotlightPosition);
        coneDirection = normalize(coneDirection);
        float spotResult = dot(lightDirection, coneDirection);
        if (spotResult > cos(coneAngle)) {
            float attenuation = 1.0 / (lightAttenuation.x + lightAttenuation.y * d + lightAttenuation.z * d * d);
            attenuation *= pow(spotResult, coneAttenuation);
            float diffuseIntensity = saturate(dot(-lightDirection, normalDirection));
            color = lightColor * diffuseIntensity;
            color *= attenuation;
        }
    }
    return float4(baseColor, 1) * diffuseIntensity + float4(color, 1);
}
