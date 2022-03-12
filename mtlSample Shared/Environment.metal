//
//  Environment.metal
//  mtlSample
//
//  Created by gzonelee on 2022/03/12.
//

#include <metal_stdlib>
using namespace metal;

#import "ShaderTypes.h"

struct VertexIn {
    float4 position [[ attribute(0) ]];
};

struct VertexOut {
    float4 position [[ position ]];
    float3 textureCoordinates;
};

vertex VertexOut vertexSkybox(const VertexIn in [[stage_in]],
                              constant float4x4 &vp [[ buffer(1) ]]) {
    VertexOut out;
    out.position = (vp * in.position).xyww;
    out.textureCoordinates = in.position.xyz;
    return out;
}


fragment float4 fragmentSkybox(VertexOut in [[stage_in]],
                               texturecube<float> cubeTexture [[ texture(0) ]]) {
    constexpr sampler s(filter::linear);
    float4 color = cubeTexture.sample(s, in.textureCoordinates);
    return color;
    
    return float4(1, 1, 0, 1);
}
