//
//  ShaderTypes.h
//  mtlSample Shared
//
//  Created by gzonelee on 2022/02/21.
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

typedef NS_ENUM(NSInteger, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexUniforms      = 5,
    BufferIndexMeshUniforms      = 6,
    BufferIndexTextures      = 7
};

typedef NS_ENUM(NSInteger, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeNormal  = 1,
    VertexAttributeTexcoord  = 2,
    VertexAttributeTangent  = 3,
    VertexAttributeBitangent  = 4,
};

typedef NS_ENUM(NSInteger, TextureIndex)
{
    TextureIndexColor    = 0,
    TextureIndexNormal    = 1,
    TextureIndexRough    = 2,
    TextureIndexMetalic    = 3,
    TextureIndexOcculusion    = 4,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    
    matrix_float4x4 viewMatrix;
} Uniforms;

typedef struct
{
    matrix_float4x4 modelMatrix;
    matrix_float3x3 normalMatrix;
} MeshUniforms;

#endif /* ShaderTypes_h */

