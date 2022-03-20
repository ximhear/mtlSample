//
//  ICB.metal
//  mtlSample
//
//  Created by gzonelee on 2022/03/20.
//

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

struct ICBContainer {
    command_buffer icb [[ id(0) ]];
};

struct Model {
    constant float* vertexBuffer;
    constant uint* indexBuffer;
    constant float* texturesBuffer;
    render_pipeline_state pipelineState;
};

kernel void encodeCommands(
                           uint modelIndex [[thread_position_in_grid]],
                           constant Uniforms& uniforms [[ buffer(BufferIndexUniforms) ]],
                           constant MTLDrawIndexedPrimitivesIndirectArguments* drawArgumentsBuffer [[ buffer(BufferIndexDrawArguments) ]],
                           constant MeshUniforms* meshUniforms [[ buffer(BufferIndexMeshUniforms) ]],
                           constant Model* models [[ buffer(BufferIndexModels) ]],
                           device ICBContainer* icbContainer [[ buffer(BufferIndexICB) ]]
                           )
{
    Model model = models[modelIndex];
    MTLDrawIndexedPrimitivesIndirectArguments drawArguments = drawArgumentsBuffer[modelIndex];
    render_command cmd(icbContainer->icb, modelIndex);
    cmd.set_render_pipeline_state(model.pipelineState);
    cmd.set_vertex_buffer(&uniforms, BufferIndexUniforms);
    cmd.set_vertex_buffer(meshUniforms, BufferIndexMeshUniforms);
    cmd.set_vertex_buffer(model.vertexBuffer, 0);
    cmd.set_fragment_buffer(model.texturesBuffer, BufferIndexTextures);
    
    cmd.draw_indexed_primitives(primitive_type::triangle,
                                drawArguments.indexCount,
                                model.indexBuffer + drawArguments.indexStart,
                                drawArguments.instanceCount,
                                drawArguments.baseVertex,
                                drawArguments.baseInstance);
}

