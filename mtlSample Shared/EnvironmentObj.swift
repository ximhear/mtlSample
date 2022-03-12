//
//  EnvironmentObj.swift
//  mtlSample
//
//  Created by gzonelee on 2022/03/12.
//

import Foundation
import MetalKit

class EnvironmentObj {
    let mesh: MTKMesh
    var texture: MTLTexture?
    var diffuseTexture: MTLTexture?
    var pipelineState: MTLRenderPipelineState?
    var depthStencilState: MTLDepthStencilState?
    
    init(textureName: String?, metalView: MTKView, device: MTLDevice, library: MTLLibrary) {
        let allocator = MTKMeshBufferAllocator(device: device)
        let cube = MDLMesh(boxWithExtent: [1, 1, 1],
                           segments: [1, 1, 1],
                           inwardNormals: true,
                           geometryType: .triangles,
                           allocator: allocator)
        
        do {
           mesh = try MTKMesh(mesh: cube, device: device)
        }
        catch {
            GZLogFunc(error)
            fatalError()
        }
        pipelineState = self.buildPipelineState(vertexDescriptor: cube.vertexDescriptor,
                                           metalView: metalView, device: device, library: library)
        depthStencilState = self.buildDepthStencilState(device: device)
        
        if let textureName = textureName {
            texture = try? loadCubeTexture(imageName: textureName, device: device)
//            loadIrridianceMap(imageName: "cube-sky.png", device: device)
            diffuseTexture = try? loadCubeTexture(imageName: "cube-sky.png", device: device)
        }
        else {
            texture = loadGeneratedSkyboxTexture(dimensions: [256, 256], device: device)
        }
    }
    
    func buildPipelineState(vertexDescriptor: MDLVertexDescriptor, metalView: MTKView, device: MTLDevice, library: MTLLibrary) -> MTLRenderPipelineState {
        let desc = MTLRenderPipelineDescriptor()
        desc.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        desc.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat
        desc.stencilAttachmentPixelFormat = metalView.depthStencilPixelFormat
        desc.vertexFunction = library.makeFunction(name: "vertexSkybox")
        desc.fragmentFunction = library.makeFunction(name: "fragmentSkybox")
        desc.vertexDescriptor = MTKMetalVertexDescriptorFromModelIO(vertexDescriptor)
        do {
            return try device.makeRenderPipelineState(descriptor: desc)
        }
        catch {
            GZLogFunc(error)
            fatalError()
        }
    }
    
    func buildDepthStencilState(device: MTLDevice) -> MTLDepthStencilState? {
        let desc = MTLDepthStencilDescriptor()
        desc.depthCompareFunction = .lessEqual
        desc.isDepthWriteEnabled = true
       
        return device.makeDepthStencilState(descriptor: desc)
    }
    
    func loadGeneratedSkyboxTexture(dimensions: SIMD2<Int32>, device: MTLDevice) -> MTLTexture? {
        var texture: MTLTexture?
        let skyTexture = MDLSkyCubeTexture(name: "environment",
                                           channelEncoding: .uInt8,
                                           textureDimensions: dimensions,
                                           turbidity: 0.38,
                                           sunElevation: 0.8,
                                           upperAtmosphereScattering: 0.3,
                                           groundAlbedo: 2)
        do {
            let loader = MTKTextureLoader(device: device)
            texture = try loader.newTexture(texture: skyTexture, options: nil)
        }
        catch {
            GZLogFunc(error)
        }
        return texture
    }
    
    func render(renderEncoder: MTLRenderCommandEncoder, uniforms: Uniforms, rotation: Float) {
        renderEncoder.setRenderPipelineState(self.pipelineState!)
        renderEncoder.setDepthStencilState(depthStencilState!)
        renderEncoder.setVertexBuffer(mesh.vertexBuffers[0].buffer, offset: 0, index: 0)
        
        var viewMatrix = uniforms.viewMatrix * matrix4x4_rotation(radians: rotation, axis: [0, 1, 0])
        viewMatrix.columns.3 = [0, 0, 0, 1]
        var viewProjectionMatrix = uniforms.projectionMatrix * viewMatrix
        renderEncoder.setVertexBytes(&viewProjectionMatrix, length: MemoryLayout<float4x4>.stride, index: 1)
        
        renderEncoder.setFragmentTexture(texture, index: 0)
        
        let submesh = mesh.submeshes[0]
        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                            indexCount: submesh.indexCount,
                                            indexType: submesh.indexType,
                                            indexBuffer: submesh.indexBuffer.buffer,
                                            indexBufferOffset: 0)
    }
    
    func loadCubeTexture(imageName: String, device: MTLDevice) throws -> MTLTexture {
        let textureLoader = MTKTextureLoader(device: device)
        if let texture = MDLTexture(cubeWithImagesNamed: [imageName]) {
            let options: [MTKTextureLoader.Option: Any] =
            [.origin: MTKTextureLoader.Origin.topLeft,
             .SRGB: false,
             .generateMipmaps: NSNumber(booleanLiteral: false)]
            return try textureLoader.newTexture(texture: texture, options: options)
        }
        let texture = try textureLoader.newTexture(name: imageName, scaleFactor: 1.0,
                                                   bundle: .main)
        return texture
    }
    
    func loadIrridianceMap(imageName: String, device: MTLDevice) {
        GZLogFunc(imageName)
        if let texture = MDLTexture(cubeWithImagesNamed: [imageName]) {
//            let options: [MTKTextureLoader.Option: Any] =
//            [.origin: MTKTextureLoader.Origin.topLeft,
//             .SRGB: false,
//             .generateMipmaps: NSNumber(booleanLiteral: false)]
            let irridiance = MDLTexture.irradianceTextureCube(with: texture, name: nil, dimensions: [64, 64], roughness: 0.6)
            let textureLoader = MTKTextureLoader(device: device)
            diffuseTexture = try? textureLoader.newTexture(texture: irridiance, options: nil)
        }
    }
}
