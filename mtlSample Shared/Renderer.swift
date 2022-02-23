//
//  Renderer.swift
//  mtlSample Shared
//
//  Created by gzonelee on 2022/02/21.
//

// Our platform independent renderer class

import Metal
import MetalKit
import simd

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<Uniforms>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
    case textureNotFound
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var colorMap: MTLTexture
    var normalMap: MTLTexture
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotation: Float = 0
    
    var meshes: [(MTKMesh, MDLMesh)]
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        
        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        
        guard let buffer = self.device.makeBuffer(length:uniformBufferSize, options:[MTLResourceOptions.storageModeShared]) else { return nil }
        dynamicUniformBuffer = buffer
        
        self.dynamicUniformBuffer.label = "UniformBuffer"
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:Uniforms.self, capacity:1)
        
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float_stencil8
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        metalKitView.sampleCount = 1
        
        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       metalKitView: metalKitView,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }
        
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDescriptor) else { return nil }
        depthState = state
        
        let usdz = "toy_biplane"
//        let usdz = "toy_car"
//        let usdz = "tv_retro"
        do {
            meshes = try Renderer.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor, usdz: usdz)
        } catch {
            GZLogFunc("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        do {
            colorMap = try Renderer.loadTexture(device: device, usdz: usdz, semantic: .baseColor)
        } catch {
            GZLogFunc("Unable to load texture. Error info: \(error)")
            return nil
        }
        do {
            normalMap = try Renderer.loadTexture(device: device, usdz: usdz, semantic: .tangentSpaceNormal)
        } catch {
            GZLogFunc("Unable to load texture. Error info: \(error)")
            return nil
        }
        
        super.init()
        
    }
    
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices
        
        let mtlVertexDescriptor = MTLVertexDescriptor()
        
        mtlVertexDescriptor.attributes[0].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[0].offset = 0
        mtlVertexDescriptor.attributes[0].bufferIndex = 0
        
        mtlVertexDescriptor.attributes[1].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[1].offset = 16
        mtlVertexDescriptor.attributes[1].bufferIndex = 0
        
        mtlVertexDescriptor.attributes[2].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[2].offset = 32
        mtlVertexDescriptor.attributes[2].bufferIndex = 0
        
        mtlVertexDescriptor.layouts[0].stride = 40
        mtlVertexDescriptor.layouts[0].stepRate = 1
        mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunction.perVertex
        

        mtlVertexDescriptor.attributes[3].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[3].offset = 0
        mtlVertexDescriptor.attributes[3].bufferIndex = 1

        mtlVertexDescriptor.layouts[1].stride = 12
        mtlVertexDescriptor.layouts[1].stepRate = 1
        mtlVertexDescriptor.layouts[1].stepFunction = MTLVertexStepFunction.perVertex


        mtlVertexDescriptor.attributes[4].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[4].offset = 0
        mtlVertexDescriptor.attributes[4].bufferIndex = 2

        mtlVertexDescriptor.layouts[2].stride = 12
        mtlVertexDescriptor.layouts[2].stepRate = 1
        mtlVertexDescriptor.layouts[2].stepFunction = MTLVertexStepFunction.perVertex
        
        return mtlVertexDescriptor
    }
    
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             metalKitView: MTKView,
                                             mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object
        
        let library = device.makeDefaultLibrary()
        
        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")
        
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.sampleCount = metalKitView.sampleCount
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalKitView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        pipelineDescriptor.stencilAttachmentPixelFormat = metalKitView.depthStencilPixelFormat
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    static var defaultVertexDescriptor: MDLVertexDescriptor = {
        let vertexDescriptor = MDLVertexDescriptor()
        var offset  = 0
        
        // position attribute
        vertexDescriptor.attributes[VertexAttribute.position.rawValue]
        = MDLVertexAttribute(name: MDLVertexAttributePosition,
                             format: .float3,
                             offset: 0,
                             bufferIndex: 0)
        offset += MemoryLayout<SIMD3<Float>>.stride
        
        // normal attribute
        vertexDescriptor.attributes[VertexAttribute.normal.rawValue] =
        MDLVertexAttribute(name: MDLVertexAttributeNormal,
                           format: .float3,
                           offset: offset,
                           bufferIndex: 0)
        offset += MemoryLayout<SIMD3<Float>>.stride
        
        // uv attribute
        vertexDescriptor.attributes[VertexAttribute.texcoord.rawValue] =
        MDLVertexAttribute(name: MDLVertexAttributeTextureCoordinate,
                           format: .float2,
                           offset: offset,
                           bufferIndex: 0)
        offset += MemoryLayout<SIMD2<Float>>.stride
        
        vertexDescriptor.layouts[0] = MDLVertexBufferLayout(stride: offset)
        return vertexDescriptor
    }()
    
    class func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor, usdz: String) throws -> [(MTKMesh, MDLMesh)] {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
//        let url = Bundle.main.url(forResource: "toy_car", withExtension: "usdz")
//        let url = Bundle.main.url(forResource: "tv_retro", withExtension: "usdz")
        let url = Bundle.main.url(forResource: usdz, withExtension: "usdz")
        GZLogFunc(url)
        GZLogFunc()
//        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
        let mdlVertexDescriptor = self.defaultVertexDescriptor
        
        let asset = MDLAsset(url: url!, vertexDescriptor: mdlVertexDescriptor, bufferAllocator: metalAllocator)
        asset.loadTextures()
        var mtkMeshes = [(MTKMesh, MDLMesh)]()
        if let meshes = asset.childObjects(of: MDLMesh.self) as? [MDLMesh], meshes.count > 0 {
            for mdlMesh in meshes {
                
                mdlMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                        tangentAttributeNamed: MDLVertexAttributeTangent,
                                        bitangentAttributeNamed: MDLVertexAttributeBitangent)
                let vvv = mdlMesh.vertexDescriptor
                if let m = try? MTKMesh(mesh:mdlMesh, device:device) {
                    mtkMeshes.append((m, mdlMesh))
                }
            }
        }
        return mtkMeshes
        
//        let mdlVertexDescriptor = MTKModelIOVertexDescriptorFromMetal(mtlVertexDescriptor)
//
//        guard let attributes = mdlVertexDescriptor.attributes as? [MDLVertexAttribute] else {
//            throw RendererError.badVertexDescriptor
//        }
//        attributes[VertexAttribute.position.rawValue].name = MDLVertexAttributePosition
//        attributes[VertexAttribute.texcoord.rawValue].name = MDLVertexAttributeTextureCoordinate
//
//        mdlMesh.vertexDescriptor = mdlVertexDescriptor
//
//        let m = try MTKMesh(mesh:mdlMesh, device:device)
//        return [m]
    }
    
    class func loadTexture(device: MTLDevice, usdz: String, semantic: MDLMaterialSemantic) throws -> MTLTexture {
        /// Create and condition mesh data to feed into a pipeline using the given vertex descriptor
        
        let metalAllocator = MTKMeshBufferAllocator(device: device)
        
//        let url = Bundle.main.url(forResource: "toy_car", withExtension: "usdz")
//        let url = Bundle.main.url(forResource: "tv_retro", withExtension: "usdz")
        let url = Bundle.main.url(forResource: usdz, withExtension: "usdz")
        
        let asset = MDLAsset(url: url!, vertexDescriptor: nil, bufferAllocator: metalAllocator)
        asset.loadTextures()
        for x in asset.childObjects(of: MDLMesh.self) {
            guard let mesh = x as? MDLMesh else {
                continue
            }
            GZLogFunc(mesh)
            for submseh in mesh.submeshes as! [MDLSubmesh] {
                if let materials = submseh.material {
                    if let t = materials.property(with: semantic)?.textureSamplerValue?.texture {
                        let textureLoaderOptions: [MTKTextureLoader.Option : Any] = [
                            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue),
                            MTKTextureLoader.Option.origin: MTKTextureLoader.Origin.bottomLeft.rawValue
                        ]
                        let textureLoader = MTKTextureLoader(device: device)
                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                            return texture
                        }
                    }
                }
            }
        }
        throw RendererError.textureNotFound
    }
    
    class func loadTexture(device: MTLDevice,
                           textureName: String) throws -> MTLTexture {
        /// Load texture data with optimal parameters for sampling
        
        let textureLoader = MTKTextureLoader(device: device)
        
        let textureLoaderOptions = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue)
        ]
        
        return try textureLoader.newTexture(name: textureName,
                                            scaleFactor: 1.0,
                                            bundle: nil,
                                            options: textureLoaderOptions)
        
    }
    
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering
        
        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:Uniforms.self, capacity:1)
    }
    
    private func updateGameState() {
        /// Update any game state before rendering
        
        uniforms[0].projectionMatrix = projectionMatrix
        
        let rotationAxis = SIMD3<Float>(0, 1, 0)
        let modelMatrix =
        simd_mul(
            simd_mul(
                matrix4x4_rotation(radians: rotation, axis: rotationAxis),
                matrix4x4_translation(0, -2.5, 0)
            ),
            matrix4x4_scale(scale: 1.00))
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -28.0)
        uniforms[0].modelViewMatrix = simd_mul(viewMatrix, modelMatrix)
        rotation += 0.01
    }
    
    func draw(in view: MTKView) {
        /// Per frame updates hare
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            
            let semaphore = inFlightSemaphore
            commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
                semaphore.signal()
            }
            
            self.updateDynamicBufferState()
            
            self.updateGameState()
            
            /// Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
            ///   holding onto the drawable and blocking the display pipeline any longer than necessary
            let renderPassDescriptor = view.currentRenderPassDescriptor
            
            if let renderPassDescriptor = renderPassDescriptor, let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                
                /// Final pass rendering code here
                renderEncoder.label = "Primary Render Encoder"
                
                renderEncoder.pushDebugGroup("Draw Box")
                
                renderEncoder.setCullMode(.back)
                
                renderEncoder.setFrontFacing(.counterClockwise)
//                renderEncoder.setTriangleFillMode(.lines)
                
                renderEncoder.setRenderPipelineState(pipelineState)
                
                renderEncoder.setDepthStencilState(depthState)
                
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                
                for mesh in meshes {
                    for (index, element) in mesh.0.vertexDescriptor.layouts.enumerated() {
                        guard let layout = element as? MDLVertexBufferLayout else {
                            return
                        }
                        
                        if layout.stride != 0 {
                            let buffer = mesh.0.vertexBuffers[index]
                            renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                        }
                    }
                    
                    renderEncoder.setFragmentTexture(colorMap, index: TextureIndex.color.rawValue)
                    renderEncoder.setFragmentTexture(normalMap, index: TextureIndex.normal.rawValue)
                    
                    for submesh in mesh.0.submeshes {
                        renderEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                            indexCount: submesh.indexCount,
                                                            indexType: submesh.indexType,
                                                            indexBuffer: submesh.indexBuffer.buffer,
                                                            indexBufferOffset: submesh.indexBuffer.offset)
                        
                    }
                }
                
                renderEncoder.popDebugGroup()
                
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        /// Respond to drawable size or orientation changes here
        
        let aspect = Float(size.width) / Float(size.height)
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)
    }
}

// Generic matrix math utility functions
func matrix4x4_rotation(radians: Float, axis: SIMD3<Float>) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix4x4_scale(scale: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(scale, 0, 0, 0),
                                         vector_float4(0, scale, 0, 0),
                                         vector_float4(0, 0, scale, 0),
                                         vector_float4(0, 0, 0, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
