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

class Mesh {
    var mesh: MTKMesh
    var submeshes: [Submesh] = []
    var meshUniforms: MeshUniforms
    let defaultTransform: float4x4
    let duration: Float
    let keyTransforms: [float4x4]
    let fps: Int
    
    init(mesh: MTKMesh, object: MDLObject, startTime: TimeInterval, endTime: TimeInterval, fps: Int) {
        self.mesh = mesh
        self.fps = fps
        self.meshUniforms = MeshUniforms(modelMatrix: .identity(), normalMatrix: .init())
        duration = Float(endTime - startTime)
        if duration > 0 {
            var timeStride = Array(stride(from: startTime, to: endTime, by: 1 / TimeInterval(fps)))
            timeStride.append(endTime)
            keyTransforms = Array(timeStride).map { time in
                GZLogFunc(time)
                return MDLTransform.globalTransform(with: object, atTime: time)
            }
        }
        else {
            keyTransforms = []
        }
        GZLogFunc(keyTransforms.count)
        self.defaultTransform = MDLTransform.globalTransform(with: object, atTime: 0)
    }
    
    func transform(time: Float) -> float4x4 {
        if keyTransforms.count > 0 {
            let frame = Int(fmod(time, duration) * Float(fps))
            if frame < keyTransforms.count {
                return keyTransforms[frame]
            }
        }
        return defaultTransform
    }
    
}

class Submesh {
    var colorMap: Int?
    var normalMap: Int?
    var roughMap: Int?
    var metalicMap: Int?
    var occlusionMap: Int?
    
    var texturesBuffer: MTLBuffer?
    
    var submesh: MTKSubmesh
    
    init(_ submesh: MTKSubmesh) {
        self.submesh = submesh
    }
    
    func makeTexturesBuffer(device: MTLDevice, fragFunction: MTLFunction,
                            textures: [Int: MTLTexture]
    ) {
        
        let argumentEncoder = fragFunction.makeArgumentEncoder(bufferIndex: BufferIndex.textures.rawValue)
        texturesBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: [])
        guard let b = texturesBuffer else { return }
        b.label = "Textures"
        argumentEncoder.setArgumentBuffer(b, offset: 0)
        if let i = colorMap {
            argumentEncoder.setTexture(textures[i], index: 0)
        }
        if let i = normalMap {
            argumentEncoder.setTexture(textures[i], index: 1)
        }
        if let i = roughMap {
            argumentEncoder.setTexture(textures[i], index: 2)
        }
        if let i = metalicMap {
            argumentEncoder.setTexture(textures[i], index: 3)
        }
        if let i = occlusionMap {
            argumentEncoder.setTexture(textures[i], index: 4)
        }
    }
    
}

class Renderer: NSObject, MTKViewDelegate {
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    
    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)
    
    var uniformBufferOffset = 0
    
    var uniformBufferIndex = 0
    
    var uniforms: UnsafeMutablePointer<Uniforms>
    
    var projectionMatrix: matrix_float4x4 = matrix_float4x4()
    
    var rotation: Float = 0
    
    var meshes: [(Mesh, MDLMesh)] = []
    
    var textures: [Int: MTLTexture] = [:]
    
    var fragFunction: MTLFunction!
    var heap: MTLHeap?
    
    var fps: Int
    var currentTime: Float = 0
    
    init?(metalKitView: MTKView) {
        self.device = metalKitView.device!
        self.fps = metalKitView.preferredFramesPerSecond
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
        
        let library = device.makeDefaultLibrary()
        fragFunction = library?.makeFunction(name: "fragmentShader")
        
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor:depthStateDescriptor) else { return nil }
        depthState = state
        
        super.init()
        
//        let usdz = "toy_biplane"
//        let usdz = "toy_car"
//        let usdz = "toy_drummer"
//        let usdz = "tv_retro"
//        let usdz = "gramophone"
        let usdz = "toy_robot_vintage"
//        let usdz = "LemonMeringuePie"
//        let usdz = "AirForce"
//        let usdz = "PegasusTrail"
//        let usdz = "chair_swan"
//        let usdz = "cup_saucer_set"
//        let usdz = "fender_stratocaster"
//        let usdz = "flower_tulip"
//        let usdz = "wateringcan"
//        let usdz = "SeinfeldSetReplica_usdz"
//        let usdz = "Procedural_Animated_Push_Pin_-_WAVE"
//        let usdz = "phoenix_bird"
//        let usdz = "Medieval_Fantasy_Book"
        do {
            meshes = try self.buildMesh(device: device, mtlVertexDescriptor: mtlVertexDescriptor, usdz: usdz)
        } catch {
            GZLogFunc("Unable to build MetalKit Mesh. Error info: \(error)")
            return nil
        }
        
        heap = buildHeap()
       
        for mesh in meshes {
            for submesh in mesh.0.submeshes {
                submesh.makeTexturesBuffer(device: device, fragFunction: fragFunction, textures: textures)
            }
        }
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
    
    var defaultVertexDescriptor: MDLVertexDescriptor = {
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
    
    func buildMesh(device: MTLDevice,
                         mtlVertexDescriptor: MTLVertexDescriptor, usdz: String) throws -> [(Mesh, MDLMesh)] {
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
        GZLogFunc(asset.startTime)
        GZLogFunc(asset.endTime)
        GZLogFunc()
        asset.loadTextures()
        var mtkMeshes = [(Mesh, MDLMesh)]()
         
        let textureLoader = MTKTextureLoader(device: device)
        let textureLoaderOptions: [MTKTextureLoader.Option : Any] = [
            MTKTextureLoader.Option.textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            MTKTextureLoader.Option.textureStorageMode: NSNumber(value: MTLStorageMode.`private`.rawValue),
            MTKTextureLoader.Option.origin: MTKTextureLoader.Origin.bottomLeft.rawValue
        ]
        
        if let meshes = asset.childObjects(of: MDLMesh.self) as? [MDLMesh], meshes.count > 0 {
            for mdlMesh in meshes {
//                GZLogFunc(mdlMesh.transform?.minimumTime)
//                GZLogFunc(mdlMesh.transform?.localTransform!(atTime:0))
                
                mdlMesh.addTangentBasis(forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                                        tangentAttributeNamed: MDLVertexAttributeTangent,
                                        bitangentAttributeNamed: MDLVertexAttributeBitangent)
                if let m = try? MTKMesh(mesh:mdlMesh, device:device) {
//                    GZLogFunc(m.submeshes.count)
//                    GZLogFunc(mdlMesh.submeshes?.count)
//                    GZLogFunc()
                    let mesh = Mesh(mesh: m, object: mdlMesh, startTime: asset.startTime,
                                    endTime: asset.endTime, fps: fps)
                    if let submeshes = mdlMesh.submeshes as? [MDLSubmesh] {
                        for (index, s) in submeshes.enumerated() {
                            let submesh = Submesh(m.submeshes[index])
                            if let materials = s.material {
                                if let t = materials.property(with: .baseColor)?.textureSamplerValue?.texture {
//                                    GZLogFunc(t.hash)
                                    if let tt = textures[t.hash] {
//                                        GZLogFunc("texture \(t.hash) already exists.")
                                        submesh.colorMap = t.hash
                                    }
                                    else {
                                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                                            textures[t.hash] = texture
                                            submesh.colorMap = t.hash
                                        }
                                    }
                                }
                                if let t = materials.property(with: .tangentSpaceNormal)?.textureSamplerValue?.texture {
                                    if let tt = textures[t.hash] {
//                                        GZLogFunc("texture \(t.name) already exists.")
                                        submesh.normalMap = t.hash
                                    }
                                    else {
                                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                                            textures[t.hash] = texture
                                            submesh.normalMap = t.hash
                                        }
                                    }
                                }
                                if let t = materials.property(with: .roughness)?.textureSamplerValue?.texture {
                                    if let tt = textures[t.hash] {
//                                        GZLogFunc("texture \(t.name) already exists.")
                                        submesh.roughMap = t.hash
                                    }
                                    else {
                                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                                            textures[t.hash] = texture
                                            submesh.roughMap = t.hash
                                        }
                                    }
                                }
                                if let t = materials.property(with: .metallic)?.textureSamplerValue?.texture {
                                    if let tt = textures[t.hash] {
//                                        GZLogFunc("texture \(t.name) already exists.")
                                        submesh.metalicMap = t.hash
                                    }
                                    else {
                                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                                            textures[t.hash] = texture
                                            submesh.metalicMap = t.hash
                                        }
                                    }
                                }
                                if let t = materials.property(with: .ambientOcclusion)?.textureSamplerValue?.texture {
                                    if let tt = textures[t.hash] {
//                                        GZLogFunc("texture \(t.name) already exists.")
                                        submesh.occlusionMap = t.hash
                                    }
                                    else {
                                        if let texture = try? textureLoader.newTexture(texture: t, options: textureLoaderOptions) {
                                            textures[t.hash] = texture
                                            submesh.occlusionMap = t.hash
                                        }
                                    }
                                }
                            }
                            mesh.submeshes.append(submesh)
                        }
                    }
                    mtkMeshes.append((mesh, mdlMesh))
                }
            }
        }
        GZLogFunc(mtkMeshes.count)
        GZLogFunc(mtkMeshes.count)
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
    
    func buildHeap() -> MTLHeap? {
        let d = MTLHeapDescriptor()
        let descriptors: [(Int, MTLTextureDescriptor)] = textures.map { ($0.key, MTLTextureDescriptor.descriptor(from: $0.value)) }
        let sizeAndAligns = descriptors.map { device.heapTextureSizeAndAlign(descriptor: $0.1) }
        d.size = sizeAndAligns.reduce(0) { a, b in
            let n = b.size - (b.size & (b.align - 1)) + b.align
            return a + n
        }
        if d.size == 0 {
            return nil
        }
        guard let h = device.makeHeap(descriptor: d) else {
            fatalError()
        }
        let heapTextures = descriptors.map { descriptor -> (Int, MTLTexture) in
            descriptor.1.storageMode = d.storageMode
            return (descriptor.0, h.makeTexture(descriptor: descriptor.1)!)
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            fatalError()
        }
        heapTextures.forEach { ht in
            let tt = textures[ht.0]!
            var region = MTLRegionMake2D(0, 0, tt.width, tt.height)
            for level in 0..<tt.mipmapLevelCount {
                for slice in 0..<tt.arrayLength {
                    blitEncoder.copy(from: tt, sourceSlice: slice, sourceLevel: level, sourceOrigin: region.origin, sourceSize: region.size, to: ht.1, destinationSlice: slice, destinationLevel: level, destinationOrigin: region.origin)
                }
                region.size.width /= 2
                region.size.height /= 2
            }
        }
        blitEncoder.endEncoding()
        commandBuffer.commit()
        textures = [:]
        heapTextures.forEach { ht in
            textures[ht.0] = ht.1
        }
        return h
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
        
//        GZLogFunc(transform)
        let viewMatrix = matrix4x4_translation(0.0, -11.00, -30.5)
        uniforms[0].viewMatrix = viewMatrix
 
//        rotation += 0.005
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
                
                    if let h = heap {
                        renderEncoder.useHeap(h)
                    }
                renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                renderEncoder.setFragmentBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
                for mesh in meshes {
                    let rotationAxis = SIMD3<Float>(0, 1, 0)
                    let modelMatrix =
                    simd_mul(
                        //            matrix4x4_scale(scale: 0.05),
                        .identity(),
                        simd_mul(
                            matrix4x4_rotation(radians: rotation, axis: rotationAxis),
                            mesh.0.transform(time: currentTime)
                            //                matrix4x4_translation(0, 0, 0)
                        )
                    )
                    mesh.0.meshUniforms = MeshUniforms(modelMatrix: modelMatrix, normalMatrix: modelMatrix.upperLeft)
                    
                    renderEncoder.setVertexBytes(&mesh.0.meshUniforms, length: MemoryLayout<MeshUniforms>.stride, index: BufferIndex.meshUniforms.rawValue)
                
                    for (index, element) in mesh.0.mesh.vertexDescriptor.layouts.enumerated() {
                        guard let layout = element as? MDLVertexBufferLayout else {
                            return
                        }
                        
                        if layout.stride != 0 {
                            let buffer = mesh.0.mesh.vertexBuffers[index]
                            renderEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
                        }
                    }
                    
                    for submesh in mesh.0.submeshes {
                        renderEncoder.setFragmentBuffer(submesh.texturesBuffer, offset: 0, index: BufferIndex.textures.rawValue)
//                        if let t = submesh.colorMap {
//                            renderEncoder.useResource(textures[t]!, usage: .read)
//                        }
//                        if let t = submesh.normalMap {
//                            renderEncoder.useResource(textures[t]!, usage: .read)
//                        }
                        
//                        renderEncoder.setFragmentTexture(submesh.colorMap, index: TextureIndex.color.rawValue)
//                        renderEncoder.setFragmentTexture(submesh.normalMap, index: TextureIndex.normal.rawValue)
                        
                        renderEncoder.drawIndexedPrimitives(type: submesh.submesh.primitiveType,
                                                            indexCount: submesh.submesh.indexCount,
                                                            indexType: submesh.submesh.indexType,
                                                            indexBuffer: submesh.submesh.indexBuffer.buffer,
                                                            indexBufferOffset: submesh.submesh.indexBuffer.offset)
                        
                    }
                }
                
                renderEncoder.popDebugGroup()
                
                renderEncoder.endEncoding()
                
                if let drawable = view.currentDrawable {
                    commandBuffer.present(drawable)
                }
            }
            
            commandBuffer.commit()
            rotation += 0.005
            currentTime += 1 / Float(fps)
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

extension float4x4 {
  // MARK:- Translate
  init(translation: float3) {
    let matrix = float4x4(
      [            1,             0,             0, 0],
      [            0,             1,             0, 0],
      [            0,             0,             1, 0],
      [translation.x, translation.y, translation.z, 1]
    )
    self = matrix
  }
  
  // MARK:- Scale
  init(scaling: float3) {
    let matrix = float4x4(
      [scaling.x,         0,         0, 0],
      [        0, scaling.y,         0, 0],
      [        0,         0, scaling.z, 0],
      [        0,         0,         0, 1]
    )
    self = matrix
  }

  init(scaling: Float) {
    self = matrix_identity_float4x4
    columns.3.w = 1 / scaling
  }
  
  // MARK:- Rotate
  init(rotationX angle: Float) {
    let matrix = float4x4(
      [1,           0,          0, 0],
      [0,  cos(angle), sin(angle), 0],
      [0, -sin(angle), cos(angle), 0],
      [0,           0,          0, 1]
    )
    self = matrix
  }
  
  init(rotationY angle: Float) {
    let matrix = float4x4(
      [cos(angle), 0, -sin(angle), 0],
      [         0, 1,           0, 0],
      [sin(angle), 0,  cos(angle), 0],
      [         0, 0,           0, 1]
    )
    self = matrix
  }
  
  init(rotationZ angle: Float) {
    let matrix = float4x4(
      [ cos(angle), sin(angle), 0, 0],
      [-sin(angle), cos(angle), 0, 0],
      [          0,          0, 1, 0],
      [          0,          0, 0, 1]
    )
    self = matrix
  }
  
  init(rotation angle: float3) {
    let rotationX = float4x4(rotationX: angle.x)
    let rotationY = float4x4(rotationY: angle.y)
    let rotationZ = float4x4(rotationZ: angle.z)
    self = rotationX * rotationY * rotationZ
  }
  
  init(rotationYXZ angle: float3) {
    let rotationX = float4x4(rotationX: angle.x)
    let rotationY = float4x4(rotationY: angle.y)
    let rotationZ = float4x4(rotationZ: angle.z)
    self = rotationY * rotationX * rotationZ
  }
  
  // MARK:- Identity
  static func identity() -> float4x4 {
    matrix_identity_float4x4
  }
  
  // MARK:- Upper left 3x3
  var upperLeft: float3x3 {
    let x = columns.0.xyz
    let y = columns.1.xyz
    let z = columns.2.xyz
    return float3x3(columns: (x, y, z))
  }
  
  // MARK: - Left handed projection matrix
  init(projectionFov fov: Float, near: Float, far: Float, aspect: Float, lhs: Bool = true) {
    let y = 1 / tan(fov * 0.5)
    let x = y / aspect
    let z = lhs ? far / (far - near) : far / (near - far)
    let X = float4( x,  0,  0,  0)
    let Y = float4( 0,  y,  0,  0)
    let Z = lhs ? float4( 0,  0,  z, 1) : float4( 0,  0,  z, -1)
    let W = lhs ? float4( 0,  0,  z * -near,  0) : float4( 0,  0,  z * near,  0)
    self.init()
    columns = (X, Y, Z, W)
  }
  
  // left-handed LookAt
  init(eye: float3, center: float3, up: float3) {
    let z = normalize(center-eye)
    let x = normalize(cross(up, z))
    let y = cross(z, x)
    
    let X = float4(x.x, y.x, z.x, 0)
    let Y = float4(x.y, y.y, z.y, 0)
    let Z = float4(x.z, y.z, z.z, 0)
    let W = float4(-dot(x, eye), -dot(y, eye), -dot(z, eye), 1)
    
    self.init()
    columns = (X, Y, Z, W)
  }
  
  // MARK:- Orthographic matrix
  init(orthoLeft left: Float, right: Float, bottom: Float, top: Float, near: Float, far: Float) {
    let X = float4(2 / (right - left), 0, 0, 0)
    let Y = float4(0, 2 / (top - bottom), 0, 0)
    let Z = float4(0, 0, 1 / (far - near), 0)
    let W = float4((left + right) / (left - right),
                   (top + bottom) / (bottom - top),
                   near / (near - far),
                   1)
    self.init()
    columns = (X, Y, Z, W)
  }
  
  // convert double4x4 to float4x4
  init(_ m: matrix_double4x4) {
    self.init()
    let matrix: float4x4 = float4x4(float4(m.columns.0),
                                    float4(m.columns.1),
                                    float4(m.columns.2),
                                    float4(m.columns.3))
    self = matrix
  }
}

extension float4 {
  var xyz: float3 {
    get {
      float3(x, y, z)
    }
    set {
      x = newValue.x
      y = newValue.y
      z = newValue.z
    }
  }
  
  // convert from double4
  init(_ d: SIMD4<Double>) {
    self.init()
    self = [Float(d.x), Float(d.y), Float(d.z), Float(d.w)]
  }
}


