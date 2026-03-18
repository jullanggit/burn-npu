import CoreML
import Foundation

// ── MLTensor FFI for Rust — all ops go to ANE/GPU/CPU via Apple's dispatch ──

final class TensorBox {
    let tensor: MLTensor
    init(_ t: MLTensor) { self.tensor = t }
}

// Thread-safe handle table
private let lock = NSLock()
private var handles: [Int32: TensorBox] = [:]
private var nextId: Int32 = 1

private func store(_ t: MLTensor) -> Int32 {
    lock.lock()
    let id = nextId; nextId += 1
    handles[id] = TensorBox(t)
    lock.unlock()
    return id
}

private func get(_ id: Int32) -> MLTensor? {
    lock.lock()
    let t = handles[id]?.tensor
    lock.unlock()
    return t
}

// ── Create / Free / Query ──

@_cdecl("npu_create_tensor")
public func npuCreate(shapePtr: UnsafePointer<Int32>, shapeDims: Int32, dataPtr: UnsafePointer<Float>, dataLen: Int32) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let data = Array(UnsafeBufferPointer(start: dataPtr, count: Int(dataLen)))
    return store(MLTensor(shape: shape, scalars: data, scalarType: Float.self))
}

@_cdecl("npu_create_int_tensor")
public func npuCreateInt(shapePtr: UnsafePointer<Int32>, shapeDims: Int32, dataPtr: UnsafePointer<Int32>, dataLen: Int32) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let data = Array(UnsafeBufferPointer(start: dataPtr, count: Int(dataLen)))
    return store(MLTensor(shape: shape, scalars: data, scalarType: Int32.self))
}

@_cdecl("npu_create_bool_tensor")
public func npuCreateBool(shapePtr: UnsafePointer<Int32>, shapeDims: Int32, dataPtr: UnsafePointer<UInt8>, dataLen: Int32) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let bools = (0..<Int(dataLen)).map { dataPtr[$0] != 0 }
    // Store as Int32 (0/1) since MLTensor doesn't have bool type
    let ints = bools.map { Int32($0 ? 1 : 0) }
    return store(MLTensor(shape: shape, scalars: ints, scalarType: Int32.self))
}

@_cdecl("npu_free_tensor")
public func npuFree(id: Int32) { lock.lock(); handles.removeValue(forKey: id); lock.unlock() }

@_cdecl("npu_get_shape")
public func npuGetShape(id: Int32, outPtr: UnsafeMutablePointer<Int32>, maxDims: Int32) -> Int32 {
    guard let t = get(id) else { return -1 }
    let s = t.shape
    for i in 0..<min(s.count, Int(maxDims)) { outPtr[i] = Int32(s[i]) }
    return Int32(s.count)
}

@_cdecl("npu_get_data")
public func npuGetData(id: Int32, outPtr: UnsafeMutablePointer<Float>, maxLen: Int32) -> Int32 {
    guard let t = get(id) else { return -1 }
    let sem = DispatchSemaphore(value: 0)
    var count: Int32 = -1
    let tensor = t
    let capturedMax = Int(maxLen)
    Thread.detachNewThread {
        let r = RunLoop.current
        nonisolated(unsafe) var done = false
        Task {
            let arr = await tensor.shapedArray(of: Float.self)
            let flat = arr.scalars
            let n = min(flat.count, capturedMax)
            for i in 0..<n { outPtr[i] = flat[i] }
            count = Int32(n)
            done = true
            sem.signal()
        }
        while !done { r.run(mode: .default, before: Date(timeIntervalSinceNow: 0.0001)) }
    }
    sem.wait()
    return count
}

@_cdecl("npu_get_int_data")
public func npuGetIntData(id: Int32, outPtr: UnsafeMutablePointer<Int32>, maxLen: Int32) -> Int32 {
    guard let t = get(id) else { return -1 }
    let sem = DispatchSemaphore(value: 0)
    var count: Int32 = -1
    let tensor = t
    let capturedMax = Int(maxLen)
    Thread.detachNewThread {
        let r = RunLoop.current
        nonisolated(unsafe) var done = false
        Task {
            let arr = await tensor.shapedArray(of: Int32.self)
            let flat = arr.scalars
            let n = min(flat.count, capturedMax)
            for i in 0..<n { outPtr[i] = flat[i] }
            count = Int32(n)
            done = true
            sem.signal()
        }
        while !done { r.run(mode: .default, before: Date(timeIntervalSinceNow: 0.0001)) }
    }
    sem.wait()
    return count
}

@_cdecl("npu_scalar_tensor")
public func npuScalar(value: Float) -> Int32 {
    store(MLTensor(shape: [], scalars: [value], scalarType: Float.self))
}

@_cdecl("npu_zeros")
public func npuZeros(shapePtr: UnsafePointer<Int32>, shapeDims: Int32) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let n = shape.reduce(1, *)
    return store(MLTensor(shape: shape, scalars: [Float](repeating: 0, count: n), scalarType: Float.self))
}

@_cdecl("npu_ones")
public func npuOnes(shapePtr: UnsafePointer<Int32>, shapeDims: Int32) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let n = shape.reduce(1, *)
    return store(MLTensor(shape: shape, scalars: [Float](repeating: 1, count: n), scalarType: Float.self))
}

@_cdecl("npu_full")
public func npuFull(shapePtr: UnsafePointer<Int32>, shapeDims: Int32, value: Float) -> Int32 {
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    let n = shape.reduce(1, *)
    return store(MLTensor(shape: shape, scalars: [Float](repeating: value, count: n), scalarType: Float.self))
}

// ── Binary arithmetic ──

@_cdecl("npu_add")
public func npuAdd(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    return store(ta + tb)
}
@_cdecl("npu_sub")
public func npuSub(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    return store(ta - tb)
}
@_cdecl("npu_mul")
public func npuMul(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    return store(ta * tb)
}
@_cdecl("npu_div")
public func npuDiv(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    return store(ta / tb)
}

// ── Scalar arithmetic ──

@_cdecl("npu_add_scalar")
public func npuAddScalar(a: Int32, s: Float) -> Int32 {
    guard let ta = get(a) else { return -1 }
    return store(ta + s)
}
@_cdecl("npu_sub_scalar")
public func npuSubScalar(a: Int32, s: Float) -> Int32 {
    guard let ta = get(a) else { return -1 }
    return store(ta - s)
}
@_cdecl("npu_mul_scalar")
public func npuMulScalar(a: Int32, s: Float) -> Int32 {
    guard let ta = get(a) else { return -1 }
    return store(ta * s)
}
@_cdecl("npu_div_scalar")
public func npuDivScalar(a: Int32, s: Float) -> Int32 {
    guard let ta = get(a) else { return -1 }
    return store(ta / s)
}
@_cdecl("npu_neg")
public func npuNeg(a: Int32) -> Int32 {
    guard let ta = get(a) else { return -1 }
    return store(-ta)
}

// ── Matmul ──

@_cdecl("npu_matmul")
public func npuMatmul(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    return store(ta.matmul(tb))
}

// ── Unary math ──

@_cdecl("npu_exp")
public func npuExp(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.exp())
}
@_cdecl("npu_log")
public func npuLog(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.log())
}
@_cdecl("npu_sqrt")
public func npuSqrt(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.squareRoot())
}
@_cdecl("npu_abs")
public func npuAbs(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    // MLTensor abs: max(x, -x)
    return store((t * t).squareRoot())
}
@_cdecl("npu_tanh")
public func npuTanh(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.tanh())
}
@_cdecl("npu_sin")
public func npuSin(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.sin())
}
@_cdecl("npu_cos")
public func npuCos(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.cos())
}
@_cdecl("npu_floor")
public func npuFloor(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.floor())
}
@_cdecl("npu_ceil")
public func npuCeil(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.ceil())
}
@_cdecl("npu_round")
public func npuRound(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.floor() + Float(0.5))
}

// ── Power ──

@_cdecl("npu_pow")
public func npuPow(x: Int32, y: Int32) -> Int32 {
    guard let tx = get(x), let ty = get(y) else { return -1 }
    return store(tx.pow(ty))
}
@_cdecl("npu_pow_scalar")
public func npuPowScalar(x: Int32, p: Float) -> Int32 {
    guard let tx = get(x) else { return -1 }
    return store(tx.pow(p))
}

// ── Clamp ──

@_cdecl("npu_clamp_min")
public func npuClampMin(x: Int32, minVal: Float) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.clamped(to: minVal...))
}
@_cdecl("npu_clamp_max")
public func npuClampMax(x: Int32, maxVal: Float) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.clamped(to: ...maxVal))
}
@_cdecl("npu_clamp")
public func npuClamp(x: Int32, minVal: Float, maxVal: Float) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.clamped(to: minVal...maxVal))
}

// ── Reduction ──

@_cdecl("npu_sum")
public func npuSum(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.sum())
}
@_cdecl("npu_sum_dim")
public func npuSumDim(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.sum(alongAxes: Int(dim)).expandingShape(at: Int(dim)))
}
@_cdecl("npu_mean")
public func npuMean(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    // keepdim: expand shape back after reduce
    return store(t.mean(alongAxes: Int(dim)).expandingShape(at: Int(dim)))
}
@_cdecl("npu_mean_all")
public func npuMeanAll(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.mean())
}
@_cdecl("npu_max")
public func npuMax(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.max())
}
@_cdecl("npu_max_dim")
public func npuMaxDim(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.max(alongAxes: Int(dim)).expandingShape(at: Int(dim)))
}
@_cdecl("npu_min")
public func npuMin(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.min())
}
@_cdecl("npu_min_dim")
public func npuMinDim(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.min(alongAxes: Int(dim)).expandingShape(at: Int(dim)))
}
@_cdecl("npu_argmax")
public func npuArgmax(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.argmax(alongAxis: Int(dim)))
}
@_cdecl("npu_argmin")
public func npuArgmin(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.argmin(alongAxis: Int(dim)))
}

// ── Softmax ──

@_cdecl("npu_softmax")
public func npuSoftmax(x: Int32, dim: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.softmax(alongAxis: Int(dim)))
}

// ── Shape ops ──

@_cdecl("npu_reshape")
public func npuReshape(x: Int32, shapePtr: UnsafePointer<Int32>, shapeDims: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    return store(t.reshaped(to: shape))
}

@_cdecl("npu_transpose")
public func npuTranspose(x: Int32, dim0: Int32, dim1: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let ndim = t.shape.count
    var perm = Array(0..<ndim)
    var d0 = Int(dim0); if d0 < 0 { d0 += ndim }
    var d1 = Int(dim1); if d1 < 0 { d1 += ndim }
    perm[d0] = d1; perm[d1] = d0
    return store(t.transposed(permutation: perm))
}

@_cdecl("npu_permute")
public func npuPermute(x: Int32, permPtr: UnsafePointer<Int32>, permLen: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let perm = (0..<Int(permLen)).map { Int(permPtr[$0]) }
    return store(t.transposed(permutation: perm))
}

@_cdecl("npu_narrow")
public func npuNarrow(x: Int32, dim: Int32, start: Int32, length: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let s = Int(start), l = Int(length)
    if dim == -1 || dim == 1 {
        return store(t[0..., s..<(s+l)])
    } else if dim == 0 {
        return store(t[s..<(s+l)])
    }
    return -1
}

@_cdecl("npu_cat")
public func npuCat(idsPtr: UnsafePointer<Int32>, count: Int32, dim: Int32) -> Int32 {
    var tensors: [MLTensor] = []
    for i in 0..<Int(count) {
        guard let t = get(idsPtr[i]) else { return -1 }
        tensors.append(t)
    }
    return store(MLTensor(concatenating: tensors, alongAxis: Int(dim)))
}

@_cdecl("npu_expand")
public func npuExpand(x: Int32, shapePtr: UnsafePointer<Int32>, shapeDims: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let shape = (0..<Int(shapeDims)).map { Int(shapePtr[$0]) }
    // broadcast by multiplying with ones of target shape
    let n = shape.reduce(1, *)
    let ones = MLTensor(shape: shape, scalars: [Float](repeating: 1, count: n), scalarType: Float.self)
    return store(t * ones)
}

// ── Indexing ──

@_cdecl("npu_index_select")
public func npuIndexSelect(x: Int32, indicesPtr: UnsafePointer<Int32>, indicesLen: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    let indices = Array(UnsafeBufferPointer(start: indicesPtr, count: Int(indicesLen)))
    let idxTensor = MLTensor(shape: [Int(indicesLen)], scalars: indices, scalarType: Int32.self)
    return store(t.gathering(atIndices: idxTensor, alongAxis: 0))
}

@_cdecl("npu_gather")
public func npuGather(x: Int32, dim: Int32, indices: Int32) -> Int32 {
    guard let t = get(x), let idx = get(indices) else { return -1 }
    return store(t.gathering(atIndices: idx, alongAxis: Int(dim)))
}

// ── Comparison (return Int32 tensor: 1=true, 0=false) ──

@_cdecl("npu_equal")
public func npuEqual(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    let zeros = MLTensor(shape: ta.shape, scalars: [Float](repeating: 0, count: ta.shape.reduce(1, *)), scalarType: Float.self)
    return store(zeros.replacing(with: Float(1), where: ta .== tb))
}

@_cdecl("npu_greater")
public func npuGreater(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    let zeros = MLTensor(shape: ta.shape, scalars: [Float](repeating: 0, count: ta.shape.reduce(1, *)), scalarType: Float.self)
    return store(zeros.replacing(with: Float(1), where: ta .> tb))
}

@_cdecl("npu_less")
public func npuLess(a: Int32, b: Int32) -> Int32 {
    guard let ta = get(a), let tb = get(b) else { return -1 }
    let zeros = MLTensor(shape: ta.shape, scalars: [Float](repeating: 0, count: ta.shape.reduce(1, *)), scalarType: Float.self)
    return store(zeros.replacing(with: Float(1), where: ta .< tb))
}

// ── Mask ──

@_cdecl("npu_mask_fill")
public func npuMaskFill(x: Int32, mask: Int32, value: Float) -> Int32 {
    guard let t = get(x), let m = get(mask) else { return -1 }
    let cond = m .> Float(0.5)
    return store(t.replacing(with: value, where: cond))
}

@_cdecl("npu_mask_where")
public func npuMaskWhere(x: Int32, mask: Int32, source: Int32) -> Int32 {
    guard let t = get(x), let m = get(mask), let s = get(source) else { return -1 }
    let cond = m .> Float(0.5)
    return store(t.replacing(with: s, where: cond))
}

// ── Slice (multi-range) ──

@_cdecl("npu_slice")
public func npuSlice(x: Int32, rangesPtr: UnsafePointer<Int32>, numRanges: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    // ranges: [start0, end0, start1, end1, ...]
    let n = Int(numRanges)
    // For 2D: t[s0..<e0, s1..<e1]
    if n == 1 {
        let s = Int(rangesPtr[0]), e = Int(rangesPtr[1])
        return store(t[s..<e])
    } else if n == 2 {
        let s0 = Int(rangesPtr[0]), e0 = Int(rangesPtr[1])
        let s1 = Int(rangesPtr[2]), e1 = Int(rangesPtr[3])
        return store(t[s0..<e0, s1..<e1])
    } else if n == 3 {
        let s0 = Int(rangesPtr[0]), e0 = Int(rangesPtr[1])
        let s1 = Int(rangesPtr[2]), e1 = Int(rangesPtr[3])
        let s2 = Int(rangesPtr[4]), e2 = Int(rangesPtr[5])
        return store(t[s0..<e0, s1..<e1, s2..<e2])
    }
    return -1
}

// ── Copy (clone handle) ──

@_cdecl("npu_clone")
public func npuClone(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    // MLTensor is value-type under the hood, storing creates a new reference
    return store(t)
}

// ── Cast ──

@_cdecl("npu_cast_to_int")
public func npuCastToInt(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.cast(to: Int32.self))
}

@_cdecl("npu_cast_to_float")
public func npuCastToFloat(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    return store(t.cast(to: Float.self))
}

// ── Erf (using Horner approximation) ──

@_cdecl("npu_erf")
public func npuErf(x: Int32) -> Int32 {
    guard let t = get(x) else { return -1 }
    // Abramowitz & Stegun approximation: erf(x) = sign(x) * (1 - y)
    let a1: Float = 0.254829592
    let a2: Float = -0.284496736
    let a3: Float = 1.421413741
    let a4: Float = -1.453152027
    let a5: Float = 1.061405429
    let p: Float = 0.3275911
    // sign: +1 where >= 0, -1 where < 0
    let ones = t * Float(0) + Float(1)  // all ones with same shape
    let sign = ones.replacing(with: Float(-1), where: t .< Float(0))
    // abs via sqrt(x*x)
    let absX = (t * t).squareRoot()
    let tt = Float(1) / (Float(1) + p * absX)
    let y = Float(1) - (((((a5 * tt + a4) * tt) + a3) * tt + a2) * tt + a1) * tt * (-absX * absX).exp()
    return store(sign * y)
}
