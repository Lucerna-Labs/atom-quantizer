#![allow(dead_code)]
//! CUDA-backed matvec for the KL discovery instrument (uses cuBLAS SGEMV).
//! Only compiled with the `cuda` feature; the codec stays zero-dep.
//!
//! Design: one `CudaMatvec` per worker thread owns a device + cuBLAS handle and
//! a persistent workspace. `matvec_batched(W, xs)` uploads W once, then loops
//! over the batch of input vectors reusing the on-GPU W. That matches how our
//! KL loop reuses W_orig across k random inputs.

#![allow(clippy::needless_range_loop)]

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

pub struct CudaMatvec {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

impl CudaMatvec {
    /// Initialize CUDA + cuBLAS on device 0. Fails softly if CUDA is unavailable.
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        let stream = ctx.default_stream();
        let blas =
            CudaBlas::new(stream.clone()).map_err(|e| format!("cuBLAS init failed: {e:?}"))?;
        Ok(CudaMatvec { stream, blas })
    }

    /// Compute `y_i = W · x_i` for each x in `xs`, where W is row-major `[out×in]`.
    /// The matrix W is uploaded ONCE and reused across the batch — this is the
    /// whole point (KL currency reuses W across k samples).
    pub fn matvec_batched(
        &self,
        w: &[f32],
        xs: &[Vec<f32>],
        out_len: usize,
        in_len: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        if w.len() != out_len * in_len {
            return Err(format!(
                "bad W shape: {} != {}·{}",
                w.len(),
                out_len,
                in_len
            ));
        }
        let s = &self.stream;
        let d_w = s.memcpy_stod(w).map_err(|e| format!("memcpy W: {e:?}"))?;
        let mut d_x = s
            .alloc_zeros::<f32>(in_len)
            .map_err(|e| format!("alloc x: {e:?}"))?;
        let mut d_y = s
            .alloc_zeros::<f32>(out_len)
            .map_err(|e| format!("alloc y: {e:?}"))?;
        let mut out = Vec::with_capacity(xs.len());
        for x in xs {
            if x.len() != in_len {
                return Err(format!("bad x len: {} != {}", x.len(), in_len));
            }
            s.memcpy_htod(x, &mut d_x)
                .map_err(|e| format!("memcpy x: {e:?}"))?;
            // cuBLAS is column-major. W is row-major [out, in]; treated as
            // column-major that's [in, out]. gemv(op=T, m=in, n=out) gives us
            // W_rowmajor · x correctly.
            let cfg = cudarc::cublas::GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: in_len as i32,
                n: out_len as i32,
                alpha: 1.0,
                lda: in_len as i32,
                incx: 1,
                beta: 0.0,
                incy: 1,
            };
            unsafe {
                self.blas
                    .gemv(cfg, &d_w, &d_x, &mut d_y)
                    .map_err(|e| format!("sgemv: {e:?}"))?;
            }
            let host_y = s
                .memcpy_dtov(&d_y)
                .map_err(|e| format!("memcpy y: {e:?}"))?;
            out.push(host_y);
        }
        s.synchronize().map_err(|e| format!("sync: {e:?}"))?;
        Ok(out)
    }

    /// Batched matvec via SGEMM: `Y[out×k] = W[out×in] · X[in×k]`. Returns a
    /// Vec of `k` `y` vectors, one per column of Y. Much better than k separate
    /// SGEMV calls because CPU/GPU launch overhead amortizes over `k`.
    pub fn matmul_batch(
        &self,
        w: &[f32],
        xs: &[Vec<f32>],
        out_len: usize,
        in_len: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let k = xs.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        if w.len() != out_len * in_len {
            return Err(format!(
                "bad W shape: {} != {}·{}",
                w.len(),
                out_len,
                in_len
            ));
        }
        // Pack X column-major: for k=col, in=row → X_cm[in, k]. Since each x_i
        // is a flat `Vec<f32>` of length in_len, we can lay them out as
        // consecutive columns: x0[0..in], x1[0..in], ..., x_{k-1}[0..in].
        let mut x_flat = Vec::with_capacity(in_len * k);
        for x in xs {
            if x.len() != in_len {
                return Err(format!("bad x len: {} != {}", x.len(), in_len));
            }
            x_flat.extend_from_slice(x);
        }
        let s = &self.stream;
        let d_w = s.memcpy_stod(w).map_err(|e| format!("memcpy W: {e:?}"))?;
        let d_x = s
            .memcpy_stod(&x_flat)
            .map_err(|e| format!("memcpy X: {e:?}"))?;
        let mut d_y: CudaSlice<f32> = s
            .alloc_zeros(out_len * k)
            .map_err(|e| format!("alloc Y: {e:?}"))?;
        // Row-major W [out, in] = column-major W^T [in, out]. To compute
        //   Y_rm [out, k] = W_rm · X_rm  where X_rm columns are the k inputs,
        // treat W_rm as W_cm^T (op=T) and X columns already contiguous.
        // In cuBLAS gemm terms: C_cm[m×n] = alpha · op(A) · op(B) + beta · C_cm.
        //   A = W_cm [in × out], op(A) = T ⇒ [out × in]
        //   B = X_cm [in × k],   op(B) = N ⇒ [in × k]
        //   ⇒ C = [out × k], stored column-major in d_y.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: out_len as i32,
            n: k as i32,
            k: in_len as i32,
            alpha: 1.0f32,
            lda: in_len as i32,
            ldb: in_len as i32,
            beta: 0.0f32,
            ldc: out_len as i32,
        };
        unsafe {
            self.blas
                .gemm(cfg, &d_w, &d_x, &mut d_y)
                .map_err(|e| format!("sgemm: {e:?}"))?;
        }
        let host_y = s
            .memcpy_dtov(&d_y)
            .map_err(|e| format!("memcpy Y: {e:?}"))?;
        s.synchronize().map_err(|e| format!("sync: {e:?}"))?;
        // Split into k column vectors of length out_len (Y stored column-major).
        let mut out = Vec::with_capacity(k);
        for col in 0..k {
            let start = col * out_len;
            out.push(host_y[start..start + out_len].to_vec());
        }
        Ok(out)
    }
}
