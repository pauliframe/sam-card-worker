"""Runtime monkeypatch: chunked scaled_dot_product_attention for Apple MPS.

MPS's math SDPA materializes the full [batch*heads, Lq, Lk] score matrix in ONE
buffer. For SAM 3.1's 16-object multiplex at 1008px that is ~25 GiB — larger than
MPS's max single allocation, and even when it fits it sits at the memory-watermark
ceiling, so every allocation triggers `release_cached_buffers()` (evict-all + resync)
and the tracker crawls (~100x slowdown) without ever OOMing.

Fix: tile SDPA over the (collapsed) batch dimension so no single score buffer is
large. Exact math — no approximation. Each caller's dtype is preserved (fp16 trunk,
fp32 decoder). attn_mask is broadcast to the query's batch shape and sliced per chunk.
Install once before running the model; CUDA tensors fall through to the original op.
"""
import torch
import torch.nn.functional as F

_ORIG_SDPA = F.scaled_dot_product_attention
_BYTES_BUDGET = 256 * 1024 * 1024  # max score-matrix bytes per chunk


def _chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, **kw):
    if query.is_cuda:
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kw)

    lead = query.shape[:-2]
    Lq, Dq = query.shape[-2], query.shape[-1]
    Lk, Dv = key.shape[-2], value.shape[-1]
    B = 1
    for d in lead:
        B *= int(d)

    q = query.reshape(B, Lq, Dq)
    k = key.reshape(B, Lk, key.shape[-1])
    v = value.reshape(B, Lk, Dv)

    # Broadcast a batched mask to the query's leading dims, then collapse to [B, Lq, Lk]
    # so chunk slices stay aligned with q. Low-dim masks broadcast on their own.
    m = attn_mask
    m_batched = m is not None and m.dim() >= 3
    if m_batched:
        try:
            m = m.expand(*lead, m.shape[-2], m.shape[-1]).reshape(B, m.shape[-2], m.shape[-1])
        except RuntimeError:
            m_batched = False  # non-standard shape; fall back to passing it whole
            m = attn_mask

    per = Lq * Lk * max(2, query.element_size())
    chunk = max(1, min(B, _BYTES_BUDGET // max(1, per)))

    if chunk >= B and not m_batched:
        out = _ORIG_SDPA(q, k, v, attn_mask=m, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale)
    else:
        outs = []
        for i in range(0, B, chunk):
            mm = m[i:i + chunk] if m_batched else (attn_mask if attn_mask is not None else None)
            outs.append(_ORIG_SDPA(q[i:i + chunk], k[i:i + chunk], v[i:i + chunk],
                                   attn_mask=mm, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale))
        out = torch.cat(outs, dim=0)

    return out.reshape(*lead, Lq, Dv)


def install():
    """Replace F.scaled_dot_product_attention with the chunked variant (idempotent)."""
    if getattr(F.scaled_dot_product_attention, "_sam3_chunked", False):
        return
    _chunked_sdpa._sam3_chunked = True
    F.scaled_dot_product_attention = _chunked_sdpa
    torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa


_ORIG_CUDA = torch.Tensor.cuda


def install_cuda_redirect(device):
    """Redirect Tensor.cuda() to `device` (e.g. 'mps'). SAM3's video tracker has
    hardcoded .cuda() calls to reload CPU-offloaded memory features 'back to GPU';
    off-CUDA we route them to the compute device instead. Idempotent."""
    if getattr(torch.Tensor.cuda, "_sam3_redirect", False):
        return

    def _cuda(self, *a, **kw):
        return self.to(device)

    _cuda._sam3_redirect = True
    torch.Tensor.cuda = _cuda
