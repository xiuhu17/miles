import torch
import torch.distributed as dist
from torch.nn import functional as F
try:
    import einops

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False

def eager_attn_bwd(q, kv, attn_bias, sinks, scale, dim_short, dropout, attn_output, probs, grad_output):

    # Rearrange query, key, value to (b, h, s, d)
    b, sq, h, d = q.shape
    _, sk, _, _ = kv.shape
    k = kv
    v = kv[:,:,:,:dim_short]
    q_tail = q[:,:,:,dim_short:]
    _q_tail_T = einops.rearrange(q_tail, 'b s h d -> b h d s').contiguous()
    _q_T = einops.rearrange(q, 'b s h d -> b h d s')
    _k_T = einops.rearrange(k, 'b s h d -> b h s d')
    _v_T = einops.rearrange(v, 'b s h d -> b h d s')

    # Backward pass for score @ value
    if sinks is None:
        attn_w = probs[..., :-1]  # Drop the sink
    grad_output = einops.rearrange(grad_output, 'b s h d -> b h s d')
    attn_w_T = einops.rearrange(attn_w, ' b h sq sk -> b h sk sq')
    grad__v = torch.matmul(attn_w_T, grad_output).contiguous() # b h sk d
    grad_attn_w = torch.matmul(grad_output, _v_T).contiguous() # b h s d  || b h d sk -> b h s sk

    # Backward pass for softmax
    if sinks is None:
        grad_probs = grad_attn_w

    # Backward pass for q @ K^T
    grad_attn_w *= scale
    grad__q = torch.matmul(grad_attn_w, _k_T).contiguous()
    grad__k = torch.matmul(_q_T, grad_attn_w).contiguous() # b h d sk
    grad__k_T = grad__k.transpose(2, 3).contiguous() # b h sk d
    grad__kv = torch.zeros((b, h, sk, d), device=q.device, dtype=q.dtype) # b h sk d
    grad__kv[:,:,:,:dim_short] = grad__v + grad__k_T[:,:,:,:dim_short]
    grad__kv[:,:,:,dim_short:] = torch.matmul(_q_tail_T, grad_attn_w).contiguous().transpose(2, 3).contiguous() # b h sk d

    # Rearrange grads to (b, s, h, d)
    grad__kv = grad__kv.transpose(1, 2).contiguous()
    grad_q = einops.rearrange(grad__q, 'b h s d -> b s h d')
    return grad_q, grad__kv, grad_sinks
