import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block, # [BLOCK_SIZE_Q, HEAD_DIM]
    l_i, # [BLOCK_SIZE_Q]
    m_i, # [BLOCK_SIZE_Q]
    Q_block, # [BLOCK_SIZE_Q, HEAD_DIM]
    K_block_ptr,
    V_block_ptr,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    offs_q: tl.constexpr, # [BLOCK_SIZE_Q]
    offs_kv: tl.constexpr, # [BLOCK_SIZE_KV]
    SEQ_LEN: tl.constexpr,
):



@triton.jit
def _attn_fwd(
    Q, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    K, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    V, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    softmax_scale, # float32
    M, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    O, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_head_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_head_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_head_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_head_dim,
    BATCH_SIZE, 
    NUM_HEADS: tl.constexpr, 
    SEQ_LEN: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # this indicates which block in the sequence length we are processing
    block_index_q = tl.program_id(0)

    # this indicates which head and batch to process. Each program is associated with a single head and batch
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS

    # this indicates which head to process
    index_head = index_batch_head % NUM_HEADS

    # offset to the correct position in the batch, head and sequence length
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch +
        index_head.to(tl.int64) * stride_Q_head      
    ) # this will take us to Q [BATCH_SIZE, NUM_HEADS, :, :]

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, # this will take us to Q [BATCH_SIZE, NUM_HEADS, SEQ_LEN, :]
        shape=(SEQ_LEN, HEAD_DIM), # resulting shape
        strides=(stride_Q_seq, stride_Q_head_dim), # strides to move in the resulting shape
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # offset to the correct block
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), # resulting sub division that we are processing
        order=(1,0), # optimize
    )

    K_block_ptr = tl.make_block_ptr (
        base=K + qvk_offset, # this will take us to K [BATCH_SIZE, NUM_HEADS, SEQ_LEN, :]
        shape=(SEQ_LEN, HEAD_DIM), # resulting shape
        strides=(stride_K_head_dim, stride_K_seq), # strides to move in the resulting shape
        offsets=(0, 0), # offset to the correct block
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), # resulting sub division that we are processing
        order=(0,1), # optimize
    )

    V_block_ptr = tl.make_block_ptr (
        base=V + qvk_offset, # this will take us to V [BATCH_SIZE, NUM_HEADS, :, :]
        shape=(SEQ_LEN, HEAD_DIM), # resulting shape
        strides=(stride_V_seq, stride_V_head_dim), # strides to move in the resulting shape
        offsets=(0, 0), # offset to the correct block
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM), # resulting sub division that we are processing
        order=(1,0), # optimize
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32 - 'inf')
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr)




    
class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, scale):
        
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        grid = lambda args: (
             triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
             BATCH_SIZE * NUM_HEADS,
             1,             
        )

        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN),
            device=Q.device,
            dtype=torch.float32
        )

        _attn_fwd [grid] (
            Q=Q,
            K=K,
            V=V,
            softmax_scale=scale,
            M=M,
            O=O,
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_head_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_head_dim = K.stride(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_head_dim = V.stride(3),
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_head_dim = O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.scale = scale
        ctx.HRAD_DIM = HEAD_DIM
        ctx.causal = causal

        return O



def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    
    # creating Q, K and V tensors
    Q = (
        torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=torch.device('cuda:0')
        ).normal_(mean=0.0, std=0.05)
        .requires_grad_()
    )
    K = (
        torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=torch.device('cuda:0')
        ).normal_(mean=0.0, std=0.05)
        .requires_grad_()
    )
    V = (
        torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=torch.device('cuda:0')
        ).normal_(mean=0.0, std=0.05)
        .requires_grad_()
    )

    scale = 1 / (HEAD_DIM**0.5) # 1 / sqrt(HEAD_DIM)
    d0 = torch.randn_like(Q) # we will see later

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=Q.device))
    P = torch.matmul(Q, K.transpose(2, 3)) * scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()

    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0)
    ref_dQ, Q_grad = Q.grad.clone(), None
    ref_dK, K_grad = K.grad.clone(), None
    ref_dV, V_grad = V.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, scale, d0)
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0 # two values are different by at most | a - b | <= rtol * | a | => | a - b | <= 0
    atol = 1e-2 # two values are different by at most 1e-2

    assert torch.allclose(ref_0, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)

