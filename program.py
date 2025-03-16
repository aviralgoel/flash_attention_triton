import torch

import triton
import triton.language as tl

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

    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=Q.device))
    P = torch.matmul(Q, K.transpose(2, 3)) * scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()
    