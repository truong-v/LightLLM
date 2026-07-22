import torch
import triton
import triton.language as tl


_FORCE_BALANCED_ROUTING_CACHE = {}


@triton.jit
def _force_balanced_routing_kernel(
    topk_ids_ptr,
    replace_token_num,
    global_rank: tl.constexpr,
    world_size: tl.constexpr,
    top_k: tl.constexpr,
    experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    assignment_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = assignment_offsets < replace_token_num * top_k
    token_offsets = assignment_offsets // top_k
    destination_ranks = (assignment_offsets + global_rank) % world_size
    local_experts = (token_offsets + global_rank) % experts_per_rank
    expert_ids = destination_ranks * experts_per_rank + local_experts
    tl.store(topk_ids_ptr + assignment_offsets, expert_ids, mask=mask)


@triton.jit
def _build_force_balanced_routing_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    assignment_num,
    routing_weight,
    global_rank: tl.constexpr,
    world_size: tl.constexpr,
    top_k: tl.constexpr,
    experts_per_rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    assignment_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = assignment_offsets < assignment_num
    token_offsets = assignment_offsets // top_k
    destination_ranks = (assignment_offsets + global_rank) % world_size
    local_experts = (token_offsets + global_rank) % experts_per_rank
    expert_ids = destination_ranks * experts_per_rank + local_experts
    tl.store(topk_ids_ptr + assignment_offsets, expert_ids, mask=mask)
    tl.store(topk_weights_ptr + assignment_offsets, routing_weight, mask=mask)


def _validate_routing_shape(*, expert_num: int, global_rank: int, world_size: int, top_k: int) -> None:
    assert world_size > 0
    assert 0 <= global_rank < world_size
    assert expert_num > 0
    assert expert_num % world_size == 0
    if not 0 < top_k <= world_size:
        raise ValueError("force-balanced routing requires top_k <= world_size")


def get_cached_force_balanced_routing(
    token_num: int,
    top_k: int,
    *,
    expert_num: int,
    global_rank: int,
    world_size: int,
    routing_weight: float,
    device: torch.device,
):
    """Return immutable full-force routing tensors shared by all MoE layers.

    Capacity is rounded up to a power of two so nearby runtime batch shapes
    share one template. The cache is stream-local: tensors are produced and
    consumed without adding cross-stream synchronization to the hot path.
    """
    if token_num < 0:
        raise ValueError(f"token_num must be non-negative, got {token_num}")
    _validate_routing_shape(
        expert_num=expert_num,
        global_rank=global_rank,
        world_size=world_size,
        top_k=top_k,
    )
    if token_num == 0:
        return (
            torch.empty((0, top_k), dtype=torch.int64, device=device),
            torch.empty((0, top_k), dtype=torch.float32, device=device),
        )

    capacity = max(128, 1 << (token_num - 1).bit_length())
    stream_id = torch.cuda.current_stream(device=device).cuda_stream
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    cache_key = (
        device_index,
        stream_id,
        capacity,
        top_k,
        expert_num,
        global_rank,
        world_size,
        float(routing_weight),
    )
    cached = _FORCE_BALANCED_ROUTING_CACHE.get(cache_key)
    if cached is None:
        topk_ids = torch.empty((capacity, top_k), dtype=torch.int64, device=device)
        topk_weights = torch.empty((capacity, top_k), dtype=torch.float32, device=device)
        assignment_num = capacity * top_k
        block_size = 256
        _build_force_balanced_routing_kernel[(triton.cdiv(assignment_num, block_size),)](
            topk_ids,
            topk_weights,
            assignment_num,
            routing_weight,
            world_size=world_size,
            top_k=top_k,
            experts_per_rank=expert_num // world_size,
            global_rank=global_rank,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        cached = (topk_ids, topk_weights)
        _FORCE_BALANCED_ROUTING_CACHE[cache_key] = cached
    return cached[0][:token_num], cached[1][:token_num]


def clear_force_balanced_routing_cache() -> None:
    """Clear cached templates. Intended for tests only."""
    _FORCE_BALANCED_ROUTING_CACHE.clear()


def force_balanced_routing(
    topk_ids: torch.Tensor,
    ratio: float,
    *,
    expert_num: int,
    global_rank: int,
    world_size: int,
) -> None:
    """Replace a local token prefix with balanced, distinct destination ranks per token row.

    ``top_k`` may be smaller than ``world_size``. Destination ranks rotate by
    source rank and token/slot offset, balancing globally across equal local
    samples while preserving unique destinations within each replaced row.
    """
    assert topk_ids.is_cuda
    assert topk_ids.is_contiguous()
    assert topk_ids.ndim == 2
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}")
    top_k = topk_ids.shape[1]
    _validate_routing_shape(
        expert_num=expert_num,
        global_rank=global_rank,
        world_size=world_size,
        top_k=top_k,
    )
    if ratio == 0.0 or topk_ids.numel() == 0:
        return

    token_num = topk_ids.shape[0]
    replace_token_num = int((global_rank + 1) * token_num * ratio) - int(global_rank * token_num * ratio)
    if replace_token_num == 0:
        return

    block_size = 256
    _force_balanced_routing_kernel[(triton.cdiv(replace_token_num * top_k, block_size),)](
        topk_ids,
        replace_token_num,
        world_size=world_size,
        top_k=top_k,
        experts_per_rank=expert_num // world_size,
        global_rank=global_rank,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
