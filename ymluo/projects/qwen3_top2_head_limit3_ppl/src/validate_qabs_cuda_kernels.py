from __future__ import annotations

import json

import torch

import qabs_cuda_kernels as kernels


def main() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    query = torch.randn((1, 4, 128), dtype=torch.float16, device=device)
    key = torch.randn((1, 4, 4097, 128), dtype=torch.float16, device=device)
    dim_count = 16
    dim_indices = torch.topk(query.float().abs(), k=dim_count, dim=-1).indices
    gathered_key = torch.gather(
        key.float(),
        -1,
        dim_indices.unsqueeze(2).expand(-1, -1, key.shape[2], -1),
    )
    expected = (gathered_key * torch.gather(query.float(), -1, dim_indices).unsqueeze(2)).sum(-1)
    actual = kernels.partial_scores(query, key, dim_count)
    packed_key, key_scales = kernels.pack_int2(key)
    int2_actual = kernels.partial_scores_int2(
        query,
        packed_key,
        key_scales,
        dim_indices,
        key.shape[2],
    )
    int2_onthefly = kernels.partial_scores_int2_onthefly(query, key, dim_count)
    selected_key = gathered_key
    selected_scale_indices = (dim_indices // 32).unsqueeze(2).expand(
        -1, -1, key.shape[2], -1
    )
    selected_scales = torch.gather(
        key_scales.float(), dim=-1, index=selected_scale_indices
    )
    normalized_key = selected_key / selected_scales
    int2_levels = torch.where(
        normalized_key < -0.7978846,
        torch.full_like(normalized_key, -1.2711063),
        torch.where(
            normalized_key < 0.0,
            torch.full_like(normalized_key, -0.3246628),
            torch.where(
                normalized_key < 0.7978846,
                torch.full_like(normalized_key, 0.3246628),
                torch.full_like(normalized_key, 1.2711063),
            ),
        ),
    )
    int2_expected = (
        int2_levels
        * selected_scales
        * torch.gather(query.float(), -1, dim_indices).unsqueeze(2)
    ).sum(-1)
    candidate_count = 287
    expected_top = torch.topk(expected, k=candidate_count, dim=-1).indices
    actual_top = torch.topk(actual, k=candidate_count, dim=-1).indices
    int2_top = torch.topk(int2_actual, k=candidate_count, dim=-1).indices
    int2_onthefly_top = torch.topk(int2_onthefly, k=candidate_count, dim=-1).indices
    compact_actual = kernels.candidate_compact_scores(
        query,
        key,
        actual_top,
        128**-0.5,
    )
    compact_key = torch.gather(
        key.float(),
        2,
        actual_top.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1]),
    )
    compact_expected = (compact_key * query.float().unsqueeze(2)).sum(-1) * (128**-0.5)
    projection_dim = 64
    projected_query = torch.randn(
        (1, 2, 4, projection_dim), dtype=torch.float16, device=device
    )
    projected_key = torch.randn(
        (1, 2, key.shape[2], projection_dim), dtype=torch.float16, device=device
    )
    projected_scale = (
        projected_key.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 7.0
    ).half()
    projected_codes = (
        torch.round(projected_key.float() / projected_scale.float()).clamp(-7, 7).to(torch.int16)
        + 7
    )
    projected_packed = (
        projected_codes[..., 0::2].to(torch.uint8)
        | (projected_codes[..., 1::2].to(torch.uint8) << 4)
    )
    capacity = key.shape[2] + 11
    projected_packed_capacity = torch.zeros(
        (1, 2, capacity, projection_dim // 2), dtype=torch.uint8, device=device
    )
    projected_scale_capacity = torch.ones(
        (1, 2, capacity, 1), dtype=torch.float16, device=device
    )
    projected_packed_capacity[..., : key.shape[2], :] = projected_packed
    projected_scale_capacity[..., : key.shape[2], :] = projected_scale
    projected_query_scale = (
        projected_query.float().abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8) / 127.0
    )
    projected_query_codes = (
        torch.round(projected_query.float() / projected_query_scale)
        .clamp(-127, 127)
        .to(torch.int8)
    )
    pca_int4_actual = kernels.pca_int4_scores(
        projected_query_codes,
        projected_packed_capacity,
        projected_scale_capacity,
        key.shape[2],
    )
    projected_dequant = (projected_codes.float() - 7.0) * projected_scale.float()
    pca_int4_expected = torch.einsum(
        "bhgm,bhkm->bhgk", projected_query_codes.float(), projected_dequant
    ).reshape(1, 8, key.shape[2])
    int8_key_capacity = torch.randint(
        -127,
        128,
        (1, 2, capacity, projection_dim),
        dtype=torch.int8,
        device=device,
    )
    pca_int8_actual = kernels.pca_int8_scores(
        projected_query_codes, int8_key_capacity, key.shape[2]
    )
    pca_int8_expected = torch.einsum(
        "bhgm,bhkm->bhgk",
        projected_query_codes.float(),
        int8_key_capacity[..., : key.shape[2], :].float(),
    ).round().int()
    padded_query_codes = torch.zeros(
        (1, 2, 16, projection_dim), dtype=torch.int8, device=device
    )
    padded_query_codes[..., : projected_query_codes.shape[2], :].copy_(
        projected_query_codes
    )
    pca_int8_wmma_actual = kernels.pca_int8_wmma_scores(
        padded_query_codes,
        int8_key_capacity,
        projected_scale_capacity,
        key.shape[2],
        projected_query_codes.shape[2],
    )
    pca_int8_wmma_expected = (
        pca_int8_expected.float()
        * projected_scale_capacity[..., : key.shape[2], 0].unsqueeze(2).float()
    ).reshape(1, 8, key.shape[2])
    retrieval_candidate_scores = torch.sort(
        torch.randn((1, 8, 40), device=device), dim=-1, descending=True
    ).values
    retrieval_candidate_indices = torch.randint(
        0, 1000, (1, 8, 40), dtype=torch.long, device=device
    )
    retrieval_previous = retrieval_candidate_indices[..., :32].roll(3, dims=-1)
    retrieval_actual = kernels.retrieval_metrics(
        retrieval_candidate_scores,
        retrieval_candidate_indices,
        retrieval_previous,
        32,
    )
    retrieval_denominator = retrieval_candidate_scores[..., 0].abs().clamp_min(1.0e-6)
    retrieval_expected = torch.stack(
        (
            (retrieval_candidate_scores[..., 0] - retrieval_candidate_scores[..., 1])
            / retrieval_denominator,
            (retrieval_candidate_scores[..., 0] - retrieval_candidate_scores[..., -1])
            / retrieval_denominator,
            (
                retrieval_candidate_indices[..., :32].unsqueeze(-1)
                == retrieval_previous.unsqueeze(-2)
            ).any(dim=-1).float().mean(dim=-1),
        ),
        dim=-1,
    )
    union_indices = torch.randint(
        0, 257, (2, 8, 25), dtype=torch.long, device=device
    )
    union_actual = kernels.candidate_union_counts(union_indices, 257, 4)
    page16_actual = kernels.candidate_bucket_union_counts(union_indices, 257, 4, 16)
    union_expected = torch.empty((2, 2), dtype=torch.int32, device=device)
    page16_expected = torch.empty((2, 2), dtype=torch.int32, device=device)
    for batch_index in range(2):
        for kv_head in range(2):
            union_expected[batch_index, kv_head] = torch.unique(
                union_indices[batch_index, kv_head * 4 : (kv_head + 1) * 4]
            ).numel()
            page16_expected[batch_index, kv_head] = torch.unique(
                union_indices[batch_index, kv_head * 4 : (kv_head + 1) * 4] // 16
            ).numel()
    gqa_key = key[:, :2]
    gqa_value = torch.randn_like(gqa_key)
    gqa_indices = torch.randint(
        0, key.shape[2], (1, 4, 20), dtype=torch.long, device=device
    )
    gqa_compact_actual = kernels.candidate_compact_scores(
        query,
        gqa_key,
        gqa_indices,
        128**-0.5,
    )
    gqa_compact_expected = torch.empty_like(gqa_compact_actual)
    for head in range(query.shape[1]):
        kv_head = head // 2
        selected_key = gqa_key[0, kv_head].index_select(
            0, gqa_indices[0, head]
        ).float()
        gqa_compact_expected[0, head] = (
            selected_key * query[0, head].float().unsqueeze(0)
        ).sum(dim=-1) * (128**-0.5)
    gqa_counts = torch.tensor([[5, 10, 15, 20]], dtype=torch.long, device=device)
    gqa_actual = kernels.final_attention_ragged(
        query,
        gqa_key,
        gqa_value,
        gqa_indices,
        gqa_counts,
        128**-0.5,
    )
    gqa_expected = torch.empty(
        (1, query.shape[1], query.shape[2]), dtype=torch.float32, device=device
    )
    for head in range(query.shape[1]):
        count = int(gqa_counts[0, head].item())
        indices = gqa_indices[0, head, :count]
        kv_head = head // 2
        selected_key = gqa_key[0, kv_head].index_select(0, indices).float()
        selected_value = gqa_value[0, kv_head].index_select(0, indices).float()
        scores = torch.matmul(selected_key, query[0, head].float()) * (128**-0.5)
        gqa_expected[0, head] = torch.matmul(torch.softmax(scores, dim=0), selected_value)
    tail_indices = torch.randint(
        0, key.shape[2], (1, 4, 12), dtype=torch.long, device=device
    )
    tail_scores = torch.randn((1, 4, 12), dtype=torch.float32, device=device)
    tail_scores[0, 1, 9] = -torch.inf
    tail_counts = torch.full((1, 4), 12, dtype=torch.long, device=device)
    tail_prefix_counts = torch.full((1, 4), 5, dtype=torch.long, device=device)
    tail_actual, reliability_actual = kernels.final_attention_tail_reliability(
        gqa_value,
        tail_indices,
        tail_scores,
        tail_counts,
        tail_prefix_counts,
    )
    mass_gate_actual, mass_gate_active_actual = kernels.final_attention_tail_mass_gate(
        gqa_value,
        tail_indices,
        tail_scores,
        tail_counts,
        tail_prefix_counts,
        mass_threshold=0.95,
        tail_shrinkage=0.5,
    )
    split_reference = kernels.final_attention_from_scores_ragged(
        gqa_value, tail_indices, tail_scores, tail_counts
    )
    split_outputs = {
        split: kernels.final_attention_from_scores_split(
            gqa_value, tail_indices, tail_scores, tail_counts, split
        )
        for split in (2, 4, 8, 16)
    }
    tail_expected = torch.empty_like(tail_actual[:, 0].float())
    reliability_expected = torch.empty_like(reliability_actual)
    mass_gate_expected = torch.empty_like(tail_actual[:, 0].float())
    mass_gate_active_expected = torch.empty_like(reliability_actual)
    for head in range(query.shape[1]):
        selected_value = gqa_value[0, head // 2].index_select(
            0, tail_indices[0, head]
        ).float()
        scores = tail_scores[0, head]
        weights = torch.exp(scores - scores[torch.isfinite(scores)].max()).masked_fill(
            ~torch.isfinite(scores), 0.0
        )
        prefix = int(tail_prefix_counts[0, head])
        base_weights = weights[:prefix]
        sample_weights = weights[prefix:]
        sample_values = selected_value[prefix:]
        base_denominator = base_weights.sum()
        base_numerator = torch.einsum(
            "n,nd->d", base_weights, selected_value[:prefix]
        )
        half_outputs = []
        for parity in (0, 1):
            positions = torch.arange(sample_weights.numel(), device=device) % 2 == parity
            valid = positions & torch.isfinite(scores[prefix:])
            multiplier = torch.isfinite(scores[prefix:]).sum() / valid.sum().clamp_min(1)
            half_denominator = base_denominator + multiplier * sample_weights[valid].sum()
            half_numerator = base_numerator + multiplier * torch.einsum(
                "n,nd->d", sample_weights[valid], sample_values[valid]
            )
            half_outputs.append(half_numerator / half_denominator)
        base_output = base_numerator / base_denominator
        delta_even = half_outputs[0] - base_output
        delta_odd = half_outputs[1] - base_output
        reliability_dims = torch.arange(8, device=device) * delta_even.numel() // 8
        delta_even = delta_even.index_select(0, reliability_dims)
        delta_odd = delta_odd.index_select(0, reliability_dims)
        signal = (delta_even + delta_odd).square().sum()
        noise = (delta_even - delta_odd).square().sum()
        reliability = signal / (signal + noise + 1.0e-12)
        reliability_expected[0, head] = reliability
        tail_expected[0, head] = (
            base_numerator
            + reliability
            * torch.einsum("n,nd->d", sample_weights, sample_values)
        ) / (base_denominator + reliability * sample_weights.sum())
        tail_denominator = sample_weights.sum()
        active = base_denominator / (base_denominator + tail_denominator) < 0.95
        mass_gate_active_expected[0, head] = float(active)
        mass_gate_expected[0, head] = (
            base_numerator
            + float(active)
            * 0.5
            * torch.einsum("n,nd->d", sample_weights, sample_values)
        ) / (
            base_denominator + float(active) * 0.5 * tail_denominator
        )
    overlaps = []
    int2_overlaps = []
    int2_onthefly_overlaps = []
    int2_implementation_overlaps = []
    for head in range(query.shape[1]):
        expected_set = set(expected_top[0, head].cpu().tolist())
        actual_set = set(actual_top[0, head].cpu().tolist())
        overlaps.append(len(expected_set & actual_set) / candidate_count)
        int2_set = set(int2_top[0, head].cpu().tolist())
        int2_onthefly_set = set(int2_onthefly_top[0, head].cpu().tolist())
        int2_overlaps.append(len(expected_set & int2_set) / candidate_count)
        int2_onthefly_overlaps.append(
            len(expected_set & int2_onthefly_set) / candidate_count
        )
        int2_implementation_overlaps.append(
            len(int2_set & int2_onthefly_set) / candidate_count
        )
    int2_overlap_by_dimension_count = {}
    for scan_dim_count in (8, 16, 32, 64, 128):
        scan_dim_indices = torch.topk(
            query.float().abs(), k=scan_dim_count, dim=-1
        ).indices
        scan_key = torch.gather(
            key.float(),
            -1,
            scan_dim_indices.unsqueeze(2).expand(-1, -1, key.shape[2], -1),
        )
        scan_expected = (
            scan_key
            * torch.gather(query.float(), -1, scan_dim_indices).unsqueeze(2)
        ).sum(-1)
        scan_int2 = kernels.partial_scores_int2_onthefly(
            query, key, scan_dim_count
        )
        scan_expected_top = torch.topk(
            scan_expected, k=candidate_count, dim=-1
        ).indices
        scan_int2_top = torch.topk(scan_int2, k=candidate_count, dim=-1).indices
        scan_overlaps = []
        for head in range(query.shape[1]):
            expected_set = set(scan_expected_top[0, head].cpu().tolist())
            int2_set = set(scan_int2_top[0, head].cpu().tolist())
            scan_overlaps.append(len(expected_set & int2_set) / candidate_count)
        int2_overlap_by_dimension_count[str(scan_dim_count)] = sum(
            scan_overlaps
        ) / len(scan_overlaps)
    payload = {
        "partial_scores_max_abs_error": float((expected - actual).abs().max().item()),
        "partial_scores_mean_abs_error": float((expected - actual).abs().mean().item()),
        "candidate_topk_overlap_mean": sum(overlaps) / len(overlaps),
        "candidate_topk_overlap_by_head": overlaps,
        "compact_scores_max_abs_error": float((compact_expected - compact_actual).abs().max().item()),
        "compact_scores_mean_abs_error": float((compact_expected - compact_actual).abs().mean().item()),
        "int2_scores_max_abs_error_vs_dequant_reference": float((int2_expected - int2_actual).abs().max().item()),
        "int2_scores_mean_abs_error_vs_dequant_reference": float((int2_expected - int2_actual).abs().mean().item()),
        "int2_candidate_overlap_vs_fp16_mean": sum(int2_overlaps) / len(int2_overlaps),
        "int2_candidate_overlap_vs_fp16_by_head": int2_overlaps,
        "int2_onthefly_candidate_overlap_vs_fp16_mean": sum(int2_onthefly_overlaps)
        / len(int2_onthefly_overlaps),
        "int2_onthefly_candidate_overlap_vs_fp16_by_head": int2_onthefly_overlaps,
        "int2_onthefly_vs_packed_candidate_overlap_mean": sum(
            int2_implementation_overlaps
        )
        / len(int2_implementation_overlaps),
        "int2_onthefly_vs_packed_scores_mean_abs_error": float(
            (int2_onthefly - int2_actual).abs().mean().item()
        ),
        "int2_candidate_overlap_by_qabs_dimension_count": int2_overlap_by_dimension_count,
        "int2_index_fraction_of_full_kv": (
            packed_key.numel() * packed_key.element_size()
            + key_scales.numel() * key_scales.element_size()
        )
        / (2 * key.numel() * key.element_size()),
        "pca_int4_scores_max_abs_error": float(
            (pca_int4_expected - pca_int4_actual).abs().max().item()
        ),
        "pca_int4_scores_mean_abs_error": float(
            (pca_int4_expected - pca_int4_actual).abs().mean().item()
        ),
        "pca_int8_scores_max_abs_error": int(
            (pca_int8_expected - pca_int8_actual).abs().max().item()
        ),
        "pca_int8_wmma_scores_max_abs_error": float(
            (pca_int8_wmma_expected - pca_int8_wmma_actual).abs().max().item()
        ),
        "retrieval_metrics_max_abs_error": float(
            (retrieval_expected - retrieval_actual).abs().max().item()
        ),
        "candidate_union_counts_max_abs_error": int(
            (union_expected - union_actual).abs().max().item()
        ),
        "candidate_page16_counts_max_abs_error": int(
            (page16_expected - page16_actual).abs().max().item()
        ),
        "gqa_compact_scores_max_abs_error": float(
            (gqa_compact_expected - gqa_compact_actual).abs().max().item()
        ),
        "gqa_compact_scores_mean_abs_error": float(
            (gqa_compact_expected - gqa_compact_actual).abs().mean().item()
        ),
        "gqa_ragged_max_abs_error": float(
            (gqa_expected - gqa_actual[:, 0].float()).abs().max().item()
        ),
        "gqa_ragged_mean_abs_error": float(
            (gqa_expected - gqa_actual[:, 0].float()).abs().mean().item()
        ),
        "tail_reliability_output_max_abs_error": float(
            (tail_expected - tail_actual[:, 0].float()).abs().max().item()
        ),
        "tail_reliability_alpha_max_abs_error": float(
            (reliability_expected - reliability_actual).abs().max().item()
        ),
        "tail_mass_gate_output_max_abs_error": float(
            (mass_gate_expected - mass_gate_actual[:, 0].float()).abs().max().item()
        ),
        "tail_mass_gate_active_max_abs_error": float(
            (mass_gate_active_expected - mass_gate_active_actual).abs().max().item()
        ),
        "split_attention_max_abs_error": {
            str(split): float((split_reference - output).float().abs().max().item())
            for split, output in split_outputs.items()
        },
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
