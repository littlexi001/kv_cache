from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import _get_build_directory, load_inline


CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor qabs_partial_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count);

torch::Tensor qabs_partial_scores_dim_major_forward(
    torch::Tensor query,
    torch::Tensor key_dim_major,
    torch::Tensor dim_indices);

torch::Tensor qabs_candidate_full_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor current_candidate,
    torch::Tensor previous_candidate,
    bool has_previous_candidate,
    torch::Tensor previous_final,
    bool has_previous_final,
    int64_t protect_sink_tokens,
    int64_t protect_recent_tokens,
    double scaling);

torch::Tensor qabs_candidate_compact_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    double scaling);

torch::Tensor qabs_candidate_prerope_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor phase_cosine,
    torch::Tensor phase_sine,
    int64_t query_position,
    double scaling);

torch::Tensor qabs_candidate_compact_scores_ragged_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    double scaling);

torch::Tensor qabs_candidate_compact_scores_range_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor start_counts,
    torch::Tensor end_counts,
    torch::Tensor output,
    double scaling);

std::vector<torch::Tensor> qabs_proxy_affine_calibrated_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor proxy_scores,
    torch::Tensor candidate_counts,
    int64_t sample_count,
    double scaling);

torch::Tensor qabs_candidate_compact_scores_masked_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_valid,
    double scaling);

std::vector<torch::Tensor> qabs_uncertainty_band_mask_forward(
    torch::Tensor candidate_scores,
    torch::Tensor error_sigma,
    int64_t final_count,
    double confidence_width);

std::vector<torch::Tensor> qabs_direct_uncertainty_candidates_forward(
    torch::Tensor scores,
    torch::Tensor error_sigma,
    int64_t final_count,
    int64_t candidate_capacity,
    double confidence_width);

std::vector<torch::Tensor> qabs_sampled_quantile_candidates_forward(
    torch::Tensor scores,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity);

torch::Tensor qabs_sample_error_sigma_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor approximate_scores,
    int64_t sample_count,
    int64_t sample_offset,
    double scaling);

torch::Tensor qabs_final_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor valid,
    double scaling);

torch::Tensor qabs_final_attention_ragged_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling);

torch::Tensor qabs_final_attention_ragged_self_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling);

torch::Tensor qabs_final_attention_ragged_self_warp_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling);

torch::Tensor qabs_final_attention_ragged_self_split_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling,
    int64_t split_count);

torch::Tensor qabs_final_attention_from_scores_ragged_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    double value_mass_threshold);

torch::Tensor qabs_final_attention_from_scores_ragged_self_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor self_scores,
    double value_mass_threshold);

torch::Tensor qabs_final_attention_from_scores_split_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    int64_t split_count);

std::vector<torch::Tensor> qabs_final_attention_tail_reliability_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor prefix_counts);

std::vector<torch::Tensor> qabs_final_attention_tail_mass_gate_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor prefix_counts,
    double mass_threshold,
    double tail_shrinkage);

torch::Tensor qabs_mass_ladder_forward(
    torch::Tensor top_scores,
    torch::Tensor sample_scores,
    torch::Tensor sample_candidate_scores,
    torch::Tensor self_scores,
    torch::Tensor keep_counts,
    int64_t history_count,
    double mass_threshold);

std::vector<torch::Tensor> qabs_pack_int2_forward(torch::Tensor key);

torch::Tensor qabs_partial_scores_int2_forward(
    torch::Tensor query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    torch::Tensor dim_indices,
    int64_t key_count);

torch::Tensor qabs_partial_scores_int2_onthefly_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count);

torch::Tensor qabs_pca_int4_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    int64_t key_count);

torch::Tensor qabs_pca_int4_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    int64_t key_count,
    int64_t prefix_dim);

torch::Tensor qabs_pca_int4_chunked_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t key_count,
    int64_t prefix_dim);

torch::Tensor qabs_pca_int4_chunked_group16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor group_scales,
    int64_t key_count,
    int64_t prefix_dim);

torch::Tensor qabs_pca_int4_chunked_logscale16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t prefix_dim);

torch::Tensor qabs_pca_nested_int2_logscale16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_high2,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t prefix_dim);

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores);

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_bound_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor chunk_squared_norms,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores,
    bool collect_statistics);

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward(
    torch::Tensor query,
    torch::Tensor basis,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores);

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_streaming_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor basis,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    double scaling,
    bool use_dp4a);

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_exact_forward(
    torch::Tensor projected_query,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    double scaling);

torch::Tensor qabs_pca_int4_logscale16_pack_into_forward(
    torch::Tensor projected_key,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t start_token);

torch::Tensor qabs_pca_int4_logscale16_chunk_norms_into_forward(
    torch::Tensor packed_key_chunked,
    torch::Tensor chunk_squared_norms,
    int64_t start_token,
    int64_t token_count);

std::vector<torch::Tensor> qabs_pca_project_query_int8_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis);

torch::Tensor qabs_pre_rope_lowfreq_int4_pack_into_forward(
    torch::Tensor post_rope_key,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t start_token,
    double rope_theta);

torch::Tensor qabs_pre_rope_lowfreq_int4_scores_forward(
    torch::Tensor post_rope_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t key_count,
    int64_t query_position,
    double rope_theta);

torch::Tensor qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward(
    torch::Tensor post_rope_key,
    torch::Tensor packed_key_chunked,
    int64_t start_token,
    double rope_theta,
    double clip_alpha);

torch::Tensor qabs_pre_rope_lowfreq_int2_fixed_scores_forward(
    torch::Tensor post_rope_query,
    torch::Tensor packed_key_chunked,
    int64_t key_count,
    int64_t query_position,
    double rope_theta);

torch::Tensor qabs_pca_int4_chunked_selected_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    int64_t key_count);

torch::Tensor qabs_pca_int4_chunked_shared_selected_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    int64_t key_count);

torch::Tensor qabs_pca_int4_chunked_shared_selected_add_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    torch::Tensor query_scales,
    torch::Tensor score_cache,
    int64_t key_count);

torch::Tensor qabs_pca_int4_chunked_contiguous_add_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor query_scales,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim);

torch::Tensor qabs_pca_int4_chunked_contiguous_delta_add_forward(
    torch::Tensor projected_query,
    torch::Tensor previous_projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim,
    int64_t selected_count);

torch::Tensor qabs_pca_int4_chunked_band_error_feedback_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count);

torch::Tensor qabs_pca_int4_chunked_band_error_feedback_masked_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor active_mask,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count);

torch::Tensor qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor active_mask,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count);

torch::Tensor qabs_one_shot_band_plan_forward(
    torch::Tensor projected_query,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor top_values,
    torch::Tensor keep_counts,
    torch::Tensor planned_bands,
    torch::Tensor crossing_risk,
    int64_t total_token_count,
    double target_recall);

torch::Tensor qabs_pca_int4_chunked_spectral_gated_delta_add_forward(
    torch::Tensor projected_query,
    torch::Tensor previous_projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor refresh_mask,
    torch::Tensor gate_signal,
    torch::Tensor refresh_indices,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim,
    double threshold,
    int64_t refresh_count);

torch::Tensor qabs_microblock_expected_max_scores_forward(
    torch::Tensor block_mean,
    torch::Tensor block_variance,
    torch::Tensor projected_query,
    int64_t block_count,
    int64_t block_size,
    int64_t last_block_size);

torch::Tensor qabs_microblock_q8_expected_max_scores_forward(
    torch::Tensor block_mean_q8,
    torch::Tensor block_mean_scales,
    torch::Tensor block_variance_q8,
    torch::Tensor block_variance_scales,
    torch::Tensor projected_query,
    int64_t block_count,
    int64_t block_size,
    int64_t last_block_size);

torch::Tensor qabs_pca_int4_logscale16_selected_block_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor query_scales,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor selected_blocks,
    int64_t key_count,
    int64_t block_size,
    int64_t start_dim,
    int64_t end_dim);

torch::Tensor qabs_microblock_local_to_token_indices_forward(
    torch::Tensor selected_blocks,
    torch::Tensor local_indices,
    int64_t key_count,
    int64_t block_size);

torch::Tensor qabs_pca_int4_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim);

torch::Tensor qabs_pca_int4_chunked_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim);

torch::Tensor qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim);

torch::Tensor qabs_pca_int8_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor quantized_key,
    int64_t key_count);

torch::Tensor qabs_pca_int8_wmma_scores_forward(
    torch::Tensor padded_query,
    torch::Tensor quantized_key,
    torch::Tensor scales,
    int64_t key_count,
    int64_t group_count);

torch::Tensor qabs_retrieval_metrics_forward(
    torch::Tensor candidate_scores,
    torch::Tensor candidate_indices,
    torch::Tensor previous_probe,
    int64_t probe_count);

torch::Tensor qabs_quota_merge_candidates_forward(
    torch::Tensor base_indices,
    torch::Tensor rescue_indices);

std::vector<torch::Tensor> qabs_append_rescue_candidates_forward(
    torch::Tensor base_indices,
    torch::Tensor rescue_indices,
    int64_t history_count);

torch::Tensor qabs_candidate_union_counts_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count);

std::vector<torch::Tensor> qabs_candidate_union_compact_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count,
    int64_t output_capacity);

torch::Tensor qabs_candidate_bucket_union_counts_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count,
    int64_t bucket_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qabs_partial_scores_forward", &qabs_partial_scores_forward, "QABS partial candidate scores");
  m.def("qabs_partial_scores_dim_major_forward", &qabs_partial_scores_dim_major_forward, "QABS partial candidate scores on dim-major K");
  m.def("qabs_candidate_full_scores_forward", &qabs_candidate_full_scores_forward, "QABS candidate full-QK scores");
  m.def("qabs_candidate_compact_scores_forward", &qabs_candidate_compact_scores_forward, "QABS compact candidate full-QK scores");
  m.def("qabs_candidate_prerope_scores_forward", &qabs_candidate_prerope_scores_forward, "QABS exact pre-RoPE scores for compact post-RoPE candidates");
  m.def("qabs_candidate_compact_scores_ragged_forward", &qabs_candidate_compact_scores_ragged_forward, "QABS ragged compact candidate full-QK scores");
  m.def("qabs_candidate_compact_scores_range_forward", &qabs_candidate_compact_scores_range_forward, "QABS incremental compact candidate full-QK scores");
  m.def("qabs_proxy_affine_calibrated_scores_forward", &qabs_proxy_affine_calibrated_scores_forward, "QABS sampled exact-QK affine calibration of candidate proxy scores");
  m.def("qabs_candidate_compact_scores_masked_forward", &qabs_candidate_compact_scores_masked_forward, "QABS masked compact candidate full-QK scores");
  m.def("qabs_uncertainty_band_mask_forward", &qabs_uncertainty_band_mask_forward, "QABS histogram uncertainty-band mask");
  m.def("qabs_direct_uncertainty_candidates_forward", &qabs_direct_uncertainty_candidates_forward, "QABS direct uncertainty-band candidates");
  m.def("qabs_sampled_quantile_candidates_forward", &qabs_sampled_quantile_candidates_forward, "QABS sampled-quantile compact candidates");
  m.def("qabs_sample_error_sigma_forward", &qabs_sample_error_sigma_forward, "QABS fused sampled proxy error sigma");
  m.def("qabs_final_attention_forward", &qabs_final_attention_forward, "QABS final sparse attention forward");
  m.def("qabs_final_attention_ragged_forward", &qabs_final_attention_ragged_forward, "QABS ragged final sparse attention forward");
  m.def("qabs_final_attention_ragged_self_forward", &qabs_final_attention_ragged_self_forward, "QABS fused exact-QK ragged attention with an implicit self token");
  m.def("qabs_final_attention_ragged_self_warp_forward", &qabs_final_attention_ragged_self_warp_forward, "QABS warp-cooperative exact-QK ragged attention with an implicit self token");
  m.def("qabs_final_attention_ragged_self_split_forward", &qabs_final_attention_ragged_self_split_forward, "QABS split-parallel exact-QK ragged attention with an implicit self token");
  m.def("qabs_final_attention_from_scores_ragged_forward", &qabs_final_attention_from_scores_ragged_forward, "QABS ragged sparse attention from precomputed scores");
  m.def("qabs_final_attention_from_scores_ragged_self_forward", &qabs_final_attention_from_scores_ragged_self_forward, "QABS ragged sparse attention with an implicit self token");
  m.def("qabs_final_attention_from_scores_split_forward", &qabs_final_attention_from_scores_split_forward, "QABS split-parallel sparse attention from precomputed scores");
  m.def("qabs_final_attention_tail_reliability_forward", &qabs_final_attention_tail_reliability_forward, "QABS split-sample reliability-shrunk tail attention");
  m.def("qabs_final_attention_tail_mass_gate_forward", &qabs_final_attention_tail_mass_gate_forward, "QABS mass-gated shrinkage tail attention");
  m.def("qabs_mass_ladder_forward", &qabs_mass_ladder_forward, "QABS sampled-tail mass ladder");
  m.def("qabs_pack_int2_forward", &qabs_pack_int2_forward, "QABS pack 2-bit key index");
  m.def("qabs_partial_scores_int2_forward", &qabs_partial_scores_int2_forward, "QABS partial scores on 2-bit key index");
  m.def("qabs_partial_scores_int2_onthefly_forward", &qabs_partial_scores_int2_onthefly_forward, "QABS on-the-fly 2-bit partial scores");
  m.def("qabs_pca_int4_scores_forward", &qabs_pca_int4_scores_forward, "PCA INT4 packed index scores");
  m.def("qabs_pca_int4_prefix_scores_forward", &qabs_pca_int4_prefix_scores_forward, "PCA INT4 prefix scores");
  m.def("qabs_pca_int4_chunked_prefix_scores_forward", &qabs_pca_int4_chunked_prefix_scores_forward, "PCA INT4 chunk-major prefix scores");
  m.def("qabs_pca_int4_chunked_group16_prefix_scores_forward", &qabs_pca_int4_chunked_group16_prefix_scores_forward, "PCA INT4 chunk-major group-16 prefix scores");
  m.def("qabs_pca_int4_chunked_logscale16_prefix_scores_forward", &qabs_pca_int4_chunked_logscale16_prefix_scores_forward, "PCA INT4 chunk-major compact log-scale group-16 prefix scores");
  m.def("qabs_pca_nested_int2_logscale16_prefix_scores_forward", &qabs_pca_nested_int2_logscale16_prefix_scores_forward, "Nested high-2-bit PCA scores with compact log scales");
  m.def("qabs_pca_int4_logscale16_sampled_quantile_forward", &qabs_pca_int4_logscale16_sampled_quantile_forward, "Fused PCA INT4 sampled-quantile candidate compaction");
  m.def("qabs_pca_int4_logscale16_sampled_quantile_bound_forward", &qabs_pca_int4_logscale16_sampled_quantile_bound_forward, "Strict Cauchy-bounded progressive PCA INT4 candidate compaction");
  m.def("qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward", &qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward, "Fused raw-query PCA projection, sampled threshold, and candidate compaction");
  m.def("qabs_pca_int4_logscale16_streaming_attention_forward", &qabs_pca_int4_logscale16_streaming_attention_forward, "Single-block PCA INT4 sampled-quantile scan and exact sparse attention");
  m.def("qabs_pca_int4_logscale16_sampled_quantile_exact_forward", &qabs_pca_int4_logscale16_sampled_quantile_exact_forward, "Fused PCA INT4 sampled-quantile scan with exact-QK candidate consumption");
  m.def("qabs_pca_int4_logscale16_pack_into_forward", &qabs_pca_int4_logscale16_pack_into_forward, "Pack PCA keys into a compact log-scale INT4 index in place");
  m.def("qabs_pca_int4_logscale16_chunk_norms_into_forward", &qabs_pca_int4_logscale16_chunk_norms_into_forward, "Store exact per-chunk squared INT4-code norms in place");
  m.def("qabs_pca_project_query_int8_forward", &qabs_pca_project_query_int8_forward, "Fused PCA query projection and symmetric INT8 quantization");
  m.def("qabs_pre_rope_lowfreq_int4_pack_into_forward", &qabs_pre_rope_lowfreq_int4_pack_into_forward, "Pack normalized pre-RoPE low-frequency keys into an INT4 index");
  m.def("qabs_pre_rope_lowfreq_int4_scores_forward", &qabs_pre_rope_lowfreq_int4_scores_forward, "Score a pre-RoPE low-frequency INT4 rescue index");
  m.def("qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward", &qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward, "Pack normalized pre-RoPE low-frequency keys into a scale-free INT2 index");
  m.def("qabs_pre_rope_lowfreq_int2_fixed_scores_forward", &qabs_pre_rope_lowfreq_int2_fixed_scores_forward, "Score a scale-free pre-RoPE low-frequency INT2 index");
  m.def("qabs_pca_int4_chunked_selected_scores_forward", &qabs_pca_int4_chunked_selected_scores_forward, "PCA INT4 arbitrary selected-dimension scores");
  m.def("qabs_pca_int4_chunked_shared_selected_scores_forward", &qabs_pca_int4_chunked_shared_selected_scores_forward, "PCA INT4 GQA-shared selected-dimension scores");
  m.def("qabs_pca_int4_chunked_shared_selected_add_forward", &qabs_pca_int4_chunked_shared_selected_add_forward, "PCA INT4 GQA-shared selected-dimension in-place score update");
  m.def("qabs_pca_int4_chunked_contiguous_add_forward", &qabs_pca_int4_chunked_contiguous_add_forward, "PCA INT4 contiguous-dimension in-place score update");
  m.def("qabs_pca_int4_chunked_contiguous_delta_add_forward", &qabs_pca_int4_chunked_contiguous_delta_add_forward, "PCA INT4 direct contiguous-delta in-place score update");
  m.def("qabs_pca_int4_chunked_band_error_feedback_forward", &qabs_pca_int4_chunked_band_error_feedback_forward, "PCA INT4 covariance-weighted band error-feedback score update");
  m.def("qabs_pca_int4_chunked_band_error_feedback_masked_forward", &qabs_pca_int4_chunked_band_error_feedback_masked_forward, "PCA INT4 masked covariance-weighted band error-feedback score update");
  m.def("qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward", &qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward, "PCA INT4 compact log-scale masked covariance band error-feedback score update");
  m.def("qabs_one_shot_band_plan_forward", &qabs_one_shot_band_plan_forward, "One-shot covariance band-count planner");
  m.def("qabs_pca_int4_chunked_spectral_gated_delta_add_forward", &qabs_pca_int4_chunked_spectral_gated_delta_add_forward, "PCA INT4 spectrally gated full-or-delta score update");
  m.def("qabs_pca_int4_candidate_range_scores_forward", &qabs_pca_int4_candidate_range_scores_forward, "PCA INT4 compact candidate range scores");
  m.def("qabs_pca_int4_chunked_candidate_range_scores_forward", &qabs_pca_int4_chunked_candidate_range_scores_forward, "PCA INT4 chunk-major candidate range scores");
  m.def("qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward", &qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward, "PCA INT4 compact log-scale candidate range scores");
  m.def("qabs_microblock_expected_max_scores_forward", &qabs_microblock_expected_max_scores_forward, "Fused QK-metric microblock expected-maximum scores");
  m.def("qabs_microblock_q8_expected_max_scores_forward", &qabs_microblock_q8_expected_max_scores_forward, "Fused quantized QK-metric microblock expected-maximum scores");
  m.def("qabs_pca_int4_logscale16_selected_block_scores_forward", &qabs_pca_int4_logscale16_selected_block_scores_forward, "Score tokens inside selected microblocks with compact log-scale PCA INT4");
  m.def("qabs_microblock_local_to_token_indices_forward", &qabs_microblock_local_to_token_indices_forward, "Map selected-block local offsets to token indices");
  m.def("qabs_pca_int8_scores_forward", &qabs_pca_int8_scores_forward, "PCA INT8 batched index scores");
  m.def("qabs_pca_int8_wmma_scores_forward", &qabs_pca_int8_wmma_scores_forward, "PCA INT8 WMMA index scores");
  m.def("qabs_retrieval_metrics_forward", &qabs_retrieval_metrics_forward, "Fused retrieval margin and stability metrics");
  m.def("qabs_quota_merge_candidates_forward", &qabs_quota_merge_candidates_forward, "Merge sorted base and rescue candidates under a fixed quota");
  m.def("qabs_append_rescue_candidates_forward", &qabs_append_rescue_candidates_forward, "Append nonduplicate rescue candidates and return a validity mask");
  m.def("qabs_candidate_union_counts_forward", &qabs_candidate_union_counts_forward, "GQA candidate union counts");
  m.def("qabs_candidate_union_compact_forward", &qabs_candidate_union_compact_forward, "GQA candidate compact union");
  m.def("qabs_candidate_bucket_union_counts_forward", &qabs_candidate_bucket_union_counts_forward, "GQA candidate page/bucket union counts");
}
"""


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <math_constants.h>

constexpr int QABS_MAX_DIMS = 128;
constexpr int QABS_MAX_RUNGS = 16;
constexpr int QABS_TILE_THREADS = 256;

__global__ void qabs_retrieval_metrics_kernel(
    const float* __restrict__ candidate_scores,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ previous_probe,
    float* __restrict__ output,
    int candidate_count,
    int probe_count,
    bool has_previous) {
  int row = blockIdx.x;
  int lane = threadIdx.x;
  int row_offset = row * candidate_count;
  int matched = 0;
  if (lane < probe_count) {
    if (!has_previous) {
      matched = 1;
    } else {
      int64_t current = candidate_indices[row_offset + lane];
      int previous_offset = row * probe_count;
      for (int index = 0; index < probe_count; ++index) {
        matched |= current == previous_probe[previous_offset + index];
      }
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    matched += __shfl_down_sync(0xffffffff, matched, offset);
  }
  if (lane == 0) {
    float top = candidate_scores[row_offset];
    float denominator = fmaxf(fabsf(top), 1.0e-6f);
    float second = candidate_scores[row_offset + min(1, candidate_count - 1)];
    float boundary = candidate_scores[row_offset + candidate_count - 1];
    output[row * 3] = (top - second) / denominator;
    output[row * 3 + 1] = (top - boundary) / denominator;
    output[row * 3 + 2] = static_cast<float>(matched) / probe_count;
  }
}

__global__ void qabs_quota_merge_candidates_kernel(
    const int64_t* __restrict__ base_indices,
    const int64_t* __restrict__ rescue_indices,
    int64_t* __restrict__ output,
    int base_count,
    int rescue_count) {
  int row = blockIdx.x;
  int base_offset = row * base_count;
  int rescue_offset = row * rescue_count;
  constexpr int warp_count = 8;
  __shared__ int warp_offsets[warp_count];
  __shared__ int write_base;
  __shared__ int tile_count;
  for (int rescue = threadIdx.x; rescue < rescue_count; rescue += blockDim.x) {
    output[base_offset + rescue] = rescue_indices[rescue_offset + rescue];
  }
  if (threadIdx.x == 0) {
    write_base = rescue_count;
  }
  __syncthreads();

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  for (int tile = 0; tile < base_count; tile += blockDim.x) {
    int base = tile + threadIdx.x;
    int64_t candidate = base < base_count ? base_indices[base_offset + base] : -1;
    bool duplicate = false;
    if (base < base_count) {
      for (int rescue = 0; rescue < rescue_count; ++rescue) {
        duplicate |= candidate == rescue_indices[rescue_offset + rescue];
      }
    }
    bool keep = base < base_count && !duplicate;
    unsigned int keep_mask = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) {
      warp_offsets[warp] = __popc(keep_mask);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int prefix = 0;
      for (int current_warp = 0; current_warp < warp_count; ++current_warp) {
        int count = warp_offsets[current_warp];
        warp_offsets[current_warp] = prefix;
        prefix += count;
      }
      tile_count = prefix;
    }
    __syncthreads();
    if (keep) {
      unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      int rank = warp_offsets[warp] + __popc(keep_mask & lower_lanes);
      int output_position = write_base + rank;
      if (output_position < base_count) {
        output[base_offset + output_position] = candidate;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      write_base += tile_count;
    }
    __syncthreads();
  }
}

__global__ void qabs_append_rescue_candidates_kernel(
    const int64_t* __restrict__ base_indices,
    const int64_t* __restrict__ rescue_indices,
    int64_t* __restrict__ output_indices,
    bool* __restrict__ output_valid,
    int base_count,
    int rescue_count,
    int history_count,
    int word_count) {
  extern __shared__ unsigned int membership[];
  int row = blockIdx.x;
  int base_offset = row * base_count;
  int rescue_offset = row * rescue_count;
  int output_offset = row * (base_count + rescue_count);
  for (int word = threadIdx.x; word < word_count; word += blockDim.x) {
    membership[word] = 0u;
  }
  __syncthreads();
  for (int base = threadIdx.x; base < base_count; base += blockDim.x) {
    int64_t candidate = base_indices[base_offset + base];
    if (candidate >= 0 && candidate < history_count) {
      atomicOr(&membership[candidate >> 5], 1u << (candidate & 31));
    }
    output_indices[output_offset + base] = candidate;
    output_valid[output_offset + base] = true;
  }
  __syncthreads();
  for (int rescue = threadIdx.x; rescue < rescue_count; rescue += blockDim.x) {
    int64_t candidate = rescue_indices[rescue_offset + rescue];
    bool valid = candidate >= 0 && candidate < history_count;
    if (valid) {
      valid = (membership[candidate >> 5] & (1u << (candidate & 31))) == 0u;
    }
    output_indices[output_offset + base_count + rescue] = candidate;
    output_valid[output_offset + base_count + rescue] = valid;
  }
}

__global__ void qabs_candidate_union_counts_kernel(
    const int64_t* __restrict__ candidate_indices,
    int32_t* __restrict__ bitset,
    int32_t* __restrict__ counts,
    int total_indices,
    int query_head_count,
    int selected_count,
    int kv_head_count,
    int group_count,
    int word_count,
    int bucket_size) {
  for (int flat = blockIdx.x * blockDim.x + threadIdx.x;
       flat < total_indices;
       flat += blockDim.x * gridDim.x) {
    int selected = flat % selected_count;
    int query_row = flat / selected_count;
    int query_head = query_row % query_head_count;
    int batch = query_row / query_head_count;
    int kv_head = query_head / group_count;
    int64_t bucket = candidate_indices[flat] / bucket_size;
    int word = static_cast<int>(bucket >> 5);
    unsigned int mask = 1u << static_cast<int>(bucket & 31);
    int bitset_offset = (batch * kv_head_count + kv_head) * word_count + word;
    unsigned int old = atomicOr(
        reinterpret_cast<unsigned int*>(bitset + bitset_offset), mask);
    if ((old & mask) == 0) {
      atomicAdd(counts + batch * kv_head_count + kv_head, 1);
    }
  }
}

__global__ void qabs_candidate_union_compact_kernel(
    const int32_t* __restrict__ bitset,
    int32_t* __restrict__ output,
    int32_t* __restrict__ write_offsets,
    int row_count,
    int history_count,
    int word_count,
    int output_capacity) {
  int row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  for (int token = threadIdx.x; token < history_count; token += blockDim.x) {
    int word = token >> 5;
    unsigned int mask = 1u << (token & 31);
    unsigned int bits = static_cast<unsigned int>(bitset[row * word_count + word]);
    if ((bits & mask) != 0) {
      int slot = atomicAdd(write_offsets + row, 1);
      if (slot < output_capacity) {
        output[row * output_capacity + slot] = token;
      }
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int8_wmma_scores_kernel(
    const int8_t* __restrict__ padded_query,
    const int8_t* __restrict__ quantized_key,
    const scale_t* __restrict__ scales,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim) {
  using namespace nvcuda;
  __shared__ int accumulator_tile[16 * 16];
  int row = blockIdx.x;
  int token_start = blockIdx.y * 16;
  wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      signed char,
      wmma::row_major>
      key_fragment;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      signed char,
      wmma::col_major>
      query_fragment;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> accumulator_fragment;
  wmma::fill_fragment(accumulator_fragment, 0);
  const int8_t* key_base = quantized_key
      + (row * capacity + token_start) * projection_dim;
  const int8_t* query_base = padded_query + row * 16 * projection_dim;
  for (int offset = 0; offset < projection_dim; offset += 16) {
    wmma::load_matrix_sync(
        key_fragment, key_base + offset, projection_dim);
    wmma::load_matrix_sync(
        query_fragment, query_base + offset, projection_dim);
    wmma::mma_sync(
        accumulator_fragment,
        key_fragment,
        query_fragment,
        accumulator_fragment);
  }
  wmma::store_matrix_sync(
      accumulator_tile,
      accumulator_fragment,
      16,
      wmma::mem_row_major);
  __syncwarp();
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int index = threadIdx.x; index < 16 * group_count; index += blockDim.x) {
    int local_token = index / group_count;
    int group = index - local_token * group_count;
    int token = token_start + local_token;
    if (token < key_count) {
      int query_head = kv_head * group_count + group;
      float scale = static_cast<float>(scales[row * capacity + token]);
      output[(batch * query_head_count + query_head) * key_count + token]
          = static_cast<float>(accumulator_tile[local_token * 16 + group]) * scale;
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key,
    const scale_t* __restrict__ scales,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int packed_dim,
    int start_dim,
    int end_dim) {
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int tokens_per_warp = 4;
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  int tile_start = blockIdx.y * warps_per_block * tokens_per_warp;
  for (int iteration = 0; iteration < tokens_per_warp; ++iteration) {
    int token = tile_start + iteration * warps_per_block + warp;
    if (token >= key_count) {
      continue;
    }
    float scale = static_cast<float>(scales[row * capacity + token]);
    int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int quad_count = (end_dim - start_dim) / 4;
    for (int quad = lane; quad < quad_count; quad += 32) {
      const uint8_t* key_row = packed_key
          + (row * capacity + token) * packed_dim + start_dim / 2 + 2 * quad;
      uint8_t first = key_row[0];
      uint8_t second = key_row[1];
      uint32_t packed_key_values =
          static_cast<uint8_t>((first & 0x0F) - 7)
          | (static_cast<uint32_t>(static_cast<uint8_t>((first >> 4) - 7)) << 8)
          | (static_cast<uint32_t>(static_cast<uint8_t>((second & 0x0F) - 7)) << 16)
          | (static_cast<uint32_t>(static_cast<uint8_t>((second >> 4) - 7)) << 24);
      for (int group = 0; group < group_count; ++group) {
        const int8_t* query_row = shared_query
            + group * projection_dim + start_dim + 4 * quad;
        int packed_query_values = *reinterpret_cast<const int*>(query_row);
        accumulators[group] = __dp4a(
            packed_query_values,
            static_cast<int>(packed_key_values),
            accumulators[group]);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      for (int group = 0; group < group_count; ++group) {
        accumulators[group] += __shfl_down_sync(
            0xffffffff, accumulators[group], offset);
      }
    }
    if (lane == 0) {
      int batch = row / kv_head_count;
      int kv_head = row - batch * kv_head_count;
      int query_head_count = kv_head_count * group_count;
      for (int group = 0; group < group_count; ++group) {
        int query_head = kv_head * group_count + group;
        output[(batch * query_head_count + query_head) * key_count + token]
            = static_cast<float>(accumulators[group]) * scale;
      }
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_prefix_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int prefix_dim) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int prefix_chunk_count = prefix_dim / 16;
  for (int chunk = 0; chunk < prefix_chunk_count; ++chunk) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      int low = static_cast<int>(packed & 0x0F) - 7;
      int high = static_cast<int>(packed >> 4) - 7;
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const int8_t* query_row = shared_query + group * projection_dim;
          accumulators[group] += static_cast<int>(query_row[dimension]) * low;
          accumulators[group] += static_cast<int>(query_row[dimension + 1]) * high;
        }
      }
    }
  }

  float scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token]
        = static_cast<float>(accumulators[group]) * scale;
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_group16_prefix_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ group_scales,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int prefix_dim) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int prefix_chunk_count = prefix_dim / 16;
  for (int chunk = 0; chunk < prefix_chunk_count; ++chunk) {
    int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      int low = static_cast<int>(packed & 0x0F) - 7;
      int high = static_cast<int>(packed >> 4) - 7;
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const int8_t* query_row = shared_query + group * projection_dim;
          accumulators[group] += static_cast<int>(query_row[dimension]) * low;
          accumulators[group] += static_cast<int>(query_row[dimension + 1]) * high;
        }
      }
    }
    float scale = static_cast<float>(
        group_scales[(row * chunk_count + chunk) * capacity + token]);
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        scores[group] += static_cast<float>(accumulators[group]) * scale;
      }
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token]
        = scores[group];
  }
}

__device__ __forceinline__ float qabs_quarter_logscale(int exponent) {
  switch (exponent) {
    case 0: return 1.0f;
    case 1: return 0.8408964153f;
    case 2: return 0.7071067812f;
    case 3: return 0.5946035575f;
    case 4: return 0.5f;
    case 5: return 0.4204482076f;
    case 6: return 0.3535533906f;
    case 7: return 0.2973017788f;
    case 8: return 0.25f;
    case 9: return 0.2102241038f;
    case 10: return 0.1767766953f;
    case 11: return 0.1486508894f;
    case 12: return 0.125f;
    case 13: return 0.1051120519f;
    case 14: return 0.0883883476f;
    default: return 0.0743254447f;
  }
}

template <typename scalar_t>
__global__ void qabs_pca_int4_logscale16_pack_into_kernel(
    const scalar_t* __restrict__ projected_key,
    uint8_t* __restrict__ packed_key_chunked,
    scalar_t* __restrict__ base_scales,
    uint8_t* __restrict__ packed_exponents,
    int new_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int start_token) {
  __shared__ float band_scales[QABS_MAX_DIMS / 16];
  __shared__ uint8_t band_exponents[QABS_MAX_DIMS / 16];
  int linear_block = blockIdx.x;
  int row = linear_block / new_count;
  int local_token = linear_block - row * new_count;
  int target_token = start_token + local_token;
  const scalar_t* key_row = projected_key
      + (row * new_count + local_token) * projection_dim;

  if (threadIdx.x == 0) {
    float base_scale = 1.0e-8f;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      float maximum = 0.0f;
      for (int local_dimension = 0; local_dimension < 16; ++local_dimension) {
        float value = fabsf(static_cast<float>(
            key_row[chunk * 16 + local_dimension]));
        maximum = fmaxf(maximum, value);
      }
      float exact_scale = fmaxf(maximum, 1.0e-8f) / 7.0f;
      band_scales[chunk] = exact_scale;
      base_scale = fmaxf(base_scale, exact_scale);
    }
    base_scales[row * capacity + target_token] =
        static_cast<scalar_t>(base_scale);
    int exponent_pair_count = (chunk_count + 1) / 2;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      float raw_exponent = log2f(base_scale / band_scales[chunk]) * 4.0f;
      int exponent = min(max(static_cast<int>(nearbyintf(raw_exponent)), 0), 15);
      band_exponents[chunk] = static_cast<uint8_t>(exponent);
      band_scales[chunk] = base_scale * qabs_quarter_logscale(exponent);
    }
    for (int pair = 0; pair < exponent_pair_count; ++pair) {
      int first = pair * 2;
      uint8_t packed = band_exponents[first];
      if (first + 1 < chunk_count) {
        packed |= static_cast<uint8_t>(band_exponents[first + 1] << 4);
      }
      packed_exponents[
          (row * capacity + target_token) * exponent_pair_count + pair] = packed;
    }
  }
  __syncthreads();

  int packed_dimension_count = projection_dim / 2;
  for (int packed_dimension = threadIdx.x;
       packed_dimension < packed_dimension_count;
       packed_dimension += blockDim.x) {
    int dimension = packed_dimension * 2;
    int chunk = dimension / 16;
    float inverse_scale = 1.0f / band_scales[chunk];
    int low = min(max(static_cast<int>(nearbyintf(
        static_cast<float>(key_row[dimension]) * inverse_scale)), -7), 7) + 7;
    int high = min(max(static_cast<int>(nearbyintf(
        static_cast<float>(key_row[dimension + 1]) * inverse_scale)), -7), 7) + 7;
    int byte_index = packed_dimension - chunk * 8;
    packed_key_chunked[
        ((row * chunk_count + chunk) * capacity + target_token) * 8 + byte_index]
        = static_cast<uint8_t>(low | (high << 4));
  }
}

__global__ void qabs_pca_int4_logscale16_chunk_norms_into_kernel(
    const uint8_t* __restrict__ packed_key_chunked,
    int16_t* __restrict__ chunk_squared_norms,
    int token_count,
    int capacity,
    int chunk_count,
    int start_token) {
  int linear_block = blockIdx.x;
  int row = linear_block / token_count;
  int local_token = linear_block - row * token_count;
  int target_token = start_token + local_token;
  for (int chunk = threadIdx.x; chunk < chunk_count; chunk += blockDim.x) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + chunk) * capacity + target_token) * 8;
    int squared_norm = 0;
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      int low = static_cast<int>(packed & 0x0F) - 7;
      int high = static_cast<int>(packed >> 4) - 7;
      squared_norm += low * low + high * high;
    }
    chunk_squared_norms[
        (row * capacity + target_token) * chunk_count + chunk]
        = static_cast<int16_t>(squared_norm);
  }
}

template <typename scalar_t>
__global__ void qabs_pca_project_query_int8_kernel(
    const scalar_t* __restrict__ grouped_query,
    const scalar_t* __restrict__ basis,
    scalar_t* __restrict__ projected_query,
    int8_t* __restrict__ query_codes,
    float* __restrict__ query_scales,
    int kv_head_count,
    int group_count,
    int head_dim,
    int projection_dim) {
  __shared__ float shared_projection[QABS_MAX_DIMS];
  __shared__ float reduction[128];
  int query_row = blockIdx.x;
  int kv_row = query_row / group_count;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  const scalar_t* query_row_ptr =
      grouped_query + query_row * head_dim;
  const scalar_t* basis_row =
      basis + kv_row * head_dim * projection_dim;

  for (int output_dimension = warp;
       output_dimension < projection_dim;
       output_dimension += 4) {
    float partial = 0.0f;
    for (int input_dimension = lane;
         input_dimension < head_dim;
         input_dimension += 32) {
      partial += static_cast<float>(query_row_ptr[input_dimension])
          * static_cast<float>(
              basis_row[input_dimension * projection_dim + output_dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
      shared_projection[output_dimension] = partial;
    }
  }
  __syncthreads();

  float local_maximum = threadIdx.x < projection_dim
      ? fabsf(shared_projection[threadIdx.x])
      : 0.0f;
  reduction[threadIdx.x] = local_maximum;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] = fmaxf(
          reduction[threadIdx.x],
          reduction[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float scale = fmaxf(reduction[0], 1.0e-8f) / 127.0f;
  if (threadIdx.x == 0) {
    query_scales[query_row] = scale;
  }
  for (int output_dimension = threadIdx.x;
       output_dimension < projection_dim;
       output_dimension += blockDim.x) {
    float value = shared_projection[output_dimension];
    projected_query[query_row * projection_dim + output_dimension] =
        static_cast<scalar_t>(value);
    int code = min(max(
        static_cast<int>(nearbyintf(value / scale)), -127), 127);
    query_codes[query_row * projection_dim + output_dimension] =
        static_cast<int8_t>(code);
  }
}

template <typename scalar_t>
__global__ void qabs_pre_rope_lowfreq_int4_pack_into_kernel(
    const scalar_t* __restrict__ post_rope_key,
    uint8_t* __restrict__ packed_key_chunked,
    scalar_t* __restrict__ scales,
    int new_count,
    int capacity,
    int head_dim,
    int start_token,
    float rope_theta) {
  __shared__ float low_values[32];
  __shared__ float normalized_scale;
  __shared__ float code_multiplier;
  int linear_block = blockIdx.x;
  int row = linear_block / new_count;
  int local_token = linear_block - row * new_count;
  int target_token = start_token + local_token;
  const scalar_t* key_row = post_rope_key
      + (row * new_count + local_token) * head_dim;
  if (threadIdx.x == 0) {
    int half = head_dim / 2;
    int first_frequency = half - 16;
    float squared_norm = 0.0f;
    float maximum = 0.0f;
    for (int local_frequency = 0; local_frequency < 16; ++local_frequency) {
      int frequency = first_frequency + local_frequency;
      float inverse_frequency = powf(
          rope_theta, -2.0f * static_cast<float>(frequency) / head_dim);
      float sine;
      float cosine;
      sincosf(
          static_cast<float>(target_token) * inverse_frequency,
          &sine,
          &cosine);
      float first = static_cast<float>(key_row[frequency]);
      float second = static_cast<float>(key_row[frequency + half]);
      float pre_first = first * cosine + second * sine;
      float pre_second = second * cosine - first * sine;
      low_values[local_frequency] = pre_first;
      low_values[16 + local_frequency] = pre_second;
      squared_norm += pre_first * pre_first + pre_second * pre_second;
      maximum = fmaxf(maximum, fmaxf(fabsf(pre_first), fabsf(pre_second)));
    }
    float inverse_norm = rsqrtf(fmaxf(squared_norm, 1.0e-12f));
    normalized_scale = fmaxf(maximum * inverse_norm, 1.0e-8f) / 7.0f;
    code_multiplier = inverse_norm / normalized_scale;
    scales[row * capacity + target_token] =
        static_cast<scalar_t>(normalized_scale);
  }
  __syncthreads();

  for (int packed_dimension = threadIdx.x;
       packed_dimension < 16;
       packed_dimension += blockDim.x) {
    int dimension = packed_dimension * 2;
    int low = min(max(static_cast<int>(nearbyintf(
        low_values[dimension] * code_multiplier)), -7), 7) + 7;
    int high = min(max(static_cast<int>(nearbyintf(
        low_values[dimension + 1] * code_multiplier)), -7), 7) + 7;
    int chunk = packed_dimension / 8;
    int byte_index = packed_dimension - chunk * 8;
    packed_key_chunked[
        ((row * 2 + chunk) * capacity + target_token) * 8 + byte_index]
        = static_cast<uint8_t>(low | (high << 4));
  }
}

template <typename scalar_t>
__global__ void qabs_pre_rope_lowfreq_int8_query_kernel(
    const scalar_t* __restrict__ post_rope_query,
    int8_t* __restrict__ query_codes,
    int kv_head_count,
    int group_count,
    int head_dim,
    int query_position,
    float rope_theta) {
  int row = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  int half = head_dim / 2;
  int first_frequency = half - 16;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    const scalar_t* query_row = post_rope_query
        + (batch * query_head_count + query_head) * head_dim;
    float values[32];
    float maximum = 0.0f;
    for (int local_frequency = 0; local_frequency < 16; ++local_frequency) {
      int frequency = first_frequency + local_frequency;
      float inverse_frequency = powf(
          rope_theta, -2.0f * static_cast<float>(frequency) / head_dim);
      float sine;
      float cosine;
      sincosf(
          static_cast<float>(query_position) * inverse_frequency,
          &sine,
          &cosine);
      float first = static_cast<float>(query_row[frequency]);
      float second = static_cast<float>(query_row[frequency + half]);
      float pre_first = first * cosine + second * sine;
      float pre_second = second * cosine - first * sine;
      values[local_frequency] = pre_first;
      values[16 + local_frequency] = pre_second;
      maximum = fmaxf(maximum, fmaxf(fabsf(pre_first), fabsf(pre_second)));
    }
    float inverse_scale = 127.0f / fmaxf(maximum, 1.0e-8f);
    int8_t* output = query_codes
        + (row * group_count + group) * 32;
    for (int dimension = 0; dimension < 32; ++dimension) {
      int code = min(max(static_cast<int>(nearbyintf(
          values[dimension] * inverse_scale)), -127), 127);
      output[dimension] = static_cast<int8_t>(code);
    }
  }
}

template <typename scalar_t>
__global__ void qabs_pre_rope_lowfreq_int2_fixed_pack_into_kernel(
    const scalar_t* __restrict__ post_rope_key,
    uint8_t* __restrict__ packed_key,
    int new_count,
    int capacity,
    int head_dim,
    int start_token,
    float rope_theta,
    float clip_alpha) {
  __shared__ float low_values[32];
  __shared__ float inverse_norm;
  int linear_block = blockIdx.x;
  int row = linear_block / new_count;
  int local_token = linear_block - row * new_count;
  int target_token = start_token + local_token;
  const scalar_t* key_row = post_rope_key
      + (row * new_count + local_token) * head_dim;
  if (threadIdx.x == 0) {
    int half = head_dim / 2;
    int first_frequency = half - 16;
    float squared_norm = 0.0f;
    for (int local_frequency = 0; local_frequency < 16; ++local_frequency) {
      int frequency = first_frequency + local_frequency;
      float inverse_frequency = powf(
          rope_theta, -2.0f * static_cast<float>(frequency) / head_dim);
      float sine;
      float cosine;
      sincosf(
          static_cast<float>(target_token) * inverse_frequency,
          &sine,
          &cosine);
      float first = static_cast<float>(key_row[frequency]);
      float second = static_cast<float>(key_row[frequency + half]);
      float pre_first = first * cosine + second * sine;
      float pre_second = second * cosine - first * sine;
      low_values[local_frequency] = pre_first;
      low_values[16 + local_frequency] = pre_second;
      squared_norm += pre_first * pre_first + pre_second * pre_second;
    }
    inverse_norm = rsqrtf(fmaxf(squared_norm, 1.0e-12f));
  }
  __syncthreads();

  float inverse_clip = sqrtf(32.0f) / clip_alpha;
  for (int packed_dimension = threadIdx.x;
       packed_dimension < 8;
       packed_dimension += blockDim.x) {
    uint8_t packed = 0;
#pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
      int dimension = packed_dimension * 4 + offset;
      float normalized = fminf(
          fmaxf(low_values[dimension] * inverse_norm * inverse_clip, -1.0f),
          1.0f);
      int code = min(max(static_cast<int>(nearbyintf(
          (normalized + 1.0f) * 1.5f)), 0), 3);
      packed |= static_cast<uint8_t>(code << (2 * offset));
    }
    packed_key[(row * capacity + target_token) * 8 + packed_dimension] = packed;
  }
}

__global__ void qabs_pre_rope_lowfreq_int2_fixed_scores_kernel(
    const int8_t* __restrict__ query_codes,
    const uint8_t* __restrict__ packed_key,
    float* __restrict__ output,
    int group_count,
    int key_count,
    int capacity) {
  __shared__ int8_t shared_query[8 * 32];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * 32; index += blockDim.x) {
    shared_query[index] = query_codes[row * group_count * 32 + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }
  const uint8_t* key_row = packed_key + (row * capacity + token) * 8;
  int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int packed_dimension = 0; packed_dimension < 8; ++packed_dimension) {
    uint8_t packed = key_row[packed_dimension];
#pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
      int dimension = packed_dimension * 4 + offset;
      int key_code = 2 * static_cast<int>((packed >> (2 * offset)) & 0x03) - 3;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          accumulators[group] += key_code
              * static_cast<int>(shared_query[group * 32 + dimension]);
        }
      }
    }
  }
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    if (group < group_count) {
      output[(row * group_count + group) * key_count + token] =
          static_cast<float>(accumulators[group]);
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_logscale16_prefix_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int prefix_dim) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float base_scale = static_cast<float>(base_scales[row * capacity + token]);
  const uint8_t* exponent_row = packed_exponents
      + (row * capacity + token) * ((chunk_count + 1) / 2);
  int prefix_chunk_count = prefix_dim / 16;
  for (int chunk = 0; chunk < prefix_chunk_count; ++chunk) {
    int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      int low = static_cast<int>(packed & 0x0F) - 7;
      int high = static_cast<int>(packed >> 4) - 7;
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const int8_t* query_row = shared_query + group * projection_dim;
          accumulators[group] += static_cast<int>(query_row[dimension]) * low;
          accumulators[group] += static_cast<int>(query_row[dimension + 1]) * high;
        }
      }
    }
    uint8_t exponent_pair = exponent_row[chunk / 2];
    int exponent = (chunk & 1) == 0
        ? static_cast<int>(exponent_pair & 0x0F)
        : static_cast<int>(exponent_pair >> 4);
    float scale = base_scale * qabs_quarter_logscale(exponent);
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        scores[group] += static_cast<float>(accumulators[group]) * scale;
      }
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token]
        = scores[group];
  }
}

__device__ __forceinline__ int qabs_nested_int2_representative_x2(int code) {
  if (code == 0) {
    return -11;
  }
  if (code == 1) {
    return -4;
  }
  if (code == 2) {
    return 3;
  }
  return 11;
}

template <typename scale_t>
__global__ void qabs_pca_nested_int2_logscale16_prefix_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_high2,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int prefix_dim) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim;
       index += blockDim.x) {
    shared_query[index] =
        projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float base_scale = static_cast<float>(base_scales[row * capacity + token]);
  const uint8_t* exponent_row = packed_exponents
      + (row * capacity + token) * ((chunk_count + 1) / 2);
  int prefix_chunk_count = prefix_dim / 16;
  for (int chunk = 0; chunk < prefix_chunk_count; ++chunk) {
    int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint8_t* key_chunk = packed_key_high2
        + ((row * chunk_count + chunk) * capacity + token) * 4;
#pragma unroll
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
#pragma unroll
      for (int offset = 0; offset < 4; ++offset) {
        int key_code = qabs_nested_int2_representative_x2(
            static_cast<int>((packed >> (2 * offset)) & 0x03));
        int dimension = chunk * 16 + byte_index * 4 + offset;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          if (group < group_count) {
            const int8_t* query_row = shared_query + group * projection_dim;
            accumulators[group] +=
                static_cast<int>(query_row[dimension]) * key_code;
          }
        }
      }
    }
    uint8_t exponent_pair = exponent_row[chunk / 2];
    int exponent = (chunk & 1) == 0
        ? static_cast<int>(exponent_pair & 0x0F)
        : static_cast<int>(exponent_pair >> 4);
    float scale = 0.5f * base_scale * qabs_quarter_logscale(exponent);
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        scores[group] += static_cast<float>(accumulators[group]) * scale;
      }
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token] =
        scores[group];
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_logscale16_sample_threshold_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    float* __restrict__ boundaries,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int sample_count,
    int sample_keep) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ float samples[8 * QABS_TILE_THREADS];
  __shared__ float replica_boundaries[8 * 4];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int sample_replicas = (sample_count + QABS_TILE_THREADS - 1)
      / QABS_TILE_THREADS;
  for (int index = tid; index < group_count * projection_dim;
       index += blockDim.x) {
    shared_query[index] =
        projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();

  for (int replica = 0; replica < sample_replicas; ++replica) {
    int sample_index = tid * sample_replicas + replica;
    bool sample_valid = sample_index < sample_count;
    int64_t numerator = static_cast<int64_t>(2 * sample_index + 1) * key_count;
    int token = static_cast<int>(numerator / (2 * sample_count));
    token = min(token, key_count - 1);
    float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (sample_valid) {
      float base_scale = static_cast<float>(base_scales[row * capacity + token]);
      const uint8_t* exponent_row = packed_exponents
          + (row * capacity + token) * ((chunk_count + 1) / 2);
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint8_t* key_chunk = packed_key_chunked
            + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
        for (int byte_index = 0; byte_index < 8; ++byte_index) {
          uint8_t packed = key_chunk[byte_index];
          int low = static_cast<int>(packed & 0x0F) - 7;
          int high = static_cast<int>(packed >> 4) - 7;
          int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count) {
              const int8_t* query_row = shared_query + group * projection_dim;
              accumulators[group] += static_cast<int>(query_row[dimension]) * low;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension + 1]) * high;
            }
          }
        }
        uint8_t exponent_pair = exponent_row[chunk / 2];
        int exponent = (chunk & 1) == 0
            ? static_cast<int>(exponent_pair & 0x0F)
            : static_cast<int>(exponent_pair >> 4);
        float scale = base_scale * qabs_quarter_logscale(exponent);
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          if (group < group_count) {
            scores[group] += static_cast<float>(accumulators[group]) * scale;
          }
        }
      }
    }
    for (int group = 0; group < group_count; ++group) {
      samples[group * QABS_TILE_THREADS + tid] =
          sample_valid ? scores[group] : -CUDART_INF_F;
    }
    __syncthreads();

    for (int group = 0; group < group_count; ++group) {
      float* group_samples = samples + group * QABS_TILE_THREADS;
      for (int width = 2; width <= QABS_TILE_THREADS; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
          int partner = tid ^ stride;
          if (partner > tid) {
            bool ascending = (tid & width) == 0;
            float left = group_samples[tid];
            float right = group_samples[partner];
            if ((ascending && left > right) || (!ascending && left < right)) {
              group_samples[tid] = right;
              group_samples[partner] = left;
            }
          }
          __syncthreads();
        }
      }
      if (tid == 0) {
        replica_boundaries[group * 4 + replica] =
            group_samples[QABS_TILE_THREADS - sample_keep];
      }
      __syncthreads();
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  if (tid == 0) {
    for (int group = 0; group < group_count; ++group) {
      float estimates[4];
      for (int replica = 0; replica < sample_replicas; ++replica) {
        estimates[replica] = replica_boundaries[group * 4 + replica];
      }
      for (int left = 1; left < sample_replicas; ++left) {
        float value = estimates[left];
        int right = left - 1;
        while (right >= 0 && estimates[right] > value) {
          estimates[right + 1] = estimates[right];
          --right;
        }
        estimates[right + 1] = value;
      }
      float boundary = estimates[sample_replicas / 2];
      if ((sample_replicas & 1) == 0) {
        boundary = 0.5f * (
            estimates[sample_replicas / 2 - 1]
            + estimates[sample_replicas / 2]);
      }
      int query_head = kv_head * group_count + group;
      boundaries[batch * query_head_count + query_head] = boundary;
    }
  }
}

template <typename scalar_t, typename scale_t>
__global__ void qabs_pca_int4_logscale16_raw_query_sample_threshold_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ basis,
    int8_t* __restrict__ projected_query,
    float* __restrict__ projected_query_scales,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    float* __restrict__ boundaries,
    int kv_head_count,
    int group_count,
    int head_dim,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int sample_count,
    int sample_keep) {
  __shared__ float shared_projection[8 * QABS_MAX_DIMS];
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ float samples[8 * QABS_TILE_THREADS];
  __shared__ float replica_boundaries[8 * 4];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  int warps_per_group = max(1, 8 / group_count);
  if (warp < group_count * warps_per_group) {
    int group = warp / warps_per_group;
    int local_warp = warp - group * warps_per_group;
    const scalar_t* query_row = query
        + (batch * query_head_count + kv_head * group_count + group) * head_dim;
    const scalar_t* basis_row =
        basis + row * head_dim * projection_dim;
    for (int output_dimension = local_warp;
         output_dimension < projection_dim;
         output_dimension += warps_per_group) {
      float partial = 0.0f;
      for (int input_dimension = lane;
           input_dimension < head_dim;
           input_dimension += 32) {
        partial += static_cast<float>(query_row[input_dimension])
            * static_cast<float>(
                basis_row[
                    input_dimension * projection_dim + output_dimension]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
      }
      if (lane == 0) {
        shared_projection[group * projection_dim + output_dimension] = partial;
      }
    }
  }
  __syncthreads();

  if (warp < group_count) {
    int group = warp;
    float maximum = 0.0f;
    for (int dimension = lane; dimension < projection_dim; dimension += 32) {
      maximum = fmaxf(
          maximum,
          fabsf(shared_projection[group * projection_dim + dimension]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    float query_scale = fmaxf(maximum, 1.0e-8f) / 127.0f;
    if (lane == 0) {
      int query_head = kv_head * group_count + group;
      projected_query_scales[
          batch * query_head_count + query_head] = query_scale;
    }
    for (int dimension = lane;
         dimension < projection_dim;
         dimension += 32) {
      int code = min(max(static_cast<int>(nearbyintf(
          shared_projection[group * projection_dim + dimension]
          / query_scale)), -127), 127);
      shared_query[group * projection_dim + dimension] =
          static_cast<int8_t>(code);
      projected_query[
          row * group_count * projection_dim
          + group * projection_dim + dimension] =
          static_cast<int8_t>(code);
    }
  }
  __syncthreads();

  int sample_replicas = (sample_count + QABS_TILE_THREADS - 1)
      / QABS_TILE_THREADS;
  for (int replica = 0; replica < sample_replicas; ++replica) {
    int sample_index = tid * sample_replicas + replica;
    bool sample_valid = sample_index < sample_count;
    int64_t numerator = static_cast<int64_t>(2 * sample_index + 1) * key_count;
    int token = static_cast<int>(numerator / (2 * sample_count));
    token = min(token, key_count - 1);
    float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (sample_valid) {
      float base_scale = static_cast<float>(base_scales[row * capacity + token]);
      const uint8_t* exponent_row = packed_exponents
          + (row * capacity + token) * ((chunk_count + 1) / 2);
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint8_t* key_chunk = packed_key_chunked
            + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
        for (int byte_index = 0; byte_index < 8; ++byte_index) {
          uint8_t packed = key_chunk[byte_index];
          int low = static_cast<int>(packed & 0x0F) - 7;
          int high = static_cast<int>(packed >> 4) - 7;
          int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count) {
              const int8_t* query_row =
                  shared_query + group * projection_dim;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension]) * low;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension + 1]) * high;
            }
          }
        }
        uint8_t exponent_pair = exponent_row[chunk / 2];
        int exponent = (chunk & 1) == 0
            ? static_cast<int>(exponent_pair & 0x0F)
            : static_cast<int>(exponent_pair >> 4);
        float scale = base_scale * qabs_quarter_logscale(exponent);
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          if (group < group_count) {
            scores[group] += static_cast<float>(accumulators[group]) * scale;
          }
        }
      }
    }
    for (int group = 0; group < group_count; ++group) {
      samples[group * QABS_TILE_THREADS + tid] =
          sample_valid ? scores[group] : -CUDART_INF_F;
    }
    __syncthreads();

    for (int group = 0; group < group_count; ++group) {
      float* group_samples = samples + group * QABS_TILE_THREADS;
      for (int width = 2; width <= QABS_TILE_THREADS; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
          int partner = tid ^ stride;
          if (partner > tid) {
            bool ascending = (tid & width) == 0;
            float left = group_samples[tid];
            float right = group_samples[partner];
            if ((ascending && left > right) || (!ascending && left < right)) {
              group_samples[tid] = right;
              group_samples[partner] = left;
            }
          }
          __syncthreads();
        }
      }
      if (tid == 0) {
        replica_boundaries[group * 4 + replica] =
            group_samples[QABS_TILE_THREADS - sample_keep];
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    for (int group = 0; group < group_count; ++group) {
      float estimates[4];
      for (int replica = 0; replica < sample_replicas; ++replica) {
        estimates[replica] = replica_boundaries[group * 4 + replica];
      }
      for (int left = 1; left < sample_replicas; ++left) {
        float value = estimates[left];
        int right = left - 1;
        while (right >= 0 && estimates[right] > value) {
          estimates[right + 1] = estimates[right];
          --right;
        }
        estimates[right + 1] = value;
      }
      float boundary = estimates[sample_replicas / 2];
      if ((sample_replicas & 1) == 0) {
        boundary = 0.5f * (
            estimates[sample_replicas / 2 - 1]
            + estimates[sample_replicas / 2]);
      }
      int query_head = kv_head * group_count + group;
      boundaries[batch * query_head_count + query_head] = boundary;
    }
  }
}

template <typename scale_t, bool use_dp4a>
__global__ void qabs_pca_int4_logscale16_threshold_compact_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const float* __restrict__ boundaries,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_proxy_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int candidate_capacity) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  for (int index = threadIdx.x; index < group_count * projection_dim;
       index += blockDim.x) {
    shared_query[index] =
        projected_query[row * group_count * projection_dim + index];
  }
  __syncthreads();
  bool valid = token < key_count;
  float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (valid) {
    float base_scale = static_cast<float>(base_scales[row * capacity + token]);
    const uint8_t* exponent_row = packed_exponents
        + (row * capacity + token) * ((chunk_count + 1) / 2);
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t* key_chunk = packed_key_chunked
          + ((row * chunk_count + chunk) * capacity + token) * 8;
      if constexpr (use_dp4a) {
#pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
          uint8_t first = key_chunk[2 * quad];
          uint8_t second = key_chunk[2 * quad + 1];
          uint32_t packed_key_values =
              static_cast<uint8_t>(static_cast<int8_t>((first & 0x0F) - 7))
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((first >> 4) - 7))) << 8)
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((second & 0x0F) - 7))) << 16)
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((second >> 4) - 7))) << 24);
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count) {
              const int8_t* query_row = shared_query
                  + group * projection_dim + chunk * 16 + quad * 4;
              int packed_query_values =
                  *reinterpret_cast<const int*>(query_row);
              accumulators[group] = __dp4a(
                  packed_query_values,
                  static_cast<int>(packed_key_values),
                  accumulators[group]);
            }
          }
        }
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < 8; ++byte_index) {
          uint8_t packed = key_chunk[byte_index];
          int low = static_cast<int>(packed & 0x0F) - 7;
          int high = static_cast<int>(packed >> 4) - 7;
          int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count) {
              const int8_t* query_row = shared_query + group * projection_dim;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension]) * low;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension + 1]) * high;
            }
          }
        }
      }
      uint8_t exponent_pair = exponent_row[chunk / 2];
      int exponent = (chunk & 1) == 0
          ? static_cast<int>(exponent_pair & 0x0F)
          : static_cast<int>(exponent_pair >> 4);
      float scale = base_scale * qabs_quarter_logscale(exponent);
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          scores[group] += static_cast<float>(accumulators[group]) * scale;
        }
      }
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int output_row = batch * query_head_count + query_head;
    bool keep = valid && scores[group] >= boundaries[output_row];
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    unsigned long long base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(
          reinterpret_cast<unsigned long long*>(candidate_counts + output_row),
          static_cast<unsigned long long>(active_count));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      unsigned long long position = base + __popc(active & lower_lanes);
      if (position < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[output_row * candidate_capacity + position] =
            static_cast<int64_t>(token);
        if (candidate_proxy_scores != nullptr) {
          candidate_proxy_scores[output_row * candidate_capacity + position] =
              scores[group];
        }
      } else {
        overflow[output_row] = true;
      }
    }
  }
}

template <typename scale_t, bool use_dp4a>
__device__ __forceinline__ float qabs_pca_int4_logscale16_one_score(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    int packed_row,
    int token,
    int capacity,
    int projection_dim,
    int chunk_count) {
  float score = 0.0f;
  float base_scale = static_cast<float>(
      base_scales[packed_row * capacity + token]);
  const uint8_t* exponent_row = packed_exponents
      + (packed_row * capacity + token) * ((chunk_count + 1) / 2);
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    int accumulator = 0;
    const uint8_t* key_chunk = packed_key_chunked
        + ((packed_row * chunk_count + chunk) * capacity + token) * 8;
    if constexpr (use_dp4a) {
#pragma unroll
      for (int quad = 0; quad < 4; ++quad) {
        uint8_t first = key_chunk[2 * quad];
        uint8_t second = key_chunk[2 * quad + 1];
        uint32_t packed_key_values =
            static_cast<uint8_t>(static_cast<int8_t>((first & 0x0F) - 7))
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((first >> 4) - 7))) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((second & 0x0F) - 7))) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((second >> 4) - 7))) << 24);
        const int8_t* query_row = projected_query
            + chunk * 16 + quad * 4;
        int packed_query_values =
            *reinterpret_cast<const int*>(query_row);
        accumulator = __dp4a(
            packed_query_values,
            static_cast<int>(packed_key_values),
            accumulator);
      }
    } else {
#pragma unroll
      for (int byte_index = 0; byte_index < 8; ++byte_index) {
        uint8_t packed = key_chunk[byte_index];
        int low = static_cast<int>(packed & 0x0F) - 7;
        int high = static_cast<int>(packed >> 4) - 7;
        int dimension = chunk * 16 + byte_index * 2;
        accumulator +=
            static_cast<int>(projected_query[dimension]) * low;
        accumulator +=
            static_cast<int>(projected_query[dimension + 1]) * high;
      }
    }
    uint8_t exponent_pair = exponent_row[chunk / 2];
    int exponent = (chunk & 1) == 0
        ? static_cast<int>(exponent_pair & 0x0F)
        : static_cast<int>(exponent_pair >> 4);
    float scale = base_scale * qabs_quarter_logscale(exponent);
    score += static_cast<float>(accumulator) * scale;
  }
  return score;
}

template <typename scalar_t, typename scale_t, bool use_dp4a>
__global__ void qabs_pca_int4_logscale16_streaming_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ basis,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    float* __restrict__ boundaries,
    bool* __restrict__ overflow,
    int8_t* __restrict__ projected_query,
    float* __restrict__ projected_query_scales,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int group_count,
    int head_dim,
    int history_count,
    int key_capacity,
    int index_capacity,
    int projection_dim,
    int chunk_count,
    int sample_count,
    int sample_keep,
    int candidate_capacity,
    float scaling) {
  __shared__ float shared_projection[QABS_MAX_DIMS];
  __shared__ int8_t shared_query[QABS_MAX_DIMS];
  __shared__ float replica_boundaries[4];
  __shared__ float shared_boundary;
  extern __shared__ float dynamic_shared[];
  float* sample_values = dynamic_shared;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int packed_row = batch * kv_head_count + kv_head;
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* basis_row =
      basis + packed_row * head_dim * projection_dim;

  for (int output_dimension = warp;
       output_dimension < projection_dim;
       output_dimension += 8) {
    float partial = 0.0f;
    for (int input_dimension = lane;
         input_dimension < head_dim;
         input_dimension += 32) {
      partial += static_cast<float>(query_row[input_dimension])
          * static_cast<float>(
              basis_row[
                  input_dimension * projection_dim + output_dimension]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if (lane == 0) {
      shared_projection[output_dimension] = partial;
    }
  }
  __syncthreads();

  float local_maximum = 0.0f;
  for (int dimension = tid;
       dimension < projection_dim;
       dimension += blockDim.x) {
    local_maximum = fmaxf(
        local_maximum,
        fabsf(shared_projection[dimension]));
  }
  sample_values[tid] = local_maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sample_values[tid] = fmaxf(
          sample_values[tid],
          sample_values[tid + stride]);
    }
    __syncthreads();
  }
  float query_scale = fmaxf(sample_values[0], 1.0e-8f) / 127.0f;
  if (tid == 0) {
    projected_query_scales[row] = query_scale;
  }
  for (int dimension = tid;
       dimension < projection_dim;
       dimension += blockDim.x) {
    int code = min(max(static_cast<int>(nearbyintf(
        shared_projection[dimension] / query_scale)), -127), 127);
    shared_query[dimension] = static_cast<int8_t>(code);
    projected_query[row * projection_dim + dimension] =
        static_cast<int8_t>(code);
  }
  __syncthreads();

  int sample_replicas = (sample_count + blockDim.x - 1) / blockDim.x;
  for (int replica = 0; replica < sample_replicas; ++replica) {
    int sample_index = tid * sample_replicas + replica;
    bool sample_valid = sample_index < sample_count;
    int64_t numerator =
        static_cast<int64_t>(2 * sample_index + 1) * history_count;
    int token = static_cast<int>(numerator / (2 * sample_count));
    token = min(token, history_count - 1);
    sample_values[tid] = sample_valid
        ? qabs_pca_int4_logscale16_one_score<scale_t, use_dp4a>(
              shared_query,
              packed_key_chunked,
              base_scales,
              packed_exponents,
              packed_row,
              token,
              index_capacity,
              projection_dim,
              chunk_count)
        : -CUDART_INF_F;
    __syncthreads();
    for (int width = 2; width <= blockDim.x; width <<= 1) {
      for (int stride = width >> 1; stride > 0; stride >>= 1) {
        int partner = tid ^ stride;
        if (partner > tid) {
          bool ascending = (tid & width) == 0;
          float left = sample_values[tid];
          float right = sample_values[partner];
          if ((ascending && left > right) || (!ascending && left < right)) {
            sample_values[tid] = right;
            sample_values[partner] = left;
          }
        }
        __syncthreads();
      }
    }
    if (tid == 0) {
      replica_boundaries[replica] =
          sample_values[blockDim.x - sample_keep];
    }
    __syncthreads();
  }
  if (tid == 0) {
    float estimates[4];
    for (int replica = 0; replica < sample_replicas; ++replica) {
      estimates[replica] = replica_boundaries[replica];
    }
    for (int left = 1; left < sample_replicas; ++left) {
      float selected = estimates[left];
      int right = left - 1;
      while (right >= 0 && estimates[right] > selected) {
        estimates[right + 1] = estimates[right];
        --right;
      }
      estimates[right + 1] = selected;
    }
    float boundary = estimates[sample_replicas / 2];
    if ((sample_replicas & 1) == 0) {
      boundary = 0.5f * (
          estimates[sample_replicas / 2 - 1]
          + estimates[sample_replicas / 2]);
    }
    boundaries[row] = boundary;
    shared_boundary = boundary;
  }
  __syncthreads();
  float boundary = shared_boundary;

  for (int tile = 0; tile < history_count; tile += blockDim.x) {
    int token = tile + tid;
    bool valid = token < history_count;
    float proxy_score = valid
        ? qabs_pca_int4_logscale16_one_score<scale_t, use_dp4a>(
              shared_query,
              packed_key_chunked,
              base_scales,
              packed_exponents,
              packed_row,
              token,
              index_capacity,
              projection_dim,
              chunk_count)
        : -CUDART_INF_F;
    bool keep = valid && proxy_score >= boundary;
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    unsigned long long base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(
          reinterpret_cast<unsigned long long*>(candidate_counts + row),
          static_cast<unsigned long long>(active_count));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower_lanes =
          lane == 0 ? 0u : ((1u << lane) - 1u);
      unsigned long long position =
          base + __popc(active & lower_lanes);
      if (position < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[row * candidate_capacity + position] =
            static_cast<int64_t>(token);
      } else {
        overflow[row] = true;
      }
    }
  }
  __syncthreads();

  int candidate_count = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int selected_count = candidate_count + 1;
  float* reduction = dynamic_shared;
  float* weights = dynamic_shared + blockDim.x;
  const scalar_t* key_base = key
      + ((batch * kv_head_count + kv_head) * key_capacity) * head_dim;
  const scalar_t* value_base = value
      + ((batch * kv_head_count + kv_head) * key_capacity) * head_dim;
  float local_score_max = -CUDART_INF_F;
  for (int selected = tid;
       selected < selected_count;
       selected += blockDim.x) {
    int64_t token = selected == candidate_count
        ? static_cast<int64_t>(history_count)
        : candidate_indices[row * candidate_capacity + selected];
    float score = 0.0f;
    const scalar_t* key_row = key_base + token * head_dim;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      score += static_cast<float>(query_row[dimension])
          * static_cast<float>(key_row[dimension]);
    }
    score *= scaling;
    weights[selected] = score;
    local_score_max = fmaxf(local_score_max, score);
  }
  reduction[tid] = local_score_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(
          reduction[tid],
          reduction[tid + stride]);
    }
    __syncthreads();
  }
  float maximum_score = reduction[0];
  float local_denominator = 0.0f;
  for (int selected = tid;
       selected < selected_count;
       selected += blockDim.x) {
    float weight = expf(weights[selected] - maximum_score);
    weights[selected] = weight;
    local_denominator += weight;
  }
  reduction[tid] = local_denominator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float inverse_denominator =
      1.0f / fmaxf(reduction[0], 1.0e-20f);
  scalar_t* output_row = output + row * head_dim;
  for (int dimension = tid;
       dimension < head_dim;
       dimension += blockDim.x) {
    float accumulator = 0.0f;
    for (int selected = 0; selected < selected_count; ++selected) {
      int64_t token = selected == candidate_count
          ? static_cast<int64_t>(history_count)
          : candidate_indices[row * candidate_capacity + selected];
      accumulator += weights[selected] * inverse_denominator
          * static_cast<float>(
              value_base[token * head_dim + dimension]);
    }
    output_row[dimension] = static_cast<scalar_t>(accumulator);
  }
}

template <typename scale_t, bool use_dp4a, bool collect_statistics>
__global__ void qabs_pca_int4_logscale16_threshold_compact_bound_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const int16_t* __restrict__ chunk_squared_norms,
    const float* __restrict__ boundaries,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_proxy_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int64_t* __restrict__ key_chunk_evaluations,
    int64_t* __restrict__ query_chunk_evaluations,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int candidate_capacity) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ float shared_query_tail_squared_norms[
      8 * (QABS_MAX_DIMS / 16)];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  for (int index = threadIdx.x; index < group_count * projection_dim;
       index += blockDim.x) {
    shared_query[index] =
        projected_query[row * group_count * projection_dim + index];
  }
  for (int index = threadIdx.x; index < group_count * chunk_count;
       index += blockDim.x) {
    int group = index / chunk_count;
    int chunk = index - group * chunk_count;
    float squared_norm = 0.0f;
    const int8_t* query_row = projected_query
        + row * group_count * projection_dim + group * projection_dim;
    for (int dimension = (chunk + 1) * 16;
         dimension < projection_dim;
         ++dimension) {
      float value = static_cast<float>(query_row[dimension]);
      squared_norm += value * value;
    }
    shared_query_tail_squared_norms[group * chunk_count + chunk] =
        squared_norm;
  }
  __syncthreads();

  bool valid = token < key_count;
  bool active[8] = {
      true, true, true, true, true, true, true, true};
  int group_chunk_evaluations[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int token_chunk_evaluations = 0;
  float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (valid) {
    float base_scale = static_cast<float>(base_scales[row * capacity + token]);
    const uint8_t* exponent_row = packed_exponents
        + (row * capacity + token) * ((chunk_count + 1) / 2);
    float chunk_scales[QABS_MAX_DIMS / 16];
#pragma unroll
    for (int chunk = 0; chunk < QABS_MAX_DIMS / 16; ++chunk) {
      if (chunk < chunk_count) {
        uint8_t exponent_pair = exponent_row[chunk / 2];
        int exponent = (chunk & 1) == 0
            ? static_cast<int>(exponent_pair & 0x0F)
            : static_cast<int>(exponent_pair >> 4);
        chunk_scales[chunk] =
            base_scale * qabs_quarter_logscale(exponent);
      }
    }

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      bool any_active = false;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count && active[group]) {
          any_active = true;
          ++group_chunk_evaluations[group];
        }
      }
      if (!any_active) {
        break;
      }
      ++token_chunk_evaluations;
      int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t* key_chunk = packed_key_chunked
          + ((row * chunk_count + chunk) * capacity + token) * 8;
      if constexpr (use_dp4a) {
#pragma unroll
        for (int quad = 0; quad < 4; ++quad) {
          uint8_t first = key_chunk[2 * quad];
          uint8_t second = key_chunk[2 * quad + 1];
          uint32_t packed_key_values =
              static_cast<uint8_t>(static_cast<int8_t>((first & 0x0F) - 7))
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((first >> 4) - 7))) << 8)
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((second & 0x0F) - 7))) << 16)
              | (static_cast<uint32_t>(static_cast<uint8_t>(
                    static_cast<int8_t>((second >> 4) - 7))) << 24);
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count && active[group]) {
              const int8_t* query_row = shared_query
                  + group * projection_dim + chunk * 16 + quad * 4;
              int packed_query_values =
                  *reinterpret_cast<const int*>(query_row);
              accumulators[group] = __dp4a(
                  packed_query_values,
                  static_cast<int>(packed_key_values),
                  accumulators[group]);
            }
          }
        }
      } else {
#pragma unroll
        for (int byte_index = 0; byte_index < 8; ++byte_index) {
          uint8_t packed = key_chunk[byte_index];
          int low = static_cast<int>(packed & 0x0F) - 7;
          int high = static_cast<int>(packed >> 4) - 7;
          int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
          for (int group = 0; group < 8; ++group) {
            if (group < group_count && active[group]) {
              const int8_t* query_row = shared_query + group * projection_dim;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension]) * low;
              accumulators[group] +=
                  static_cast<int>(query_row[dimension + 1]) * high;
            }
          }
        }
      }
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count && active[group]) {
          scores[group] += static_cast<float>(accumulators[group])
              * chunk_scales[chunk];
        }
      }

      if (chunk + 1 < chunk_count) {
        float remaining_key_scale_squared = 0.0f;
        for (int remaining = chunk + 1;
             remaining < chunk_count;
             ++remaining) {
          float code_squared_norm = static_cast<float>(
              chunk_squared_norms[
                  (row * capacity + token) * chunk_count + remaining]);
          remaining_key_scale_squared +=
              code_squared_norm
              * chunk_scales[remaining] * chunk_scales[remaining];
        }
        int batch = row / kv_head_count;
        int kv_head = row - batch * kv_head_count;
        int query_head_count = kv_head_count * group_count;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          if (group < group_count && active[group]) {
            int query_head = kv_head * group_count + group;
            int output_row = batch * query_head_count + query_head;
            float boundary = boundaries[output_row];
            float gap = boundary - scores[group];
            float roundoff_guard = 1.0e-5f * (
                fabsf(scores[group]) + fabsf(boundary) + 1.0f);
            float guarded_gap = gap - roundoff_guard;
            float tail_bound_squared =
                shared_query_tail_squared_norms[
                    group * chunk_count + chunk]
                * remaining_key_scale_squared;
            if (guarded_gap > 0.0f
                && guarded_gap * guarded_gap > tail_bound_squared) {
              active[group] = false;
            }
          }
        }
      }
    }
  }

  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int output_row = batch * query_head_count + query_head;
    bool keep = valid && active[group]
        && scores[group] >= boundaries[output_row];
    unsigned int active_mask = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active_mask);
    unsigned long long base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(
          reinterpret_cast<unsigned long long*>(candidate_counts + output_row),
          static_cast<unsigned long long>(active_count));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      unsigned long long position = base + __popc(active_mask & lower_lanes);
      if (position < static_cast<unsigned long long>(candidate_capacity)) {
        candidate_indices[output_row * candidate_capacity + position] =
            static_cast<int64_t>(token);
        if (candidate_proxy_scores != nullptr) {
          candidate_proxy_scores[output_row * candidate_capacity + position] =
              scores[group];
        }
      } else {
        overflow[output_row] = true;
      }
    }
  }

  if constexpr (collect_statistics) {
    int key_evaluations = valid ? token_chunk_evaluations : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      key_evaluations += __shfl_down_sync(
          0xffffffffu, key_evaluations, offset);
    }
    if (lane == 0 && key_evaluations > 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              key_chunk_evaluations + row),
          static_cast<unsigned long long>(key_evaluations));
    }
    for (int group = 0; group < group_count; ++group) {
      int evaluations = valid ? group_chunk_evaluations[group] : 0;
      for (int offset = 16; offset > 0; offset >>= 1) {
        evaluations += __shfl_down_sync(
            0xffffffffu, evaluations, offset);
      }
      if (lane == 0 && evaluations > 0) {
        int query_head = kv_head * group_count + group;
        int output_row = batch * query_head_count + query_head;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(
                query_chunk_evaluations + output_row),
            static_cast<unsigned long long>(evaluations));
      }
    }
  }
}

template <typename scale_t, typename scalar_t>
__global__ void qabs_pca_int4_logscale16_threshold_exact_kernel(
    const int8_t* __restrict__ projected_query,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const float* __restrict__ boundaries,
    int64_t* __restrict__ candidate_indices,
    float* __restrict__ candidate_exact_scores,
    int64_t* __restrict__ candidate_counts,
    bool* __restrict__ overflow,
    int kv_head_count,
    int group_count,
    int key_count,
    int key_capacity,
    int index_capacity,
    int projection_dim,
    int chunk_count,
    int head_dim,
    int candidate_capacity,
    float scaling) {
  __shared__ int8_t shared_projected_query[8 * QABS_MAX_DIMS];
  __shared__ scalar_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int index = tid; index < group_count * projection_dim;
       index += blockDim.x) {
    shared_projected_query[index] =
        projected_query[row * group_count * projection_dim + index];
  }
  for (int index = tid; index < group_count * head_dim;
       index += blockDim.x) {
    int group = index / head_dim;
    int dimension = index - group * head_dim;
    int query_head = kv_head * group_count + group;
    shared_query[index] = query[
        (batch * query_head_count + query_head) * head_dim + dimension];
  }
  __syncthreads();

  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int token = tile_start + warp;
       token < min(key_count, tile_start + QABS_TILE_THREADS);
       token += warps_per_block) {
    float base_scale = static_cast<float>(
        base_scales[row * index_capacity + token]);
    const uint8_t* exponent_row = packed_exponents
        + (row * index_capacity + token) * ((chunk_count + 1) / 2);
    unsigned int keep_groups = 0;
    for (int group = 0; group < group_count; ++group) {
      int partial_accumulator = 0;
      if (lane < chunk_count * 4) {
        int chunk = lane >> 2;
        int quad = lane & 3;
        const uint8_t* key_chunk = packed_key_chunked
            + ((row * chunk_count + chunk) * index_capacity + token) * 8;
        uint8_t first = key_chunk[2 * quad];
        uint8_t second = key_chunk[2 * quad + 1];
        uint32_t packed_key_values =
            static_cast<uint8_t>(static_cast<int8_t>((first & 0x0F) - 7))
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((first >> 4) - 7))) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((second & 0x0F) - 7))) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(
                  static_cast<int8_t>((second >> 4) - 7))) << 24);
        const int8_t* query_values = shared_projected_query
            + group * projection_dim + chunk * 16 + quad * 4;
        int packed_query_values =
            *reinterpret_cast<const int*>(query_values);
        partial_accumulator = __dp4a(
            packed_query_values,
            static_cast<int>(packed_key_values),
            0);
      }
      partial_accumulator += __shfl_down_sync(
          0xffffffffu, partial_accumulator, 2, 4);
      partial_accumulator += __shfl_down_sync(
          0xffffffffu, partial_accumulator, 1, 4);
      float chunk_contribution = 0.0f;
      if (lane < chunk_count * 4 && (lane & 3) == 0) {
        int chunk = lane >> 2;
        uint8_t exponent_pair = exponent_row[chunk / 2];
        int exponent = (chunk & 1) == 0
            ? static_cast<int>(exponent_pair & 0x0F)
            : static_cast<int>(exponent_pair >> 4);
        chunk_contribution = static_cast<float>(partial_accumulator)
            * base_scale * qabs_quarter_logscale(exponent);
      }
      float approximate_score = 0.0f;
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        approximate_score += __shfl_sync(
            0xffffffffu, chunk_contribution, chunk * 4);
      }
      if (lane == 0) {
        int query_head = kv_head * group_count + group;
        int output_row = batch * query_head_count + query_head;
        if (approximate_score >= boundaries[output_row]) {
          keep_groups |= 1u << group;
        }
      }
    }
    keep_groups = __shfl_sync(0xffffffffu, keep_groups, 0);
    if (keep_groups == 0) {
      continue;
    }

    const scalar_t* key_row = key
        + (row * key_capacity + token) * head_dim;
    for (int group = 0; group < group_count; ++group) {
      if ((keep_groups & (1u << group)) == 0) {
        continue;
      }
      const scalar_t* query_row = shared_query + group * head_dim;
      float exact_score = 0.0f;
      for (int dimension = lane; dimension < head_dim; dimension += 32) {
        exact_score += static_cast<float>(query_row[dimension])
            * static_cast<float>(key_row[dimension]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        exact_score += __shfl_down_sync(
            0xffffffffu, exact_score, offset);
      }
      if (lane == 0) {
        int query_head = kv_head * group_count + group;
        int output_row = batch * query_head_count + query_head;
        unsigned long long position = atomicAdd(
            reinterpret_cast<unsigned long long*>(
                candidate_counts + output_row),
            1ull);
        if (position < static_cast<unsigned long long>(candidate_capacity)) {
          candidate_indices[output_row * candidate_capacity + position] =
              static_cast<int64_t>(token);
          candidate_exact_scores[
              output_row * candidate_capacity + position] =
                  exact_score * scaling;
        } else {
          overflow[output_row] = true;
        }
      }
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_selected_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    const int32_t* __restrict__ dimension_indices,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int selected_count) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ int32_t shared_dimensions[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  for (int index = threadIdx.x; index < group_count * selected_count; index += blockDim.x) {
    shared_dimensions[index] = dimension_indices[row * group_count * selected_count + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int accumulator = 0;
    const int8_t* query_row = shared_query + group * projection_dim;
    const int32_t* group_dimensions = shared_dimensions + group * selected_count;
    for (int selected = 0; selected < selected_count; ++selected) {
      int dimension = group_dimensions[selected];
      int chunk = dimension / 16;
      int local_dimension = dimension - chunk * 16;
      int byte_index = local_dimension / 2;
      uint8_t packed = packed_key_chunked[
          ((row * chunk_count + chunk) * capacity + token) * 8 + byte_index];
      int key_value = (local_dimension & 1)
          ? static_cast<int>(packed >> 4) - 7
          : static_cast<int>(packed & 0x0F) - 7;
      accumulator += static_cast<int>(query_row[dimension]) * key_value;
    }
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token]
        = static_cast<float>(accumulator) * scale;
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_shared_selected_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    const int32_t* __restrict__ dimension_indices,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int selected_count) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ int32_t shared_dimensions[QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  for (int index = threadIdx.x; index < selected_count; index += blockDim.x) {
    shared_dimensions[index] = dimension_indices[row * selected_count + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int selected = 0; selected < selected_count; ++selected) {
    int dimension = shared_dimensions[selected];
    int chunk = dimension / 16;
    int local_dimension = dimension - chunk * 16;
    int byte_index = local_dimension / 2;
    uint8_t packed = packed_key_chunked[
        ((row * chunk_count + chunk) * capacity + token) * 8 + byte_index];
    int key_value = (local_dimension & 1)
        ? static_cast<int>(packed >> 4) - 7
        : static_cast<int>(packed & 0x0F) - 7;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        accumulators[group] += static_cast<int>(
            shared_query[group * projection_dim + dimension]) * key_value;
      }
    }
  }

  float scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    output[(batch * query_head_count + query_head) * key_count + token]
        = static_cast<float>(accumulators[group]) * scale;
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_shared_selected_add_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    const int32_t* __restrict__ dimension_indices,
    const float* __restrict__ query_scales,
    scale_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int selected_count) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  __shared__ int32_t shared_dimensions[QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * projection_dim + index];
  }
  for (int index = threadIdx.x; index < selected_count; index += blockDim.x) {
    shared_dimensions[index] = dimension_indices[row * selected_count + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int selected = 0; selected < selected_count; ++selected) {
    int dimension = shared_dimensions[selected];
    int chunk = dimension / 16;
    int local_dimension = dimension - chunk * 16;
    int byte_index = local_dimension / 2;
    uint8_t packed = packed_key_chunked[
        ((row * chunk_count + chunk) * capacity + token) * 8 + byte_index];
    int key_value = (local_dimension & 1)
        ? static_cast<int>(packed >> 4) - 7
        : static_cast<int>(packed & 0x0F) - 7;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        accumulators[group] += static_cast<int>(
            shared_query[group * projection_dim + dimension]) * key_value;
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    float delta_score = static_cast<float>(accumulators[group])
        * key_scale * query_scales[row * group_count + group];
    score_cache[cache_index] = static_cast<scale_t>(
        static_cast<float>(score_cache[cache_index]) + delta_score);
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_contiguous_add_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    const float* __restrict__ query_scales,
    scale_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int selected_count,
    int chunk_count,
    int start_chunk) {
  __shared__ int8_t shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * selected_count; index += blockDim.x) {
    shared_query[index] = projected_query[row * group_count * selected_count + index];
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  int accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int selected_chunk_count = selected_count / 16;
  for (int chunk = 0; chunk < selected_chunk_count; ++chunk) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + start_chunk + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      int low = static_cast<int>(packed & 0x0F) - 7;
      int high = static_cast<int>(packed >> 4) - 7;
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const int8_t* query_row = shared_query + group * selected_count;
          accumulators[group] += static_cast<int>(query_row[dimension]) * low;
          accumulators[group] += static_cast<int>(query_row[dimension + 1]) * high;
        }
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    float delta_score = static_cast<float>(accumulators[group])
        * key_scale * query_scales[row * group_count + group];
    score_cache[cache_index] = static_cast<scale_t>(
        static_cast<float>(score_cache[cache_index]) + delta_score);
  }
}

template <typename scalar_t>
__global__ void qabs_pca_int4_chunked_contiguous_delta_add_kernel(
    const scalar_t* __restrict__ projected_query,
    const scalar_t* __restrict__ previous_projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int selected_count,
    int chunk_count,
    int start_chunk) {
  __shared__ float shared_delta[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  int start_dim = start_chunk * 16;
  for (int index = threadIdx.x; index < group_count * selected_count; index += blockDim.x) {
    int group = index / selected_count;
    int local_dimension = index - group * selected_count;
    int query_index = row * group_count * projection_dim
        + group * projection_dim + start_dim + local_dimension;
    shared_delta[index] = static_cast<float>(projected_query[query_index])
        - static_cast<float>(previous_projected_query[query_index]);
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int selected_chunk_count = selected_count / 16;
  for (int chunk = 0; chunk < selected_chunk_count; ++chunk) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + start_chunk + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      float low = static_cast<float>(static_cast<int>(packed & 0x0F) - 7);
      float high = static_cast<float>(static_cast<int>(packed >> 4) - 7);
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const float* query_row = shared_delta + group * selected_count;
          accumulators[group] += query_row[dimension] * low;
          accumulators[group] += query_row[dimension + 1] * high;
        }
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    score_cache[cache_index] = static_cast<scalar_t>(
        static_cast<float>(score_cache[cache_index])
        + accumulators[group] * key_scale);
  }
}

template <typename scalar_t>
__global__ void qabs_spectral_band_select_kernel(
    const scalar_t* __restrict__ projected_query,
    const float* __restrict__ spectral_weights,
    const scalar_t* __restrict__ anchor_query,
    const uint8_t* __restrict__ active_mask,
    int32_t* __restrict__ selected_chunk,
    float* __restrict__ gate_signal,
    int group_count,
    int projection_dim,
    int chunk_count) {
  int row = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  if (active_mask != nullptr && active_mask[row] == 0) {
    selected_chunk[row] = -1;
    gate_signal[row] = 0.0f;
    return;
  }
  float best_energy = -1.0f;
  float total_residual_energy = 0.0f;
  float current_energy = 0.0f;
  int best_chunk = 0;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    float chunk_energy = 0.0f;
    for (int group = 0; group < group_count; ++group) {
      int query_offset = (row * group_count + group) * projection_dim;
      int weight_offset = row * projection_dim;
      for (int local_dimension = 0; local_dimension < 16; ++local_dimension) {
        int dimension = chunk * 16 + local_dimension;
        float current = static_cast<float>(projected_query[query_offset + dimension]);
        float anchor = static_cast<float>(anchor_query[query_offset + dimension]);
        float weight = spectral_weights[weight_offset + dimension];
        float residual = current - anchor;
        chunk_energy += residual * residual * weight;
        current_energy += current * current * weight;
      }
    }
    total_residual_energy += chunk_energy;
    if (chunk_energy > best_energy) {
      best_energy = chunk_energy;
      best_chunk = chunk;
    }
  }
  selected_chunk[row] = best_chunk;
  gate_signal[row] = sqrtf(
      fmaxf(total_residual_energy - best_energy, 0.0f)
      / fmaxf(current_energy, 1.0e-12f));
}

__device__ __forceinline__ float qabs_tail_density_crossing_risk(
    const float* __restrict__ top_values,
    int keep_count,
    int candidate_count,
    int total_token_count,
    float sigma) {
  int density_start = keep_count + (candidate_count - keep_count) / 2;
  if (density_start >= candidate_count) {
    density_start = max(1, candidate_count / 2);
  }
  float target_floor = top_values[keep_count - 1];
  float density_ceiling = top_values[density_start - 1];
  float candidate_floor = top_values[candidate_count - 1];
  float local_density = static_cast<float>(candidate_count - density_start)
      / fmaxf(density_ceiling - candidate_floor, 1.0e-12f);
  float normalized_gap = fmaxf(target_floor - candidate_floor, 0.0f)
      / (CUDART_SQRT_TWO_F * fmaxf(sigma, 1.0e-12f));
  float gaussian_pdf = expf(-0.5f * normalized_gap * normalized_gap)
      * 0.3989422804014327f;
  float gaussian_tail = 0.5f * erfcf(normalized_gap * 0.7071067811865475f);
  float integrated_tail = CUDART_SQRT_TWO_F * fmaxf(sigma, 1.0e-12f)
      * fmaxf(gaussian_pdf - normalized_gap * gaussian_tail, 0.0f);
  return fminf(
      local_density * integrated_tail,
      static_cast<float>(total_token_count - candidate_count));
}

template <typename scalar_t>
__global__ void qabs_one_shot_band_plan_kernel(
    const scalar_t* __restrict__ projected_query,
    const float* __restrict__ spectral_weights,
    const scalar_t* __restrict__ anchor_query,
    const float* __restrict__ top_values,
    const int64_t* __restrict__ keep_counts,
    int32_t* __restrict__ planned_bands,
    float* __restrict__ crossing_risk,
    int kv_head_count,
    int group_count,
    int projection_dim,
    int candidate_count,
    int total_token_count,
    float target_recall) {
  int row = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  constexpr int max_chunks = QABS_MAX_DIMS / 16;
  float group_band_energy[8][max_chunks];
  int chunk_count = projection_dim / 16;
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;

  for (int group = 0; group < group_count; ++group) {
    int query_offset = (row * group_count + group) * projection_dim;
    int weight_offset = row * projection_dim;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      float energy = 0.0f;
#pragma unroll
      for (int local_dimension = 0; local_dimension < 16; ++local_dimension) {
        int dimension = chunk * 16 + local_dimension;
        float residual = static_cast<float>(projected_query[query_offset + dimension])
            - static_cast<float>(anchor_query[query_offset + dimension]);
        energy += residual * residual * spectral_weights[weight_offset + dimension];
      }
      group_band_energy[group][chunk] = energy;
    }
  }

  int planned = 1;
  for (int round = 0; round < chunk_count - 1; ++round) {
    bool active = false;
    for (int group = 0; group < group_count; ++group) {
      float residual_variance = 0.0f;
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        residual_variance += group_band_energy[group][chunk];
      }
      int query_head = kv_head * group_count + group;
      int score_row = batch * query_head_count + query_head;
      int keep_count = min(
          max(static_cast<int>(keep_counts[score_row]), 1),
          candidate_count);
      float risk = qabs_tail_density_crossing_risk(
          top_values + score_row * candidate_count,
          keep_count,
          candidate_count,
          total_token_count,
          sqrtf(fmaxf(residual_variance, 0.0f)));
      crossing_risk[score_row] = risk;
      active |= risk > (1.0f - target_recall) * static_cast<float>(keep_count);
    }
    if (!active) {
      break;
    }
    int selected_chunk = 0;
    float best_energy = -1.0f;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      float energy = 0.0f;
      for (int group = 0; group < group_count; ++group) {
        energy += group_band_energy[group][chunk];
      }
      if (energy > best_energy) {
        best_energy = energy;
        selected_chunk = chunk;
      }
    }
    for (int group = 0; group < group_count; ++group) {
      group_band_energy[group][selected_chunk] = 0.0f;
    }
    ++planned;
  }

  for (int group = 0; group < group_count; ++group) {
    float residual_variance = 0.0f;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      residual_variance += group_band_energy[group][chunk];
    }
    int query_head = kv_head * group_count + group;
    int score_row = batch * query_head_count + query_head;
    int keep_count = min(
        max(static_cast<int>(keep_counts[score_row]), 1),
        candidate_count);
    crossing_risk[score_row] = qabs_tail_density_crossing_risk(
        top_values + score_row * candidate_count,
        keep_count,
        candidate_count,
        total_token_count,
        sqrtf(fmaxf(residual_variance, 0.0f)));
  }
  planned_bands[row] = planned;
}

template <typename scalar_t>
__global__ void qabs_pca_int4_chunked_band_delta_add_kernel(
    const scalar_t* __restrict__ projected_query,
    const scalar_t* __restrict__ anchor_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scalar_t* __restrict__ scales,
    const int32_t* __restrict__ selected_chunk,
    scalar_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count) {
  __shared__ float shared_delta[8 * 16];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  int chunk = selected_chunk[row];
  if (chunk < 0) {
    return;
  }
  int start_dim = chunk * 16;
  for (int index = threadIdx.x; index < group_count * 16; index += blockDim.x) {
    int group = index / 16;
    int local_dimension = index - group * 16;
    int query_index = row * group_count * projection_dim
        + group * projection_dim + start_dim + local_dimension;
    shared_delta[index] = static_cast<float>(projected_query[query_index])
        - static_cast<float>(anchor_query[query_index]);
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  const uint8_t* key_chunk = packed_key_chunked
      + ((row * chunk_count + chunk) * capacity + token) * 8;
  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int byte_index = 0; byte_index < 8; ++byte_index) {
    uint8_t packed = key_chunk[byte_index];
    float low = static_cast<float>(static_cast<int>(packed & 0x0F) - 7);
    float high = static_cast<float>(static_cast<int>(packed >> 4) - 7);
    int dimension = byte_index * 2;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        const float* query_row = shared_delta + group * 16;
        accumulators[group] += query_row[dimension] * low;
        accumulators[group] += query_row[dimension + 1] * high;
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    score_cache[cache_index] = static_cast<scalar_t>(
        static_cast<float>(score_cache[cache_index])
        + accumulators[group] * key_scale);
  }
}

template <typename scalar_t>
__global__ void qabs_pca_int4_chunked_logscale16_band_delta_add_kernel(
    const scalar_t* __restrict__ projected_query,
    const scalar_t* __restrict__ anchor_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scalar_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const int32_t* __restrict__ selected_chunk,
    scalar_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count) {
  __shared__ float shared_delta[8 * 16];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  int chunk = selected_chunk[row];
  if (chunk < 0) {
    return;
  }
  int start_dim = chunk * 16;
  for (int index = threadIdx.x; index < group_count * 16; index += blockDim.x) {
    int group = index / 16;
    int local_dimension = index - group * 16;
    int query_index = row * group_count * projection_dim
        + group * projection_dim + start_dim + local_dimension;
    shared_delta[index] = static_cast<float>(projected_query[query_index])
        - static_cast<float>(anchor_query[query_index]);
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  const uint8_t* key_chunk = packed_key_chunked
      + ((row * chunk_count + chunk) * capacity + token) * 8;
  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int byte_index = 0; byte_index < 8; ++byte_index) {
    uint8_t packed = key_chunk[byte_index];
    float low = static_cast<float>(static_cast<int>(packed & 0x0F) - 7);
    float high = static_cast<float>(static_cast<int>(packed >> 4) - 7);
    int dimension = byte_index * 2;
#pragma unroll
    for (int group = 0; group < 8; ++group) {
      if (group < group_count) {
        const float* query_row = shared_delta + group * 16;
        accumulators[group] += query_row[dimension] * low;
        accumulators[group] += query_row[dimension + 1] * high;
      }
    }
  }

  int exponent_pair_count = (chunk_count + 1) / 2;
  uint8_t exponent_pair = packed_exponents[
      (row * capacity + token) * exponent_pair_count + chunk / 2];
  int exponent = (chunk & 1) == 0
      ? static_cast<int>(exponent_pair & 0x0F)
      : static_cast<int>(exponent_pair >> 4);
  float key_scale = static_cast<float>(base_scales[row * capacity + token])
      * qabs_quarter_logscale(exponent);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    score_cache[cache_index] = static_cast<scalar_t>(
        static_cast<float>(score_cache[cache_index])
        + accumulators[group] * key_scale);
  }
}

template <typename scalar_t>
__global__ void qabs_spectral_band_anchor_update_kernel(
    const scalar_t* __restrict__ projected_query,
    const int32_t* __restrict__ selected_chunk,
    scalar_t* __restrict__ anchor_query,
    int group_count,
    int projection_dim) {
  int row = blockIdx.x;
  int chunk = selected_chunk[row];
  if (chunk < 0) {
    return;
  }
  for (int index = threadIdx.x; index < group_count * 16; index += blockDim.x) {
    int group = index / 16;
    int local_dimension = index - group * 16;
    int query_index = row * group_count * projection_dim
        + group * projection_dim + chunk * 16 + local_dimension;
    anchor_query[query_index] = projected_query[query_index];
  }
}

template <typename scalar_t>
__global__ void qabs_spectral_residual_gate_kernel(
    const scalar_t* __restrict__ projected_query,
    const float* __restrict__ spectral_weights,
    scalar_t* __restrict__ anchor_query,
    uint8_t* __restrict__ refresh_mask,
    float* __restrict__ gate_signal,
    int group_count,
    int projection_dim,
    int omitted_dim,
    float threshold) {
  int row = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  float group_risk_sum = 0.0f;
  for (int group = 0; group < group_count; ++group) {
    float residual_energy = 0.0f;
    float current_energy = 0.0f;
    int query_offset = (row * group_count + group) * projection_dim;
    int weight_offset = row * projection_dim;
    for (int dimension = 0; dimension < projection_dim; ++dimension) {
      float current = static_cast<float>(
          projected_query[query_offset + dimension]);
      float weight = spectral_weights[weight_offset + dimension];
      current_energy += current * current * weight;
      if (dimension < omitted_dim) {
        float residual = current - static_cast<float>(
            anchor_query[query_offset + dimension]);
        residual_energy += residual * residual * weight;
      }
    }
    group_risk_sum += sqrtf(
        residual_energy / fmaxf(current_energy, 1.0e-12f));
  }
  float risk = group_risk_sum / static_cast<float>(group_count);
  bool refresh = risk > threshold;
  refresh_mask[row] = refresh ? 1 : 0;
  gate_signal[row] = risk;
  if (refresh) {
    int query_count = group_count * projection_dim;
    for (int index = 0; index < query_count; ++index) {
      anchor_query[row * query_count + index] =
          projected_query[row * query_count + index];
    }
  }
}

template <typename scalar_t>
__global__ void qabs_spectral_residual_topk_gate_kernel(
    const scalar_t* __restrict__ projected_query,
    const float* __restrict__ spectral_weights,
    scalar_t* __restrict__ anchor_query,
    uint8_t* __restrict__ refresh_mask,
    float* __restrict__ gate_signal,
    int32_t* __restrict__ refresh_indices,
    int kv_head_count,
    int group_count,
    int projection_dim,
    int omitted_dim,
    int refresh_count) {
  int batch = blockIdx.x;
  extern __shared__ float query_risk[];
  int row_offset = batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int query_head = threadIdx.x; query_head < query_head_count;
       query_head += blockDim.x) {
    int kv_head = query_head / group_count;
    int group = query_head - kv_head * group_count;
    int row = row_offset + kv_head;
    float residual_energy = 0.0f;
    float current_energy = 0.0f;
    int query_offset = (row * group_count + group) * projection_dim;
    int weight_offset = row * projection_dim;
    for (int dimension = 0; dimension < projection_dim; ++dimension) {
      float current = static_cast<float>(
          projected_query[query_offset + dimension]);
      float weight = spectral_weights[weight_offset + dimension];
      current_energy += current * current * weight;
      if (dimension < omitted_dim) {
        float residual = current - static_cast<float>(
            anchor_query[query_offset + dimension]);
        residual_energy += residual * residual * weight;
      }
    }
    query_risk[query_head] = sqrtf(
        residual_energy / fmaxf(current_energy, 1.0e-12f));
  }
  __syncthreads();
  for (int kv_head = threadIdx.x; kv_head < kv_head_count;
       kv_head += blockDim.x) {
    float group_risk_sum = 0.0f;
    for (int group = 0; group < group_count; ++group) {
      group_risk_sum += query_risk[kv_head * group_count + group];
    }
    int row = row_offset + kv_head;
    gate_signal[row] = group_risk_sum / static_cast<float>(group_count);
    refresh_mask[row] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int selection = 0; selection < refresh_count; ++selection) {
      int best_head = -1;
      float best_risk = -1.0f;
      for (int kv_head = 0; kv_head < kv_head_count; ++kv_head) {
        int row = row_offset + kv_head;
        if (refresh_mask[row] == 0 && gate_signal[row] > best_risk) {
          best_risk = gate_signal[row];
          best_head = kv_head;
        }
      }
      if (best_head >= 0) {
        refresh_mask[row_offset + best_head] = 1;
        refresh_indices[batch * refresh_count + selection] = best_head;
      }
    }
  }
  __syncthreads();
  int element_count = query_head_count * projection_dim;
  for (int index = threadIdx.x; index < element_count; index += blockDim.x) {
    int query_head = index / projection_dim;
    int kv_head = query_head / group_count;
    int row = row_offset + kv_head;
    if (refresh_mask[row] != 0) {
      int batch_offset = batch * element_count;
      anchor_query[batch_offset + index] = projected_query[batch_offset + index];
    }
  }
}

template <typename scalar_t>
__global__ void qabs_pca_int4_chunked_mixed_delta_add_kernel(
    const scalar_t* __restrict__ projected_query,
    const scalar_t* __restrict__ previous_projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ refresh_mask,
    scalar_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int key_count,
    int capacity,
    int projection_dim,
    int tail_start_chunk,
    int chunk_count) {
  __shared__ float shared_query[8 * QABS_MAX_DIMS];
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  bool refresh = refresh_mask[row] != 0;
  int start_chunk = refresh ? 0 : tail_start_chunk;
  int start_dim = start_chunk * 16;
  int selected_count = refresh ? projection_dim : projection_dim - start_dim;
  for (int index = threadIdx.x; index < group_count * selected_count; index += blockDim.x) {
    int group = index / selected_count;
    int local_dimension = index - group * selected_count;
    int query_index = row * group_count * projection_dim
        + group * projection_dim + start_dim + local_dimension;
    float current = static_cast<float>(projected_query[query_index]);
    shared_query[index] = refresh
        ? current
        : current - static_cast<float>(previous_projected_query[query_index]);
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  int selected_chunk_count = selected_count / 16;
  for (int chunk = 0; chunk < selected_chunk_count; ++chunk) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + start_chunk + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      float low = static_cast<float>(static_cast<int>(packed & 0x0F) - 7);
      float high = static_cast<float>(static_cast<int>(packed >> 4) - 7);
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const float* query_row = shared_query + group * selected_count;
          accumulators[group] += query_row[dimension] * low;
          accumulators[group] += query_row[dimension + 1] * high;
        }
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int batch = row / kv_head_count;
  int kv_head = row - batch * kv_head_count;
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    float score = accumulators[group] * key_scale;
    score_cache[cache_index] = static_cast<scalar_t>(
        refresh
            ? score
            : static_cast<float>(score_cache[cache_index]) + score);
  }
}

template <typename scalar_t>
__global__ void qabs_pca_int4_chunked_selective_full_scores_kernel(
    const scalar_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scalar_t* __restrict__ scales,
    const int32_t* __restrict__ refresh_indices,
    scalar_t* __restrict__ score_cache,
    int kv_head_count,
    int group_count,
    int refresh_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count) {
  __shared__ float shared_query[8 * QABS_MAX_DIMS];
  int refresh_row = blockIdx.x;
  int batch = refresh_row / refresh_count;
  int selection = refresh_row - batch * refresh_count;
  int kv_head = refresh_indices[batch * refresh_count + selection];
  int row = batch * kv_head_count + kv_head;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  for (int index = threadIdx.x; index < group_count * projection_dim; index += blockDim.x) {
    shared_query[index] = static_cast<float>(
        projected_query[row * group_count * projection_dim + index]);
  }
  __syncthreads();
  if (token >= key_count) {
    return;
  }

  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const uint8_t* key_chunk = packed_key_chunked
        + ((row * chunk_count + chunk) * capacity + token) * 8;
#pragma unroll
    for (int byte_index = 0; byte_index < 8; ++byte_index) {
      uint8_t packed = key_chunk[byte_index];
      float low = static_cast<float>(static_cast<int>(packed & 0x0F) - 7);
      float high = static_cast<float>(static_cast<int>(packed >> 4) - 7);
      int dimension = chunk * 16 + byte_index * 2;
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        if (group < group_count) {
          const float* query_row = shared_query + group * projection_dim;
          accumulators[group] += query_row[dimension] * low;
          accumulators[group] += query_row[dimension + 1] * high;
        }
      }
    }
  }

  float key_scale = static_cast<float>(scales[row * capacity + token]);
  int query_head_count = kv_head_count * group_count;
  for (int group = 0; group < group_count; ++group) {
    int query_head = kv_head * group_count + group;
    int cache_index = (batch * query_head_count + query_head) * capacity + token;
    score_cache[cache_index] = static_cast<scalar_t>(
        accumulators[group] * key_scale);
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_candidate_range_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ scales,
    const int64_t* __restrict__ candidate_indices,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int candidate_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int start_dim,
    int end_dim) {
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int candidates_per_warp = 4;
  __shared__ int8_t shared_query[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int kv_row = batch * kv_head_count + kv_head;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  const int8_t* query_source = projected_query
      + (kv_row * group_count + group) * projection_dim;
  for (int dim = threadIdx.x; dim < projection_dim; dim += blockDim.x) {
    shared_query[dim] = query_source[dim];
  }
  __syncthreads();

  int tile_start = blockIdx.y * warps_per_block * candidates_per_warp;
  int start_chunk = start_dim / 16;
  int end_chunk = end_dim / 16;
  for (int iteration = 0; iteration < candidates_per_warp; ++iteration) {
    int candidate = tile_start + iteration * warps_per_block + warp;
    if (candidate >= candidate_count) {
      continue;
    }
    int64_t token = candidate_indices[query_row * candidate_count + candidate];
    if (token < 0 || token >= key_count) {
      if (lane == 0) {
        output[query_row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    int accumulator = 0;
    if (lane < 8) {
      for (int chunk = start_chunk; chunk < end_chunk; ++chunk) {
        uint8_t packed = packed_key_chunked[
            ((kv_row * chunk_count + chunk) * capacity + token) * 8 + lane];
        int dimension = chunk * 16 + lane * 2;
        accumulator += static_cast<int>(shared_query[dimension])
            * (static_cast<int>(packed & 0x0F) - 7);
        accumulator += static_cast<int>(shared_query[dimension + 1])
            * (static_cast<int>(packed >> 4) - 7);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulator += __shfl_down_sync(0xffffffff, accumulator, offset);
    }
    if (lane == 0) {
      float scale = static_cast<float>(scales[kv_row * capacity + token]);
      output[query_row * candidate_count + candidate]
          = static_cast<float>(accumulator) * scale;
    }
  }
}

template <typename scale_t>
__global__ void qabs_pca_int4_chunked_logscale16_candidate_range_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const int64_t* __restrict__ candidate_indices,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int candidate_count,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int start_dim,
    int end_dim) {
  __shared__ int8_t shared_query[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int kv_row = batch * kv_head_count + kv_head;
  const int8_t* query_source = projected_query
      + (kv_row * group_count + group) * projection_dim;
  for (int dim = threadIdx.x; dim < projection_dim; dim += blockDim.x) {
    shared_query[dim] = query_source[dim];
  }
  __syncthreads();

  int candidate = blockIdx.y * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  int start_chunk = start_dim / 16;
  int end_chunk = end_dim / 16;
  int exponent_pairs = (chunk_count + 1) / 2;
  int64_t token = candidate_indices[query_row * candidate_count + candidate];
  if (token < 0 || token >= key_count) {
    output[query_row * candidate_count + candidate] = -CUDART_INF_F;
    return;
  }
  float total = 0.0f;
  float base_scale = static_cast<float>(base_scales[kv_row * capacity + token]);
  const uint8_t* exponent_row = packed_exponents
      + (kv_row * capacity + token) * exponent_pairs;
  for (int chunk = start_chunk; chunk < end_chunk; ++chunk) {
    int accumulator = 0;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      uint8_t packed = packed_key_chunked[
          ((kv_row * chunk_count + chunk) * capacity + token) * 8 + lane];
      int dimension = chunk * 16 + lane * 2;
      accumulator += static_cast<int>(shared_query[dimension])
          * (static_cast<int>(packed & 0x0F) - 7);
      accumulator += static_cast<int>(shared_query[dimension + 1])
          * (static_cast<int>(packed >> 4) - 7);
    }
    uint8_t exponent_pair = exponent_row[chunk / 2];
    int exponent = (chunk & 1) == 0
        ? static_cast<int>(exponent_pair & 0x0F)
        : static_cast<int>(exponent_pair >> 4);
    total += static_cast<float>(accumulator)
        * base_scale * qabs_quarter_logscale(exponent);
  }
  output[query_row * candidate_count + candidate] = total;
}

template <typename scalar_t>
__global__ void qabs_microblock_expected_max_scores_kernel(
    const scalar_t* __restrict__ block_mean,
    const scalar_t* __restrict__ block_variance,
    const scalar_t* __restrict__ projected_query,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int capacity_blocks,
    int block_count,
    int projection_dim,
    int block_size,
    int last_block_size) {
  __shared__ float shared_query[QABS_MAX_DIMS];
  __shared__ float shared_query_squared[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int kv_row = batch * kv_head_count + kv_head;
  const scalar_t* query_source = projected_query
      + (kv_row * group_count + group) * projection_dim;
  for (int dimension = threadIdx.x;
       dimension < projection_dim;
       dimension += blockDim.x) {
    float value = static_cast<float>(query_source[dimension]);
    shared_query[dimension] = value;
    shared_query_squared[dimension] = value * value;
  }
  __syncthreads();

  int block = blockIdx.y * blockDim.x + threadIdx.x;
  if (block >= block_count) {
    return;
  }
  const scalar_t* mean_source = block_mean
      + (kv_row * capacity_blocks + block) * projection_dim;
  const scalar_t* variance_source = block_variance
      + (kv_row * capacity_blocks + block) * projection_dim;
  float center = 0.0f;
  float directional_variance = 0.0f;
  for (int dimension = 0; dimension < projection_dim; ++dimension) {
    center = fmaf(
        static_cast<float>(mean_source[dimension]),
        shared_query[dimension],
        center);
    directional_variance = fmaf(
        static_cast<float>(variance_source[dimension]),
        shared_query_squared[dimension],
        directional_variance);
  }
  int length = block == block_count - 1 ? last_block_size : block_size;
  float multiplier = sqrtf(2.0f * logf(static_cast<float>(max(length, 2))));
  output[query_row * block_count + block]
      = center + sqrtf(max(directional_variance, 0.0f)) * multiplier;
}

template <typename scalar_t>
__global__ void qabs_microblock_q8_expected_max_scores_kernel(
    const int8_t* __restrict__ block_mean_q8,
    const scalar_t* __restrict__ block_mean_scales,
    const uint8_t* __restrict__ block_variance_q8,
    const scalar_t* __restrict__ block_variance_scales,
    const scalar_t* __restrict__ projected_query,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int capacity_blocks,
    int block_count,
    int projection_dim,
    int block_size,
    int last_block_size) {
  __shared__ float shared_query[QABS_MAX_DIMS];
  __shared__ float shared_query_squared[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int kv_row = batch * kv_head_count + kv_head;
  const scalar_t* query_source = projected_query
      + (kv_row * group_count + group) * projection_dim;
  for (int dimension = threadIdx.x;
       dimension < projection_dim;
       dimension += blockDim.x) {
    float value = static_cast<float>(query_source[dimension]);
    shared_query[dimension] = value;
    shared_query_squared[dimension] = value * value;
  }
  __syncthreads();

  int block = blockIdx.y * blockDim.x + threadIdx.x;
  if (block >= block_count) {
    return;
  }
  int summary_row = kv_row * capacity_blocks + block;
  const int8_t* mean_source =
      block_mean_q8 + summary_row * projection_dim;
  const uint8_t* variance_source =
      block_variance_q8 + summary_row * projection_dim;
  float center_code_sum = 0.0f;
  float variance_code_sum = 0.0f;
  for (int dimension = 0; dimension < projection_dim; ++dimension) {
    center_code_sum = fmaf(
        static_cast<float>(mean_source[dimension]),
        shared_query[dimension],
        center_code_sum);
    variance_code_sum = fmaf(
        static_cast<float>(variance_source[dimension]),
        shared_query_squared[dimension],
        variance_code_sum);
  }
  float center = center_code_sum
      * static_cast<float>(block_mean_scales[summary_row]);
  float directional_variance = variance_code_sum
      * static_cast<float>(block_variance_scales[summary_row]);
  int length = block == block_count - 1 ? last_block_size : block_size;
  float multiplier = sqrtf(2.0f * logf(static_cast<float>(max(length, 2))));
  output[query_row * block_count + block]
      = center + sqrtf(max(directional_variance, 0.0f)) * multiplier;
}

template <typename scale_t>
__global__ void qabs_pca_int4_logscale16_selected_block_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const float* __restrict__ query_scales,
    const uint8_t* __restrict__ packed_key_chunked,
    const scale_t* __restrict__ base_scales,
    const uint8_t* __restrict__ packed_exponents,
    const int64_t* __restrict__ selected_blocks,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int selected_block_count,
    int block_size,
    int key_count,
    int capacity,
    int projection_dim,
    int chunk_count,
    int start_dim,
    int end_dim) {
  __shared__ int8_t shared_query[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int kv_row = batch * kv_head_count + kv_head;
  const int8_t* query_source = projected_query
      + (kv_row * group_count + group) * projection_dim;
  for (int dimension = threadIdx.x;
       dimension < projection_dim;
       dimension += blockDim.x) {
    shared_query[dimension] = query_source[dimension];
  }
  __syncthreads();

  int candidate_count = selected_block_count * block_size;
  int candidate = blockIdx.y * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  int block_slot = candidate / block_size;
  int offset = candidate - block_slot * block_size;
  int64_t selected_block =
      selected_blocks[query_row * selected_block_count + block_slot];
  int64_t token = selected_block * block_size + offset;
  if (token < 0 || token >= key_count) {
    output[query_row * candidate_count + candidate] = -CUDART_INF_F;
    return;
  }
  int start_chunk = start_dim / 16;
  int end_chunk = end_dim / 16;
  int exponent_pairs = (chunk_count + 1) / 2;
  float base_scale = static_cast<float>(base_scales[kv_row * capacity + token]);
  const uint8_t* exponent_row = packed_exponents
      + (kv_row * capacity + token) * exponent_pairs;
  float total = 0.0f;
  for (int chunk = start_chunk; chunk < end_chunk; ++chunk) {
    int accumulator = 0;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      uint8_t packed = packed_key_chunked[
          ((kv_row * chunk_count + chunk) * capacity + token) * 8 + lane];
      int dimension = chunk * 16 + lane * 2;
      accumulator += static_cast<int>(shared_query[dimension])
          * (static_cast<int>(packed & 0x0F) - 7);
      accumulator += static_cast<int>(shared_query[dimension + 1])
          * (static_cast<int>(packed >> 4) - 7);
    }
    uint8_t exponent_pair = exponent_row[chunk / 2];
    int exponent = (chunk & 1) == 0
        ? static_cast<int>(exponent_pair & 0x0F)
        : static_cast<int>(exponent_pair >> 4);
    total += static_cast<float>(accumulator)
        * base_scale * qabs_quarter_logscale(exponent);
  }
  output[query_row * candidate_count + candidate]
      = total * query_scales[query_row];
}

__global__ void qabs_microblock_local_to_token_indices_kernel(
    const int64_t* __restrict__ selected_blocks,
    const int64_t* __restrict__ local_indices,
    int64_t* __restrict__ output,
    int selected_block_count,
    int candidate_count,
    int key_count,
    int block_size) {
  int query_row = blockIdx.x;
  int candidate = blockIdx.y * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  int64_t local = local_indices[query_row * candidate_count + candidate];
  int64_t block_slot = local / block_size;
  int64_t offset = local - block_slot * block_size;
  int64_t block = selected_blocks[
      query_row * selected_block_count + block_slot];
  output[query_row * candidate_count + candidate]
      = min(block * block_size + offset, static_cast<int64_t>(key_count - 1));
}

template <typename scale_t>
__global__ void qabs_pca_int4_candidate_range_scores_kernel(
    const int8_t* __restrict__ projected_query,
    const uint8_t* __restrict__ packed_key,
    const scale_t* __restrict__ scales,
    const int64_t* __restrict__ candidate_indices,
    float* __restrict__ output,
    int kv_head_count,
    int group_count,
    int candidate_count,
    int key_count,
    int capacity,
    int projection_dim,
    int packed_dim,
    int start_dim,
    int end_dim) {
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int candidates_per_warp = 4;
  __shared__ int8_t shared_query[QABS_MAX_DIMS];
  int query_row = blockIdx.x;
  int query_head_count = kv_head_count * group_count;
  int batch = query_row / query_head_count;
  int query_head = query_row - batch * query_head_count;
  int kv_head = query_head / group_count;
  int group = query_head - kv_head * group_count;
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  const int8_t* query_source = projected_query
      + ((batch * kv_head_count + kv_head) * group_count + group) * projection_dim;
  for (int dim = threadIdx.x; dim < projection_dim; dim += blockDim.x) {
    shared_query[dim] = query_source[dim];
  }
  __syncthreads();

  int tile_start = blockIdx.y * warps_per_block * candidates_per_warp;
  int quad_count = (end_dim - start_dim) / 4;
  for (int iteration = 0; iteration < candidates_per_warp; ++iteration) {
    int candidate = tile_start + iteration * warps_per_block + warp;
    if (candidate >= candidate_count) {
      continue;
    }
    int64_t token = candidate_indices[query_row * candidate_count + candidate];
    if (token < 0 || token >= key_count) {
      if (lane == 0) {
        output[query_row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    int accumulator = 0;
    const uint8_t* key_row = packed_key
        + ((batch * kv_head_count + kv_head) * capacity + token) * packed_dim
        + start_dim / 2;
    for (int quad = lane; quad < quad_count; quad += 32) {
      uint8_t first = key_row[2 * quad];
      uint8_t second = key_row[2 * quad + 1];
      uint32_t packed_key_values =
          static_cast<uint8_t>((first & 0x0F) - 7)
          | (static_cast<uint32_t>(static_cast<uint8_t>((first >> 4) - 7)) << 8)
          | (static_cast<uint32_t>(static_cast<uint8_t>((second & 0x0F) - 7)) << 16)
          | (static_cast<uint32_t>(static_cast<uint8_t>((second >> 4) - 7)) << 24);
      const int8_t* query_values = shared_query + start_dim + 4 * quad;
      int packed_query_values = *reinterpret_cast<const int*>(query_values);
      accumulator = __dp4a(
          packed_query_values,
          static_cast<int>(packed_key_values),
          accumulator);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulator += __shfl_down_sync(0xffffffff, accumulator, offset);
    }
    if (lane == 0) {
      float scale = static_cast<float>(
          scales[(batch * kv_head_count + kv_head) * capacity + token]);
      output[query_row * candidate_count + candidate]
          = static_cast<float>(accumulator) * scale;
    }
  }
}

__device__ __forceinline__ float qabs_int2_level(uint8_t code) {
  if (code == 0) {
    return -1.2711063f;
  }
  if (code == 1) {
    return -0.3246628f;
  }
  if (code == 2) {
    return 0.3246628f;
  }
  return 1.2711063f;
}

template <typename scalar_t>
__global__ void qabs_pack_int2_kernel(
    const scalar_t* __restrict__ key,
    uint8_t* __restrict__ packed_key,
    scalar_t* __restrict__ scales,
    int key_count,
    int head_dim,
    int group_count,
    int scale_group_count) {
  __shared__ float token_group_scales[4][4];
  int flat_block = blockIdx.x;
  int row = flat_block / group_count;
  int group = flat_block - row * group_count;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  int first_token = group * 4;
  const scalar_t* key_row = key + row * key_count * head_dim;
  scalar_t* scale_row = scales + row * key_count * scale_group_count;

  for (int offset = 0; offset < 4; ++offset) {
    int token = first_token + offset;
    float local_square = 0.0f;
    if (token < key_count && tid < head_dim) {
      float value = static_cast<float>(key_row[token * head_dim + tid]);
      local_square = value * value;
    }
    for (int stride = 16; stride > 0; stride >>= 1) {
      local_square += __shfl_down_sync(0xffffffff, local_square, stride);
    }
    if (lane == 0 && warp < scale_group_count) {
      int dimensions = min(32, head_dim - warp * 32);
      float scale = fmaxf(
          sqrtf(local_square / static_cast<float>(dimensions)), 1.0e-8f);
      token_group_scales[offset][warp] = scale;
      if (token < key_count) {
        scale_row[token * scale_group_count + warp] = static_cast<scalar_t>(scale);
      }
    }
    __syncthreads();
  }

  if (tid < head_dim) {
    uint8_t packed = 0;
    for (int offset = 0; offset < 4; ++offset) {
      int token = first_token + offset;
      uint8_t code = 0;
      if (token < key_count) {
        float normalized = static_cast<float>(key_row[token * head_dim + tid])
            / token_group_scales[offset][tid / 32];
        code = normalized < -0.7978846f ? 0 : (normalized < 0.0f ? 1 : (normalized < 0.7978846f ? 2 : 3));
      }
      packed |= static_cast<uint8_t>(code << (2 * offset));
    }
    packed_key[(row * head_dim + tid) * group_count + group] = packed;
  }
}

template <typename scalar_t>
__global__ void qabs_partial_scores_int2_kernel(
    const scalar_t* __restrict__ query,
    const uint8_t* __restrict__ packed_key,
    const scalar_t* __restrict__ scales,
    const int64_t* __restrict__ dim_indices,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int selected_count,
    int group_count,
    int scale_group_count) {
  int row = blockIdx.x;
  int token = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= key_count) {
    return;
  }
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* scale_row = scales + row * key_count * scale_group_count;
  const int64_t* dim_row = dim_indices + row * selected_count;
  int group = token / 4;
  int shift = 2 * (token % 4);
  float acc = 0.0f;
  for (int selected = 0; selected < selected_count; ++selected) {
    int64_t dim = dim_row[selected];
    float scale = static_cast<float>(
        scale_row[token * scale_group_count + dim / 32]);
    uint8_t packed = packed_key[(row * head_dim + dim) * group_count + group];
    uint8_t code = static_cast<uint8_t>((packed >> shift) & 0x3);
    acc += static_cast<float>(query_row[dim]) * qabs_int2_level(code) * scale;
  }
  output[row * key_count + token] = acc;
}

template <typename scalar_t>
__global__ void qabs_partial_scores_int2_onthefly_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int selected_count,
    int head_count) {
  __shared__ int selected_idx[QABS_MAX_DIMS];
  __shared__ float selected_query[QABS_MAX_DIMS];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  int batch = row / head_count;
  int head = row - batch * head_count;
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* key_row = key + (batch * head_count + head) * key_count * head_dim;

  if (tid == 0) {
    for (int selected = 0; selected < selected_count; ++selected) {
      int best_index = 0;
      float best_abs = -1.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        bool used = false;
        for (int previous = 0; previous < selected; ++previous) {
          used = used || selected_idx[previous] == dim;
        }
        if (!used) {
          float value = static_cast<float>(query_row[dim]);
          if (fabsf(value) > best_abs) {
            best_abs = fabsf(value);
            best_index = dim;
          }
        }
      }
      selected_idx[selected] = best_index;
      selected_query[selected] = static_cast<float>(query_row[best_index]);
    }
  }
  __syncthreads();

  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int local_token = warp; local_token < QABS_TILE_THREADS; local_token += warps_per_block) {
    int token = tile_start + local_token;
    if (token >= key_count) {
      continue;
    }
    const scalar_t* token_key = key_row + token * head_dim;
    float group_scales[4];
    int scale_group_count = (head_dim + 31) / 32;
    for (int group = 0; group < scale_group_count; ++group) {
      int dim = group * 32 + lane;
      float square_sum = 0.0f;
      if (dim < head_dim) {
        float value = static_cast<float>(token_key[dim]);
        square_sum = value * value;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
      }
      int dimensions = min(32, head_dim - group * 32);
      group_scales[group] = fmaxf(
          sqrtf(__shfl_sync(0xffffffff, square_sum, 0)
                / static_cast<float>(dimensions)),
          1.0e-8f);
    }
    float acc = 0.0f;
    for (int selected = lane; selected < selected_count; selected += 32) {
      int dim = selected_idx[selected];
      float scale = group_scales[dim / 32];
      float normalized = static_cast<float>(token_key[dim]) / scale;
      uint8_t code = normalized < -0.7978846f ? 0 : (normalized < 0.0f ? 1 : (normalized < 0.7978846f ? 2 : 3));
      acc += selected_query[selected] * qabs_int2_level(code) * scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
      output[row * key_count + token] = acc;
    }
  }
}

template <typename scalar_t>
__global__ void qabs_partial_scores_dim_major_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_dim_major,
    const int64_t* __restrict__ dim_indices,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int selected_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_d,
    int64_t key_stride_k,
    int head_count) {
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int token = tile * blockDim.x + tid;
  if (token >= key_count) {
    return;
  }

  const scalar_t* q_row = query + row * head_dim;
  const int64_t* dim_row = dim_indices + row * selected_count;
  const scalar_t* k_base = key_dim_major + batch * key_stride_b + head * key_stride_h;
  float acc = 0.0f;
  for (int s = 0; s < selected_count; ++s) {
    int64_t dim = dim_row[s];
    acc += static_cast<float>(q_row[dim]) * static_cast<float>(k_base[dim * key_stride_d + token * key_stride_k]);
  }
  output[row * key_count + token] = acc;
}

template <typename scalar_t>
__global__ void qabs_partial_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float* __restrict__ output,
    int row_count,
    int key_count,
    int head_dim,
    int dim_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int head_count) {
  __shared__ int selected_idx[QABS_MAX_DIMS];
  __shared__ float selected_q[QABS_MAX_DIMS];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int selected_count = min(min(max(dim_count, 1), head_dim), QABS_MAX_DIMS);
  const scalar_t* q_row = query + row * head_dim;

  if (tid == 0) {
    for (int s = 0; s < selected_count; ++s) {
      int best_idx = 0;
      float best_abs = -1.0f;
      for (int d = 0; d < head_dim; ++d) {
        bool used = false;
        for (int u = 0; u < s; ++u) {
          used = used || selected_idx[u] == d;
        }
        if (used) {
          continue;
        }
        float value = static_cast<float>(q_row[d]);
        float magnitude = fabsf(value);
        if (magnitude > best_abs) {
          best_abs = magnitude;
          best_idx = d;
        }
      }
      selected_idx[s] = best_idx;
      selected_q[s] = static_cast<float>(q_row[best_idx]);
    }
  }
  __syncthreads();

  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int local_token = warp; local_token < QABS_TILE_THREADS; local_token += warps_per_block) {
    int token = tile_start + local_token;
    if (token >= key_count) {
      continue;
    }
    const scalar_t* k_row = key + batch * key_stride_b + head * key_stride_h + token * key_stride_k;
    float acc = 0.0f;
    for (int selected = lane; selected < selected_count; selected += 32) {
      acc += selected_q[selected] * static_cast<float>(k_row[selected_idx[selected] * key_stride_d]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
      output[row * key_count + token] = acc;
    }
  }
}

template <typename scalar_t>
__global__ void qabs_candidate_full_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const bool* __restrict__ current_candidate,
    const bool* __restrict__ previous_candidate,
    bool has_previous_candidate,
    const bool* __restrict__ previous_final,
    bool has_previous_final,
    float* __restrict__ output,
    int key_count,
    int head_dim,
    int protect_sink_tokens,
    int protect_recent_tokens,
    float scaling,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int head_count) {
  int row = blockIdx.x;
  int tile = blockIdx.y;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;
  int token = tile * blockDim.x + tid;
  if (token >= key_count) {
    return;
  }

  int offset = row * key_count + token;
  bool selected = current_candidate[offset];
  if (has_previous_candidate) {
    selected = selected || previous_candidate[offset];
  }
  if (has_previous_final) {
    selected = selected || previous_final[offset];
  }
  if (protect_sink_tokens > 0 && token < protect_sink_tokens) {
    selected = true;
  }
  if (protect_recent_tokens > 0 && token >= max(0, key_count - protect_recent_tokens)) {
    selected = true;
  }
  if (!selected) {
    output[offset] = -CUDART_INF_F;
    return;
  }

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_row = key + batch * key_stride_b + head * key_stride_h + token * key_stride_k;
  float acc = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    acc += static_cast<float>(q_row[d]) * static_cast<float>(k_row[d * key_stride_d]);
  }
  output[offset] = acc * scaling;
}

__global__ void qabs_uncertainty_band_mask_kernel(
    const float* __restrict__ candidate_scores,
    const float* __restrict__ error_sigma,
    bool* __restrict__ candidate_valid,
    int64_t* __restrict__ candidate_counts,
    float* __restrict__ proxy_boundaries,
    int candidate_count,
    int final_count,
    float confidence_width) {
  __shared__ float shared_min[QABS_TILE_THREADS];
  __shared__ float shared_max[QABS_TILE_THREADS];
  __shared__ int histogram[256];
  __shared__ int selected_count;
  __shared__ float score_cutoff;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* score_row = candidate_scores + row * candidate_count;
  bool* valid_row = candidate_valid + row * candidate_count;
  float local_min = CUDART_INF_F;
  float local_max = -CUDART_INF_F;
  for (int candidate = tid; candidate < candidate_count; candidate += blockDim.x) {
    float score = score_row[candidate];
    local_min = fminf(local_min, score);
    local_max = fmaxf(local_max, score);
  }
  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  histogram[tid] = 0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }
  float minimum = shared_min[0];
  float maximum = shared_max[0];
  float range = fmaxf(maximum - minimum, 1.0e-12f);
  for (int candidate = tid; candidate < candidate_count; candidate += blockDim.x) {
    int bin = static_cast<int>(
        floorf((score_row[candidate] - minimum) * 255.0f / range));
    bin = max(0, min(255, bin));
    atomicAdd(histogram + bin, 1);
  }
  __syncthreads();
  if (tid == 0) {
    int cumulative = 0;
    int boundary_bin = 0;
    int target = max(1, min(final_count, candidate_count));
    for (int bin = 255; bin >= 0; --bin) {
      cumulative += histogram[bin];
      if (cumulative >= target) {
        boundary_bin = bin;
        break;
      }
    }
    float boundary = minimum + range * static_cast<float>(boundary_bin) / 255.0f;
    proxy_boundaries[row] = boundary;
    score_cutoff = boundary - confidence_width * error_sigma[row];
    selected_count = 0;
  }
  __syncthreads();
  int local_count = 0;
  for (int candidate = tid; candidate < candidate_count; candidate += blockDim.x) {
    bool valid = score_row[candidate] >= score_cutoff;
    valid_row[candidate] = valid;
    local_count += static_cast<int>(valid);
  }
  atomicAdd(&selected_count, local_count);
  __syncthreads();
  if (tid == 0) {
    candidate_counts[row] = static_cast<int64_t>(selected_count);
  }
}

__device__ __forceinline__ unsigned int qabs_float_to_ordered(float value) {
  unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float qabs_ordered_to_float(unsigned int ordered) {
  unsigned int bits = (ordered & 0x80000000u)
      ? (ordered ^ 0x80000000u)
      : ~ordered;
  return __uint_as_float(bits);
}

__global__ void qabs_direct_uncertainty_candidates_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ error_sigma,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    float* __restrict__ proxy_boundaries,
    bool* __restrict__ overflow,
    int history_count,
    int final_count,
    int candidate_capacity,
    float confidence_width) {
  __shared__ int histogram[256];
  __shared__ int boundary_high;
  __shared__ int boundary_mid;
  __shared__ int count_above_high;
  __shared__ int selected_count;
  __shared__ float score_cutoff;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  const float* score_row = scores + row * history_count;
  int64_t* index_row = candidate_indices + row * candidate_capacity;

  histogram[tid] = 0;
  if (tid == 0) {
    boundary_high = 0;
    boundary_mid = 0;
    count_above_high = 0;
    selected_count = 0;
  }
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    unsigned int ordered = qabs_float_to_ordered(score_row[token]);
    atomicAdd(histogram + (ordered >> 24), 1);
  }
  __syncthreads();
  if (tid == 0) {
    int cumulative = 0;
    for (int bin = 255; bin >= 0; --bin) {
      int next = cumulative + histogram[bin];
      if (next >= final_count) {
        boundary_high = bin;
        count_above_high = cumulative;
        break;
      }
      cumulative = next;
    }
  }
  __syncthreads();

  histogram[tid] = 0;
  __syncthreads();
  for (int token = tid; token < history_count; token += blockDim.x) {
    unsigned int ordered = qabs_float_to_ordered(score_row[token]);
    if (static_cast<int>(ordered >> 24) == boundary_high) {
      atomicAdd(histogram + ((ordered >> 16) & 0xffu), 1);
    }
  }
  __syncthreads();
  if (tid == 0) {
    int target_in_bin = max(1, final_count - count_above_high);
    int cumulative = 0;
    for (int bin = 255; bin >= 0; --bin) {
      cumulative += histogram[bin];
      if (cumulative >= target_in_bin) {
        boundary_mid = bin;
        break;
      }
    }
    unsigned int ordered_boundary =
        (static_cast<unsigned int>(boundary_high) << 24)
        | (static_cast<unsigned int>(boundary_mid) << 16);
    float boundary = qabs_ordered_to_float(ordered_boundary);
    proxy_boundaries[row] = boundary;
    score_cutoff = boundary - confidence_width * error_sigma[row];
  }
  __syncthreads();

  for (int token = tid; token < history_count; token += blockDim.x) {
    bool keep = score_row[token] >= score_cutoff;
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    int base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(&selected_count, active_count);
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      int position = base + __popc(active & lower_lanes);
      if (position < candidate_capacity) {
        index_row[position] = static_cast<int64_t>(token);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    candidate_counts[row] = static_cast<int64_t>(
        min(selected_count, candidate_capacity));
    overflow[row] = selected_count > candidate_capacity;
  }
}

__global__ void qabs_sampled_quantile_candidates_kernel(
    const float* __restrict__ scores,
    int64_t* __restrict__ candidate_indices,
    int64_t* __restrict__ candidate_counts,
    float* __restrict__ boundaries,
    bool* __restrict__ overflow,
    int history_count,
    int sample_count,
    int sample_keep,
    int candidate_capacity) {
  __shared__ float samples[QABS_TILE_THREADS];
  __shared__ int selected_count;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  const float* score_row = scores + row * history_count;
  int64_t* index_row = candidate_indices + row * candidate_capacity;

  float sample = -CUDART_INF_F;
  if (tid < sample_count) {
    int64_t numerator = static_cast<int64_t>(2 * tid + 1) * history_count;
    int token = static_cast<int>(numerator / (2 * sample_count));
    sample = score_row[min(token, history_count - 1)];
  }
  samples[tid] = sample;
  __syncthreads();

  // Sort the fixed 256-entry shared array. Unused entries are -inf, so the
  // requested upper quantile remains at the end of the ascending array.
  for (int width = 2; width <= QABS_TILE_THREADS; width <<= 1) {
    for (int stride = width >> 1; stride > 0; stride >>= 1) {
      int partner = tid ^ stride;
      if (partner > tid) {
        bool ascending = (tid & width) == 0;
        float left = samples[tid];
        float right = samples[partner];
        if ((ascending && left > right) || (!ascending && left < right)) {
          samples[tid] = right;
          samples[partner] = left;
        }
      }
      __syncthreads();
    }
  }

  float boundary = samples[QABS_TILE_THREADS - sample_keep];
  if (tid == 0) {
    boundaries[row] = boundary;
    selected_count = 0;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < history_count;
       tile_start += blockDim.x) {
    int token = tile_start + tid;
    bool keep = token < history_count && score_row[token] >= boundary;
    unsigned int active = __ballot_sync(0xffffffffu, keep);
    int active_count = __popc(active);
    int base = 0;
    if (lane == 0 && active_count > 0) {
      base = atomicAdd(&selected_count, active_count);
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (keep) {
      unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
      int position = base + __popc(active & lower_lanes);
      if (position < candidate_capacity) {
        index_row[position] = static_cast<int64_t>(token);
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    candidate_counts[row] = static_cast<int64_t>(
        min(selected_count, candidate_capacity));
    overflow[row] = selected_count > candidate_capacity;
  }
}

template <typename scalar_t>
__global__ void qabs_sample_error_sigma_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const float* __restrict__ approximate_scores,
    float* __restrict__ error_sigma,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int history_count,
    int head_dim,
    int sample_count,
    int sample_stride,
    int sample_offset,
    float scaling) {
  __shared__ float warp_sums[8];
  __shared__ float warp_square_sums[8];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int group_count = query_head_count / kv_head_count;
  int kv_head = query_head / group_count;
  const scalar_t* query_row = query + row * head_dim;
  float local_sum = 0.0f;
  float local_square_sum = 0.0f;
  for (int sample = warp; sample < sample_count; sample += 8) {
    int token = sample_offset + sample * sample_stride;
    token = min(token, history_count - 1);
    const scalar_t* key_row = key
        + ((batch * kv_head_count + kv_head) * key_count + token) * head_dim;
    float dot = 0.0f;
    for (int dim = lane; dim < head_dim; dim += 32) {
      dot += static_cast<float>(query_row[dim])
          * static_cast<float>(key_row[dim]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffu, dot, offset);
    }
    if (lane == 0) {
      float proxy = approximate_scores[row * history_count + token];
      float error = (dot - proxy) * scaling;
      local_sum += error;
      local_square_sum += error * error;
    }
  }
  if (lane == 0) {
    warp_sums[warp] = local_sum;
    warp_square_sums[warp] = local_square_sum;
  }
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int index = 0; index < 8; ++index) {
      sum += warp_sums[index];
      square_sum += warp_square_sums[index];
    }
    float mean = sum / static_cast<float>(sample_count);
    float variance = fmaxf(
        square_sum / static_cast<float>(sample_count) - mean * mean,
        1.0e-16f);
    error_sigma[row] = sqrtf(variance);
  }
}

template <typename scalar_t>
__global__ void qabs_candidate_compact_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ candidate_counts,
    bool has_candidate_counts,
    const bool* __restrict__ candidate_valid,
    bool has_candidate_valid,
    float* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_count,
    int head_dim,
    float scaling) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int group_count = query_head_count / kv_head_count;
  int kv_head = query_head / group_count;
  const scalar_t* q_row = query + row * head_dim;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int local_candidate = warp; local_candidate < QABS_TILE_THREADS; local_candidate += warps_per_block) {
    int candidate = tile_start + local_candidate;
    if (candidate >= candidate_count) {
      continue;
    }
    int active_count = has_candidate_counts
        ? static_cast<int>(candidate_counts[row])
        : candidate_count;
    if (candidate >= active_count) {
      if (lane == 0) {
        output[row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    if (has_candidate_valid
        && !candidate_valid[row * candidate_count + candidate]) {
      if (lane == 0) {
        output[row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    int64_t key_index = candidate_indices[row * candidate_count + candidate];
    const scalar_t* k_row = key + ((batch * kv_head_count + kv_head) * key_count + key_index) * head_dim;
    float acc = 0.0f;
    for (int dim = lane; dim < head_dim; dim += 32) {
      acc += static_cast<float>(q_row[dim]) * static_cast<float>(k_row[dim]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
      output[row * candidate_count + candidate] = acc * scaling;
    }
  }
}

template <typename scalar_t>
__global__ void qabs_candidate_prerope_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const int64_t* __restrict__ candidate_indices,
    const float* __restrict__ phase_cosine,
    const float* __restrict__ phase_sine,
    float* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_count,
    int head_dim,
    int query_position,
    int phase_count,
    float scaling) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int group_count = query_head_count / kv_head_count;
  int kv_head = query_head / group_count;
  const scalar_t* q_row = query + row * head_dim;
  int half = head_dim / 2;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int local_candidate = warp;
       local_candidate < QABS_TILE_THREADS;
       local_candidate += warps_per_block) {
    int candidate = tile_start + local_candidate;
    if (candidate >= candidate_count) {
      continue;
    }
    int64_t key_index =
        candidate_indices[row * candidate_count + candidate];
    if (key_index < 0 || key_index >= key_count) {
      if (lane == 0) {
        output[row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    const scalar_t* k_row =
        key
        + ((batch * kv_head_count + kv_head) * key_count + key_index)
            * head_dim;
    int distance = query_position - static_cast<int>(key_index);
    if (distance < 0 || distance >= phase_count) {
      if (lane == 0) {
        output[row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    float acc = 0.0f;
    for (int pair = lane; pair < half; pair += 32) {
      float cosine = phase_cosine[distance * half + pair];
      float sine = phase_sine[distance * half + pair];
      float key_first = static_cast<float>(k_row[pair]);
      float key_second = static_cast<float>(k_row[pair + half]);
      float rotated_first = key_first * cosine - key_second * sine;
      float rotated_second = key_second * cosine + key_first * sine;
      acc += static_cast<float>(q_row[pair]) * rotated_first;
      acc += static_cast<float>(q_row[pair + half]) * rotated_second;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
      output[row * candidate_count + candidate] = acc * scaling;
    }
  }
}

template <typename scalar_t>
__global__ void qabs_candidate_compact_scores_range_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const int64_t* __restrict__ candidate_indices,
    const int64_t* __restrict__ start_counts,
    const int64_t* __restrict__ end_counts,
    float* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_count,
    int head_dim,
    float scaling) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int group_count = query_head_count / kv_head_count;
  int kv_head = query_head / group_count;
  int start = static_cast<int>(start_counts[row]);
  int end = static_cast<int>(end_counts[row]);
  const scalar_t* q_row = query + row * head_dim;
  int tile_start = blockIdx.y * QABS_TILE_THREADS;
  for (int local_candidate = warp;
       local_candidate < QABS_TILE_THREADS;
       local_candidate += warps_per_block) {
    int candidate = tile_start + local_candidate;
    if (candidate >= candidate_count || candidate < start || candidate >= end) {
      continue;
    }
    int64_t key_index = candidate_indices[row * candidate_count + candidate];
    if (key_index < 0 || key_index >= key_count) {
      if (lane == 0) {
        output[row * candidate_count + candidate] = -CUDART_INF_F;
      }
      continue;
    }
    const scalar_t* k_row =
        key + ((batch * kv_head_count + kv_head) * key_count + key_index) * head_dim;
    float acc = 0.0f;
    for (int dim = lane; dim < head_dim; dim += 32) {
      acc += static_cast<float>(q_row[dim]) * static_cast<float>(k_row[dim]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
      output[row * candidate_count + candidate] = acc * scaling;
    }
  }
}

template <typename scalar_t>
__global__ void qabs_proxy_affine_calibrated_scores_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const int64_t* __restrict__ candidate_indices,
    const float* __restrict__ proxy_scores,
    const int64_t* __restrict__ candidate_counts,
    float* __restrict__ calibrated_scores,
    float* __restrict__ calibration_parameters,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int candidate_capacity,
    int head_dim,
    int sample_count,
    float scaling) {
  extern __shared__ float shared[];
  float* proxy_sum = shared;
  float* exact_sum = proxy_sum + blockDim.x;
  float* proxy_square_sum = exact_sum + blockDim.x;
  float* proxy_exact_sum = proxy_square_sum + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int group_count = query_head_count / kv_head_count;
  int kv_head = query_head / group_count;
  int active_count = min(
      max(static_cast<int>(candidate_counts[row]), 0),
      candidate_capacity);
  int active_samples = min(active_count, sample_count);
  const scalar_t* query_row = query + row * head_dim;
  const scalar_t* key_base =
      key + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* index_row =
      candidate_indices + row * candidate_capacity;
  const float* proxy_row = proxy_scores + row * candidate_capacity;
  float* output_row = calibrated_scores + row * candidate_capacity;

  float proxy_value = 0.0f;
  float exact_value = 0.0f;
  if (tid < active_samples) {
    int64_t numerator =
        static_cast<int64_t>(2 * tid + 1) * active_count;
    int position = static_cast<int>(
        numerator / (2 * active_samples));
    position = min(position, active_count - 1);
    int64_t token = index_row[position];
    proxy_value = proxy_row[position];
    if (token >= 0 && token < key_count) {
      const scalar_t* key_row = key_base + token * head_dim;
      float accumulator = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        accumulator += static_cast<float>(query_row[dim])
            * static_cast<float>(key_row[dim]);
      }
      exact_value = accumulator * scaling;
    }
  }
  proxy_sum[tid] = proxy_value;
  exact_sum[tid] = exact_value;
  proxy_square_sum[tid] = proxy_value * proxy_value;
  proxy_exact_sum[tid] = proxy_value * exact_value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      proxy_sum[tid] += proxy_sum[tid + stride];
      exact_sum[tid] += exact_sum[tid + stride];
      proxy_square_sum[tid] += proxy_square_sum[tid + stride];
      proxy_exact_sum[tid] += proxy_exact_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float inverse_count = 1.0f / max(active_samples, 1);
    float proxy_mean = proxy_sum[0] * inverse_count;
    float exact_mean = exact_sum[0] * inverse_count;
    float proxy_variance = proxy_square_sum[0]
        - proxy_sum[0] * proxy_mean;
    float covariance = proxy_exact_sum[0]
        - proxy_sum[0] * exact_mean;
    float slope = proxy_variance > 1.0e-6f
        ? covariance / proxy_variance
        : 1.0f;
    slope = fminf(4.0f, fmaxf(0.25f, slope));
    float intercept = exact_mean - slope * proxy_mean;
    proxy_sum[0] = slope;
    exact_sum[0] = intercept;
    calibration_parameters[row * 2] = slope;
    calibration_parameters[row * 2 + 1] = intercept;
  }
  __syncthreads();
  float slope = proxy_sum[0];
  float intercept = exact_sum[0];
  for (int candidate = tid; candidate < candidate_capacity;
       candidate += blockDim.x) {
    output_row[candidate] = candidate < active_count
        ? slope * proxy_row[candidate] + intercept
        : -CUDART_INF_F;
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ valid,
    scalar_t* __restrict__ output,
    int batch_count,
    int head_count,
    int key_count,
    int select_count,
    int head_dim,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base = key + ((batch * head_count + head) * key_count) * head_dim;
  const scalar_t* v_base = value + ((batch * head_count + head) * key_count) * head_dim;
  const int64_t* idx_row = indices + row * select_count;
  const uint8_t* valid_row = valid + row * select_count;
  scalar_t* out_row = output + row * head_dim;

  for (int s = tid; s < select_count; s += blockDim.x) {
    weights[s] = 0.0f;
  }
  __syncthreads();

  float max_score = -CUDART_INF_F;
  for (int s = 0; s < select_count; ++s) {
    float local = 0.0f;
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        local += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0 && is_valid) {
      float score = reduction[0] * scaling;
      max_score = fmaxf(max_score, score);
    }
    __syncthreads();
  }

  __shared__ float shared_max;
  __shared__ float shared_denom;
  if (tid == 0) {
    shared_max = isfinite(max_score) ? max_score : 0.0f;
    shared_denom = 0.0f;
  }
  __syncthreads();

  float denom_local = 0.0f;
  for (int s = 0; s < select_count; ++s) {
    float local = 0.0f;
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        local += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
    }
    reduction[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0 && is_valid) {
      float weight = expf(reduction[0] * scaling - shared_max);
      weights[s] = weight;
      denom_local += weight;
    }
    __syncthreads();
  }
  if (tid == 0) {
    shared_denom = fmaxf(denom_local, 1.0e-20f);
  }
  __syncthreads();

  for (int s = tid; s < select_count; s += blockDim.x) {
    weights[s] = weights[s] / shared_denom;
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < select_count; ++s) {
      int64_t idx = idx_row[s];
      bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
      if (is_valid) {
        acc += weights[s] * static_cast<float>(v_base[idx * head_dim + d]);
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_token_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ valid,
    scalar_t* __restrict__ output,
    int batch_count,
    int head_count,
    int key_count,
    int select_count,
    int head_dim,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / head_count;
  int head = row - batch * head_count;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base = key + ((batch * head_count + head) * key_count) * head_dim;
  const scalar_t* v_base = value + ((batch * head_count + head) * key_count) * head_dim;
  const int64_t* idx_row = indices + row * select_count;
  const uint8_t* valid_row = valid + row * select_count;
  scalar_t* out_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int s = tid; s < select_count; s += blockDim.x) {
    int64_t idx = idx_row[s];
    bool is_valid = valid_row[s] != 0 && idx >= 0 && idx < key_count;
    float score = -CUDART_INF_F;
    if (is_valid) {
      const scalar_t* k_vec = k_base + idx * head_dim;
      float acc = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        acc += static_cast<float>(q_row[d]) * static_cast<float>(k_vec[d]);
      }
      score = acc * scaling;
      local_max = fmaxf(local_max, score);
    }
    weights[s] = score;
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_denom = 0.0f;
  for (int s = tid; s < select_count; s += blockDim.x) {
    float weight = isfinite(weights[s]) ? expf(weights[s] - max_score) : 0.0f;
    weights[s] = weight;
    local_denom += weight;
  }
  reduction[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float denom = fmaxf(reduction[0], 1.0e-20f);

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < select_count; ++s) {
      float weight = weights[s];
      if (weight != 0.0f) {
        int64_t idx = idx_row[s];
        if (idx >= 0 && idx < key_count) {
          acc += (weight / denom) * static_cast<float>(v_base[idx * head_dim + d]);
        }
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, bool append_self>
__global__ void qabs_final_attention_ragged_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ counts,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = head / groups;
  int candidate_count = static_cast<int>(counts[row]);
  candidate_count = append_self
      ? min(max(candidate_count, 0), max_select_count)
      : min(max(candidate_count, 1), max_select_count);
  int row_select_count = candidate_count + (append_self ? 1 : 0);

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base =
      key + batch * key_stride_b + kv_head * key_stride_h;
  const scalar_t* v_base =
      value + batch * value_stride_b + kv_head * value_stride_h;
  const int64_t* idx_row = indices + row * max_select_count;
  scalar_t* out_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int s = tid; s < row_select_count; s += blockDim.x) {
    int64_t idx = append_self && s == candidate_count
        ? static_cast<int64_t>(key_count - 1)
        : idx_row[s];
    float score = -CUDART_INF_F;
    if (idx >= 0 && idx < key_count) {
      const scalar_t* k_vec = k_base + idx * key_stride_k;
      float acc = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        acc += static_cast<float>(q_row[d])
            * static_cast<float>(k_vec[d * key_stride_d]);
      }
      score = acc * scaling;
      local_max = fmaxf(local_max, score);
    }
    weights[s] = score;
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_denom = 0.0f;
  for (int s = tid; s < row_select_count; s += blockDim.x) {
    float weight = isfinite(weights[s]) ? expf(weights[s] - max_score) : 0.0f;
    weights[s] = weight;
    local_denom += weight;
  }
  reduction[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float denom = fmaxf(reduction[0], 1.0e-20f);

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < row_select_count; ++s) {
      int64_t idx = append_self && s == candidate_count
          ? static_cast<int64_t>(key_count - 1)
          : idx_row[s];
      if (idx >= 0 && idx < key_count) {
        acc += (weights[s] / denom)
            * static_cast<float>(
                v_base[idx * value_stride_k + d * value_stride_d]);
      }
    }
    out_row[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_ragged_self_warp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ counts,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warp_count = blockDim.x >> 5;
  int batch = row / query_head_count;
  int head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = head / groups;
  int candidate_count = static_cast<int>(counts[row]);
  candidate_count = min(max(candidate_count, 0), max_select_count);
  int row_select_count = candidate_count + 1;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base =
      key + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const scalar_t* v_base =
      value + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* idx_row = indices + row * max_select_count;
  scalar_t* out_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int selected = warp; selected < row_select_count;
       selected += warp_count) {
    int64_t token = selected == candidate_count
        ? static_cast<int64_t>(key_count - 1)
        : idx_row[selected];
    float score = 0.0f;
    if (token >= 0 && token < key_count) {
      const scalar_t* key_row = k_base + token * head_dim;
      for (int dim = lane; dim < head_dim; dim += 32) {
        score += static_cast<float>(q_row[dim])
            * static_cast<float>(key_row[dim]);
      }
    } else {
      score = -CUDART_INF_F;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xffffffffu, score, offset);
    }
    if (lane == 0) {
      score = isfinite(score) ? score * scaling : -CUDART_INF_F;
      weights[selected] = score;
      local_max = fmaxf(local_max, score);
    }
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_denom = 0.0f;
  for (int selected = tid; selected < row_select_count;
       selected += blockDim.x) {
    float weight = isfinite(weights[selected])
        ? expf(weights[selected] - max_score)
        : 0.0f;
    weights[selected] = weight;
    local_denom += weight;
  }
  reduction[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float inverse_denom = 1.0f / fmaxf(reduction[0], 1.0e-20f);

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int selected = 0; selected < row_select_count; ++selected) {
      int64_t token = selected == candidate_count
          ? static_cast<int64_t>(key_count - 1)
          : idx_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weights[selected] * inverse_denom
            * static_cast<float>(v_base[token * head_dim + dim]);
      }
    }
    out_row[dim] = static_cast<scalar_t>(accumulator);
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_ragged_self_split_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ counts,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int split_count,
    int64_t key_stride_b,
    int64_t key_stride_h,
    int64_t key_stride_k,
    int64_t key_stride_d,
    int64_t value_stride_b,
    int64_t value_stride_h,
    int64_t value_stride_k,
    int64_t value_stride_d,
    float scaling) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = head / groups;
  int candidate_count = min(
      max(static_cast<int>(counts[row]), 0), max_select_count);
  int row_select_count = candidate_count + 1;
  int chunk = (row_select_count + split_count - 1) / split_count;
  int start = min(split * chunk, row_select_count);
  int end = min(start + chunk, row_select_count);
  int local_count = end - start;

  const scalar_t* q_row = query + row * head_dim;
  const scalar_t* k_base =
      key + batch * key_stride_b + kv_head * key_stride_h;
  const scalar_t* v_base =
      value + batch * value_stride_b + kv_head * value_stride_h;
  const int64_t* idx_row = indices + row * max_select_count;
  float* partial_row =
      partial_output + (row * split_count + split) * head_dim;

  float local_max = -CUDART_INF_F;
  for (int local = tid; local < local_count; local += blockDim.x) {
    int selected = start + local;
    int64_t token = selected == candidate_count
        ? static_cast<int64_t>(key_count - 1)
        : idx_row[selected];
    float score = -CUDART_INF_F;
    if (token >= 0 && token < key_count) {
      const scalar_t* key_row = k_base + token * key_stride_k;
      float accumulator = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        accumulator += static_cast<float>(q_row[dim])
            * static_cast<float>(key_row[dim * key_stride_d]);
      }
      score = accumulator * scaling;
      local_max = fmaxf(local_max, score);
    }
    weights[local] = score;
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float block_max = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_sum = 0.0f;
  for (int local = tid; local < local_count; local += blockDim.x) {
    float weight = isfinite(weights[local])
        ? expf(weights[local] - block_max)
        : 0.0f;
    weights[local] = weight;
    local_sum += weight;
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_max[row * split_count + split] = block_max;
    partial_sum[row * split_count + split] = reduction[0];
  }

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int local = 0; local < local_count; ++local) {
      int selected = start + local;
      int64_t token = selected == candidate_count
          ? static_cast<int64_t>(key_count - 1)
          : idx_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weights[local]
            * static_cast<float>(
                v_base[
                    token * value_stride_k + dim * value_stride_d]);
      }
    }
    partial_row[dim] = accumulator;
  }
}

template <typename scalar_t>
__global__ void qabs_reduce_attention_splits_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    scalar_t* __restrict__ output,
    int head_dim,
    int split_count) {
  __shared__ float global_max;
  __shared__ float inverse_denom;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (tid == 0) {
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < split_count; ++split) {
      if (partial_sum[row * split_count + split] > 0.0f) {
        maximum = fmaxf(
            maximum, partial_max[row * split_count + split]);
      }
    }
    maximum = isfinite(maximum) ? maximum : 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      denominator += partial_sum[row * split_count + split]
          * expf(partial_max[row * split_count + split] - maximum);
    }
    global_max = maximum;
    inverse_denom = 1.0f / fmaxf(denominator, 1.0e-20f);
  }
  __syncthreads();

  const float* partial_row =
      partial_output + row * split_count * head_dim;
  scalar_t* output_row = output + row * head_dim;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      float scale = expf(
          partial_max[row * split_count + split] - global_max);
      accumulator += scale * partial_row[split * head_dim + dim];
    }
    output_row[dim] = static_cast<scalar_t>(
        accumulator * inverse_denom);
  }
}

template <typename scalar_t, bool bound_values, bool append_self>
__global__ void qabs_final_attention_from_scores_ragged_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    const int64_t* __restrict__ counts,
    const float* __restrict__ self_scores,
    scalar_t* __restrict__ output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    float value_mass_threshold) {
  extern __shared__ float shared[];
  __shared__ int value_history_count;
  __shared__ float value_denom;
  __shared__ float value_weight_cutoff;
  float* reduction = shared;
  float* weights = shared + blockDim.x;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int candidate_count = static_cast<int>(counts[row]);
  candidate_count = append_self
      ? min(max(candidate_count, 0), max_select_count)
      : min(max(candidate_count, 1), max_select_count);
  int row_select_count = candidate_count + (append_self ? 1 : 0);
  const scalar_t* value_base = value
      + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* index_row = indices + row * max_select_count;
  const float* score_row = scores + row * max_select_count;
  scalar_t* output_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float score = append_self && selected == candidate_count
        ? self_scores[row]
        : score_row[selected];
    weights[selected] = score;
    local_max = fmaxf(local_max, score);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;

  float local_denom = 0.0f;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float weight = expf(weights[selected] - max_score);
    weights[selected] = weight;
    local_denom += weight;
  }
  reduction[tid] = local_denom;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  float inverse_denom = 1.0f / fmaxf(reduction[0], 1.0e-20f);
  if (tid == 0) {
    value_history_count = row_select_count - 1;
    value_denom = reduction[0];
    value_weight_cutoff = bound_values
        ? (1.0f - value_mass_threshold) * reduction[0]
            / static_cast<float>(row_select_count)
        : 0.0f;
  }
  __syncthreads();
  if (bound_values) {
    if (tid == 0) {
      int low = 0;
      int high = row_select_count - 1;
      while (low < high) {
        int middle = low + (high - low) / 2;
        if (weights[middle] >= value_weight_cutoff) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
      value_history_count = low;
    }
    __syncthreads();
    float local_value_denom = 0.0f;
    for (int selected = tid; selected < value_history_count; selected += blockDim.x) {
      local_value_denom += weights[selected];
    }
    if (tid == 0) {
      local_value_denom += weights[row_select_count - 1];
    }
    reduction[tid] = local_value_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      value_denom = fmaxf(reduction[0], 1.0e-20f);
    }
    __syncthreads();
  }
  inverse_denom = 1.0f / value_denom;

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int selected = 0; selected < value_history_count; ++selected) {
      int64_t token = index_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weights[selected] * inverse_denom
            * static_cast<float>(value_base[token * head_dim + dim]);
      }
    }
    int self_position = row_select_count - 1;
    int64_t self_token = append_self
        ? static_cast<int64_t>(key_count - 1)
        : index_row[self_position];
    if (self_token >= 0 && self_token < key_count) {
      accumulator += weights[self_position] * inverse_denom
          * static_cast<float>(value_base[self_token * head_dim + dim]);
    }
    output_row[dim] = static_cast<scalar_t>(accumulator);
  }
}

__global__ void qabs_softmax_weights_ragged_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ counts,
    float* __restrict__ weights,
    int max_select_count) {
  extern __shared__ float reduction[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int row_count = min(max(static_cast<int>(counts[row]), 1), max_select_count);
  const float* score_row = scores + row * max_select_count;
  float* weight_row = weights + row * max_select_count;
  float local_max = -CUDART_INF_F;
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    local_max = fmaxf(local_max, score_row[selected]);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  float local_denominator = 0.0f;
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    float weight = expf(score_row[selected] - max_score);
    weight_row[selected] = weight;
    local_denominator += weight;
  }
  reduction[tid] = local_denominator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  float inverse_denominator = 1.0f / fmaxf(reduction[0], 1.0e-20f);
  for (int selected = tid; selected < row_count; selected += blockDim.x) {
    weight_row[selected] *= inverse_denominator;
  }
}

template <typename scalar_t>
__global__ void qabs_split_value_attention_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int64_t* __restrict__ counts,
    float* __restrict__ partial_output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    int split_count) {
  int block = blockIdx.x;
  int row = block / split_count;
  int split = block - row * split_count;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int row_count = min(max(static_cast<int>(counts[row]), 1), max_select_count);
  int chunk = (row_count + split_count - 1) / split_count;
  int start = min(split * chunk, row_count);
  int end = min(start + chunk, row_count);
  const scalar_t* value_base = value
      + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* index_row = indices + row * max_select_count;
  const float* weight_row = weights + row * max_select_count;
  float* partial_row = partial_output
      + (row * split_count + split) * head_dim;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int selected = start; selected < end; ++selected) {
      int64_t token = index_row[selected];
      if (token >= 0 && token < key_count) {
        accumulator += weight_row[selected]
            * static_cast<float>(value_base[token * head_dim + dim]);
      }
    }
    partial_row[dim] = accumulator;
  }
}

template <typename scalar_t>
__global__ void qabs_reduce_value_splits_kernel(
    const float* __restrict__ partial_output,
    scalar_t* __restrict__ output,
    int head_dim,
    int split_count) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* partial_row = partial_output + row * split_count * head_dim;
  scalar_t* output_row = output + row * head_dim;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float accumulator = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      accumulator += partial_row[split * head_dim + dim];
    }
    output_row[dim] = static_cast<scalar_t>(accumulator);
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_tail_reliability_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ prefix_counts,
    scalar_t* __restrict__ output,
    float* __restrict__ reliability_output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = reduction + blockDim.x;
  float* base_numerators = weights + max_select_count;
  float* tail_numerators = base_numerators + head_dim;
  float* delta_even = tail_numerators + head_dim;
  float* delta_odd = delta_even + 8;
  __shared__ float base_denominator;
  __shared__ float tail_denominator_even;
  __shared__ float tail_denominator_odd;
  __shared__ float half_multiplier_even;
  __shared__ float half_multiplier_odd;
  __shared__ float reliability;
  __shared__ int valid_even;
  __shared__ int valid_odd;

  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int row_select_count = static_cast<int>(counts[row]);
  row_select_count = min(max(row_select_count, 1), max_select_count);
  int prefix_count = static_cast<int>(prefix_counts[row]);
  prefix_count = min(max(prefix_count, 1), row_select_count);
  const scalar_t* value_base = value
      + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* index_row = indices + row * max_select_count;
  const float* score_row = scores + row * max_select_count;
  scalar_t* output_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float score = score_row[selected];
    weights[selected] = score;
    local_max = fmaxf(local_max, score);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  if (tid == 0) {
    valid_even = 0;
    valid_odd = 0;
  }
  __syncthreads();
  float local_base_denominator = 0.0f;
  float local_tail_even = 0.0f;
  float local_tail_odd = 0.0f;
  int local_valid_even = 0;
  int local_valid_odd = 0;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float score = weights[selected];
    float weight = isfinite(score) ? expf(score - max_score) : 0.0f;
    weights[selected] = weight;
    if (selected < prefix_count) {
      local_base_denominator += weight;
    } else if (((selected - prefix_count) & 1) == 0) {
      local_tail_even += weight;
      local_valid_even += isfinite(score) ? 1 : 0;
    } else {
      local_tail_odd += weight;
      local_valid_odd += isfinite(score) ? 1 : 0;
    }
  }
  atomicAdd(&valid_even, local_valid_even);
  atomicAdd(&valid_odd, local_valid_odd);

  reduction[tid] = local_base_denominator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) base_denominator = fmaxf(reduction[0], 1.0e-20f);
  __syncthreads();
  reduction[tid] = local_tail_even;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) tail_denominator_even = reduction[0];
  __syncthreads();
  reduction[tid] = local_tail_odd;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    tail_denominator_odd = reduction[0];
    int total_valid = max(valid_even + valid_odd, 1);
    half_multiplier_even = static_cast<float>(total_valid)
        / static_cast<float>(max(valid_even, 1));
    half_multiplier_odd = static_cast<float>(total_valid)
        / static_cast<float>(max(valid_odd, 1));
  }
  __syncthreads();

  int reliability_dim_count = min(8, head_dim);
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    int reliability_index = -1;
    for (int index = 0; index < reliability_dim_count; ++index) {
      if (dim == index * head_dim / reliability_dim_count) {
        reliability_index = index;
      }
    }
    float base_numerator = 0.0f;
    float tail_numerator = 0.0f;
    float tail_numerator_even = 0.0f;
    float tail_numerator_odd = 0.0f;
    for (int selected = 0; selected < row_select_count; ++selected) {
      int64_t token = index_row[selected];
      if (token < 0 || token >= key_count) continue;
      float contribution = weights[selected]
          * static_cast<float>(value_base[token * head_dim + dim]);
      if (selected < prefix_count) {
        base_numerator += contribution;
      } else {
        tail_numerator += contribution;
        if (reliability_index >= 0) {
          if (((selected - prefix_count) & 1) == 0) {
            tail_numerator_even += contribution;
          } else {
            tail_numerator_odd += contribution;
          }
        }
      }
    }
    base_numerators[dim] = base_numerator;
    tail_numerators[dim] = tail_numerator;
    if (reliability_index >= 0) {
      float base_output = base_numerator / base_denominator;
      float even_output = (
          base_numerator + half_multiplier_even * tail_numerator_even)
          / fmaxf(
              base_denominator
                  + half_multiplier_even * tail_denominator_even,
              1.0e-20f);
      float odd_output = (
          base_numerator + half_multiplier_odd * tail_numerator_odd)
          / fmaxf(
              base_denominator
                  + half_multiplier_odd * tail_denominator_odd,
              1.0e-20f);
      delta_even[reliability_index] = even_output - base_output;
      delta_odd[reliability_index] = odd_output - base_output;
    }
  }
  __syncthreads();

  float local_signal = 0.0f;
  float local_noise = 0.0f;
  if (tid < reliability_dim_count) {
    float signal = delta_even[tid] + delta_odd[tid];
    float noise = delta_even[tid] - delta_odd[tid];
    local_signal += signal * signal;
    local_noise += noise * noise;
  }
  reduction[tid] = local_signal;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  float signal_power = reduction[0];
  reduction[tid] = local_noise;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    reliability = signal_power / (signal_power + reduction[0] + 1.0e-12f);
    reliability_output[row] = reliability;
  }
  __syncthreads();
  float final_denominator = base_denominator + reliability
      * (tail_denominator_even + tail_denominator_odd);
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float numerator = base_numerators[dim] + reliability * tail_numerators[dim];
    output_row[dim] = static_cast<scalar_t>(
        numerator / fmaxf(final_denominator, 1.0e-20f));
  }
}

template <typename scalar_t>
__global__ void qabs_final_attention_tail_mass_gate_kernel(
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ indices,
    const float* __restrict__ scores,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ prefix_counts,
    scalar_t* __restrict__ output,
    float* __restrict__ active_output,
    int query_head_count,
    int kv_head_count,
    int key_count,
    int max_select_count,
    int head_dim,
    float mass_threshold,
    float tail_shrinkage) {
  extern __shared__ float shared[];
  float* reduction = shared;
  float* weights = reduction + blockDim.x;
  __shared__ float base_denominator;
  __shared__ float final_denominator;
  __shared__ int tail_active;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int batch = row / query_head_count;
  int query_head = row - batch * query_head_count;
  int groups = query_head_count / kv_head_count;
  int kv_head = query_head / groups;
  int row_select_count = min(
      max(static_cast<int>(counts[row]), 1), max_select_count);
  int prefix_count = min(
      max(static_cast<int>(prefix_counts[row]), 1), row_select_count);
  const scalar_t* value_base = value
      + ((batch * kv_head_count + kv_head) * key_count) * head_dim;
  const int64_t* index_row = indices + row * max_select_count;
  const float* score_row = scores + row * max_select_count;
  scalar_t* output_row = output + row * head_dim;

  float local_max = -CUDART_INF_F;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float score = score_row[selected];
    weights[selected] = score;
    local_max = fmaxf(local_max, score);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    __syncthreads();
  }
  float max_score = isfinite(reduction[0]) ? reduction[0] : 0.0f;
  for (int selected = tid; selected < row_select_count; selected += blockDim.x) {
    float score = weights[selected];
    weights[selected] = isfinite(score) ? expf(score - max_score) : 0.0f;
  }
  __syncthreads();

  float local_base = 0.0f;
  for (int selected = tid; selected < prefix_count; selected += blockDim.x) {
    local_base += weights[selected];
  }
  reduction[tid] = local_base;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) base_denominator = fmaxf(reduction[0], 1.0e-20f);
  __syncthreads();
  float local_tail = 0.0f;
  for (int selected = prefix_count + tid; selected < row_select_count;
       selected += blockDim.x) {
    local_tail += weights[selected];
  }
  reduction[tid] = local_tail;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) reduction[tid] += reduction[tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    float tail_denominator = reduction[0];
    float selected_mass = base_denominator
        / fmaxf(base_denominator + tail_denominator, 1.0e-20f);
    tail_active = selected_mass < mass_threshold ? 1 : 0;
    final_denominator = base_denominator
        + (tail_active ? tail_shrinkage * tail_denominator : 0.0f);
    active_output[row] = static_cast<float>(tail_active);
  }
  __syncthreads();

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    float numerator = 0.0f;
    for (int selected = 0; selected < prefix_count; ++selected) {
      int64_t token = index_row[selected];
      if (token >= 0 && token < key_count) {
        numerator += weights[selected]
            * static_cast<float>(value_base[token * head_dim + dim]);
      }
    }
    if (tail_active) {
      for (int selected = prefix_count; selected < row_select_count; ++selected) {
        int64_t token = index_row[selected];
        if (token >= 0 && token < key_count) {
          numerator += tail_shrinkage * weights[selected]
              * static_cast<float>(value_base[token * head_dim + dim]);
        }
      }
    }
    output_row[dim] = static_cast<scalar_t>(
        numerator / fmaxf(final_denominator, 1.0e-20f));
  }
}

__global__ void qabs_mass_ladder_kernel(
    const float* __restrict__ top_scores,
    const float* __restrict__ sample_scores,
    const float* __restrict__ sample_candidate_scores,
    const float* __restrict__ self_scores,
    const int64_t* __restrict__ keep_counts,
    float* __restrict__ output,
    int max_keep_count,
    int sample_count,
    int rung_count,
    int history_count,
    float mass_threshold) {
  extern __shared__ float reduction[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  const float* top_row = top_scores + row * max_keep_count;
  const float* sample_row = sample_scores + row * sample_count;
  const float* sample_candidate_row = sample_candidate_scores + row * sample_count;

  float local_max = tid == 0 ? fmaxf(top_row[0], self_scores[row]) : -CUDART_INF_F;
  for (int sample = tid; sample < sample_count; sample += blockDim.x) {
    local_max = fmaxf(local_max, sample_row[sample]);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  float reference_max = reduction[0];

  float selected_local[QABS_MAX_RUNGS];
  float residual_local[QABS_MAX_RUNGS];
  for (int rung = 0; rung < rung_count; ++rung) {
    selected_local[rung] = tid == 0 ? expf(self_scores[row] - reference_max) : 0.0f;
    residual_local[rung] = 0.0f;
  }
  for (int selected = tid; selected < max_keep_count; selected += blockDim.x) {
    float weight = expf(top_row[selected] - reference_max);
    for (int rung = 0; rung < rung_count; ++rung) {
      if (selected < keep_counts[rung]) {
        selected_local[rung] += weight;
      }
    }
  }
  for (int sample = tid; sample < sample_count; sample += blockDim.x) {
    float weight = expf(sample_row[sample] - reference_max);
    float candidate_score = sample_candidate_row[sample];
    for (int rung = 0; rung < rung_count; ++rung) {
      float cutoff = top_row[keep_counts[rung] - 1];
      if (candidate_score < cutoff) {
        residual_local[rung] += weight;
      }
    }
  }

  int chosen_rung = -1;
  float chosen_mass = 0.0f;
  float chosen_log_total = 0.0f;
  float chosen_top1_mass = 0.0f;
  float residual_scale = static_cast<float>(history_count) / static_cast<float>(sample_count);
  float top1_weight = expf(fmaxf(top_row[0], self_scores[row]) - reference_max);
  for (int rung = 0; rung < rung_count; ++rung) {
    reduction[tid] = selected_local[rung];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    float selected_sum = reduction[0];
    reduction[tid] = residual_local[rung];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      float estimated_total = fmaxf(selected_sum + reduction[0] * residual_scale, 1.0e-20f);
      float mass = fminf(selected_sum / estimated_total, 1.0f);
      if (chosen_rung < 0 && (mass >= mass_threshold || rung == rung_count - 1)) {
        chosen_rung = rung;
        chosen_mass = mass;
        chosen_log_total = reference_max + logf(estimated_total);
        chosen_top1_mass = fminf(top1_weight / estimated_total, 1.0f);
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    float* output_row = output + row * 4;
    output_row[0] = static_cast<float>(chosen_rung);
    output_row[1] = chosen_mass;
    output_row[2] = chosen_log_total;
    output_row[3] = chosen_top1_mass;
  }
}

torch::Tensor qabs_partial_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(2) == key.size(3), "query/key shape mismatch");

  auto query_c = query.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key.size(2));
  int selected_dim_count = static_cast<int>(std::min<int64_t>(std::max<int64_t>(dim_count, 1), std::min<int64_t>(head_dim, QABS_MAX_DIMS)));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_forward", [&] {
    qabs_partial_scores_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        output.data_ptr<float>(),
        batch_count * head_count,
        key_count,
        head_dim,
        selected_dim_count,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_partial_scores_dim_major_forward(
    torch::Tensor query,
    torch::Tensor key_dim_major,
    torch::Tensor dim_indices) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key_dim_major.is_cuda(), "key_dim_major must be CUDA");
  TORCH_CHECK(dim_indices.is_cuda(), "dim_indices must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key_dim_major.dim() == 4, "key_dim_major must have shape [batch, heads, dim, key]");
  TORCH_CHECK(dim_indices.dim() == 3, "dim_indices must have shape [batch, heads, selected_dim]");
  TORCH_CHECK(query.scalar_type() == key_dim_major.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(dim_indices.scalar_type() == at::kLong, "dim_indices must be int64");
  TORCH_CHECK(query.size(0) == key_dim_major.size(0) && query.size(1) == key_dim_major.size(1) && query.size(2) == key_dim_major.size(2), "query/key shape mismatch");
  TORCH_CHECK(dim_indices.size(0) == query.size(0) && dim_indices.size(1) == query.size(1), "dim_indices shape mismatch");

  auto query_c = query.contiguous();
  auto dim_indices_c = dim_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_dim_major.size(3));
  int selected_count = static_cast<int>(dim_indices_c.size(2));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_dim_major_forward", [&] {
    qabs_partial_scores_dim_major_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_dim_major.data_ptr<scalar_t>(),
        dim_indices_c.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        key_count,
        head_dim,
        selected_count,
        key_dim_major.stride(0),
        key_dim_major.stride(1),
        key_dim_major.stride(2),
        key_dim_major.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_full_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor current_candidate,
    torch::Tensor previous_candidate,
    bool has_previous_candidate,
    torch::Tensor previous_final,
    bool has_previous_final,
    int64_t protect_sink_tokens,
    int64_t protect_recent_tokens,
    double scaling) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(current_candidate.is_cuda(), "current_candidate must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(current_candidate.dim() == 3, "current_candidate must have shape [batch, heads, key]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(current_candidate.scalar_type() == at::kBool, "current_candidate must be bool");
  TORCH_CHECK(!has_previous_candidate || previous_candidate.scalar_type() == at::kBool, "previous_candidate must be bool");
  TORCH_CHECK(!has_previous_final || previous_final.scalar_type() == at::kBool, "previous_final must be bool");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(1) == key.size(1) && query.size(2) == key.size(3), "query/key shape mismatch");
  TORCH_CHECK(current_candidate.size(0) == key.size(0) && current_candidate.size(1) == key.size(1) && current_candidate.size(2) == key.size(2), "candidate shape mismatch");

  auto query_c = query.contiguous();
  auto current_c = current_candidate.contiguous();
  auto previous_candidate_c = has_previous_candidate ? previous_candidate.contiguous() : previous_candidate;
  auto previous_final_c = has_previous_final ? previous_final.contiguous() : previous_final;
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key.size(2));
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);

  const bool* previous_candidate_ptr = has_previous_candidate ? previous_candidate_c.data_ptr<bool>() : current_c.data_ptr<bool>();
  const bool* previous_final_ptr = has_previous_final ? previous_final_c.data_ptr<bool>() : current_c.data_ptr<bool>();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_candidate_full_scores_forward", [&] {
    qabs_candidate_full_scores_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        current_c.data_ptr<bool>(),
        previous_candidate_ptr,
        has_previous_candidate,
        previous_final_ptr,
        has_previous_final,
        output.data_ptr<float>(),
        key_count,
        head_dim,
        static_cast<int>(protect_sink_tokens),
        static_cast<int>(protect_recent_tokens),
        static_cast<float>(scaling),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> qabs_uncertainty_band_mask_forward(
    torch::Tensor candidate_scores,
    torch::Tensor error_sigma,
    int64_t final_count,
    double confidence_width) {
  TORCH_CHECK(candidate_scores.is_cuda() && error_sigma.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(candidate_scores.dim() == 3, "candidate scores must be rank three");
  TORCH_CHECK(error_sigma.dim() == 2, "error sigma must be rank two");
  TORCH_CHECK(candidate_scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(error_sigma.scalar_type() == at::kFloat, "error sigma must be float32");
  auto scores_c = candidate_scores.contiguous();
  auto sigma_c = error_sigma.contiguous();
  c10::cuda::CUDAGuard device_guard(scores_c.device());
  int batch_count = static_cast<int>(scores_c.size(0));
  int head_count = static_cast<int>(scores_c.size(1));
  int candidate_count = static_cast<int>(scores_c.size(2));
  TORCH_CHECK(
      sigma_c.size(0) == batch_count && sigma_c.size(1) == head_count,
      "error sigma shape mismatch");
  TORCH_CHECK(final_count > 0 && final_count <= candidate_count, "invalid final count");
  auto valid = torch::empty(
      scores_c.sizes(), scores_c.options().dtype(at::kBool));
  auto counts = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kFloat));
  qabs_uncertainty_band_mask_kernel<<<
      batch_count * head_count,
      QABS_TILE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          sigma_c.data_ptr<float>(),
          valid.data_ptr<bool>(),
          counts.data_ptr<int64_t>(),
          boundaries.data_ptr<float>(),
          candidate_count,
          static_cast<int>(final_count),
          static_cast<float>(confidence_width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {valid, counts, boundaries};
}

std::vector<torch::Tensor> qabs_direct_uncertainty_candidates_forward(
    torch::Tensor scores,
    torch::Tensor error_sigma,
    int64_t final_count,
    int64_t candidate_capacity,
    double confidence_width) {
  TORCH_CHECK(scores.is_cuda() && error_sigma.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(scores.dim() == 3, "scores must be rank three");
  TORCH_CHECK(error_sigma.dim() == 2, "error sigma must be rank two");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(error_sigma.scalar_type() == at::kFloat, "error sigma must be float32");
  auto scores_c = scores.contiguous();
  auto sigma_c = error_sigma.contiguous();
  c10::cuda::CUDAGuard device_guard(scores_c.device());
  int batch_count = static_cast<int>(scores_c.size(0));
  int head_count = static_cast<int>(scores_c.size(1));
  int history_count = static_cast<int>(scores_c.size(2));
  TORCH_CHECK(
      sigma_c.size(0) == batch_count && sigma_c.size(1) == head_count,
      "error sigma shape mismatch");
  TORCH_CHECK(final_count > 0 && final_count <= history_count, "invalid final count");
  TORCH_CHECK(
      candidate_capacity >= final_count && candidate_capacity <= history_count,
      "invalid candidate capacity");
  auto indices = torch::empty(
      {batch_count, head_count, candidate_capacity},
      scores_c.options().dtype(at::kLong));
  auto counts = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kFloat));
  auto overflow = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kBool));
  qabs_direct_uncertainty_candidates_kernel<<<
      batch_count * head_count,
      QABS_TILE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          sigma_c.data_ptr<float>(),
          indices.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          boundaries.data_ptr<float>(),
          overflow.data_ptr<bool>(),
          history_count,
          static_cast<int>(final_count),
          static_cast<int>(candidate_capacity),
          static_cast<float>(confidence_width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, counts, boundaries, overflow};
}

std::vector<torch::Tensor> qabs_sampled_quantile_candidates_forward(
    torch::Tensor scores,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity) {
  TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(scores.dim() == 3, "scores must be rank three");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(
      sample_count > 0 && sample_count <= QABS_TILE_THREADS,
      "sample count must be in [1, 256]");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");
  auto scores_c = scores.contiguous();
  c10::cuda::CUDAGuard device_guard(scores_c.device());
  int batch_count = static_cast<int>(scores_c.size(0));
  int head_count = static_cast<int>(scores_c.size(1));
  int history_count = static_cast<int>(scores_c.size(2));
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "invalid candidate capacity");
  int sample_keep = max(
      1, static_cast<int>(ceil(selected_fraction * sample_count)));
  // Ragged consumers gather the whole capacity before applying counts.
  auto indices = torch::zeros(
      {batch_count, head_count, candidate_capacity},
      scores_c.options().dtype(at::kLong));
  auto counts = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kFloat));
  auto overflow = torch::empty(
      {batch_count, head_count}, scores_c.options().dtype(at::kBool));
  qabs_sampled_quantile_candidates_kernel<<<
      batch_count * head_count,
      QABS_TILE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          indices.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          boundaries.data_ptr<float>(),
          overflow.data_ptr<bool>(),
          history_count,
          static_cast<int>(sample_count),
          sample_keep,
          static_cast<int>(candidate_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, counts, boundaries, overflow};
}

torch::Tensor qabs_sample_error_sigma_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor approximate_scores,
    int64_t sample_count,
    int64_t sample_offset,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && approximate_scores.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must be rank three");
  TORCH_CHECK(key.dim() == 4, "key must be rank four");
  TORCH_CHECK(approximate_scores.dim() == 3, "scores must be rank three");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(
      approximate_scores.scalar_type() == at::kFloat,
      "approximate scores must be float32");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto scores_c = approximate_scores.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int key_count = static_cast<int>(key_c.size(2));
  int history_count = static_cast<int>(scores_c.size(2));
  int head_dim = static_cast<int>(query_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      scores_c.size(0) == batch_count
          && scores_c.size(1) == query_head_count,
      "score shape mismatch");
  TORCH_CHECK(key_count >= history_count, "key source is shorter than history");
  TORCH_CHECK(sample_count > 0 && sample_count <= history_count, "invalid sample count");
  int sample_stride = std::max<int>(1, history_count / sample_count);
  int bounded_offset = static_cast<int>(sample_offset) % sample_stride;
  auto output = torch::empty(
      {batch_count, query_head_count}, scores_c.options().dtype(at::kFloat));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_sample_error_sigma_forward",
      [&] {
        qabs_sample_error_sigma_kernel<scalar_t><<<
            batch_count * query_head_count,
            QABS_TILE_THREADS,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                scores_c.data_ptr<float>(),
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                history_count,
                head_dim,
                static_cast<int>(sample_count),
                sample_stride,
                bounded_offset,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_compact_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    double scaling) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate_indices must have shape [batch, heads, candidate]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "candidate_indices must be int64");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count, "candidate shape mismatch");
  TORCH_CHECK(candidate_count > 0, "candidate_count must be positive");
  auto output = torch::empty({batch_count, query_head_count, candidate_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * query_head_count, (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_candidate_compact_scores_forward", [&] {
    qabs_candidate_compact_scores_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        indices_c.data_ptr<int64_t>(),
        nullptr,
        false,
        nullptr,
        false,
        output.data_ptr<float>(),
        query_head_count,
        kv_head_count,
        key_count,
        candidate_count,
        head_dim,
        static_cast<float>(scaling));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_prerope_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor phase_cosine,
    torch::Tensor phase_sine,
    int64_t query_position,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda()
          && phase_cosine.is_cuda() && phase_sine.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(
      candidate_indices.dim() == 3,
      "candidate_indices must have shape [batch, heads, candidate]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "candidate indices must be int64");
  TORCH_CHECK(
      phase_cosine.scalar_type() == at::kFloat
          && phase_sine.scalar_type() == at::kFloat,
      "phase tables must be float32");
  TORCH_CHECK(
      phase_cosine.dim() == 2 && phase_sine.dim() == 2,
      "phase tables must be rank two");
  TORCH_CHECK(
      phase_cosine.sizes() == phase_sine.sizes(),
      "phase table shapes must match");
  TORCH_CHECK(query_position >= 0, "query position must be non-negative");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto phase_cosine_c = phase_cosine.contiguous();
  auto phase_sine_c = phase_sine.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "head dim must be positive and even");
  TORCH_CHECK(
      phase_cosine_c.size(1) == head_dim / 2,
      "phase table width must equal half the head dimension");
  TORCH_CHECK(
      query_position < phase_cosine_c.size(0),
      "phase table is shorter than the query position");
  TORCH_CHECK(candidate_count > 0, "candidate count must be positive");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_candidate_prerope_scores_forward",
      [&] {
        qabs_candidate_prerope_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                phase_cosine_c.data_ptr<float>(),
                phase_sine_c.data_ptr<float>(),
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_count,
                head_dim,
                static_cast<int>(query_position),
                static_cast<int>(phase_cosine_c.size(0)),
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_compact_scores_ragged_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_counts,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda()
          && candidate_counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(candidate_counts.dim() == 2, "candidate counts must be rank two");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(candidate_counts.scalar_type() == at::kLong, "counts must be int64");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto counts_c = candidate_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count && counts_c.size(1) == query_head_count,
      "candidate count shape mismatch");
  TORCH_CHECK(candidate_count > 0, "candidate_count must be positive");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_candidate_compact_scores_ragged_forward",
      [&] {
        qabs_candidate_compact_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                true,
                nullptr,
                false,
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_count,
                head_dim,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_candidate_compact_scores_range_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor start_counts,
    torch::Tensor end_counts,
    torch::Tensor output,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda()
          && start_counts.is_cuda() && end_counts.is_cuda() && output.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(start_counts.dim() == 2 && end_counts.dim() == 2, "range counts must be rank two");
  TORCH_CHECK(output.sizes() == candidate_indices.sizes(), "output shape mismatch");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(
      start_counts.scalar_type() == at::kLong && end_counts.scalar_type() == at::kLong,
      "range counts must be int64");
  TORCH_CHECK(output.scalar_type() == at::kFloat, "output must be float32");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto starts_c = start_counts.contiguous();
  auto ends_c = end_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(
      starts_c.size(0) == batch_count && starts_c.size(1) == query_head_count
          && ends_c.sizes() == starts_c.sizes(),
      "range count shape mismatch");
  TORCH_CHECK(candidate_count > 0, "candidate_count must be positive");
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_candidate_compact_scores_range_forward",
      [&] {
        qabs_candidate_compact_scores_range_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                starts_c.data_ptr<int64_t>(),
                ends_c.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_count,
                head_dim,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> qabs_proxy_affine_calibrated_scores_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor proxy_scores,
    torch::Tensor candidate_counts,
    int64_t sample_count,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda()
          && proxy_scores.is_cuda() && candidate_counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(
      candidate_indices.dim() == 3,
      "candidate indices must be rank three");
  TORCH_CHECK(
      proxy_scores.sizes() == candidate_indices.sizes(),
      "proxy score shape mismatch");
  TORCH_CHECK(
      candidate_counts.dim() == 2,
      "candidate counts must be rank two");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(proxy_scores.scalar_type() == at::kFloat, "proxy scores must be float32");
  TORCH_CHECK(candidate_counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(
      sample_count >= 8 && sample_count <= 128,
      "sample count must be in [8, 128]");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto proxy_c = proxy_scores.contiguous();
  auto counts_c = candidate_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_capacity = static_cast<int>(indices_c.size(2));
  int rows = batch_count * query_head_count;
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count
          && counts_c.size(1) == query_head_count,
      "candidate count shape mismatch");
  TORCH_CHECK(candidate_capacity > 0, "candidate capacity must be positive");

  auto calibrated_scores = torch::empty_like(proxy_c);
  auto calibration_parameters = torch::empty(
      {batch_count, query_head_count, 2},
      proxy_c.options());
  int threads = 128;
  size_t shared_bytes = static_cast<size_t>(
      4 * threads) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_proxy_affine_calibrated_scores_forward",
      [&] {
        qabs_proxy_affine_calibrated_scores_kernel<scalar_t>
            <<<rows, threads, shared_bytes,
               at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                proxy_c.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                calibrated_scores.data_ptr<float>(),
                calibration_parameters.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_capacity,
                head_dim,
                static_cast<int>(sample_count),
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {calibrated_scores, calibration_parameters};
}

torch::Tensor qabs_candidate_compact_scores_masked_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor candidate_indices,
    torch::Tensor candidate_valid,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && candidate_indices.is_cuda()
          && candidate_valid.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(candidate_valid.sizes() == candidate_indices.sizes(), "valid shape mismatch");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(candidate_valid.scalar_type() == at::kBool, "valid must be bool");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto valid_c = candidate_valid.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(candidate_count > 0, "candidate_count must be positive");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_candidate_compact_scores_masked_forward",
      [&] {
        qabs_candidate_compact_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                nullptr,
                false,
                valid_c.data_ptr<bool>(),
                true,
                output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                candidate_count,
                head_dim,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor valid,
    double scaling) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(valid.is_cuda(), "valid must be CUDA");
  TORCH_CHECK(query.device() == key.device(), "query/key device mismatch");
  TORCH_CHECK(query.device() == value.device(), "query/value device mismatch");
  TORCH_CHECK(query.device() == indices.device(), "query/indices device mismatch");
  TORCH_CHECK(query.device() == valid.device(), "query/valid device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, heads, selected]");
  TORCH_CHECK(valid.sizes() == indices.sizes(), "valid must match indices shape");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(valid.scalar_type() == at::kByte, "valid must be uint8");

  auto query_c = query.contiguous();
  auto key_c = key;
  auto value_c = value;
  auto indices_c = indices.contiguous();
  auto valid_c = valid.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(key_c.size(0) == batch_count && key_c.size(1) == head_count && key_c.size(3) == head_dim, "key shape mismatch");
  TORCH_CHECK(indices_c.size(0) == batch_count && indices_c.size(1) == head_count, "indices shape mismatch");
  TORCH_CHECK(select_count > 0, "select_count must be positive");

  auto output = torch::empty({batch_count, head_count, head_dim}, query_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * head_count;
  size_t shared_bytes = static_cast<size_t>(threads + select_count) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_final_attention_forward", [&] {
    qabs_final_attention_token_kernel<scalar_t><<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        value_c.data_ptr<scalar_t>(),
        indices_c.data_ptr<int64_t>(),
        valid_c.data_ptr<uint8_t>(),
        output.data_ptr<scalar_t>(),
        batch_count,
        head_count,
        key_count,
        select_count,
        head_dim,
        static_cast<float>(scaling));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_ragged_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
  TORCH_CHECK(query.device() == key.device() && query.device() == value.device(), "query/key/value device mismatch");
  TORCH_CHECK(query.device() == indices.device() && query.device() == counts.device(), "query/index device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, heads, max_selected]");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, heads]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(), "query/key/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");

  auto query_c = query.contiguous();
  auto key_c = key;
  auto value_c = value;
  auto indices_c = indices.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count, "indices shape mismatch");
  TORCH_CHECK(counts_c.size(0) == batch_count && counts_c.size(1) == query_head_count, "counts shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");

  auto output = torch::empty({batch_count, query_head_count, head_dim}, query_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(threads + max_select_count) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_final_attention_ragged_forward", [&] {
    qabs_final_attention_ragged_kernel<scalar_t, false><<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        value_c.data_ptr<scalar_t>(),
        indices_c.data_ptr<int64_t>(),
        counts_c.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(),
        query_head_count,
        kv_head_count,
        key_count,
        max_select_count,
        head_dim,
        key_c.stride(0),
        key_c.stride(1),
        key_c.stride(2),
        key_c.stride(3),
        value_c.stride(0),
        value_c.stride(1),
        value_c.stride(2),
        value_c.stride(3),
        static_cast<float>(scaling));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_ragged_self_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && indices.is_cuda() && counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query/key/value device mismatch");
  TORCH_CHECK(
      query.device() == indices.device() && query.device() == counts.device(),
      "query/index device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(
      indices.dim() == 3,
      "indices must have shape [batch, heads, max_selected]");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, heads]");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type(),
      "query/key/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "indices shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count
          && counts_c.size(1) == query_head_count,
      "counts shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");

  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, query_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_select_count + 1) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_final_attention_ragged_self_forward",
      [&] {
        qabs_final_attention_ragged_kernel<scalar_t, true>
            <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                key_c.stride(0),
                key_c.stride(1),
                key_c.stride(2),
                key_c.stride(3),
                value_c.stride(0),
                value_c.stride(1),
                value_c.stride(2),
                value_c.stride(3),
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_ragged_self_warp_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && indices.is_cuda() && counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query/key/value device mismatch");
  TORCH_CHECK(
      query.device() == indices.device() && query.device() == counts.device(),
      "query/index device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(
      indices.dim() == 3,
      "indices must have shape [batch, heads, max_selected]");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, heads]");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type(),
      "query/key/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "indices shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count
          && counts_c.size(1) == query_head_count,
      "counts shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");

  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, query_c.options());
  int threads = 128;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_select_count + 1) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_final_attention_ragged_self_warp_forward",
      [&] {
        qabs_final_attention_ragged_self_warp_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_ragged_self_split_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor counts,
    double scaling,
    int64_t split_count) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && indices.is_cuda() && counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query/key/value device mismatch");
  TORCH_CHECK(
      query.device() == indices.device() && query.device() == counts.device(),
      "query/index device mismatch");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(value.sizes() == key.sizes(), "value must match key shape");
  TORCH_CHECK(
      indices.dim() == 3,
      "indices must have shape [batch, heads, max_selected]");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, heads]");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type(),
      "query/key/value dtype mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(
      split_count >= 2 && split_count <= 16,
      "split count must be in [2, 16]");

  auto query_c = query.contiguous();
  auto key_c = key;
  auto value_c = value;
  auto indices_c = indices.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());

  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int max_select_count = static_cast<int>(indices_c.size(2));
  int splits = static_cast<int>(split_count);
  int rows = batch_count * query_head_count;
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && query_head_count % kv_head_count == 0,
      "key shape mismatch");
  TORCH_CHECK(
      indices_c.size(0) == batch_count
          && indices_c.size(1) == query_head_count,
      "indices shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count
          && counts_c.size(1) == query_head_count,
      "counts shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");

  auto partial_output = torch::empty(
      {rows, splits, head_dim}, query_c.options().dtype(at::kFloat));
  auto partial_max = torch::empty(
      {rows, splits}, query_c.options().dtype(at::kFloat));
  auto partial_sum = torch::empty_like(partial_max);
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, query_c.options());
  int threads = 128;
  int max_local_count = (
      max_select_count + 1 + splits - 1) / splits;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_local_count) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_final_attention_ragged_self_split_forward",
      [&] {
        qabs_final_attention_ragged_self_split_kernel<scalar_t>
            <<<rows * splits, threads, shared_bytes,
               at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                splits,
                key_c.stride(0),
                key_c.stride(1),
                key_c.stride(2),
                key_c.stride(3),
                value_c.stride(0),
                value_c.stride(1),
                value_c.stride(2),
                value_c.stride(3),
                static_cast<float>(scaling));
        qabs_reduce_attention_splits_kernel<scalar_t>
            <<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                head_dim,
                splits);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_from_scores_ragged_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    double value_mass_threshold) {
  TORCH_CHECK(
      value.is_cuda() && indices.is_cuda() && scores.is_cuda() && counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "score shape mismatch");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, query_heads]");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(value_c.device());
  int batch_count = static_cast<int>(value_c.size(0));
  int kv_head_count = static_cast<int>(value_c.size(1));
  int key_count = static_cast<int>(value_c.size(2));
  int head_dim = static_cast<int>(value_c.size(3));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      indices_c.size(0) == batch_count && query_head_count % kv_head_count == 0,
      "index shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count && counts_c.size(1) == query_head_count,
      "count shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");
  TORCH_CHECK(
      value_mass_threshold > 0.0 && value_mass_threshold <= 1.0,
      "value mass threshold must be in (0, 1]");
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, value_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(threads + max_select_count)
      * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_c.scalar_type(),
      "qabs_final_attention_from_scores_ragged_forward",
      [&] {
        if (value_mass_threshold < 0.999999) {
          qabs_final_attention_from_scores_ragged_kernel<scalar_t, true, false>
              <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                  value_c.data_ptr<scalar_t>(),
                  indices_c.data_ptr<int64_t>(),
                  scores_c.data_ptr<float>(),
                  counts_c.data_ptr<int64_t>(),
                  nullptr,
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  key_count,
                  max_select_count,
                  head_dim,
                  static_cast<float>(value_mass_threshold));
        } else {
          qabs_final_attention_from_scores_ragged_kernel<scalar_t, false, false>
              <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                  value_c.data_ptr<scalar_t>(),
                  indices_c.data_ptr<int64_t>(),
                  scores_c.data_ptr<float>(),
                  counts_c.data_ptr<int64_t>(),
                  nullptr,
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  key_count,
                  max_select_count,
                  head_dim,
                  1.0f);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_from_scores_ragged_self_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor self_scores,
    double value_mass_threshold) {
  TORCH_CHECK(
      value.is_cuda() && indices.is_cuda() && scores.is_cuda()
          && counts.is_cuda() && self_scores.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "score shape mismatch");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, query_heads]");
  TORCH_CHECK(self_scores.dim() == 3 && self_scores.size(2) == 1,
              "self_scores must have shape [batch, query_heads, 1]");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(self_scores.scalar_type() == at::kFloat, "self_scores must be float32");
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  auto self_scores_c = self_scores.contiguous();
  c10::cuda::CUDAGuard device_guard(value_c.device());
  int batch_count = static_cast<int>(value_c.size(0));
  int kv_head_count = static_cast<int>(value_c.size(1));
  int key_count = static_cast<int>(value_c.size(2));
  int head_dim = static_cast<int>(value_c.size(3));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      indices_c.size(0) == batch_count && query_head_count % kv_head_count == 0,
      "index shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count && counts_c.size(1) == query_head_count,
      "count shape mismatch");
  TORCH_CHECK(
      self_scores_c.size(0) == batch_count
          && self_scores_c.size(1) == query_head_count,
      "self score shape mismatch");
  TORCH_CHECK(max_select_count > 0, "max_select_count must be positive");
  TORCH_CHECK(
      value_mass_threshold > 0.0 && value_mass_threshold <= 1.0,
      "value mass threshold must be in (0, 1]");
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, value_c.options());
  int threads = 1;
  while (threads < head_dim) {
    threads <<= 1;
  }
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(threads + max_select_count + 1)
      * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_c.scalar_type(),
      "qabs_final_attention_from_scores_ragged_self_forward",
      [&] {
        if (value_mass_threshold < 0.999999) {
          qabs_final_attention_from_scores_ragged_kernel<scalar_t, true, true>
              <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                  value_c.data_ptr<scalar_t>(),
                  indices_c.data_ptr<int64_t>(),
                  scores_c.data_ptr<float>(),
                  counts_c.data_ptr<int64_t>(),
                  self_scores_c.data_ptr<float>(),
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  key_count,
                  max_select_count,
                  head_dim,
                  static_cast<float>(value_mass_threshold));
        } else {
          qabs_final_attention_from_scores_ragged_kernel<scalar_t, false, true>
              <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                  value_c.data_ptr<scalar_t>(),
                  indices_c.data_ptr<int64_t>(),
                  scores_c.data_ptr<float>(),
                  counts_c.data_ptr<int64_t>(),
                  self_scores_c.data_ptr<float>(),
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  key_count,
                  max_select_count,
                  head_dim,
                  1.0f);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_final_attention_from_scores_split_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    int64_t split_count) {
  TORCH_CHECK(
      value.is_cuda() && indices.is_cuda() && scores.is_cuda() && counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "score shape mismatch");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, query_heads]");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(split_count >= 1 && split_count <= 16, "split count must be in [1, 16]");
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  c10::cuda::CUDAGuard device_guard(value_c.device());
  int batch_count = static_cast<int>(value_c.size(0));
  int kv_head_count = static_cast<int>(value_c.size(1));
  int key_count = static_cast<int>(value_c.size(2));
  int head_dim = static_cast<int>(value_c.size(3));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int max_select_count = static_cast<int>(indices_c.size(2));
  int rows = batch_count * query_head_count;
  TORCH_CHECK(
      indices_c.size(0) == batch_count && query_head_count % kv_head_count == 0,
      "index shape mismatch");
  TORCH_CHECK(
      counts_c.size(0) == batch_count && counts_c.size(1) == query_head_count,
      "count shape mismatch");
  auto weights = torch::empty_like(scores_c);
  auto partial_output = torch::empty(
      {rows, split_count, head_dim}, value_c.options().dtype(at::kFloat));
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, value_c.options());
  int threads = 1;
  while (threads < head_dim) threads <<= 1;
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  qabs_softmax_weights_ragged_kernel
      <<<rows, threads, threads * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          counts_c.data_ptr<int64_t>(),
          weights.data_ptr<float>(),
          max_select_count);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_c.scalar_type(),
      "qabs_final_attention_from_scores_split_forward",
      [&] {
        qabs_split_value_attention_kernel<scalar_t>
            <<<rows * split_count, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                weights.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                partial_output.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                split_count);
        qabs_reduce_value_splits_kernel<scalar_t>
            <<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                partial_output.data_ptr<float>(),
                output.data_ptr<scalar_t>(),
                head_dim,
                split_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> qabs_final_attention_tail_reliability_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor prefix_counts) {
  TORCH_CHECK(
      value.is_cuda() && indices.is_cuda() && scores.is_cuda()
          && counts.is_cuda() && prefix_counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "score shape mismatch");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, query_heads]");
  TORCH_CHECK(prefix_counts.sizes() == counts.sizes(), "prefix count shape mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(prefix_counts.scalar_type() == at::kLong, "prefix counts must be int64");
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  auto prefix_counts_c = prefix_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(value_c.device());
  int batch_count = static_cast<int>(value_c.size(0));
  int kv_head_count = static_cast<int>(value_c.size(1));
  int key_count = static_cast<int>(value_c.size(2));
  int head_dim = static_cast<int>(value_c.size(3));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      indices_c.size(0) == batch_count && query_head_count % kv_head_count == 0,
      "index shape mismatch");
  TORCH_CHECK(max_select_count > 1, "tail reliability needs prefix and sample tokens");
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, value_c.options());
  auto reliability = torch::empty(
      {batch_count, query_head_count}, value_c.options().dtype(at::kFloat));
  int threads = 1;
  while (threads < head_dim) threads <<= 1;
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_select_count + 2 * head_dim + 16) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_c.scalar_type(),
      "qabs_final_attention_tail_reliability_forward",
      [&] {
        qabs_final_attention_tail_reliability_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                scores_c.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                prefix_counts_c.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                reliability.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, reliability};
}

std::vector<torch::Tensor> qabs_final_attention_tail_mass_gate_forward(
    torch::Tensor value,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor counts,
    torch::Tensor prefix_counts,
    double mass_threshold,
    double tail_shrinkage) {
  TORCH_CHECK(
      value.is_cuda() && indices.is_cuda() && scores.is_cuda()
          && counts.is_cuda() && prefix_counts.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(value.dim() == 4, "value must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(scores.sizes() == indices.sizes(), "score shape mismatch");
  TORCH_CHECK(counts.dim() == 2, "counts must have shape [batch, query_heads]");
  TORCH_CHECK(prefix_counts.sizes() == counts.sizes(), "prefix count shape mismatch");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "indices must be int64");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
  TORCH_CHECK(prefix_counts.scalar_type() == at::kLong, "prefix counts must be int64");
  TORCH_CHECK(mass_threshold > 0.0 && mass_threshold <= 1.0, "invalid mass threshold");
  TORCH_CHECK(tail_shrinkage >= 0.0 && tail_shrinkage <= 1.0, "invalid tail shrinkage");
  auto value_c = value.contiguous();
  auto indices_c = indices.contiguous();
  auto scores_c = scores.contiguous();
  auto counts_c = counts.contiguous();
  auto prefix_counts_c = prefix_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(value_c.device());
  int batch_count = static_cast<int>(value_c.size(0));
  int kv_head_count = static_cast<int>(value_c.size(1));
  int key_count = static_cast<int>(value_c.size(2));
  int head_dim = static_cast<int>(value_c.size(3));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int max_select_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      indices_c.size(0) == batch_count && query_head_count % kv_head_count == 0,
      "index shape mismatch");
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim}, value_c.options());
  auto active = torch::empty(
      {batch_count, query_head_count}, value_c.options().dtype(at::kFloat));
  int threads = 1;
  while (threads < head_dim) threads <<= 1;
  threads = threads < 32 ? 32 : threads;
  threads = threads > 256 ? 256 : threads;
  int blocks = batch_count * query_head_count;
  size_t shared_bytes = static_cast<size_t>(
      threads + max_select_count) * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      value_c.scalar_type(),
      "qabs_final_attention_tail_mass_gate_forward",
      [&] {
        qabs_final_attention_tail_mass_gate_kernel<scalar_t>
            <<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                value_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                scores_c.data_ptr<float>(),
                counts_c.data_ptr<int64_t>(),
                prefix_counts_c.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                active.data_ptr<float>(),
                query_head_count,
                kv_head_count,
                key_count,
                max_select_count,
                head_dim,
                static_cast<float>(mass_threshold),
                static_cast<float>(tail_shrinkage));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, active};
}

torch::Tensor qabs_mass_ladder_forward(
    torch::Tensor top_scores,
    torch::Tensor sample_scores,
    torch::Tensor sample_candidate_scores,
    torch::Tensor self_scores,
    torch::Tensor keep_counts,
    int64_t history_count,
    double mass_threshold) {
  TORCH_CHECK(top_scores.is_cuda() && sample_scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(sample_candidate_scores.is_cuda() && self_scores.is_cuda(), "scores must be CUDA");
  TORCH_CHECK(keep_counts.is_cuda(), "keep_counts must be CUDA");
  TORCH_CHECK(top_scores.scalar_type() == at::kFloat, "top_scores must be float32");
  TORCH_CHECK(sample_scores.scalar_type() == at::kFloat, "sample_scores must be float32");
  TORCH_CHECK(sample_candidate_scores.scalar_type() == at::kFloat, "sample_candidate_scores must be float32");
  TORCH_CHECK(self_scores.scalar_type() == at::kFloat, "self_scores must be float32");
  TORCH_CHECK(keep_counts.scalar_type() == at::kLong, "keep_counts must be int64");
  TORCH_CHECK(top_scores.dim() == 3 && sample_scores.dim() == 3, "score tensors must be [batch, heads, count]");
  TORCH_CHECK(sample_candidate_scores.sizes() == sample_scores.sizes(), "sample candidate shape mismatch");
  TORCH_CHECK(self_scores.dim() == 2, "self_scores must be [batch, heads]");
  TORCH_CHECK(keep_counts.dim() == 1, "keep_counts must be one dimensional");
  TORCH_CHECK(top_scores.size(0) == sample_scores.size(0) && top_scores.size(1) == sample_scores.size(1), "score shape mismatch");
  TORCH_CHECK(self_scores.size(0) == top_scores.size(0) && self_scores.size(1) == top_scores.size(1), "self score shape mismatch");
  TORCH_CHECK(keep_counts.size(0) > 0 && keep_counts.size(0) <= QABS_MAX_RUNGS, "unsupported rung count");
  TORCH_CHECK(history_count > 0, "history_count must be positive");

  auto top_c = top_scores.contiguous();
  auto sample_c = sample_scores.contiguous();
  auto sample_candidate_c = sample_candidate_scores.contiguous();
  auto self_c = self_scores.contiguous();
  auto keep_c = keep_counts.contiguous();
  c10::cuda::CUDAGuard device_guard(top_c.device());
  int batch_count = static_cast<int>(top_c.size(0));
  int head_count = static_cast<int>(top_c.size(1));
  int max_keep_count = static_cast<int>(top_c.size(2));
  int sample_count = static_cast<int>(sample_c.size(2));
  int rung_count = static_cast<int>(keep_c.size(0));
  TORCH_CHECK(sample_count > 0 && max_keep_count > 0, "score counts must be positive");
  auto output = torch::empty({batch_count, head_count, 4}, top_c.options());
  int threads = 256;
  int blocks = batch_count * head_count;
  qabs_mass_ladder_kernel<<<blocks, threads, threads * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
      top_c.data_ptr<float>(),
      sample_c.data_ptr<float>(),
      sample_candidate_c.data_ptr<float>(),
      self_c.data_ptr<float>(),
      keep_c.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      max_keep_count,
      sample_count,
      rung_count,
      static_cast<int>(history_count),
      static_cast<float>(mass_threshold));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int8_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor quantized_key,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && quantized_key.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4,
      "projected_query must have shape [batch, kv_heads, groups, projection]");
  TORCH_CHECK(
      quantized_key.dim() == 4,
      "quantized_key must have shape [batch, kv_heads, capacity, projection]");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar
          && quantized_key.scalar_type() == at::kChar,
      "query/key must be int8");
  auto query_c = projected_query.contiguous();
  auto key_c = quantized_key.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int capacity = static_cast<int>(key_c.size(2));
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(1) == kv_head_count
          && key_c.size(3) == projection_dim,
      "key shape mismatch");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim % 4 == 0,
      "projection dimension must be divisible by four");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key_count");
  auto output = torch::empty(
      {batch_count, kv_head_count, group_count, key_count},
      query_c.options().dtype(at::kInt));
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  int alpha = 1;
  int beta = 0;
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      static_cast<int>(key_count),
      group_count,
      projection_dim,
      &alpha,
      key_c.data_ptr<int8_t>(),
      CUDA_R_8I,
      projection_dim,
      static_cast<long long>(capacity) * projection_dim,
      query_c.data_ptr<int8_t>(),
      CUDA_R_8I,
      projection_dim,
      static_cast<long long>(group_count) * projection_dim,
      &beta,
      output.data_ptr<int32_t>(),
      CUDA_R_32I,
      static_cast<int>(key_count),
      static_cast<long long>(group_count) * static_cast<int>(key_count),
      batch_count * kv_head_count,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "INT8 batched GEMM failed");
  return output;
}

torch::Tensor qabs_retrieval_metrics_forward(
    torch::Tensor candidate_scores,
    torch::Tensor candidate_indices,
    torch::Tensor previous_probe,
    int64_t probe_count) {
  TORCH_CHECK(candidate_scores.is_cuda() && candidate_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(candidate_scores.dim() == 3 && candidate_indices.dim() == 3, "candidate tensors must be rank three");
  TORCH_CHECK(candidate_scores.scalar_type() == at::kFloat, "candidate scores must be float32");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "candidate indices must be int64");
  auto scores_c = candidate_scores.contiguous();
  auto indices_c = candidate_indices.contiguous();
  auto previous_c = previous_probe.contiguous();
  c10::cuda::CUDAGuard device_guard(scores_c.device());
  int batch_count = static_cast<int>(scores_c.size(0));
  int head_count = static_cast<int>(scores_c.size(1));
  int candidate_count = static_cast<int>(scores_c.size(2));
  int active_probe_count = static_cast<int>(std::min<int64_t>(probe_count, candidate_count));
  TORCH_CHECK(active_probe_count > 0 && active_probe_count <= 32, "probe_count must be in [1, 32]");
  TORCH_CHECK(indices_c.sizes() == scores_c.sizes(), "candidate shape mismatch");
  bool has_previous = previous_c.numel() > 0;
  if (has_previous) {
    TORCH_CHECK(previous_c.is_cuda() && previous_c.scalar_type() == at::kLong, "previous probe must be CUDA int64");
    TORCH_CHECK(
        previous_c.size(0) == batch_count && previous_c.size(1) == head_count
            && previous_c.size(2) == active_probe_count,
        "previous probe shape mismatch");
  }
  auto output = torch::empty(
      {batch_count, head_count, 3}, scores_c.options().dtype(at::kFloat));
  qabs_retrieval_metrics_kernel<<<
      batch_count * head_count,
      32,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          scores_c.data_ptr<float>(),
          indices_c.data_ptr<int64_t>(),
          has_previous ? previous_c.data_ptr<int64_t>() : nullptr,
          output.data_ptr<float>(),
          candidate_count,
          active_probe_count,
          has_previous);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_quota_merge_candidates_forward(
    torch::Tensor base_indices,
    torch::Tensor rescue_indices) {
  TORCH_CHECK(base_indices.is_cuda() && rescue_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(base_indices.dim() == 3 && rescue_indices.dim() == 3, "candidate tensors must be rank three");
  TORCH_CHECK(base_indices.scalar_type() == at::kLong, "base indices must be int64");
  TORCH_CHECK(rescue_indices.scalar_type() == at::kLong, "rescue indices must be int64");
  TORCH_CHECK(base_indices.device() == rescue_indices.device(), "candidate tensors must share a device");
  TORCH_CHECK(
      base_indices.size(0) == rescue_indices.size(0)
          && base_indices.size(1) == rescue_indices.size(1),
      "candidate batch and head shapes must match");
  auto base_c = base_indices.contiguous();
  auto rescue_c = rescue_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(base_c.device());
  int batch_count = static_cast<int>(base_c.size(0));
  int head_count = static_cast<int>(base_c.size(1));
  int base_count = static_cast<int>(base_c.size(2));
  int rescue_count = static_cast<int>(rescue_c.size(2));
  TORCH_CHECK(base_count > 0, "base candidate count must be positive");
  TORCH_CHECK(rescue_count > 0 && rescue_count < base_count, "rescue count must be smaller than base count");
  auto output = torch::empty_like(base_c);
  qabs_quota_merge_candidates_kernel<<<
      batch_count * head_count,
      256,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          base_c.data_ptr<int64_t>(),
          rescue_c.data_ptr<int64_t>(),
          output.data_ptr<int64_t>(),
          base_count,
          rescue_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> qabs_append_rescue_candidates_forward(
    torch::Tensor base_indices,
    torch::Tensor rescue_indices,
    int64_t history_count) {
  TORCH_CHECK(base_indices.is_cuda() && rescue_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(base_indices.dim() == 3 && rescue_indices.dim() == 3, "candidate tensors must be rank three");
  TORCH_CHECK(base_indices.scalar_type() == at::kLong, "base indices must be int64");
  TORCH_CHECK(rescue_indices.scalar_type() == at::kLong, "rescue indices must be int64");
  TORCH_CHECK(base_indices.device() == rescue_indices.device(), "candidate tensors must share a device");
  TORCH_CHECK(
      base_indices.size(0) == rescue_indices.size(0)
          && base_indices.size(1) == rescue_indices.size(1),
      "candidate batch and head shapes must match");
  TORCH_CHECK(history_count > 0, "history count must be positive");
  auto base_c = base_indices.contiguous();
  auto rescue_c = rescue_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(base_c.device());
  int batch_count = static_cast<int>(base_c.size(0));
  int head_count = static_cast<int>(base_c.size(1));
  int base_count = static_cast<int>(base_c.size(2));
  int rescue_count = static_cast<int>(rescue_c.size(2));
  TORCH_CHECK(base_count > 0 && rescue_count > 0, "candidate counts must be positive");
  int output_count = base_count + rescue_count;
  int word_count = (static_cast<int>(history_count) + 31) / 32;
  size_t shared_bytes = static_cast<size_t>(word_count) * sizeof(unsigned int);
  TORCH_CHECK(shared_bytes <= 48 * 1024, "history is too long for the shared membership bitset");
  auto output_indices = torch::empty(
      {batch_count, head_count, output_count}, base_c.options());
  auto output_valid = torch::empty(
      {batch_count, head_count, output_count},
      base_c.options().dtype(at::kBool));
  qabs_append_rescue_candidates_kernel<<<
      batch_count * head_count,
      256,
      shared_bytes,
      at::cuda::getCurrentCUDAStream()>>>(
          base_c.data_ptr<int64_t>(),
          rescue_c.data_ptr<int64_t>(),
          output_indices.data_ptr<int64_t>(),
          output_valid.data_ptr<bool>(),
          base_count,
          rescue_count,
          static_cast<int>(history_count),
          word_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_indices, output_valid};
}

torch::Tensor qabs_candidate_bucket_union_counts_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count,
    int64_t bucket_size) {
  TORCH_CHECK(candidate_indices.is_cuda(), "candidate_indices must be CUDA");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate_indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "candidate_indices must be int64");
  auto indices_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(indices_c.device());
  int batch_count = static_cast<int>(indices_c.size(0));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int selected_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(group_count > 0 && query_head_count % group_count == 0, "invalid GQA group count");
  TORCH_CHECK(history_count > 0, "history_count must be positive");
  TORCH_CHECK(bucket_size > 0, "bucket_size must be positive");
  int kv_head_count = query_head_count / static_cast<int>(group_count);
  int bucket_count = (
      static_cast<int>(history_count) + static_cast<int>(bucket_size) - 1)
      / static_cast<int>(bucket_size);
  int word_count = (bucket_count + 31) / 32;
  auto bitset = torch::zeros(
      {batch_count, kv_head_count, word_count},
      indices_c.options().dtype(at::kInt));
  auto counts = torch::zeros(
      {batch_count, kv_head_count}, indices_c.options().dtype(at::kInt));
  int total_indices = batch_count * query_head_count * selected_count;
  int threads = 256;
  int blocks = std::min(4096, (total_indices + threads - 1) / threads);
  qabs_candidate_union_counts_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          indices_c.data_ptr<int64_t>(),
          bitset.data_ptr<int32_t>(),
          counts.data_ptr<int32_t>(),
          total_indices,
          query_head_count,
          selected_count,
          kv_head_count,
          static_cast<int>(group_count),
          word_count,
          static_cast<int>(bucket_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

torch::Tensor qabs_candidate_union_counts_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count) {
  return qabs_candidate_bucket_union_counts_forward(
      candidate_indices, history_count, group_count, 1);
}

std::vector<torch::Tensor> qabs_candidate_union_compact_forward(
    torch::Tensor candidate_indices,
    int64_t history_count,
    int64_t group_count,
    int64_t output_capacity) {
  TORCH_CHECK(candidate_indices.is_cuda(), "candidate_indices must be CUDA");
  TORCH_CHECK(
      candidate_indices.dim() == 3,
      "candidate_indices must have shape [batch, query_heads, selected]");
  TORCH_CHECK(
      candidate_indices.scalar_type() == at::kLong,
      "candidate_indices must be int64");
  TORCH_CHECK(history_count > 0, "history_count must be positive");
  TORCH_CHECK(output_capacity > 0, "output_capacity must be positive");
  auto indices_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(indices_c.device());
  int batch_count = static_cast<int>(indices_c.size(0));
  int query_head_count = static_cast<int>(indices_c.size(1));
  int selected_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(
      group_count > 0 && query_head_count % group_count == 0,
      "invalid GQA group count");
  int kv_head_count = query_head_count / static_cast<int>(group_count);
  int row_count = batch_count * kv_head_count;
  int word_count = (static_cast<int>(history_count) + 31) / 32;
  auto bitset = torch::zeros(
      {batch_count, kv_head_count, word_count},
      indices_c.options().dtype(at::kInt));
  auto counts = torch::zeros(
      {batch_count, kv_head_count}, indices_c.options().dtype(at::kInt));
  auto write_offsets = torch::zeros_like(counts);
  auto output = torch::full(
      {batch_count, kv_head_count, output_capacity},
      -1,
      indices_c.options().dtype(at::kInt));
  int total_indices = batch_count * query_head_count * selected_count;
  int threads = 256;
  int blocks = std::min(4096, (total_indices + threads - 1) / threads);
  qabs_candidate_union_counts_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          indices_c.data_ptr<int64_t>(),
          bitset.data_ptr<int32_t>(),
          counts.data_ptr<int32_t>(),
          total_indices,
          query_head_count,
          selected_count,
          kv_head_count,
          static_cast<int>(group_count),
          word_count,
          1);
  qabs_candidate_union_compact_kernel<<<
      row_count,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          bitset.data_ptr<int32_t>(),
          output.data_ptr<int32_t>(),
          write_offsets.data_ptr<int32_t>(),
          row_count,
          static_cast<int>(history_count),
          word_count,
          static_cast<int>(output_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, counts};
}

torch::Tensor qabs_pca_int8_wmma_scores_forward(
    torch::Tensor padded_query,
    torch::Tensor quantized_key,
    torch::Tensor scales,
    int64_t key_count,
    int64_t group_count) {
  TORCH_CHECK(
      padded_query.is_cuda() && quantized_key.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      padded_query.dim() == 4 && padded_query.size(2) == 16,
      "padded_query must have shape [batch, kv_heads, 16, projection]");
  TORCH_CHECK(
      quantized_key.dim() == 4,
      "quantized_key must have shape [batch, kv_heads, capacity, projection]");
  TORCH_CHECK(
      scales.dim() == 4 && scales.size(3) == 1,
      "scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(
      padded_query.scalar_type() == at::kChar
          && quantized_key.scalar_type() == at::kChar,
      "query/key must be int8");
  auto query_c = padded_query.contiguous();
  auto key_c = quantized_key.contiguous();
  auto scales_c = scales.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int projection_dim = static_cast<int>(query_c.size(3));
  int capacity = static_cast<int>(key_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 16, "group_count must be in [1, 16]");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= 128 && projection_dim % 16 == 0,
      "projection dimension must be divisible by 16");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key_count");
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(1) == kv_head_count
          && key_c.size(3) == projection_dim,
      "key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * static_cast<int>(group_count);
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      scales_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * kv_head_count, (key_count + 15) / 16);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int8_wmma_scores_forward",
      [&] {
        qabs_pca_int8_wmma_scores_kernel<scalar_t><<<
            blocks,
            32,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                key_c.data_ptr<int8_t>(),
                scales_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                static_cast<int>(group_count),
                static_cast<int>(key_count),
                capacity,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4,
      "projected_query must have shape [batch, kv_heads, groups, projection]");
  TORCH_CHECK(
      packed_key.dim() == 4,
      "packed_key must have shape [batch, kv_heads, capacity, packed_projection]");
  TORCH_CHECK(
      scales.dim() == 4 && scales.size(3) == 1,
      "scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed_key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected_query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key.contiguous();
  auto scales_c = scales.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int capacity = static_cast<int>(packed_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= 128 && projection_dim % 4 == 0,
      "invalid projection dimension");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key_count");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count
          && packed_c.size(3) == projection_dim / 2,
      "packed key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int tokens_per_warp = 4;
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + warps_per_block * tokens_per_warp - 1)
          / (warps_per_block * tokens_per_warp));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_scores_forward",
      [&] {
        qabs_pca_int4_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                projection_dim / 2,
                0,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    int64_t key_count,
    int64_t prefix_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4,
      "projected_query must have shape [batch, kv_heads, groups, projection]");
  TORCH_CHECK(
      packed_key.dim() == 4,
      "packed_key must have shape [batch, kv_heads, capacity, packed_projection]");
  TORCH_CHECK(
      scales.dim() == 4 && scales.size(3) == 1,
      "scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed_key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected_query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key.contiguous();
  auto scales_c = scales.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int packed_dim = static_cast<int>(packed_c.size(3));
  int capacity = static_cast<int>(packed_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= 128 && projection_dim % 4 == 0,
      "invalid projection dimension");
  TORCH_CHECK(
      prefix_dim > 0 && prefix_dim <= projection_dim && prefix_dim % 4 == 0,
      "invalid prefix dimension");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key_count");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count
          && packed_dim == projection_dim / 2,
      "packed key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int tokens_per_warp = 4;
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + warps_per_block * tokens_per_warp - 1)
          / (warps_per_block * tokens_per_warp));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_prefix_scores_forward",
      [&] {
        qabs_pca_int4_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                packed_dim,
                0,
                static_cast<int>(prefix_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t key_count,
    int64_t prefix_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      prefix_dim > 0 && prefix_dim <= projection_dim && prefix_dim % 16 == 0,
      "prefix dimension must be a positive multiple of 16");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_prefix_scores_forward",
      [&] {
        qabs_pca_int4_chunked_prefix_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(prefix_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_group16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor group_scales,
    int64_t key_count,
    int64_t prefix_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && group_scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(
      group_scales.dim() == 4,
      "group scales must have shape [batch, kv_heads, chunks, capacity]");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte,
      "chunked key must be uint8");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = group_scales.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      prefix_dim > 0 && prefix_dim <= projection_dim && prefix_dim % 16 == 0,
      "prefix dimension must be a positive multiple of 16");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == chunk_count
          && scales_c.size(3) == capacity,
      "group scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_group16_prefix_scores_forward",
      [&] {
        qabs_pca_int4_chunked_group16_prefix_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(prefix_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_logscale16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t prefix_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "base scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(
      packed_exponents.dim() == 4,
      "packed exponents must have shape [batch, kv_heads, capacity, pairs]");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      prefix_dim > 0 && prefix_dim <= projection_dim && prefix_dim % 16 == 0,
      "prefix dimension must be a positive multiple of 16");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_logscale16_prefix_scores_forward",
      [&] {
        qabs_pca_int4_chunked_logscale16_prefix_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(prefix_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_nested_int2_logscale16_prefix_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_high2,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t prefix_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_high2.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_high2.dim() == 5 && packed_key_high2.size(4) == 4,
      "high-2-bit key must have shape [batch, kv_heads, chunks, capacity, 4]");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "base scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(
      packed_key_high2.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_high2.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      prefix_dim > 0 && prefix_dim <= projection_dim && prefix_dim % 16 == 0,
      "prefix dimension must be a positive multiple of 16");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_nested_int2_logscale16_prefix_scores_forward",
      [&] {
        qabs_pca_nested_int2_logscale16_prefix_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(prefix_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      sample_count == QABS_TILE_THREADS / 2
          || (
              sample_count >= QABS_TILE_THREADS
              && sample_count <= 4 * QABS_TILE_THREADS
              && sample_count % QABS_TILE_THREADS == 0),
      "sample count must be 128 or a multiple of 256 up to 1024");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");

  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "invalid group count");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= key_count,
      "invalid candidate capacity");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key batch/head mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  int query_head_count = kv_head_count * group_count;
  int sample_replicas = (
      static_cast<int>(sample_count) + QABS_TILE_THREADS - 1)
      / QABS_TILE_THREADS;
  int sample_keep = max(
      1,
      static_cast<int>(ceil(
          selected_fraction * sample_count / sample_replicas)));
  // Unused ragged slots are still read by the generic attention-mask gather.
  // Zero is a valid history position and keeps those masked-out slots in range.
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      query_c.options().dtype(at::kLong));
  auto proxy_scores = write_proxy_scores
      ? torch::empty(
            {batch_count, query_head_count, candidate_capacity},
            query_c.options().dtype(at::kFloat))
      : torch::empty({0}, query_c.options().dtype(at::kFloat));
  auto counts = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, query_head_count}, query_c.options().dtype(at::kFloat));
  auto overflow = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kBool));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_logscale16_sampled_quantile_forward",
      [&] {
        qabs_pca_int4_logscale16_sample_threshold_kernel<scalar_t>
            <<<batch_count * kv_head_count,
               QABS_TILE_THREADS,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(sample_count),
                sample_keep);
        dim3 blocks(
            batch_count * kv_head_count,
            (static_cast<int>(key_count) + QABS_TILE_THREADS - 1)
                / QABS_TILE_THREADS);
        if (use_dp4a) {
          qabs_pca_int4_logscale16_threshold_compact_kernel<scalar_t, true>
              <<<blocks,
                 QABS_TILE_THREADS,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  kv_head_count,
                  group_count,
                  static_cast<int>(key_count),
                  capacity,
                  projection_dim,
                  chunk_count,
                  static_cast<int>(candidate_capacity));
        } else {
          qabs_pca_int4_logscale16_threshold_compact_kernel<scalar_t, false>
              <<<blocks,
                 QABS_TILE_THREADS,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(candidate_capacity));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, proxy_scores, counts, boundaries, overflow};
}

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward(
    torch::Tensor query,
    torch::Tensor basis,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores) {
  TORCH_CHECK(
      query.is_cuda() && basis.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.dim() == 3 && basis.dim() == 4,
      "query and basis shapes are invalid");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(
      query.scalar_type() == basis.scalar_type()
          && query.scalar_type() == base_scales.scalar_type(),
      "query, basis, and scale dtypes must match");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      sample_count == QABS_TILE_THREADS / 2
          || (
              sample_count >= QABS_TILE_THREADS
              && sample_count <= 4 * QABS_TILE_THREADS
              && sample_count % QABS_TILE_THREADS == 0),
      "sample count must be 128 or a multiple of 256 up to 1024");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");

  auto query_c = query.contiguous();
  auto basis_c = basis.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int kv_head_count = static_cast<int>(packed_c.size(1));
  int projection_dim = static_cast<int>(basis_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(
      query_head_count % kv_head_count == 0,
      "query heads must be divisible by KV heads");
  int group_count = query_head_count / kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "invalid group count");
  TORCH_CHECK(
      basis_c.size(0) == batch_count
          && basis_c.size(1) == kv_head_count
          && basis_c.size(2) == head_dim,
      "query/basis shape mismatch");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= key_count,
      "invalid candidate capacity");
  TORCH_CHECK(
      packed_c.size(0) == batch_count,
      "chunked key batch mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");

  int sample_replicas = (
      static_cast<int>(sample_count) + QABS_TILE_THREADS - 1)
      / QABS_TILE_THREADS;
  int sample_keep = max(
      1,
      static_cast<int>(ceil(
          selected_fraction * sample_count / sample_replicas)));
  auto projected_query = torch::empty(
      {batch_count, kv_head_count, group_count, projection_dim},
      query_c.options().dtype(at::kChar));
  auto projected_query_scales = torch::empty(
      {batch_count, query_head_count, 1},
      query_c.options().dtype(at::kFloat));
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      query_c.options().dtype(at::kLong));
  auto proxy_scores = write_proxy_scores
      ? torch::empty(
            {batch_count, query_head_count, candidate_capacity},
            query_c.options().dtype(at::kFloat))
      : torch::empty({0}, query_c.options().dtype(at::kFloat));
  auto counts = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, query_head_count}, query_c.options().dtype(at::kFloat));
  auto overflow = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kBool));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward",
      [&] {
        qabs_pca_int4_logscale16_raw_query_sample_threshold_kernel<
            scalar_t, scalar_t>
            <<<batch_count * kv_head_count,
               QABS_TILE_THREADS,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                projected_query.data_ptr<int8_t>(),
                projected_query_scales.data_ptr<float>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                kv_head_count,
                group_count,
                head_dim,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(sample_count),
                sample_keep);
        dim3 blocks(
            batch_count * kv_head_count,
            (static_cast<int>(key_count) + QABS_TILE_THREADS - 1)
                / QABS_TILE_THREADS);
        if (use_dp4a) {
          qabs_pca_int4_logscale16_threshold_compact_kernel<scalar_t, true>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  projected_query.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        } else {
          qabs_pca_int4_logscale16_threshold_compact_kernel<scalar_t, false>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  projected_query.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      indices,
      proxy_scores,
      counts,
      boundaries,
      overflow,
      projected_query,
      projected_query_scales};
}

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_streaming_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor basis,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t history_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    double scaling,
    bool use_dp4a) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda()
          && basis.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 4 && value.sizes() == key.sizes(),
      "query/key/value shapes are invalid");
  TORCH_CHECK(
      basis.dim() == 4 && packed_key_chunked.dim() == 5
          && packed_key_chunked.size(4) == 8,
      "basis or packed key shape is invalid");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1
          && packed_exponents.dim() == 4,
      "scale or exponent shape is invalid");
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type()
          && query.scalar_type() == value.scalar_type()
          && query.scalar_type() == basis.scalar_type()
          && query.scalar_type() == base_scales.scalar_type(),
      "floating point dtypes must match");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      sample_count == QABS_TILE_THREADS / 2
          || (
              sample_count >= QABS_TILE_THREADS
              && sample_count <= 4 * QABS_TILE_THREADS
              && sample_count % QABS_TILE_THREADS == 0),
      "sample count must be 128 or a multiple of 256 up to 1024");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");

  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto value_c = value.contiguous();
  auto basis_c = basis.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_capacity = static_cast<int>(key_c.size(2));
  int index_capacity = static_cast<int>(packed_c.size(3));
  int projection_dim = static_cast<int>(basis_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  TORCH_CHECK(
      query_head_count % kv_head_count == 0,
      "query heads must be divisible by KV heads");
  int group_count = query_head_count / kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "invalid group count");
  TORCH_CHECK(
      key_c.size(0) == batch_count && key_c.size(3) == head_dim
          && key_capacity == history_count + 1,
      "key shape or history count mismatch");
  TORCH_CHECK(
      basis_c.size(0) == batch_count
          && basis_c.size(1) == kv_head_count
          && basis_c.size(2) == head_dim,
      "basis shape mismatch");
  TORCH_CHECK(
      projection_dim == chunk_count * 16
          && projection_dim <= QABS_MAX_DIMS,
      "projection dimension is unsupported");
  TORCH_CHECK(
      history_count > 0 && history_count <= index_capacity,
      "history count exceeds packed index capacity");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= history_count,
      "candidate capacity is invalid");
  TORCH_CHECK(
      packed_c.size(0) == batch_count
          && packed_c.size(1) == kv_head_count,
      "packed key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == index_capacity,
      "base scale shape mismatch");

  int sample_replicas = (
      static_cast<int>(sample_count) + QABS_TILE_THREADS - 1)
      / QABS_TILE_THREADS;
  int sample_keep = max(
      1,
      static_cast<int>(ceil(
          selected_fraction * sample_count / sample_replicas)));
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      query_c.options().dtype(at::kLong));
  auto counts = torch::zeros(
      {batch_count, query_head_count},
      query_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, query_head_count},
      query_c.options().dtype(at::kFloat));
  auto overflow = torch::zeros(
      {batch_count, query_head_count},
      query_c.options().dtype(at::kBool));
  auto projected_query = torch::empty(
      {batch_count, kv_head_count, group_count, projection_dim},
      query_c.options().dtype(at::kChar));
  auto projected_query_scales = torch::empty(
      {batch_count, query_head_count, 1},
      query_c.options().dtype(at::kFloat));
  auto output = torch::empty(
      {batch_count, query_head_count, head_dim},
      query_c.options());
  size_t shared_bytes = static_cast<size_t>(
      QABS_TILE_THREADS + candidate_capacity + 1) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pca_int4_logscale16_streaming_attention_forward",
      [&] {
        if (use_dp4a) {
          qabs_pca_int4_logscale16_streaming_attention_kernel<
              scalar_t, scalar_t, true>
              <<<batch_count * query_head_count,
                 QABS_TILE_THREADS,
                 shared_bytes,
                 at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<scalar_t>(),
                  key_c.data_ptr<scalar_t>(),
                  value_c.data_ptr<scalar_t>(),
                  basis_c.data_ptr<scalar_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  indices.data_ptr<int64_t>(),
                  counts.data_ptr<int64_t>(),
                  boundaries.data_ptr<float>(),
                  overflow.data_ptr<bool>(),
                  projected_query.data_ptr<int8_t>(),
                  projected_query_scales.data_ptr<float>(),
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  group_count,
                  head_dim,
                  static_cast<int>(history_count),
                  key_capacity,
                  index_capacity,
                  projection_dim,
                  chunk_count,
                  static_cast<int>(sample_count),
                  sample_keep,
                  static_cast<int>(candidate_capacity),
                  static_cast<float>(scaling));
        } else {
          qabs_pca_int4_logscale16_streaming_attention_kernel<
              scalar_t, scalar_t, false>
              <<<batch_count * query_head_count,
                 QABS_TILE_THREADS,
                 shared_bytes,
                 at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<scalar_t>(),
                  key_c.data_ptr<scalar_t>(),
                  value_c.data_ptr<scalar_t>(),
                  basis_c.data_ptr<scalar_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  indices.data_ptr<int64_t>(),
                  counts.data_ptr<int64_t>(),
                  boundaries.data_ptr<float>(),
                  overflow.data_ptr<bool>(),
                  projected_query.data_ptr<int8_t>(),
                  projected_query_scales.data_ptr<float>(),
                  output.data_ptr<scalar_t>(),
                  query_head_count,
                  kv_head_count,
                  group_count,
                  head_dim,
                  static_cast<int>(history_count),
                  key_capacity,
                  index_capacity,
                  projection_dim,
                  chunk_count,
                  static_cast<int>(sample_count),
                  sample_keep,
                  static_cast<int>(candidate_capacity),
                  static_cast<float>(scaling));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      output,
      indices,
      counts,
      boundaries,
      overflow,
      projected_query,
      projected_query_scales};
}

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_bound_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor chunk_squared_norms,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    bool use_dp4a,
    bool write_proxy_scores,
    bool collect_statistics) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda()
          && chunk_squared_norms.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(
      chunk_squared_norms.dim() == 4,
      "invalid chunk squared-norm shape");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      chunk_squared_norms.scalar_type() == at::kShort,
      "chunk squared norms must be int16");
  TORCH_CHECK(
      sample_count >= QABS_TILE_THREADS
          && sample_count <= 4 * QABS_TILE_THREADS
          && sample_count % QABS_TILE_THREADS == 0,
      "sample count must be 256, 512, 768, or 1024");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");

  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  auto chunk_norms_c = chunk_squared_norms.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "invalid group count");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= key_count,
      "invalid candidate capacity");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key batch/head mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  TORCH_CHECK(
      chunk_norms_c.size(0) == batch_count
          && chunk_norms_c.size(1) == kv_head_count
          && chunk_norms_c.size(2) == capacity
          && chunk_norms_c.size(3) == chunk_count,
      "chunk squared-norm shape mismatch");

  int query_head_count = kv_head_count * group_count;
  int sample_replicas = static_cast<int>(sample_count) / QABS_TILE_THREADS;
  int sample_keep = max(
      1,
      static_cast<int>(ceil(
          selected_fraction * sample_count / sample_replicas)));
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      query_c.options().dtype(at::kLong));
  auto proxy_scores = write_proxy_scores
      ? torch::empty(
            {batch_count, query_head_count, candidate_capacity},
            query_c.options().dtype(at::kFloat))
      : torch::empty({0}, query_c.options().dtype(at::kFloat));
  auto counts = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, query_head_count}, query_c.options().dtype(at::kFloat));
  auto overflow = torch::zeros(
      {batch_count, query_head_count}, query_c.options().dtype(at::kBool));
  auto key_chunk_evaluations = collect_statistics
      ? torch::zeros(
            {batch_count, kv_head_count}, query_c.options().dtype(at::kLong))
      : torch::empty({0}, query_c.options().dtype(at::kLong));
  auto query_chunk_evaluations = collect_statistics
      ? torch::zeros(
            {batch_count, query_head_count}, query_c.options().dtype(at::kLong))
      : torch::empty({0}, query_c.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_logscale16_sampled_quantile_bound_forward",
      [&] {
        qabs_pca_int4_logscale16_sample_threshold_kernel<scalar_t>
            <<<batch_count * kv_head_count,
               QABS_TILE_THREADS,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(sample_count),
                sample_keep);
        dim3 blocks(
            batch_count * kv_head_count,
            (static_cast<int>(key_count) + QABS_TILE_THREADS - 1)
                / QABS_TILE_THREADS);
        if (use_dp4a && collect_statistics) {
          qabs_pca_int4_logscale16_threshold_compact_bound_kernel<
              scalar_t, true, true>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  chunk_norms_c.data_ptr<int16_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  key_chunk_evaluations.data_ptr<int64_t>(),
                  query_chunk_evaluations.data_ptr<int64_t>(),
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        } else if (use_dp4a) {
          qabs_pca_int4_logscale16_threshold_compact_bound_kernel<
              scalar_t, true, false>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  chunk_norms_c.data_ptr<int16_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  nullptr, nullptr,
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        } else if (collect_statistics) {
          qabs_pca_int4_logscale16_threshold_compact_bound_kernel<
              scalar_t, false, true>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  chunk_norms_c.data_ptr<int16_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  key_chunk_evaluations.data_ptr<int64_t>(),
                  query_chunk_evaluations.data_ptr<int64_t>(),
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        } else {
          qabs_pca_int4_logscale16_threshold_compact_bound_kernel<
              scalar_t, false, false>
              <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<int8_t>(),
                  packed_c.data_ptr<uint8_t>(),
                  scales_c.data_ptr<scalar_t>(),
                  exponents_c.data_ptr<uint8_t>(),
                  chunk_norms_c.data_ptr<int16_t>(),
                  boundaries.data_ptr<float>(),
                  indices.data_ptr<int64_t>(),
                  write_proxy_scores ? proxy_scores.data_ptr<float>() : nullptr,
                  counts.data_ptr<int64_t>(),
                  overflow.data_ptr<bool>(),
                  nullptr, nullptr,
                  kv_head_count, group_count, static_cast<int>(key_count),
                  capacity, projection_dim, chunk_count,
                  static_cast<int>(candidate_capacity));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      indices,
      proxy_scores,
      counts,
      boundaries,
      overflow,
      key_chunk_evaluations,
      query_chunk_evaluations};
}

std::vector<torch::Tensor>
qabs_pca_int4_logscale16_sampled_quantile_exact_forward(
    torch::Tensor projected_query,
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t key_count,
    int64_t sample_count,
    double selected_fraction,
    int64_t candidate_capacity,
    double scaling) {
  TORCH_CHECK(
      projected_query.is_cuda() && query.is_cuda() && key.is_cuda()
          && packed_key_chunked.is_cuda() && base_scales.is_cuda()
          && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, kv_heads, key, dim]");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(
      projected_query.scalar_type() == at::kChar,
      "projected query must be int8");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  TORCH_CHECK(
      query.scalar_type() == base_scales.scalar_type(),
      "query/base-scale dtype mismatch");
  TORCH_CHECK(
      sample_count >= QABS_TILE_THREADS
          && sample_count <= 4 * QABS_TILE_THREADS
          && sample_count % QABS_TILE_THREADS == 0,
      "sample count must be 256, 512, 768, or 1024");
  TORCH_CHECK(
      selected_fraction > 0.0 && selected_fraction < 1.0,
      "selected fraction must be in (0, 1)");

  auto projected_c = projected_query.contiguous();
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  c10::cuda::CUDAGuard device_guard(projected_c.device());
  int batch_count = static_cast<int>(projected_c.size(0));
  int kv_head_count = static_cast<int>(projected_c.size(1));
  int group_count = static_cast<int>(projected_c.size(2));
  int projection_dim = static_cast<int>(projected_c.size(3));
  int query_head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int index_capacity = static_cast<int>(packed_c.size(3));
  int key_capacity = static_cast<int>(key_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "invalid group count");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(head_dim <= QABS_MAX_DIMS, "head dimension exceeds kernel limit");
  TORCH_CHECK(
      query_head_count == kv_head_count * group_count,
      "query/GQA head mismatch");
  TORCH_CHECK(
      query_c.size(0) == batch_count && key_c.size(0) == batch_count
          && key_c.size(1) == kv_head_count && key_c.size(3) == head_dim,
      "query/key shape mismatch");
  TORCH_CHECK(
      key_count > 0 && key_count <= index_capacity && key_count <= key_capacity,
      "invalid key count");
  TORCH_CHECK(
      candidate_capacity > 0 && candidate_capacity <= key_count,
      "invalid candidate capacity");
  TORCH_CHECK(
      scales_c.size(0) == batch_count
          && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == index_capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == index_capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");

  int sample_replicas = static_cast<int>(sample_count) / QABS_TILE_THREADS;
  int sample_keep = max(
      1,
      static_cast<int>(ceil(
          selected_fraction * sample_count / sample_replicas)));
  auto indices = torch::zeros(
      {batch_count, query_head_count, candidate_capacity},
      projected_c.options().dtype(at::kLong));
  auto exact_scores = torch::empty(
      {batch_count, query_head_count, candidate_capacity},
      projected_c.options().dtype(at::kFloat));
  auto counts = torch::zeros(
      {batch_count, query_head_count}, projected_c.options().dtype(at::kLong));
  auto boundaries = torch::empty(
      {batch_count, query_head_count}, projected_c.options().dtype(at::kFloat));
  auto overflow = torch::zeros(
      {batch_count, query_head_count}, projected_c.options().dtype(at::kBool));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pca_int4_logscale16_sampled_quantile_exact_forward",
      [&] {
        qabs_pca_int4_logscale16_sample_threshold_kernel<scalar_t>
            <<<batch_count * kv_head_count,
               QABS_TILE_THREADS,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                projected_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                index_capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(sample_count),
                sample_keep);
        dim3 blocks(
            batch_count * kv_head_count,
            (static_cast<int>(key_count) + QABS_TILE_THREADS - 1)
                / QABS_TILE_THREADS);
        qabs_pca_int4_logscale16_threshold_exact_kernel<scalar_t, scalar_t>
            <<<blocks,
               QABS_TILE_THREADS,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                projected_c.data_ptr<int8_t>(),
                query_c.data_ptr<scalar_t>(),
                key_c.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                boundaries.data_ptr<float>(),
                indices.data_ptr<int64_t>(),
                exact_scores.data_ptr<float>(),
                counts.data_ptr<int64_t>(),
                overflow.data_ptr<bool>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                key_capacity,
                index_capacity,
                projection_dim,
                chunk_count,
                head_dim,
                static_cast<int>(candidate_capacity),
                static_cast<float>(scaling));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {indices, exact_scores, counts, boundaries, overflow};
}

torch::Tensor qabs_pca_int4_logscale16_pack_into_forward(
    torch::Tensor projected_key,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    int64_t start_token) {
  TORCH_CHECK(
      projected_key.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_key.dim() == 4, "projected key must be rank four");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "base scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_exponents.dim() == 4, "packed exponents must be rank four");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(
      projected_key.scalar_type() == base_scales.scalar_type(),
      "projected key and base scale dtypes must match");
  auto key_c = projected_key.contiguous();
  TORCH_CHECK(packed_key_chunked.is_contiguous(), "chunked key must be contiguous");
  TORCH_CHECK(base_scales.is_contiguous(), "base scales must be contiguous");
  TORCH_CHECK(packed_exponents.is_contiguous(), "packed exponents must be contiguous");
  c10::cuda::CUDAGuard device_guard(key_c.device());
  int batch_count = static_cast<int>(key_c.size(0));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int new_count = static_cast<int>(key_c.size(2));
  int projection_dim = static_cast<int>(key_c.size(3));
  int chunk_count = static_cast<int>(packed_key_chunked.size(2));
  int capacity = static_cast<int>(packed_key_chunked.size(3));
  TORCH_CHECK(new_count > 0, "projected key must contain at least one token");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= QABS_MAX_DIMS
          && projection_dim % 16 == 0,
      "projection dimension must be a supported multiple of 16");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      start_token >= 0 && start_token + new_count <= capacity,
      "packed index range exceeds capacity");
  TORCH_CHECK(
      packed_key_chunked.size(0) == batch_count
          && packed_key_chunked.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      base_scales.size(0) == batch_count
          && base_scales.size(1) == kv_head_count
          && base_scales.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      packed_exponents.size(0) == batch_count
          && packed_exponents.size(1) == kv_head_count
          && packed_exponents.size(2) == capacity
          && packed_exponents.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  int block_count = batch_count * kv_head_count * new_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key_c.scalar_type(),
      "qabs_pca_int4_logscale16_pack_into_forward",
      [&] {
        qabs_pca_int4_logscale16_pack_into_kernel<scalar_t>
            <<<block_count, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
                key_c.data_ptr<scalar_t>(),
                packed_key_chunked.data_ptr<uint8_t>(),
                base_scales.data_ptr<scalar_t>(),
                packed_exponents.data_ptr<uint8_t>(),
                new_count,
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(start_token));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed_key_chunked;
}

torch::Tensor qabs_pca_int4_logscale16_chunk_norms_into_forward(
    torch::Tensor packed_key_chunked,
    torch::Tensor chunk_squared_norms,
    int64_t start_token,
    int64_t token_count) {
  TORCH_CHECK(
      packed_key_chunked.is_cuda() && chunk_squared_norms.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8
          && packed_key_chunked.scalar_type() == at::kByte,
      "packed key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(
      chunk_squared_norms.dim() == 4
          && chunk_squared_norms.scalar_type() == at::kShort,
      "chunk squared norms must be a rank-four int16 tensor");
  TORCH_CHECK(packed_key_chunked.is_contiguous(), "packed key must be contiguous");
  TORCH_CHECK(chunk_squared_norms.is_contiguous(), "chunk norms must be contiguous");
  c10::cuda::CUDAGuard device_guard(packed_key_chunked.device());
  int batch_count = static_cast<int>(packed_key_chunked.size(0));
  int kv_head_count = static_cast<int>(packed_key_chunked.size(1));
  int chunk_count = static_cast<int>(packed_key_chunked.size(2));
  int capacity = static_cast<int>(packed_key_chunked.size(3));
  TORCH_CHECK(
      token_count > 0,
      "token count must be positive");
  TORCH_CHECK(
      start_token >= 0 && start_token + token_count <= capacity,
      "chunk-norm range exceeds capacity");
  TORCH_CHECK(
      chunk_squared_norms.size(0) == batch_count
          && chunk_squared_norms.size(1) == kv_head_count
          && chunk_squared_norms.size(2) == capacity
          && chunk_squared_norms.size(3) == chunk_count,
      "chunk squared-norm shape mismatch");
  int block_count =
      batch_count * kv_head_count * static_cast<int>(token_count);
  qabs_pca_int4_logscale16_chunk_norms_into_kernel
      <<<block_count, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
          packed_key_chunked.data_ptr<uint8_t>(),
          chunk_squared_norms.data_ptr<int16_t>(),
          static_cast<int>(token_count),
          capacity,
          chunk_count,
          static_cast<int>(start_token));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return chunk_squared_norms;
}

std::vector<torch::Tensor> qabs_pca_project_query_int8_forward(
    torch::Tensor grouped_query,
    torch::Tensor basis) {
  TORCH_CHECK(grouped_query.is_cuda() && basis.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(
      grouped_query.dim() == 4,
      "grouped query must have shape [batch, kv_heads, groups, head_dim]");
  TORCH_CHECK(
      basis.dim() == 4,
      "basis must have shape [batch, kv_heads, head_dim, projection_dim]");
  TORCH_CHECK(
      grouped_query.scalar_type() == basis.scalar_type(),
      "query and basis dtypes must match");
  auto query_c = grouped_query.contiguous();
  auto basis_c = basis.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int head_dim = static_cast<int>(query_c.size(3));
  int projection_dim = static_cast<int>(basis_c.size(3));
  TORCH_CHECK(
      basis_c.size(0) == batch_count
          && basis_c.size(1) == kv_head_count
          && basis_c.size(2) == head_dim,
      "query/basis shape mismatch");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= QABS_MAX_DIMS,
      "unsupported projection dimension");
  auto projected = torch::empty(
      {batch_count, kv_head_count, group_count, projection_dim},
      query_c.options());
  auto codes = torch::empty(
      {batch_count, kv_head_count, group_count, projection_dim},
      query_c.options().dtype(at::kChar));
  auto scales = torch::empty(
      {batch_count, kv_head_count, group_count, 1},
      query_c.options().dtype(at::kFloat));
  int block_count = batch_count * kv_head_count * group_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pca_project_query_int8_forward",
      [&] {
        qabs_pca_project_query_int8_kernel<scalar_t>
            <<<block_count, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                basis_c.data_ptr<scalar_t>(),
                projected.data_ptr<scalar_t>(),
                codes.data_ptr<int8_t>(),
                scales.data_ptr<float>(),
                kv_head_count,
                group_count,
                head_dim,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {projected, codes, scales};
}

torch::Tensor qabs_pre_rope_lowfreq_int4_pack_into_forward(
    torch::Tensor post_rope_key,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t start_token,
    double rope_theta) {
  TORCH_CHECK(
      post_rope_key.is_cuda() && packed_key_chunked.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(post_rope_key.dim() == 4, "post-RoPE key must be rank four");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5
          && packed_key_chunked.size(2) == 2
          && packed_key_chunked.size(4) == 8,
      "low-frequency key must have two packed 16-D chunks");
  TORCH_CHECK(
      scales.dim() == 4 && scales.size(3) == 1,
      "low-frequency scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "packed key must be uint8");
  TORCH_CHECK(post_rope_key.scalar_type() == scales.scalar_type(), "key and scale dtypes must match");
  auto key_c = post_rope_key.contiguous();
  TORCH_CHECK(packed_key_chunked.is_contiguous(), "packed key must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  c10::cuda::CUDAGuard device_guard(key_c.device());
  int batch_count = static_cast<int>(key_c.size(0));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int new_count = static_cast<int>(key_c.size(2));
  int head_dim = static_cast<int>(key_c.size(3));
  int capacity = static_cast<int>(packed_key_chunked.size(3));
  TORCH_CHECK(new_count > 0, "post-RoPE key must contain at least one token");
  TORCH_CHECK(head_dim >= 32 && head_dim % 2 == 0, "unsupported head dimension");
  TORCH_CHECK(
      start_token >= 0 && start_token + new_count <= capacity,
      "packed low-frequency index exceeds capacity");
  TORCH_CHECK(
      packed_key_chunked.size(0) == batch_count
          && packed_key_chunked.size(1) == kv_head_count,
      "packed low-frequency key shape mismatch");
  TORCH_CHECK(
      scales.size(0) == batch_count
          && scales.size(1) == kv_head_count
          && scales.size(2) == capacity,
      "low-frequency scale shape mismatch");
  int block_count = batch_count * kv_head_count * new_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key_c.scalar_type(),
      "qabs_pre_rope_lowfreq_int4_pack_into_forward",
      [&] {
        qabs_pre_rope_lowfreq_int4_pack_into_kernel<scalar_t>
            <<<block_count, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
                key_c.data_ptr<scalar_t>(),
                packed_key_chunked.data_ptr<uint8_t>(),
                scales.data_ptr<scalar_t>(),
                new_count,
                capacity,
                head_dim,
                static_cast<int>(start_token),
                static_cast<float>(rope_theta));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed_key_chunked;
}

torch::Tensor qabs_pre_rope_lowfreq_int4_scores_forward(
    torch::Tensor post_rope_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    int64_t key_count,
    int64_t query_position,
    double rope_theta) {
  TORCH_CHECK(
      post_rope_query.is_cuda() && packed_key_chunked.is_cuda() && scales.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(post_rope_query.dim() == 3, "post-RoPE query must be rank three");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5
          && packed_key_chunked.size(2) == 2
          && packed_key_chunked.size(4) == 8,
      "low-frequency key must have two packed 16-D chunks");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "packed key must be uint8");
  TORCH_CHECK(post_rope_query.scalar_type() == scales.scalar_type(), "query and scale dtypes must match");
  auto query_c = post_rope_query.contiguous();
  TORCH_CHECK(packed_key_chunked.is_contiguous(), "packed key must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int kv_head_count = static_cast<int>(packed_key_chunked.size(1));
  int capacity = static_cast<int>(packed_key_chunked.size(3));
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA layout");
  int group_count = query_head_count / kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(head_dim >= 32 && head_dim % 2 == 0, "unsupported head dimension");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(query_position >= 0, "query position must be nonnegative");
  TORCH_CHECK(
      packed_key_chunked.size(0) == batch_count
          && scales.size(0) == batch_count
          && scales.size(1) == kv_head_count
          && scales.size(2) == capacity,
      "low-frequency index shape mismatch");

  auto query_codes = torch::empty(
      {batch_count, kv_head_count, group_count, 32},
      query_c.options().dtype(at::kChar));
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  int row_count = batch_count * kv_head_count;
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pre_rope_lowfreq_int4_scores_forward",
      [&] {
        qabs_pre_rope_lowfreq_int8_query_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                query_codes.data_ptr<int8_t>(),
                kv_head_count,
                group_count,
                head_dim,
                static_cast<int>(query_position),
                static_cast<float>(rope_theta));
        qabs_pca_int4_chunked_prefix_scores_kernel<scalar_t>
            <<<score_blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_codes.data_ptr<int8_t>(),
                packed_key_chunked.data_ptr<uint8_t>(),
                scales.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                32,
                2,
                32);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward(
    torch::Tensor post_rope_key,
    torch::Tensor packed_key,
    int64_t start_token,
    double rope_theta,
    double clip_alpha) {
  TORCH_CHECK(post_rope_key.is_cuda() && packed_key.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(post_rope_key.dim() == 4, "post-RoPE key must be rank four");
  TORCH_CHECK(
      packed_key.dim() == 4 && packed_key.size(3) == 8,
      "INT2 low-frequency key must have shape [batch, kv_heads, capacity, 8]");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed key must be uint8");
  TORCH_CHECK(clip_alpha > 0.0, "clip alpha must be positive");
  auto key_c = post_rope_key.contiguous();
  TORCH_CHECK(packed_key.is_contiguous(), "packed key must be contiguous");
  c10::cuda::CUDAGuard device_guard(key_c.device());
  int batch_count = static_cast<int>(key_c.size(0));
  int kv_head_count = static_cast<int>(key_c.size(1));
  int new_count = static_cast<int>(key_c.size(2));
  int head_dim = static_cast<int>(key_c.size(3));
  int capacity = static_cast<int>(packed_key.size(2));
  TORCH_CHECK(new_count > 0, "post-RoPE key must contain at least one token");
  TORCH_CHECK(head_dim >= 32 && head_dim % 2 == 0, "unsupported head dimension");
  TORCH_CHECK(
      start_token >= 0 && start_token + new_count <= capacity,
      "packed low-frequency index exceeds capacity");
  TORCH_CHECK(
      packed_key.size(0) == batch_count && packed_key.size(1) == kv_head_count,
      "packed low-frequency key shape mismatch");
  int block_count = batch_count * kv_head_count * new_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key_c.scalar_type(),
      "qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward",
      [&] {
        qabs_pre_rope_lowfreq_int2_fixed_pack_into_kernel<scalar_t>
            <<<block_count, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
                key_c.data_ptr<scalar_t>(),
                packed_key.data_ptr<uint8_t>(),
                new_count,
                capacity,
                head_dim,
                static_cast<int>(start_token),
                static_cast<float>(rope_theta),
                static_cast<float>(clip_alpha));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed_key;
}

torch::Tensor qabs_pre_rope_lowfreq_int2_fixed_scores_forward(
    torch::Tensor post_rope_query,
    torch::Tensor packed_key,
    int64_t key_count,
    int64_t query_position,
    double rope_theta) {
  TORCH_CHECK(post_rope_query.is_cuda() && packed_key.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(post_rope_query.dim() == 3, "post-RoPE query must be rank three");
  TORCH_CHECK(
      packed_key.dim() == 4 && packed_key.size(3) == 8,
      "INT2 low-frequency key must have shape [batch, kv_heads, capacity, 8]");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed key must be uint8");
  auto query_c = post_rope_query.contiguous();
  TORCH_CHECK(packed_key.is_contiguous(), "packed key must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int query_head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int kv_head_count = static_cast<int>(packed_key.size(1));
  int capacity = static_cast<int>(packed_key.size(2));
  TORCH_CHECK(query_head_count % kv_head_count == 0, "invalid GQA layout");
  int group_count = query_head_count / kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(head_dim >= 32 && head_dim % 2 == 0, "unsupported head dimension");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(query_position >= 0, "query position must be nonnegative");
  TORCH_CHECK(
      packed_key.size(0) == batch_count,
      "low-frequency index batch mismatch");
  auto query_codes = torch::empty(
      {batch_count, kv_head_count, group_count, 32},
      query_c.options().dtype(at::kChar));
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  int row_count = batch_count * kv_head_count;
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_pre_rope_lowfreq_int2_fixed_scores_forward",
      [&] {
        qabs_pre_rope_lowfreq_int8_query_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                query_codes.data_ptr<int8_t>(),
                kv_head_count,
                group_count,
                head_dim,
                static_cast<int>(query_position),
                static_cast<float>(rope_theta));
      });
  qabs_pre_rope_lowfreq_int2_fixed_scores_kernel<<<
      score_blocks,
      QABS_TILE_THREADS,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          query_codes.data_ptr<int8_t>(),
          packed_key.data_ptr<uint8_t>(),
          output.data_ptr<float>(),
          group_count,
          static_cast<int>(key_count),
          capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_selected_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && dimension_indices.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(dimension_indices.dim() == 4, "dimension indices must be rank four");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected query must be int8");
  TORCH_CHECK(dimension_indices.scalar_type() == at::kInt, "dimension indices must be int32");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto dimensions_c = dimension_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int selected_count = static_cast<int>(dimensions_c.size(3));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(selected_count > 0 && selected_count <= projection_dim, "invalid selected count");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      dimensions_c.size(0) == batch_count
          && dimensions_c.size(1) == kv_head_count
          && dimensions_c.size(2) == group_count,
      "dimension index shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_selected_scores_forward",
      [&] {
        qabs_pca_int4_chunked_selected_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                dimensions_c.data_ptr<int32_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                selected_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_shared_selected_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && dimension_indices.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(dimension_indices.dim() == 3, "dimension indices must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected query must be int8");
  TORCH_CHECK(dimension_indices.scalar_type() == at::kInt, "dimension indices must be int32");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto dimensions_c = dimension_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int selected_count = static_cast<int>(dimensions_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(selected_count > 0 && selected_count <= projection_dim, "invalid selected count");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      dimensions_c.size(0) == batch_count
          && dimensions_c.size(1) == kv_head_count,
      "dimension index shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, key_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_shared_selected_scores_forward",
      [&] {
        qabs_pca_int4_chunked_shared_selected_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                dimensions_c.data_ptr<int32_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                selected_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_shared_selected_add_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor dimension_indices,
    torch::Tensor query_scales,
    torch::Tensor score_cache,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && dimension_indices.is_cuda()
          && query_scales.is_cuda() && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(dimension_indices.dim() == 3, "dimension indices must be rank three");
  TORCH_CHECK(query_scales.dim() == 4 && query_scales.size(3) == 1, "invalid query scale shape");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected query must be int8");
  TORCH_CHECK(dimension_indices.scalar_type() == at::kInt, "dimension indices must be int32");
  TORCH_CHECK(query_scales.scalar_type() == at::kFloat, "query scales must be float32");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto dimensions_c = dimension_indices.contiguous();
  auto query_scales_c = query_scales.contiguous();
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int selected_count = static_cast<int>(dimensions_c.size(2));
  int query_head_count = kv_head_count * group_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(selected_count > 0 && selected_count <= projection_dim, "invalid selected count");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      dimensions_c.size(0) == batch_count
          && dimensions_c.size(1) == kv_head_count,
      "dimension index shape mismatch");
  TORCH_CHECK(
      query_scales_c.size(0) == batch_count
          && query_scales_c.size(1) == kv_head_count
          && query_scales_c.size(2) == group_count,
      "query scale shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_shared_selected_add_forward",
      [&] {
        qabs_pca_int4_chunked_shared_selected_add_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                dimensions_c.data_ptr<int32_t>(),
                query_scales_c.data_ptr<float>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                selected_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_contiguous_add_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor query_scales,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && query_scales.is_cuda() && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(query_scales.dim() == 4 && query_scales.size(3) == 1, "invalid query scale shape");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected query must be int8");
  TORCH_CHECK(query_scales.scalar_type() == at::kFloat, "query scales must be float32");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto query_scales_c = query_scales.contiguous();
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int selected_count = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int projection_dim = chunk_count * 16;
  int query_head_count = kv_head_count * group_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(selected_count > 0 && selected_count % 16 == 0, "selected count must be a positive multiple of 16");
  TORCH_CHECK(start_dim >= 0 && start_dim % 16 == 0, "start dimension must be a nonnegative multiple of 16");
  TORCH_CHECK(start_dim + selected_count <= projection_dim, "selected range exceeds projection");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      query_scales_c.size(0) == batch_count
          && query_scales_c.size(1) == kv_head_count
          && query_scales_c.size(2) == group_count,
      "query scale shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_contiguous_add_forward",
      [&] {
        qabs_pca_int4_chunked_contiguous_add_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                query_scales_c.data_ptr<float>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                selected_count,
                chunk_count,
                static_cast<int>(start_dim / 16));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_contiguous_delta_add_forward(
    torch::Tensor projected_query,
    torch::Tensor previous_projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim,
    int64_t selected_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && previous_projected_query.is_cuda()
          && packed_key_chunked.is_cuda() && scales.is_cuda()
          && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4
          && previous_projected_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(projected_query.scalar_type() == previous_projected_query.scalar_type(), "query dtypes must match");
  TORCH_CHECK(projected_query.scalar_type() == scales.scalar_type(), "query and key scale dtypes must match");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto previous_query_c = previous_projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(selected_count > 0 && selected_count % 16 == 0, "selected count must be a positive multiple of 16");
  TORCH_CHECK(start_dim >= 0 && start_dim % 16 == 0, "start dimension must be a nonnegative multiple of 16");
  TORCH_CHECK(start_dim + selected_count <= projection_dim, "selected range exceeds projection");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 blocks(
      batch_count * kv_head_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_contiguous_delta_add_forward",
      [&] {
        qabs_pca_int4_chunked_contiguous_delta_add_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                previous_query_c.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                static_cast<int>(selected_count),
                chunk_count,
                static_cast<int>(start_dim / 16));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_band_error_feedback_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && spectral_weights.is_cuda()
          && anchor_query.is_cuda() && selected_chunk.is_cuda()
          && gate_signal.is_cuda() && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4 && anchor_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(spectral_weights.dim() == 3, "spectral weights must be rank three");
  TORCH_CHECK(selected_chunk.dim() == 2, "selected chunk must be rank two");
  TORCH_CHECK(gate_signal.dim() == 2, "gate signal must be rank two");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(spectral_weights.scalar_type() == at::kFloat, "spectral weights must be float32");
  TORCH_CHECK(selected_chunk.scalar_type() == at::kInt, "selected chunk must be int32");
  TORCH_CHECK(gate_signal.scalar_type() == at::kFloat, "gate signal must be float32");
  TORCH_CHECK(projected_query.scalar_type() == anchor_query.scalar_type(), "anchor query dtype must match");
  TORCH_CHECK(projected_query.scalar_type() == scales.scalar_type(), "query and key scale dtypes must match");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto weights_c = spectral_weights.contiguous();
  TORCH_CHECK(anchor_query.is_contiguous(), "anchor query must be contiguous");
  TORCH_CHECK(selected_chunk.is_contiguous(), "selected chunk must be contiguous");
  TORCH_CHECK(gate_signal.is_contiguous(), "gate signal must be contiguous");
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int row_count = batch_count * kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      weights_c.size(0) == batch_count
          && weights_c.size(1) == kv_head_count
          && weights_c.size(2) == projection_dim,
      "spectral weight shape mismatch");
  TORCH_CHECK(
      selected_chunk.size(0) == batch_count
          && selected_chunk.size(1) == kv_head_count
          && gate_signal.sizes() == selected_chunk.sizes(),
      "band output shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_band_error_feedback_forward",
      [&] {
        qabs_spectral_band_select_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                weights_c.data_ptr<float>(),
                anchor_query.data_ptr<scalar_t>(),
                nullptr,
                selected_chunk.data_ptr<int32_t>(),
                gate_signal.data_ptr<float>(),
                group_count,
                projection_dim,
                chunk_count);
        qabs_pca_int4_chunked_band_delta_add_kernel<scalar_t>
            <<<score_blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                anchor_query.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                selected_chunk.data_ptr<int32_t>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count);
        qabs_spectral_band_anchor_update_kernel<scalar_t>
            <<<row_count, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                selected_chunk.data_ptr<int32_t>(),
                anchor_query.data_ptr<scalar_t>(),
                group_count,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_band_error_feedback_masked_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor active_mask,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && spectral_weights.is_cuda()
          && anchor_query.is_cuda() && active_mask.is_cuda()
          && selected_chunk.is_cuda() && gate_signal.is_cuda()
          && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4 && anchor_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(spectral_weights.dim() == 3, "spectral weights must be rank three");
  TORCH_CHECK(active_mask.dim() == 2, "active mask must be rank two");
  TORCH_CHECK(selected_chunk.dim() == 2, "selected chunk must be rank two");
  TORCH_CHECK(gate_signal.dim() == 2, "gate signal must be rank two");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(active_mask.scalar_type() == at::kByte, "active mask must be uint8");
  TORCH_CHECK(spectral_weights.scalar_type() == at::kFloat, "spectral weights must be float32");
  TORCH_CHECK(selected_chunk.scalar_type() == at::kInt, "selected chunk must be int32");
  TORCH_CHECK(gate_signal.scalar_type() == at::kFloat, "gate signal must be float32");
  TORCH_CHECK(projected_query.scalar_type() == anchor_query.scalar_type(), "anchor query dtype must match");
  TORCH_CHECK(projected_query.scalar_type() == scales.scalar_type(), "query and key scale dtypes must match");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto weights_c = spectral_weights.contiguous();
  auto active_c = active_mask.contiguous();
  TORCH_CHECK(anchor_query.is_contiguous(), "anchor query must be contiguous");
  TORCH_CHECK(selected_chunk.is_contiguous(), "selected chunk must be contiguous");
  TORCH_CHECK(gate_signal.is_contiguous(), "gate signal must be contiguous");
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int row_count = batch_count * kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      active_c.size(0) == batch_count && active_c.size(1) == kv_head_count,
      "active mask shape mismatch");
  TORCH_CHECK(
      weights_c.size(0) == batch_count
          && weights_c.size(1) == kv_head_count
          && weights_c.size(2) == projection_dim,
      "spectral weight shape mismatch");
  TORCH_CHECK(
      selected_chunk.size(0) == batch_count
          && selected_chunk.size(1) == kv_head_count
          && gate_signal.sizes() == selected_chunk.sizes(),
      "band output shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_band_error_feedback_masked_forward",
      [&] {
        qabs_spectral_band_select_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                weights_c.data_ptr<float>(),
                anchor_query.data_ptr<scalar_t>(),
                active_c.data_ptr<uint8_t>(),
                selected_chunk.data_ptr<int32_t>(),
                gate_signal.data_ptr<float>(),
                group_count,
                projection_dim,
                chunk_count);
        qabs_pca_int4_chunked_band_delta_add_kernel<scalar_t>
            <<<score_blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                anchor_query.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                selected_chunk.data_ptr<int32_t>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count);
        qabs_spectral_band_anchor_update_kernel<scalar_t>
            <<<row_count, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                selected_chunk.data_ptr<int32_t>(),
                anchor_query.data_ptr<scalar_t>(),
                group_count,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor active_mask,
    torch::Tensor selected_chunk,
    torch::Tensor gate_signal,
    torch::Tensor score_cache,
    int64_t key_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda()
          && spectral_weights.is_cuda() && anchor_query.is_cuda()
          && active_mask.is_cuda() && selected_chunk.is_cuda()
          && gate_signal.is_cuda() && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4 && anchor_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "base scales must have shape [batch, kv_heads, capacity, 1]");
  TORCH_CHECK(packed_exponents.dim() == 4, "packed exponents must be rank four");
  TORCH_CHECK(spectral_weights.dim() == 3, "spectral weights must be rank three");
  TORCH_CHECK(active_mask.dim() == 2, "active mask must be rank two");
  TORCH_CHECK(selected_chunk.dim() == 2, "selected chunk must be rank two");
  TORCH_CHECK(gate_signal.dim() == 2, "gate signal must be rank two");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "packed key and exponents must be uint8");
  TORCH_CHECK(active_mask.scalar_type() == at::kByte, "active mask must be uint8");
  TORCH_CHECK(spectral_weights.scalar_type() == at::kFloat, "spectral weights must be float32");
  TORCH_CHECK(selected_chunk.scalar_type() == at::kInt, "selected chunk must be int32");
  TORCH_CHECK(gate_signal.scalar_type() == at::kFloat, "gate signal must be float32");
  TORCH_CHECK(projected_query.scalar_type() == anchor_query.scalar_type(), "anchor query dtype must match");
  TORCH_CHECK(projected_query.scalar_type() == base_scales.scalar_type(), "query and base scale dtypes must match");
  TORCH_CHECK(score_cache.scalar_type() == base_scales.scalar_type(), "score cache and base scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  auto weights_c = spectral_weights.contiguous();
  auto active_c = active_mask.contiguous();
  TORCH_CHECK(anchor_query.is_contiguous(), "anchor query must be contiguous");
  TORCH_CHECK(selected_chunk.is_contiguous(), "selected chunk must be contiguous");
  TORCH_CHECK(gate_signal.is_contiguous(), "gate signal must be contiguous");
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int row_count = batch_count * kv_head_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      active_c.size(0) == batch_count && active_c.size(1) == kv_head_count,
      "active mask shape mismatch");
  TORCH_CHECK(
      weights_c.size(0) == batch_count
          && weights_c.size(1) == kv_head_count
          && weights_c.size(2) == projection_dim,
      "spectral weight shape mismatch");
  TORCH_CHECK(
      selected_chunk.size(0) == batch_count
          && selected_chunk.size(1) == kv_head_count
          && gate_signal.sizes() == selected_chunk.sizes(),
      "band output shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "base scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "packed exponent shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward",
      [&] {
        qabs_spectral_band_select_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                weights_c.data_ptr<float>(),
                anchor_query.data_ptr<scalar_t>(),
                active_c.data_ptr<uint8_t>(),
                selected_chunk.data_ptr<int32_t>(),
                gate_signal.data_ptr<float>(),
                group_count,
                projection_dim,
                chunk_count);
        qabs_pca_int4_chunked_logscale16_band_delta_add_kernel<scalar_t>
            <<<score_blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                anchor_query.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                selected_chunk.data_ptr<int32_t>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count);
        qabs_spectral_band_anchor_update_kernel<scalar_t>
            <<<row_count, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                selected_chunk.data_ptr<int32_t>(),
                anchor_query.data_ptr<scalar_t>(),
                group_count,
                projection_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_one_shot_band_plan_forward(
    torch::Tensor projected_query,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor top_values,
    torch::Tensor keep_counts,
    torch::Tensor planned_bands,
    torch::Tensor crossing_risk,
    int64_t total_token_count,
    double target_recall) {
  TORCH_CHECK(
      projected_query.is_cuda() && spectral_weights.is_cuda()
          && anchor_query.is_cuda() && top_values.is_cuda()
          && keep_counts.is_cuda() && planned_bands.is_cuda()
          && crossing_risk.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      spectral_weights.device() == projected_query.device()
          && anchor_query.device() == projected_query.device()
          && top_values.device() == projected_query.device()
          && keep_counts.device() == projected_query.device()
          && planned_bands.device() == projected_query.device()
          && crossing_risk.device() == projected_query.device(),
      "inputs must be on the same CUDA device");
  TORCH_CHECK(
      projected_query.dim() == 4 && anchor_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(spectral_weights.dim() == 3, "spectral weights must be rank three");
  TORCH_CHECK(top_values.dim() == 3, "top values must be rank three");
  TORCH_CHECK(keep_counts.dim() == 2, "keep counts must be rank two");
  TORCH_CHECK(planned_bands.dim() == 2, "planned bands must be rank two");
  TORCH_CHECK(crossing_risk.dim() == 2, "crossing risk must be rank two");
  TORCH_CHECK(spectral_weights.scalar_type() == at::kFloat, "spectral weights must be float32");
  TORCH_CHECK(top_values.scalar_type() == at::kFloat, "top values must be float32");
  TORCH_CHECK(keep_counts.scalar_type() == at::kLong, "keep counts must be int64");
  TORCH_CHECK(planned_bands.scalar_type() == at::kInt, "planned bands must be int32");
  TORCH_CHECK(crossing_risk.scalar_type() == at::kFloat, "crossing risk must be float32");
  TORCH_CHECK(projected_query.scalar_type() == anchor_query.scalar_type(), "anchor query dtype must match");
  TORCH_CHECK(target_recall > 0.0 && target_recall < 1.0, "target recall must be in (0, 1)");
  auto query_c = projected_query.contiguous();
  auto weights_c = spectral_weights.contiguous();
  auto anchor_c = anchor_query.contiguous();
  auto top_c = top_values.contiguous();
  auto keep_c = keep_counts.contiguous();
  TORCH_CHECK(planned_bands.is_contiguous(), "planned bands must be contiguous");
  TORCH_CHECK(crossing_risk.is_contiguous(), "crossing risk must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int candidate_count = static_cast<int>(top_c.size(2));
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim > 0 && projection_dim <= QABS_MAX_DIMS && projection_dim % 16 == 0, "invalid projection dimension");
  TORCH_CHECK(candidate_count > 1 && candidate_count <= total_token_count, "invalid candidate count");
  TORCH_CHECK(
      weights_c.size(0) == batch_count
          && weights_c.size(1) == kv_head_count
          && weights_c.size(2) == projection_dim,
      "spectral weight shape mismatch");
  TORCH_CHECK(
      top_c.size(0) == batch_count && top_c.size(1) == query_head_count,
      "top-value shape mismatch");
  TORCH_CHECK(
      keep_c.size(0) == batch_count && keep_c.size(1) == query_head_count,
      "keep-count shape mismatch");
  TORCH_CHECK(
      planned_bands.size(0) == batch_count
          && planned_bands.size(1) == kv_head_count,
      "planned-band shape mismatch");
  TORCH_CHECK(crossing_risk.sizes() == keep_c.sizes(), "risk shape mismatch");
  int row_count = batch_count * kv_head_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_one_shot_band_plan_forward",
      [&] {
        qabs_one_shot_band_plan_kernel<scalar_t>
            <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                weights_c.data_ptr<float>(),
                anchor_c.data_ptr<scalar_t>(),
                top_c.data_ptr<float>(),
                keep_c.data_ptr<int64_t>(),
                planned_bands.data_ptr<int32_t>(),
                crossing_risk.data_ptr<float>(),
                kv_head_count,
                group_count,
                projection_dim,
                candidate_count,
                static_cast<int>(total_token_count),
                static_cast<float>(target_recall));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return planned_bands;
}

torch::Tensor qabs_pca_int4_chunked_spectral_gated_delta_add_forward(
    torch::Tensor projected_query,
    torch::Tensor previous_projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor spectral_weights,
    torch::Tensor anchor_query,
    torch::Tensor refresh_mask,
    torch::Tensor gate_signal,
    torch::Tensor refresh_indices,
    torch::Tensor score_cache,
    int64_t key_count,
    int64_t start_dim,
    double threshold,
    int64_t refresh_count) {
  TORCH_CHECK(
      projected_query.is_cuda() && previous_projected_query.is_cuda()
          && packed_key_chunked.is_cuda() && scales.is_cuda()
          && spectral_weights.is_cuda() && anchor_query.is_cuda()
          && refresh_mask.is_cuda() && gate_signal.is_cuda()
          && refresh_indices.is_cuda()
          && score_cache.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      projected_query.dim() == 4
          && previous_projected_query.sizes() == projected_query.sizes()
          && anchor_query.sizes() == projected_query.sizes(),
      "invalid projected query shapes");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "chunked key must have shape [batch, kv_heads, chunks, capacity, 8]");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(spectral_weights.dim() == 3, "spectral weights must be rank three");
  TORCH_CHECK(refresh_mask.dim() == 2, "refresh mask must be rank two");
  TORCH_CHECK(gate_signal.dim() == 2, "gate signal must be rank two");
  TORCH_CHECK(refresh_indices.dim() == 2, "refresh indices must be rank two");
  TORCH_CHECK(score_cache.dim() == 3, "score cache must be rank three");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "chunked key must be uint8");
  TORCH_CHECK(refresh_mask.scalar_type() == at::kByte, "refresh mask must be uint8");
  TORCH_CHECK(spectral_weights.scalar_type() == at::kFloat, "spectral weights must be float32");
  TORCH_CHECK(gate_signal.scalar_type() == at::kFloat, "gate signal must be float32");
  TORCH_CHECK(refresh_indices.scalar_type() == at::kInt, "refresh indices must be int32");
  TORCH_CHECK(projected_query.scalar_type() == previous_projected_query.scalar_type(), "query dtypes must match");
  TORCH_CHECK(projected_query.scalar_type() == anchor_query.scalar_type(), "anchor query dtype must match");
  TORCH_CHECK(projected_query.scalar_type() == scales.scalar_type(), "query and key scale dtypes must match");
  TORCH_CHECK(score_cache.scalar_type() == scales.scalar_type(), "score cache and key scales must have the same dtype");
  auto query_c = projected_query.contiguous();
  auto previous_query_c = previous_projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto weights_c = spectral_weights.contiguous();
  TORCH_CHECK(anchor_query.is_contiguous(), "anchor query must be contiguous");
  TORCH_CHECK(refresh_mask.is_contiguous(), "refresh mask must be contiguous");
  TORCH_CHECK(gate_signal.is_contiguous(), "gate signal must be contiguous");
  TORCH_CHECK(refresh_indices.is_contiguous(), "refresh indices must be contiguous");
  TORCH_CHECK(score_cache.is_contiguous(), "score cache must be contiguous");
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  TORCH_CHECK(group_count > 0 && group_count <= 8, "group_count must be in [1, 8]");
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(start_dim > 0 && start_dim % 16 == 0, "tail start must be a positive multiple of 16");
  TORCH_CHECK(start_dim < projection_dim, "tail start must be within projection");
  TORCH_CHECK(
      threshold > 0.0 || refresh_count > 0,
      "a spectral threshold or top-k refresh count is required");
  TORCH_CHECK(
      refresh_count >= 0 && refresh_count <= kv_head_count,
      "invalid top-k refresh count");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      weights_c.size(0) == batch_count
          && weights_c.size(1) == kv_head_count
          && weights_c.size(2) == projection_dim,
      "spectral weight shape mismatch");
  TORCH_CHECK(
      refresh_mask.size(0) == batch_count
          && refresh_mask.size(1) == kv_head_count
          && gate_signal.sizes() == refresh_mask.sizes()
          && refresh_indices.size(0) == batch_count
          && refresh_indices.size(1) == kv_head_count,
      "gate output shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count,
      "chunked key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(
      score_cache.size(0) == batch_count
          && score_cache.size(1) == query_head_count
          && score_cache.size(2) == capacity,
      "score cache shape mismatch");
  int row_count = batch_count * kv_head_count;
  dim3 score_blocks(
      row_count,
      (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_spectral_gated_delta_add_forward",
      [&] {
        if (refresh_count > 0) {
          qabs_spectral_residual_topk_gate_kernel<scalar_t>
              <<<batch_count,
                 QABS_TILE_THREADS,
                 query_head_count * sizeof(float),
                 at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<scalar_t>(),
                  weights_c.data_ptr<float>(),
                  anchor_query.data_ptr<scalar_t>(),
                  refresh_mask.data_ptr<uint8_t>(),
                  gate_signal.data_ptr<float>(),
                  refresh_indices.data_ptr<int32_t>(),
                  kv_head_count,
                  group_count,
                  projection_dim,
                  static_cast<int>(start_dim),
                  static_cast<int>(refresh_count));
        } else {
          qabs_spectral_residual_gate_kernel<scalar_t>
              <<<row_count, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                  query_c.data_ptr<scalar_t>(),
                  weights_c.data_ptr<float>(),
                  anchor_query.data_ptr<scalar_t>(),
                  refresh_mask.data_ptr<uint8_t>(),
                  gate_signal.data_ptr<float>(),
                  group_count,
                  projection_dim,
                  static_cast<int>(start_dim),
                  static_cast<float>(threshold));
        }
        qabs_pca_int4_chunked_mixed_delta_add_kernel<scalar_t>
            <<<score_blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<scalar_t>(),
                previous_query_c.data_ptr<scalar_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                refresh_mask.data_ptr<uint8_t>(),
                score_cache.data_ptr<scalar_t>(),
                kv_head_count,
                group_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                static_cast<int>(start_dim / 16),
                chunk_count);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score_cache;
}

torch::Tensor qabs_pca_int4_chunked_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor scales,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && scales.is_cuda() && candidate_indices.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "query must be int8");
  TORCH_CHECK(packed_key_chunked.scalar_type() == at::kByte, "key must be uint8");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = scales.contiguous();
  auto indices_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      start_dim >= 0 && end_dim <= projection_dim && start_dim < end_dim
          && start_dim % 16 == 0 && end_dim % 16 == 0,
      "range must align to 16-dimension chunks");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_candidate_range_scores_forward",
      [&] {
        qabs_pca_int4_chunked_candidate_range_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                indices_c.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                candidate_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(start_dim),
                static_cast<int>(end_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key_chunked.is_cuda()
          && base_scales.is_cuda() && packed_exponents.is_cuda()
          && candidate_indices.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base-scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "query must be int8");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "key and exponents must be uint8");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "indices must be int64");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  auto indices_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int candidate_count = static_cast<int>(indices_c.size(2));
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      start_dim >= 0 && end_dim <= projection_dim && start_dim < end_dim
          && start_dim % 16 == 0 && end_dim % 16 == 0,
      "range must align to 16-dimension chunks");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      indices_c.size(0) == batch_count && indices_c.size(1) == query_head_count,
      "candidate shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "exponent shape mismatch");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward",
      [&] {
        qabs_pca_int4_chunked_logscale16_candidate_range_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                indices_c.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                candidate_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(start_dim),
                static_cast<int>(end_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_logscale16_selected_block_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor query_scales,
    torch::Tensor packed_key_chunked,
    torch::Tensor base_scales,
    torch::Tensor packed_exponents,
    torch::Tensor selected_blocks,
    int64_t key_count,
    int64_t block_size,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && query_scales.is_cuda()
          && packed_key_chunked.is_cuda() && base_scales.is_cuda()
          && packed_exponents.is_cuda() && selected_blocks.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(
      packed_key_chunked.dim() == 5 && packed_key_chunked.size(4) == 8,
      "invalid chunked key shape");
  TORCH_CHECK(
      base_scales.dim() == 4 && base_scales.size(3) == 1,
      "invalid base-scale shape");
  TORCH_CHECK(packed_exponents.dim() == 4, "invalid exponent shape");
  TORCH_CHECK(selected_blocks.dim() == 3, "selected blocks must be rank three");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "query must be int8");
  TORCH_CHECK(query_scales.scalar_type() == at::kFloat, "query scales must be float");
  TORCH_CHECK(
      packed_key_chunked.scalar_type() == at::kByte
          && packed_exponents.scalar_type() == at::kByte,
      "key and exponents must be uint8");
  TORCH_CHECK(selected_blocks.scalar_type() == at::kLong, "blocks must be int64");
  TORCH_CHECK(block_size > 0, "invalid block size");
  auto query_c = projected_query.contiguous();
  auto query_scales_c = query_scales.contiguous();
  auto packed_c = packed_key_chunked.contiguous();
  auto scales_c = base_scales.contiguous();
  auto exponents_c = packed_exponents.contiguous();
  auto blocks_c = selected_blocks.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int chunk_count = static_cast<int>(packed_c.size(2));
  int capacity = static_cast<int>(packed_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int selected_block_count = static_cast<int>(blocks_c.size(2));
  int candidate_count = selected_block_count * static_cast<int>(block_size);
  TORCH_CHECK(projection_dim == chunk_count * 16, "projection/chunk mismatch");
  TORCH_CHECK(
      start_dim >= 0 && end_dim <= projection_dim && start_dim < end_dim
          && start_dim % 16 == 0 && end_dim % 16 == 0,
      "range must align to 16-dimension chunks");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key count");
  TORCH_CHECK(
      blocks_c.size(0) == batch_count && blocks_c.size(1) == query_head_count,
      "selected-block shape mismatch");
  TORCH_CHECK(
      query_scales_c.numel() == batch_count * query_head_count,
      "query-scale shape mismatch");
  TORCH_CHECK(
      exponents_c.size(0) == batch_count
          && exponents_c.size(1) == kv_head_count
          && exponents_c.size(2) == capacity
          && exponents_c.size(3) == (chunk_count + 1) / 2,
      "exponent shape mismatch");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  dim3 grid(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_logscale16_selected_block_scores_forward",
      [&] {
        qabs_pca_int4_logscale16_selected_block_scores_kernel<scalar_t>
            <<<grid, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                query_scales_c.data_ptr<float>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                exponents_c.data_ptr<uint8_t>(),
                blocks_c.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                selected_block_count,
                static_cast<int>(block_size),
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                chunk_count,
                static_cast<int>(start_dim),
                static_cast<int>(end_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_microblock_local_to_token_indices_forward(
    torch::Tensor selected_blocks,
    torch::Tensor local_indices,
    int64_t key_count,
    int64_t block_size) {
  TORCH_CHECK(selected_blocks.is_cuda() && local_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(
      selected_blocks.dim() == 3 && local_indices.dim() == 3,
      "block and local indices must be rank three");
  TORCH_CHECK(
      selected_blocks.scalar_type() == at::kLong
          && local_indices.scalar_type() == at::kLong,
      "indices must be int64");
  TORCH_CHECK(
      selected_blocks.size(0) == local_indices.size(0)
          && selected_blocks.size(1) == local_indices.size(1),
      "row shape mismatch");
  TORCH_CHECK(key_count > 0 && block_size > 0, "invalid key/block size");
  auto blocks_c = selected_blocks.contiguous();
  auto local_c = local_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(blocks_c.device());
  int batch_count = static_cast<int>(blocks_c.size(0));
  int query_head_count = static_cast<int>(blocks_c.size(1));
  int selected_block_count = static_cast<int>(blocks_c.size(2));
  int candidate_count = static_cast<int>(local_c.size(2));
  auto output = torch::empty_like(local_c);
  dim3 grid(
      batch_count * query_head_count,
      (candidate_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  qabs_microblock_local_to_token_indices_kernel
      <<<grid, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          blocks_c.data_ptr<int64_t>(),
          local_c.data_ptr<int64_t>(),
          output.data_ptr<int64_t>(),
          selected_block_count,
          candidate_count,
          static_cast<int>(key_count),
          static_cast<int>(block_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_microblock_expected_max_scores_forward(
    torch::Tensor block_mean,
    torch::Tensor block_variance,
    torch::Tensor projected_query,
    int64_t block_count,
    int64_t block_size,
    int64_t last_block_size) {
  TORCH_CHECK(
      block_mean.is_cuda() && block_variance.is_cuda()
          && projected_query.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      block_mean.dim() == 4 && block_variance.dim() == 4,
      "block summaries must be rank four");
  TORCH_CHECK(projected_query.dim() == 4, "projected query must be rank four");
  TORCH_CHECK(
      block_mean.sizes() == block_variance.sizes(),
      "block mean/variance shape mismatch");
  TORCH_CHECK(
      block_mean.scalar_type() == block_variance.scalar_type()
          && block_mean.scalar_type() == projected_query.scalar_type(),
      "block summary/query dtype mismatch");
  auto mean_c = block_mean.contiguous();
  auto variance_c = block_variance.contiguous();
  auto query_c = projected_query.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int capacity_blocks = static_cast<int>(mean_c.size(2));
  TORCH_CHECK(
      mean_c.size(0) == batch_count && mean_c.size(1) == kv_head_count
          && mean_c.size(3) == projection_dim,
      "block summary/query shape mismatch");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= QABS_MAX_DIMS,
      "invalid projection dimension");
  TORCH_CHECK(
      block_count > 0 && block_count <= capacity_blocks,
      "invalid block count");
  TORCH_CHECK(block_size > 0, "invalid block size");
  TORCH_CHECK(
      last_block_size > 0 && last_block_size <= block_size,
      "invalid last block size");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, block_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (block_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_microblock_expected_max_scores_forward",
      [&] {
        qabs_microblock_expected_max_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                mean_c.data_ptr<scalar_t>(),
                variance_c.data_ptr<scalar_t>(),
                query_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                capacity_blocks,
                static_cast<int>(block_count),
                projection_dim,
                static_cast<int>(block_size),
                static_cast<int>(last_block_size));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_microblock_q8_expected_max_scores_forward(
    torch::Tensor block_mean_q8,
    torch::Tensor block_mean_scales,
    torch::Tensor block_variance_q8,
    torch::Tensor block_variance_scales,
    torch::Tensor projected_query,
    int64_t block_count,
    int64_t block_size,
    int64_t last_block_size) {
  TORCH_CHECK(
      block_mean_q8.is_cuda() && block_mean_scales.is_cuda()
          && block_variance_q8.is_cuda() && block_variance_scales.is_cuda()
          && projected_query.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(
      block_mean_q8.dim() == 4 && block_variance_q8.dim() == 4,
      "quantized block summaries must be rank four");
  TORCH_CHECK(
      block_mean_scales.dim() == 4 && block_mean_scales.size(3) == 1
          && block_variance_scales.dim() == 4
          && block_variance_scales.size(3) == 1,
      "block scales must be rank four with scalar rows");
  TORCH_CHECK(projected_query.dim() == 4, "projected query must be rank four");
  TORCH_CHECK(
      block_mean_q8.scalar_type() == at::kChar
          && block_variance_q8.scalar_type() == at::kByte,
      "mean/variance codes must be int8/uint8");
  TORCH_CHECK(
      block_mean_scales.scalar_type() == block_variance_scales.scalar_type()
          && block_mean_scales.scalar_type() == projected_query.scalar_type(),
      "block scale/query dtype mismatch");
  auto mean_c = block_mean_q8.contiguous();
  auto mean_scales_c = block_mean_scales.contiguous();
  auto variance_c = block_variance_q8.contiguous();
  auto variance_scales_c = block_variance_scales.contiguous();
  auto query_c = projected_query.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int capacity_blocks = static_cast<int>(mean_c.size(2));
  TORCH_CHECK(
      mean_c.size(0) == batch_count && mean_c.size(1) == kv_head_count
          && mean_c.size(3) == projection_dim
          && variance_c.sizes() == mean_c.sizes(),
      "quantized block summary/query shape mismatch");
  TORCH_CHECK(
      mean_scales_c.size(0) == batch_count
          && mean_scales_c.size(1) == kv_head_count
          && mean_scales_c.size(2) == capacity_blocks
          && variance_scales_c.sizes() == mean_scales_c.sizes(),
      "block-scale shape mismatch");
  TORCH_CHECK(
      projection_dim > 0 && projection_dim <= QABS_MAX_DIMS,
      "invalid projection dimension");
  TORCH_CHECK(
      block_count > 0 && block_count <= capacity_blocks,
      "invalid block count");
  TORCH_CHECK(block_size > 0, "invalid block size");
  TORCH_CHECK(
      last_block_size > 0 && last_block_size <= block_size,
      "invalid last block size");
  int query_head_count = kv_head_count * group_count;
  auto output = torch::empty(
      {batch_count, query_head_count, block_count},
      query_c.options().dtype(at::kFloat));
  dim3 blocks(
      batch_count * query_head_count,
      (block_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query_c.scalar_type(),
      "qabs_microblock_q8_expected_max_scores_forward",
      [&] {
        qabs_microblock_q8_expected_max_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                mean_c.data_ptr<int8_t>(),
                mean_scales_c.data_ptr<scalar_t>(),
                variance_c.data_ptr<uint8_t>(),
                variance_scales_c.data_ptr<scalar_t>(),
                query_c.data_ptr<scalar_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                capacity_blocks,
                static_cast<int>(block_count),
                projection_dim,
                static_cast<int>(block_size),
                static_cast<int>(last_block_size));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_pca_int4_candidate_range_scores_forward(
    torch::Tensor projected_query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    torch::Tensor candidate_indices,
    int64_t key_count,
    int64_t start_dim,
    int64_t end_dim) {
  TORCH_CHECK(
      projected_query.is_cuda() && packed_key.is_cuda() && scales.is_cuda()
          && candidate_indices.is_cuda(),
      "inputs must be CUDA");
  TORCH_CHECK(projected_query.dim() == 4, "invalid projected query shape");
  TORCH_CHECK(packed_key.dim() == 4, "invalid packed key shape");
  TORCH_CHECK(scales.dim() == 4 && scales.size(3) == 1, "invalid scale shape");
  TORCH_CHECK(candidate_indices.dim() == 3, "candidate indices must be rank three");
  TORCH_CHECK(projected_query.scalar_type() == at::kChar, "projected_query must be int8");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed_key must be uint8");
  TORCH_CHECK(candidate_indices.scalar_type() == at::kLong, "candidate indices must be int64");
  auto query_c = projected_query.contiguous();
  auto packed_c = packed_key.contiguous();
  auto scales_c = scales.contiguous();
  auto candidates_c = candidate_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int kv_head_count = static_cast<int>(query_c.size(1));
  int group_count = static_cast<int>(query_c.size(2));
  int projection_dim = static_cast<int>(query_c.size(3));
  int query_head_count = kv_head_count * group_count;
  int candidate_count = static_cast<int>(candidates_c.size(2));
  int capacity = static_cast<int>(packed_c.size(2));
  int packed_dim = static_cast<int>(packed_c.size(3));
  TORCH_CHECK(
      candidates_c.size(0) == batch_count
          && candidates_c.size(1) == query_head_count,
      "candidate index shape mismatch");
  TORCH_CHECK(
      packed_c.size(0) == batch_count && packed_c.size(1) == kv_head_count
          && packed_dim == projection_dim / 2,
      "packed key shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == kv_head_count
          && scales_c.size(2) == capacity,
      "scale shape mismatch");
  TORCH_CHECK(key_count > 0 && key_count <= capacity, "invalid key_count");
  TORCH_CHECK(
      start_dim >= 0 && start_dim < end_dim && end_dim <= projection_dim
          && start_dim % 4 == 0 && end_dim % 4 == 0,
      "invalid projection range");
  auto output = torch::empty(
      {batch_count, query_head_count, candidate_count},
      query_c.options().dtype(at::kFloat));
  constexpr int warps_per_block = QABS_TILE_THREADS / 32;
  constexpr int candidates_per_warp = 4;
  dim3 blocks(
      batch_count * query_head_count,
      (candidate_count + warps_per_block * candidates_per_warp - 1)
          / (warps_per_block * candidates_per_warp));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_c.scalar_type(),
      "qabs_pca_int4_candidate_range_scores_forward",
      [&] {
        qabs_pca_int4_candidate_range_scores_kernel<scalar_t>
            <<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                query_c.data_ptr<int8_t>(),
                packed_c.data_ptr<uint8_t>(),
                scales_c.data_ptr<scalar_t>(),
                candidates_c.data_ptr<int64_t>(),
                output.data_ptr<float>(),
                kv_head_count,
                group_count,
                candidate_count,
                static_cast<int>(key_count),
                capacity,
                projection_dim,
                packed_dim,
                static_cast<int>(start_dim),
                static_cast<int>(end_dim));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> qabs_pack_int2_forward(torch::Tensor key) {
  TORCH_CHECK(key.is_cuda(), "key must be CUDA");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  auto key_c = key.contiguous();
  c10::cuda::CUDAGuard device_guard(key_c.device());
  int batch_count = static_cast<int>(key_c.size(0));
  int head_count = static_cast<int>(key_c.size(1));
  int key_count = static_cast<int>(key_c.size(2));
  int head_dim = static_cast<int>(key_c.size(3));
  TORCH_CHECK(head_dim > 0 && head_dim <= 128, "head_dim must be in [1, 128]");
  int group_count = (key_count + 3) / 4;
  int scale_group_count = (head_dim + 31) / 32;
  auto packed = torch::empty({batch_count, head_count, head_dim, group_count}, key_c.options().dtype(at::kByte));
  auto scales = torch::empty({batch_count, head_count, key_count, scale_group_count}, key_c.options());
  int blocks = batch_count * head_count * group_count;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_c.scalar_type(), "qabs_pack_int2_forward", [&] {
    qabs_pack_int2_kernel<scalar_t><<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        key_c.data_ptr<scalar_t>(),
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<scalar_t>(),
        key_count,
        head_dim,
        group_count,
        scale_group_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

torch::Tensor qabs_partial_scores_int2_forward(
    torch::Tensor query,
    torch::Tensor packed_key,
    torch::Tensor scales,
    torch::Tensor dim_indices,
    int64_t key_count) {
  TORCH_CHECK(query.is_cuda() && packed_key.is_cuda() && scales.is_cuda() && dim_indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(packed_key.dim() == 4, "packed_key must have shape [batch, heads, dim, groups]");
  TORCH_CHECK(scales.dim() == 4, "scales must have shape [batch, heads, key, scale_groups]");
  TORCH_CHECK(dim_indices.dim() == 3, "dim_indices must have shape [batch, heads, selected]");
  TORCH_CHECK(packed_key.scalar_type() == at::kByte, "packed_key must be uint8");
  TORCH_CHECK(query.scalar_type() == scales.scalar_type(), "query/scale dtype mismatch");
  TORCH_CHECK(dim_indices.scalar_type() == at::kLong, "dim_indices must be int64");
  auto query_c = query.contiguous();
  auto packed_c = packed_key.contiguous();
  auto scales_c = scales.contiguous();
  auto dims_c = dim_indices.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int selected_count = static_cast<int>(dims_c.size(2));
  int group_count = static_cast<int>(packed_c.size(3));
  int scale_group_count = static_cast<int>(scales_c.size(3));
  TORCH_CHECK(key_count > 0 && key_count <= scales_c.size(2), "invalid key_count");
  TORCH_CHECK(packed_c.size(0) == batch_count && packed_c.size(1) == head_count && packed_c.size(2) == head_dim, "packed shape mismatch");
  TORCH_CHECK(
      scales_c.size(0) == batch_count && scales_c.size(1) == head_count
          && scale_group_count == (head_dim + 31) / 32,
      "scale shape mismatch");
  TORCH_CHECK(dims_c.size(0) == batch_count && dims_c.size(1) == head_count, "dim shape mismatch");
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (static_cast<int>(key_count) + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_int2_forward", [&] {
    qabs_partial_scores_int2_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        packed_c.data_ptr<uint8_t>(),
        scales_c.data_ptr<scalar_t>(),
        dims_c.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        static_cast<int>(key_count),
        head_dim,
        selected_count,
        group_count,
        scale_group_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor qabs_partial_scores_int2_onthefly_forward(
    torch::Tensor query,
    torch::Tensor key,
    int64_t dim_count) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(), "query and key must be CUDA");
  TORCH_CHECK(query.dim() == 3, "query must have shape [batch, heads, dim]");
  TORCH_CHECK(key.dim() == 4, "key must have shape [batch, heads, key, dim]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  auto query_c = query.contiguous();
  auto key_c = key.contiguous();
  c10::cuda::CUDAGuard device_guard(query_c.device());
  int batch_count = static_cast<int>(query_c.size(0));
  int head_count = static_cast<int>(query_c.size(1));
  int head_dim = static_cast<int>(query_c.size(2));
  int key_count = static_cast<int>(key_c.size(2));
  int selected_count = static_cast<int>(std::min<int64_t>(std::max<int64_t>(dim_count, 1), std::min<int64_t>(head_dim, QABS_MAX_DIMS)));
  TORCH_CHECK(key_c.size(0) == batch_count && key_c.size(1) == head_count && key_c.size(3) == head_dim, "key shape mismatch");
  auto output = torch::empty({batch_count, head_count, key_count}, query_c.options().dtype(at::kFloat));
  dim3 blocks(batch_count * head_count, (key_count + QABS_TILE_THREADS - 1) / QABS_TILE_THREADS);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query_c.scalar_type(), "qabs_partial_scores_int2_onthefly_forward", [&] {
    qabs_partial_scores_int2_onthefly_kernel<scalar_t><<<blocks, QABS_TILE_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        query_c.data_ptr<scalar_t>(),
        key_c.data_ptr<scalar_t>(),
        output.data_ptr<float>(),
        key_count,
        head_dim,
        selected_count,
        head_count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


@lru_cache(maxsize=1)
def _load_extension():
    extension_name = "qabs_sparse_attention_ext_v103"
    build_directory = Path(_get_build_directory(extension_name, verbose=False))
    binaries = sorted(build_directory.glob(f"{extension_name}*.so"))
    if binaries:
        spec = importlib.util.spec_from_file_location(extension_name, binaries[0])
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return load_inline(
        name=extension_name,
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcublas"],
        with_cuda=True,
        verbose=False,
    )


def partial_scores(query: torch.Tensor, key_history: torch.Tensor, dim_count: int) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_forward(query, key_history, int(dim_count))


def partial_scores_dim_major(
    query: torch.Tensor,
    key_dim_major: torch.Tensor,
    dim_indices: torch.Tensor,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_dim_major_forward(query, key_dim_major, dim_indices)


def candidate_full_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    current_candidate: torch.Tensor,
    previous_candidate: torch.Tensor | None,
    previous_final: torch.Tensor | None,
    protect_sink_tokens: int,
    protect_recent_tokens: int,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    empty_candidate = torch.empty(0, dtype=torch.bool, device=query.device)
    return module.qabs_candidate_full_scores_forward(
        query,
        key_history,
        current_candidate,
        previous_candidate if previous_candidate is not None else empty_candidate,
        previous_candidate is not None,
        previous_final if previous_final is not None else empty_candidate,
        previous_final is not None,
        int(protect_sink_tokens),
        int(protect_recent_tokens),
        float(scaling),
    )


def candidate_compact_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_compact_scores_forward(
        query,
        key_history,
        candidate_indices,
        float(scaling),
    )


def candidate_prerope_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    phase_cosine: torch.Tensor,
    phase_sine: torch.Tensor,
    query_position: int,
    scaling: float = 1.0,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_prerope_scores_forward(
        query,
        key_history,
        candidate_indices,
        phase_cosine,
        phase_sine,
        int(query_position),
        float(scaling),
    )


def candidate_compact_scores_ragged(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_compact_scores_ragged_forward(
        query,
        key_history,
        candidate_indices,
        candidate_counts,
        float(scaling),
    )


def candidate_compact_scores_range(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    start_counts: torch.Tensor,
    end_counts: torch.Tensor,
    output: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_compact_scores_range_forward(
        query,
        key_history,
        candidate_indices,
        start_counts,
        end_counts,
        output,
        float(scaling),
    )


def proxy_affine_calibrated_scores(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    proxy_scores: torch.Tensor,
    candidate_counts: torch.Tensor,
    sample_count: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    calibrated_scores, parameters = (
        module.qabs_proxy_affine_calibrated_scores_forward(
            query,
            key_history,
            candidate_indices,
            proxy_scores.float().contiguous(),
            candidate_counts,
            int(sample_count),
            float(scaling),
        )
    )
    return calibrated_scores, parameters


def candidate_compact_scores_masked(
    query: torch.Tensor,
    key_history: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_compact_scores_masked_forward(
        query,
        key_history,
        candidate_indices,
        candidate_valid,
        float(scaling),
    )


def uncertainty_band_mask(
    candidate_scores: torch.Tensor,
    error_sigma: torch.Tensor,
    final_count: int,
    confidence_width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_extension()
    valid, counts, boundaries = module.qabs_uncertainty_band_mask_forward(
        candidate_scores.float().contiguous(),
        error_sigma.float().contiguous(),
        int(final_count),
        float(confidence_width),
    )
    return valid, counts, boundaries


def direct_uncertainty_candidates(
    scores: torch.Tensor,
    error_sigma: torch.Tensor,
    final_count: int,
    candidate_capacity: int,
    confidence_width: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = _load_extension()
    indices, counts, boundaries, overflow = (
        module.qabs_direct_uncertainty_candidates_forward(
            scores.float().contiguous(),
            error_sigma.float().contiguous(),
            int(final_count),
            int(candidate_capacity),
            float(confidence_width),
        )
    )
    return indices, counts, boundaries, overflow


def sampled_quantile_candidates(
    scores: torch.Tensor,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_extension()
    indices, counts, boundaries, overflow = (
        module.qabs_sampled_quantile_candidates_forward(
            scores.float().contiguous(),
            int(sample_count),
            float(selected_fraction),
            int(candidate_capacity),
        )
    )
    return indices, counts, boundaries, overflow


def sample_error_sigma(
    query: torch.Tensor,
    key: torch.Tensor,
    approximate_scores: torch.Tensor,
    sample_count: int,
    sample_offset: int,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_sample_error_sigma_forward(
        query,
        key,
        approximate_scores.float().contiguous(),
        int(sample_count),
        int(sample_offset),
        float(scaling),
    )


def final_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    valid_u8 = valid.to(dtype=torch.uint8)
    output = module.qabs_final_attention_forward(query, key, value, indices, valid_u8, float(scaling))
    return output[:, None, :, :].contiguous()


def final_attention_ragged(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_ragged_forward(
        query,
        key,
        value,
        indices,
        counts,
        float(scaling),
    )
    return output[:, None, :, :].contiguous()


def final_attention_ragged_self(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_ragged_self_forward(
        query,
        key,
        value,
        indices,
        counts,
        float(scaling),
    )
    return output[:, None, :, :].contiguous()


def final_attention_ragged_self_warp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_ragged_self_warp_forward(
        query,
        key,
        value,
        indices,
        counts,
        float(scaling),
    )
    return output[:, None, :, :].contiguous()


def final_attention_ragged_self_split(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    scaling: float,
    split_count: int,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_ragged_self_split_forward(
        query,
        key,
        value,
        indices,
        counts,
        float(scaling),
        int(split_count),
    )
    return output[:, None, :, :].contiguous()


def final_attention_from_scores_ragged(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    value_mass_threshold: float = 1.0,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_from_scores_ragged_forward(
        value,
        indices,
        scores.float().contiguous(),
        counts,
        float(value_mass_threshold),
    )
    return output[:, None, :, :].contiguous()


def final_attention_from_scores_ragged_self(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    self_scores: torch.Tensor,
    value_mass_threshold: float = 1.0,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_from_scores_ragged_self_forward(
        value,
        indices,
        scores.float().contiguous(),
        counts,
        self_scores.float().contiguous(),
        float(value_mass_threshold),
    )
    return output[:, None, :, :].contiguous()


def final_attention_from_scores_split(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    split_count: int = 4,
) -> torch.Tensor:
    module = _load_extension()
    output = module.qabs_final_attention_from_scores_split_forward(
        value,
        indices,
        scores.float().contiguous(),
        counts,
        int(split_count),
    )
    return output[:, None, :, :].contiguous()


def final_attention_tail_reliability(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    prefix_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    output, reliability = module.qabs_final_attention_tail_reliability_forward(
        value,
        indices,
        scores.float().contiguous(),
        counts,
        prefix_counts,
    )
    return output[:, None, :, :].contiguous(), reliability


def final_attention_tail_mass_gate(
    value: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    counts: torch.Tensor,
    prefix_counts: torch.Tensor,
    mass_threshold: float = 0.95,
    tail_shrinkage: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    output, active = module.qabs_final_attention_tail_mass_gate_forward(
        value,
        indices,
        scores.float().contiguous(),
        counts,
        prefix_counts,
        float(mass_threshold),
        float(tail_shrinkage),
    )
    return output[:, None, :, :].contiguous(), active


def mass_ladder(
    top_scores: torch.Tensor,
    sample_scores: torch.Tensor,
    sample_candidate_scores: torch.Tensor,
    self_scores: torch.Tensor,
    keep_counts: torch.Tensor,
    history_count: int,
    mass_threshold: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_mass_ladder_forward(
        top_scores.float().contiguous(),
        sample_scores.float().contiguous(),
        sample_candidate_scores.float().contiguous(),
        self_scores.float().contiguous(),
        keep_counts.contiguous(),
        int(history_count),
        float(mass_threshold),
    )


def pack_int2(key_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    packed, scales = module.qabs_pack_int2_forward(key_history)
    return packed, scales


def partial_scores_int2(
    query: torch.Tensor,
    packed_key: torch.Tensor,
    scales: torch.Tensor,
    dim_indices: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_int2_forward(
        query,
        packed_key,
        scales,
        dim_indices,
        int(key_count),
    )


def partial_scores_int2_onthefly(
    query: torch.Tensor,
    key_history: torch.Tensor,
    dim_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_partial_scores_int2_onthefly_forward(
        query,
        key_history,
        int(dim_count),
    )


def pca_int4_scores(
    projected_query: torch.Tensor,
    packed_key: torch.Tensor,
    scales: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_scores_forward(
        projected_query,
        packed_key,
        scales,
        int(key_count),
    )


def pca_int4_prefix_scores(
    projected_query: torch.Tensor,
    packed_key: torch.Tensor,
    scales: torch.Tensor,
    key_count: int,
    prefix_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_prefix_scores_forward(
        projected_query,
        packed_key,
        scales,
        int(key_count),
        int(prefix_dim),
    )


def pca_int4_chunked_prefix_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    key_count: int,
    prefix_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_prefix_scores_forward(
        projected_query,
        packed_key_chunked,
        scales,
        int(key_count),
        int(prefix_dim),
    )


def pca_int4_chunked_group16_prefix_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    group_scales: torch.Tensor,
    key_count: int,
    prefix_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_group16_prefix_scores_forward(
        projected_query,
        packed_key_chunked,
        group_scales,
        int(key_count),
        int(prefix_dim),
    )


def pca_int4_chunked_logscale16_prefix_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    key_count: int,
    prefix_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_logscale16_prefix_scores_forward(
        projected_query,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(key_count),
        int(prefix_dim),
    )


def pca_nested_int2_logscale16_prefix_scores(
    projected_query: torch.Tensor,
    packed_key_high2: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    key_count: int,
    prefix_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_nested_int2_logscale16_prefix_scores_forward(
        projected_query,
        packed_key_high2,
        base_scales,
        packed_exponents,
        int(key_count),
        int(prefix_dim),
    )


def pca_int4_logscale16_sampled_quantile_candidates(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    key_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    use_dp4a: bool = False,
    write_proxy_scores: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_sampled_quantile_forward(
        projected_query,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(key_count),
        int(sample_count),
        float(selected_fraction),
        int(candidate_capacity),
        bool(use_dp4a),
        bool(write_proxy_scores),
    )


def pca_int4_logscale16_sampled_quantile_bound_candidates(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    chunk_squared_norms: torch.Tensor,
    key_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    use_dp4a: bool = False,
    write_proxy_scores: bool = True,
    collect_statistics: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Progressive scan with a strict Cauchy upper bound.

    The returned candidates are identical to the full-dimensional quantized
    scan. The final two tensors count key-chunk reads and per-query-head chunk
    evaluations when ``collect_statistics`` is enabled.
    """
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_sampled_quantile_bound_forward(
        projected_query,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        chunk_squared_norms,
        int(key_count),
        int(sample_count),
        float(selected_fraction),
        int(candidate_capacity),
        bool(use_dp4a),
        bool(write_proxy_scores),
        bool(collect_statistics),
    )


def pca_int4_logscale16_raw_query_sampled_quantile_candidates(
    query: torch.Tensor,
    basis: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    key_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    use_dp4a: bool = False,
    write_proxy_scores: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_raw_query_sampled_quantile_forward(
        query,
        basis,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(key_count),
        int(sample_count),
        float(selected_fraction),
        int(candidate_capacity),
        bool(use_dp4a),
        bool(write_proxy_scores),
    )


def pca_int4_logscale16_streaming_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    basis: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    history_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    scaling: float,
    use_dp4a: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_streaming_attention_forward(
        query,
        key,
        value,
        basis,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(history_count),
        int(sample_count),
        float(selected_fraction),
        int(candidate_capacity),
        float(scaling),
        bool(use_dp4a),
    )


def pca_int4_logscale16_sampled_quantile_exact_candidates(
    projected_query: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    key_count: int,
    sample_count: int,
    selected_fraction: float,
    candidate_capacity: int,
    scaling: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_sampled_quantile_exact_forward(
        projected_query,
        query,
        key,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(key_count),
        int(sample_count),
        float(selected_fraction),
        int(candidate_capacity),
        float(scaling),
    )


def pca_int4_logscale16_pack_into(
    projected_key: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    start_token: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_pack_into_forward(
        projected_key,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        int(start_token),
    )


def pca_int4_logscale16_chunk_norms_into(
    packed_key_chunked: torch.Tensor,
    chunk_squared_norms: torch.Tensor,
    start_token: int,
    token_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_chunk_norms_into_forward(
        packed_key_chunked,
        chunk_squared_norms,
        int(start_token),
        int(token_count),
    )


def pca_project_query_int8(
    grouped_query: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_extension()
    return module.qabs_pca_project_query_int8_forward(
        grouped_query,
        basis,
    )


def pre_rope_lowfreq_int4_pack_into(
    post_rope_key: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    start_token: int,
    rope_theta: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pre_rope_lowfreq_int4_pack_into_forward(
        post_rope_key,
        packed_key_chunked,
        scales,
        int(start_token),
        float(rope_theta),
    )


def pre_rope_lowfreq_int4_scores(
    post_rope_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    key_count: int,
    query_position: int,
    rope_theta: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pre_rope_lowfreq_int4_scores_forward(
        post_rope_query,
        packed_key_chunked,
        scales,
        int(key_count),
        int(query_position),
        float(rope_theta),
    )


def pre_rope_lowfreq_int2_fixed_pack_into(
    post_rope_key: torch.Tensor,
    packed_key: torch.Tensor,
    start_token: int,
    rope_theta: float,
    clip_alpha: float = 1.5,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pre_rope_lowfreq_int2_fixed_pack_into_forward(
        post_rope_key,
        packed_key,
        int(start_token),
        float(rope_theta),
        float(clip_alpha),
    )


def pre_rope_lowfreq_int2_fixed_scores(
    post_rope_query: torch.Tensor,
    packed_key: torch.Tensor,
    key_count: int,
    query_position: int,
    rope_theta: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pre_rope_lowfreq_int2_fixed_scores_forward(
        post_rope_query,
        packed_key,
        int(key_count),
        int(query_position),
        float(rope_theta),
    )


def pca_int4_chunked_selected_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    dimension_indices: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_selected_scores_forward(
        projected_query,
        packed_key_chunked,
        scales,
        dimension_indices.to(dtype=torch.int32),
        int(key_count),
    )


def pca_int4_chunked_shared_selected_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    dimension_indices: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_shared_selected_scores_forward(
        projected_query,
        packed_key_chunked,
        scales,
        dimension_indices.to(dtype=torch.int32),
        int(key_count),
    )


def pca_int4_chunked_shared_selected_add(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    dimension_indices: torch.Tensor,
    query_scales: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_shared_selected_add_forward(
        projected_query,
        packed_key_chunked,
        scales,
        dimension_indices.to(dtype=torch.int32),
        query_scales.to(dtype=torch.float32),
        score_cache,
        int(key_count),
    )


def pca_int4_chunked_contiguous_add(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    query_scales: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
    start_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_contiguous_add_forward(
        projected_query,
        packed_key_chunked,
        scales,
        query_scales.to(dtype=torch.float32),
        score_cache,
        int(key_count),
        int(start_dim),
    )


def pca_int4_chunked_contiguous_delta_add(
    projected_query: torch.Tensor,
    previous_projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
    start_dim: int,
    selected_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_contiguous_delta_add_forward(
        projected_query,
        previous_projected_query,
        packed_key_chunked,
        scales,
        score_cache,
        int(key_count),
        int(start_dim),
        int(selected_count),
    )


def pca_int4_chunked_band_error_feedback(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    spectral_weights: torch.Tensor,
    anchor_query: torch.Tensor,
    selected_chunk: torch.Tensor,
    gate_signal: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_band_error_feedback_forward(
        projected_query,
        packed_key_chunked,
        scales,
        spectral_weights,
        anchor_query,
        selected_chunk,
        gate_signal,
        score_cache,
        int(key_count),
    )


def pca_int4_chunked_band_error_feedback_masked(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    spectral_weights: torch.Tensor,
    anchor_query: torch.Tensor,
    active_mask: torch.Tensor,
    selected_chunk: torch.Tensor,
    gate_signal: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_band_error_feedback_masked_forward(
        projected_query,
        packed_key_chunked,
        scales,
        spectral_weights,
        anchor_query,
        active_mask,
        selected_chunk,
        gate_signal,
        score_cache,
        int(key_count),
    )


def pca_int4_chunked_logscale16_band_error_feedback_masked(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    spectral_weights: torch.Tensor,
    anchor_query: torch.Tensor,
    active_mask: torch.Tensor,
    selected_chunk: torch.Tensor,
    gate_signal: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_logscale16_band_error_feedback_masked_forward(
        projected_query,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        spectral_weights,
        anchor_query,
        active_mask,
        selected_chunk,
        gate_signal,
        score_cache,
        int(key_count),
    )


def one_shot_band_plan(
    projected_query: torch.Tensor,
    spectral_weights: torch.Tensor,
    anchor_query: torch.Tensor,
    top_values: torch.Tensor,
    keep_counts: torch.Tensor,
    planned_bands: torch.Tensor,
    crossing_risk: torch.Tensor,
    total_token_count: int,
    target_recall: float,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_one_shot_band_plan_forward(
        projected_query,
        spectral_weights,
        anchor_query,
        top_values,
        keep_counts,
        planned_bands,
        crossing_risk,
        int(total_token_count),
        float(target_recall),
    )


def pca_int4_chunked_spectral_gated_delta_add(
    projected_query: torch.Tensor,
    previous_projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    spectral_weights: torch.Tensor,
    anchor_query: torch.Tensor,
    refresh_mask: torch.Tensor,
    gate_signal: torch.Tensor,
    refresh_indices: torch.Tensor,
    score_cache: torch.Tensor,
    key_count: int,
    start_dim: int,
    threshold: float,
    refresh_count: int = 0,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_spectral_gated_delta_add_forward(
        projected_query,
        previous_projected_query,
        packed_key_chunked,
        scales,
        spectral_weights,
        anchor_query,
        refresh_mask,
        gate_signal,
        refresh_indices,
        score_cache,
        int(key_count),
        int(start_dim),
        float(threshold),
        int(refresh_count),
    )


def pca_int4_candidate_range_scores(
    projected_query: torch.Tensor,
    packed_key: torch.Tensor,
    scales: torch.Tensor,
    candidate_indices: torch.Tensor,
    key_count: int,
    start_dim: int,
    end_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_candidate_range_scores_forward(
        projected_query,
        packed_key,
        scales,
        candidate_indices,
        int(key_count),
        int(start_dim),
        int(end_dim),
    )


def pca_int4_chunked_candidate_range_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    scales: torch.Tensor,
    candidate_indices: torch.Tensor,
    key_count: int,
    start_dim: int,
    end_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_candidate_range_scores_forward(
        projected_query,
        packed_key_chunked,
        scales,
        candidate_indices,
        int(key_count),
        int(start_dim),
        int(end_dim),
    )


def pca_int4_chunked_logscale16_candidate_range_scores(
    projected_query: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    candidate_indices: torch.Tensor,
    key_count: int,
    start_dim: int,
    end_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_chunked_logscale16_candidate_range_scores_forward(
        projected_query,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        candidate_indices,
        int(key_count),
        int(start_dim),
        int(end_dim),
    )


def microblock_expected_max_scores(
    block_mean: torch.Tensor,
    block_variance: torch.Tensor,
    projected_query: torch.Tensor,
    block_count: int,
    block_size: int,
    last_block_size: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_microblock_expected_max_scores_forward(
        block_mean,
        block_variance,
        projected_query,
        int(block_count),
        int(block_size),
        int(last_block_size),
    )


def microblock_q8_expected_max_scores(
    block_mean_q8: torch.Tensor,
    block_mean_scales: torch.Tensor,
    block_variance_q8: torch.Tensor,
    block_variance_scales: torch.Tensor,
    projected_query: torch.Tensor,
    block_count: int,
    block_size: int,
    last_block_size: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_microblock_q8_expected_max_scores_forward(
        block_mean_q8,
        block_mean_scales,
        block_variance_q8,
        block_variance_scales,
        projected_query,
        int(block_count),
        int(block_size),
        int(last_block_size),
    )


def pca_int4_logscale16_selected_block_scores(
    projected_query: torch.Tensor,
    query_scales: torch.Tensor,
    packed_key_chunked: torch.Tensor,
    base_scales: torch.Tensor,
    packed_exponents: torch.Tensor,
    selected_blocks: torch.Tensor,
    key_count: int,
    block_size: int,
    start_dim: int,
    end_dim: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int4_logscale16_selected_block_scores_forward(
        projected_query,
        query_scales,
        packed_key_chunked,
        base_scales,
        packed_exponents,
        selected_blocks,
        int(key_count),
        int(block_size),
        int(start_dim),
        int(end_dim),
    )


def microblock_local_to_token_indices(
    selected_blocks: torch.Tensor,
    local_indices: torch.Tensor,
    key_count: int,
    block_size: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_microblock_local_to_token_indices_forward(
        selected_blocks,
        local_indices,
        int(key_count),
        int(block_size),
    )


def pca_int8_scores(
    projected_query: torch.Tensor,
    quantized_key: torch.Tensor,
    key_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int8_scores_forward(
        projected_query,
        quantized_key,
        int(key_count),
    )


def pca_int8_wmma_scores(
    padded_query: torch.Tensor,
    quantized_key: torch.Tensor,
    scales: torch.Tensor,
    key_count: int,
    group_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_pca_int8_wmma_scores_forward(
        padded_query,
        quantized_key,
        scales,
        int(key_count),
        int(group_count),
    )


def retrieval_metrics(
    candidate_scores: torch.Tensor,
    candidate_indices: torch.Tensor,
    previous_probe: torch.Tensor | None,
    probe_count: int = 32,
) -> torch.Tensor:
    module = _load_extension()
    empty = torch.empty(0, dtype=torch.long, device=candidate_scores.device)
    return module.qabs_retrieval_metrics_forward(
        candidate_scores.float().contiguous(),
        candidate_indices.contiguous(),
        previous_probe.contiguous() if previous_probe is not None else empty,
        int(probe_count),
    )


def quota_merge_candidates(
    base_indices: torch.Tensor,
    rescue_indices: torch.Tensor,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_quota_merge_candidates_forward(
        base_indices.contiguous(),
        rescue_indices.contiguous(),
    )


def append_rescue_candidates(
    base_indices: torch.Tensor,
    rescue_indices: torch.Tensor,
    history_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    indices, valid = module.qabs_append_rescue_candidates_forward(
        base_indices.contiguous(),
        rescue_indices.contiguous(),
        int(history_count),
    )
    return indices, valid


def candidate_union_counts(
    candidate_indices: torch.Tensor,
    history_count: int,
    group_count: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_union_counts_forward(
        candidate_indices.contiguous(),
        int(history_count),
        int(group_count),
    )


def candidate_union_compact(
    candidate_indices: torch.Tensor,
    history_count: int,
    group_count: int,
    output_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_extension()
    output, counts = module.qabs_candidate_union_compact_forward(
        candidate_indices.contiguous(),
        int(history_count),
        int(group_count),
        int(output_capacity),
    )
    return output, counts


def candidate_bucket_union_counts(
    candidate_indices: torch.Tensor,
    history_count: int,
    group_count: int,
    bucket_size: int,
) -> torch.Tensor:
    module = _load_extension()
    return module.qabs_candidate_bucket_union_counts_forward(
        candidate_indices.contiguous(),
        int(history_count),
        int(group_count),
        int(bucket_size),
    )
