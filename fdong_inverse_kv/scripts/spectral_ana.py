import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import QwenTokenizedData
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler

from itertools import chain

MATRICES = ['gate_proj', 'up_proj', 'down_proj']
TOPK = -4

@torch.no_grad()
def perform_svd_and_cache(cache_file, unique_experts, shared_experts = None):
    u, s, vh = {matrix: [] for matrix in MATRICES}, {matrix: [] for matrix in MATRICES}, {matrix: [] for matrix in MATRICES}
    iterchain = chain([shared_experts], unique_experts) if shared_experts is not None else unique_experts

    # 对每个专家的每个线性层执行SVD
    for i, expert in enumerate(iterchain):
        for matrix in MATRICES:
            if matrix == 'gate_proj':
                linear_layer = expert.gate_proj
            elif matrix == 'up_proj':
                linear_layer = expert.up_proj
            elif matrix == 'down_proj':
                linear_layer = expert.down_proj
            U, S, Vh = torch.svd(linear_layer.weight.grad.to(torch.float32))
            u[matrix].append(U)
            s[matrix].append(S)
            vh[matrix].append(Vh.T)
            print(f"Expert {i}, Matrix {matrix}", end='\r')

    # 缓存结果
    torch.save((u, s, vh), cache_file)
    return (u, s, vh)

def load_svd_results(cache_file):
    if not os.path.exists(cache_file):
        return None

    u, s, Vh = torch.load(cache_file, weights_only=True)
    return (u, s, Vh)

def subspace_angles_torch(QA, QB):
    QA_H_QB = QA.T @ QB
    sigma = torch.linalg.svdvals(QA_H_QB)
    angle = torch.acos(torch.clamp(sigma, -1., 1.)).mean()

    return angle.item()

@torch.no_grad()
def compute_primary_angles(singular_vectors, top_k, is_u_transpose=False):
    sigvec_i, sigvec_j = singular_vectors[0], singular_vectors[1]
    if is_u_transpose:
        sigvec_i, sigvec_j = sigvec_i.T, sigvec_j.T
    min_dim = min(sigvec_i.shape[1], sigvec_j.shape[1])
    min_idx = 10 # min(sigvec_i.shape[0], sigvec_j.shape[0])

    # 假设每个是 (k_i, d)，我们取前 top_k 行，并确保 d 一致
    if is_u_transpose:
        if top_k < 0:
            S = torch.stack([v[:min_dim, -top_k:min_idx].T for v in singular_vectors], dim=0)  # (n, k, d)
        else:
            S = torch.stack([v[:min_dim, :top_k].T for v in singular_vectors], dim=0)  # (n, k, d)
    else:
        if top_k < 0:
            S = torch.stack([v[-top_k:min_idx, :min_dim] for v in singular_vectors], dim=0)  # (n, k, d)
        else:
            S = torch.stack([v[:top_k, :min_dim] for v in singular_vectors], dim=0)  # (n, k, d)

    n, k, d = S.shape  # (n_experts, k, d)

    # ---- Step 2: 批量计算 C_ij = S_i @ S_j^T for all i,j ----
    # S: (n, k, d)
    # We want C[i, j] = S[i] @ S[j].T  => shape (n, n, k, k)
    # Use broadcasting: (n, 1, k, d) @ (1, n, d, k) -> (n, n, k, k)
    S_i = S.unsqueeze(1)          # (n, 1, k, d)
    S_j_T = S.unsqueeze(0).transpose(-1, -2)  # (1, n, d, k)
    C = torch.matmul(S_i, S_j_T)  # (n, n, k, k)

    # ---- Step 3: 批量 SVD 得到奇异值 ----
    # torch.linalg.svdvals supports batched input (..., k, k)
    sigma = torch.linalg.svdvals(C)  # (n, n, k), descending order

    # Clamp for numerical stability
    sigma = torch.clamp(sigma, -1.0, 1.0)

    # ---- Step 4: 计算角度 ----
    angles = sigma # torch.acos(sigma) / torch.pi * 90  # (n, n, k)

    angle_matrix, _ = angles.max(dim=-1)  # (n, n)

    # Optional: set diagonal to 0 (angle between same subspace)
    angle_matrix.fill_diagonal_(1.0)

    return angle_matrix.cpu().numpy()


def draw_heatmap(data, title, save_path):
    vmin = 0  # 热力条下限
    vmax = 1  # 热力条上限
    cmap = plt.cm.YlOrRd_r  # 颜色渐变模式

    # 比例一致，方便对齐图片
    fig, ax = plt.subplots(figsize=(8, 7))
    # yyf 指定好看字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Serif']

    # 设置全局字号
    plt.rcParams.update({'font.size': 22})
    heatmap = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(heatmap)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # 设置坐标轴名字
    plt.xlabel('Expert Index', fontsize=40)
    plt.ylabel('Expert Index', fontsize=40)

    # 防止坐标轴名字被裁掉
    plt.tight_layout()

    # 高 dpi 保证清晰度
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    
def prepare_data(local_rank, world_size, model_dir, data_dir, seq_len):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    dataset = QwenTokenizedData(data_dir, seq_len, tokenizer)

    print(f"Construct dataset, total {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)

    return dataloader


def forward_step(local_rank, source, target, model, token_loss_fn):
    output = model(source, output_hidden_states = False,)

    target = target.reshape(-1)
    loss = token_loss_fn(output.logits.view(-1, output.logits.size(-1)), target)

    return loss


def decompose_gradient(model_name):
    dataloader = prepare_data(0, 1, MODEL_PATHS[model_name], "/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10", 1024)
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643)
    model =  AutoModelForCausalLM.from_pretrained(MODEL_PATHS[model_name], trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)

    for batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        if batch_idx > 10:
            break

        start_time = time.time()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda', enabled=True):
            loss = forward_step(0, source, target, model, token_loss_fn)

        loss.backward()

        print(f"batch: {batch_idx}, loss: {loss:.3f}, batch_time:{time.time() - start_time:.3f}", flush=True)

    for layer in range(0, 27):
        cache_file = os.path.join("../tensors/gradients/qwen1.5", f'layer{layer}.tensor')

        moe_layer = model.model.layers[layer].mlp  # experts, shared_experts: gate_proj, up_proj, down_proj
        svd_results = perform_svd_and_cache(cache_file, moe_layer.experts, moe_layer.shared_expert)


def ana_expert_spectrum_angle(model_name, layers = [], topk=32):
    # 加载预训练模型
    model = None
    cache_dir = f"../tensors/gradients/{model_name}"
    figure_dir = f"../figures/gradient_angle/{model_name}/top{topk}"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    if isinstance(layers, int):
        layers = [layers]

    for layer in layers:
        cache_file = os.path.join(cache_dir, f'layer{layer}.tensor')
        svd_results = load_svd_results(cache_file)
        
        U, S, Vh = svd_results
        for matrix in MATRICES:
            print(f"Computing subspace angle Layer {layer}, Matrix {matrix}:")
            if matrix == 'gate_proj' or matrix == 'up_proj':
                primary_angle_heatmap = compute_primary_angles(Vh[matrix], topk)
            else: # down_proj
                primary_angle_heatmap = compute_primary_angles(U[matrix], topk, is_u_transpose=True)
            primary_angle_heatmap = primary_angle_heatmap
            np.save(f'{figure_dir}/layer{layer}_{matrix}_topk{topk}_primary_angle_heatmap.npy', primary_angle_heatmap)
            draw_heatmap(primary_angle_heatmap, f'Layer {layer} {matrix} Primary Angles', f'{figure_dir}/layer{layer}_{matrix}_topk{topk}_primary_angle_heatmap.png')


def draw_expert_spectrum_angle(model_name, layers = [], topk=32):
    figure_dir = f"../figures/spectrum_angle/{model_name}"
    for layer in layers:
        for matrix in MATRICES:
            heatmap_data_file = f'{figure_dir}/layer{layer}_{matrix}_topk{topk}_primary_angle_heatmap.npy'
            if not os.path.exists(heatmap_data_file):
                continue
            heatmap_data = np.load(heatmap_data_file)
            draw_heatmap(heatmap_data, f'Layer {layer} {matrix} Primary Angles', f'{figure_dir}/layer{layer}_{matrix}_topk{topk}_primary_angle_heatmap.png')
            print(f"Layer {layer}, Matrix {matrix} Primary Angles saved.")


MODEL_PATHS = {
    "deepseek-16b-2.8b": "../DeepSeek-16B-2.8B",
    "qwen-1.5": "/mnt/workspace/Qwen1.5-MoE-A2.7B",
}


if __name__ == "__main__":
    # decompose_gradient("qwen-1.5")
    ana_expert_spectrum_angle("qwen-1.5", layers=range(0, 24), topk=TOPK)
    # draw_expert_spectrum_angle("deepseek-16b-2.8b", layers=range(1, 27), topk=4)


