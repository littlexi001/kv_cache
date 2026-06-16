from transformers import AutoTokenizer, AutoConfig
import torch
from models import MyQwen3ForCausalLM


# MoE 的参数，目前训练的 2.8B-0.8B 模型的参数就是如下
USE_MOE = True
MOE_INTERMEDIATE_SIZE = 1536
EXPERT_PER_TOKEN = 2
NUM_EXPERTS = 16
GATING_REFERENCE = "oracle" # oracle / switch

DEVICE = 0
CONFIG_DIR = "/mnt/workspace/Qwen3-0.6B" # 就直接用 qwen3 的 dir 就可以
# CKPT_DIR = f"/home/jxzhou/PLM_PER/qwen/checkpoints/qwen{GATING_REFERENCE}" # qwenswitch

D_TYPE = torch.bfloat16


@torch.no_grad()
def prepare_model():
    weights = torch.load(f"{MODEL_DIR}", weights_only = True, map_location="cpu")

    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code = True)
    config.moe_intermediate_size = MOE_INTERMEDIATE_SIZE
    config.num_experts_per_tok = EXPERT_PER_TOKEN
    config.num_experts = NUM_EXPERTS
    config.gating_reference = GATING_REFERENCE
    config.norm_topk_prob = True
    config.use_moe = USE_MOE

    print("Constructing student model ...")

    model = MyQwen3ForCausalLM(config).to(D_TYPE).to(DEVICE)
    model.eval()
    model.load_state_dict(weights)

    print(f'rank {DEVICE} student model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def generate_main(model, tokenizer):
    # 初始化聊天历史
    chat_history_ids = None

    print("🤖 小助手已启动！输入 'exit' 来退出对话。")

    while True:
        # 获取用户输入
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break

        # 对用户输入进行编码
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, return_tensors="pt"
        ).to(DEVICE)

        # 拼接之前的聊天历史
        chat_history_ids = None
        chat_history_ids = (
            torch.cat([chat_history_ids, new_input_ids], dim=-1)
            if chat_history_ids is not None
            else new_input_ids
        )

        hist_len = chat_history_ids.shape[-1]
        if hist_len > 2048:
            # 如果聊天历史超过 2048，截断前面的部分
            chat_history_ids = chat_history_ids[:, -2048:]
        # 生成回复
        chat_history_ids = model.generate(
            chat_history_ids,
            max_new_tokens=512 - hist_len,          # 控制最大长度
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,   # 防止重复
            do_sample=True,           # 启用采样生成更自然的回答
            top_k=30,                # Top-k 采样
            temperature=0.7,          # 控制生成多样性
        )

        # 解码并打印回复
        response = tokenizer.decode(
            chat_history_ids[:, new_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        print(f"Bot: {response}")


if __name__ == "__main__":
    model = prepare_model()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)    

    generate_main(model, tokenizer)

