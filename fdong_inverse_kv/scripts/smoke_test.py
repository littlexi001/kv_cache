"""Fast CPU test for shapes, gradients, causal centering, and logging metrics."""

import torch
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from models.myqwen import exclusive_causal_mean_center


def main():
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("../configs/qwen3_0.6b")
    config.vocab_size = 128
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.inverse_kv_router_input = "k"
    config.inverse_kv_center_router_input = True
    config.inverse_kv_router_normalization = "l2"
    config.inverse_kv_router_bias = False
    config.inverse_kv_num_experts = 4
    config.inverse_kv_expert_intermediate_size = 32
    config.inverse_kv_local_window = 4
    config.inverse_kv_sink_tokens = 1

    states = torch.randn(2, 4, 8, 16, requires_grad=True)
    centered = exclusive_causal_mean_center(states)
    assert torch.allclose(centered[:, :, 0], states[:, :, 0])
    centered[:, :, -1].sum().backward()
    assert states.grad[:, :, :-1].abs().sum().item() == 0.0
    assert states.grad[:, :, -1].abs().sum().item() > 0.0

    model = MyQwen3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    assert output.logits.shape == (2, 16, config.vocab_size)

    changed_suffix = input_ids.clone()
    changed_suffix[:, 9:] = torch.randint(0, config.vocab_size, changed_suffix[:, 9:].shape)
    changed_output = model(
        input_ids=changed_suffix,
        attention_mask=torch.ones_like(changed_suffix),
        use_cache=False,
    )
    if not torch.allclose(output.logits[:, :9], changed_output.logits[:, :9], atol=1e-5, rtol=1e-5):
        raise AssertionError("Future tokens changed prefix outputs; causal contract is broken")

    model.train()
    loss = output.logits.float().square().mean()
    loss.backward()

    router_grad = sum(
        parameter.grad.abs().sum().item()
        for name, parameter in model.named_parameters()
        if ".router." in name and parameter.grad is not None
    )
    if router_grad <= 0:
        raise AssertionError("Router did not receive a gradient from the NTP path")
    metrics = model.routing_metrics()
    required = {"candidate_ratio", "router_load_entropy", "effective_experts", "router_margin"}
    if not required.issubset(metrics):
        raise AssertionError(f"Missing metrics: {required - set(metrics)}")

    # CPU autocast reproduces the mixed float32/bfloat16 aggregation used by
    # CUDA BF16 training. Test both decoder branches because each dispatches
    # expert outputs into a preallocated tensor.
    for architecture in ("ordinary_moe", "shared_bucket"):
        config.inverse_kv_architecture = architecture
        autocast_model = MyQwen3ForCausalLM(config)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_logits = autocast_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            ).logits
            autocast_loss = autocast_logits.float().square().mean()
        autocast_loss.backward()
        if not torch.isfinite(autocast_loss):
            raise AssertionError(f"{architecture} BF16 autocast produced a non-finite loss")
    print({"loss": float(loss.detach()), "router_grad_l1": router_grad, **metrics})


if __name__ == "__main__":
    main()
