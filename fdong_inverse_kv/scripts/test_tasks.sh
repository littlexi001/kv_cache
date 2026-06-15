# # arc_c, arc_e, boolq, hellaswag, lambada_openai, piqa, race, siqa
# # task_name="mmlu"
# task_name="arc_e"

# # task_name="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,piqa,race,siqa"


# lm_eval --model hf \
#     --model_args pretrained=/mnt/workspace/Qwen1.5-MoE-A2.7B,dtype=bfloat16 \
#     --tasks $task_name \
#     --batch_size 16 \
#     >>../logs/all_layer_metis_qwen1.5-moe_a2.7b_mmlu_test.log 2>&1 &
#     # --device cuda:0 \
#     # --batch_size 16



# arc_c, arc_e, boolq, hellaswag, lambada_openai, piqa, race, siqa
# task_name="mmlu"
task_name="arc_easy"

# task_name="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,piqa,race,siqa"


lm_eval --model hf \
    --model_args pretrained=/mnt/workspace/Qwen3-0.6B,dtype=bfloat16,checkpoint_path=/mnt/workspace/lym_code/checkpoints/BF16-global-shard-0.6B-pruned_data-PER-1.0-LR-1e-4-BS-512-DATA_SHUFFLE-false/step_20000.pt \
    --tasks $task_name \
    --batch_size 16 \
    >>../logs/all_layer_metis_qwen0.6b_arc_e_test.log 2>&1 &
    # --device cuda:0 \
    # --batch_size 16

