# arc_c, arc_e, boolq, hellaswag, lambada_openai, piqa, race, siqa
task_name="arc_easy,hellaswag,lambada_openai,piqa"
# task_name="arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,piqa,race,siqa"



lm_eval --model hf \
    --model_args pretrained=/mnt/workspace/Qwen3-0.6B,dtype=bfloat16" \
    --tasks $task_name \
    --device cuda:3 \
    --batch_size auto \
    # --output_path "${model_name}.json" \
    # --log_samples \

# /home/jxzhou/PLM_PER/qwen/Qwen3-0.6B \
# /home/fdong/qwen/Qwen3-MoE-2.8B-0.8B,dtype=bfloat16,checkpoint_path=/home/fdong/lowmem_qwen/checkpoints/pretrain-switch.0.13000.pth \

