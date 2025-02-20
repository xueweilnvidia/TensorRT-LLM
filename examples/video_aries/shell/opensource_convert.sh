checkpointdir=jizhicfs/flyerxu/opensource/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/tp8_checkpoints
ckpt_path=/jizhicfs/flyerxu/opensource/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
tp_size=8
max_batch=1
#蒸馏版本
python3 convert_checkpoint.py --flux_ckpt ${ckpt_path} --dtype bfloat16 --output_dir ${checkpointdir} --input_h 720 --input_w 1280 --guidance_embed --tp_size ${tp_size} --workers ${tp_size}

#attn plugin
engine_output_dir=opensource_tp${tp_size}_distill_video720_plugin_maxbatch${max_batch}_engine_output
trtllm-build --checkpoint_dir ${checkpointdir} --max_batch_size ${max_batch} --gemm_plugin bfloat16 --remove_input_padding disable --bert_attention_plugin bfloat16 --output_dir ${engine_output_dir} --workers ${tp_size} 
