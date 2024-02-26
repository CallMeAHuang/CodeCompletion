# nohup run_bash/bm25_knm_gpt2/block_blank_300.sh > log_files/bm25_knm_gpt2_blank_300.log 2>&1 &
model_type=gpt2
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}_blank
task_name=bm25_knm_gpt2_blank_300_debug

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn_blank.py \
--dstore_file ./dataset/intra_scenario_completion/java/token_completion_blank/Android/train.txt \
--model_type ${model_type} \
--pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
--lit_file ./dataset/intra_scenario_completion/java/literals.json \
--data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
--langs java \
--task_name ${task_name} \
--output_dir ${output_dir} \
--use_bm25 \
--use_clearml \
--max_chunk_len ${chunk_len} \
--use_knm \
--data_process \
--do_search

# --do_generate \