# nohup run_bash/bm25_knm_unix/block_300_oe.sh > log_files/bm25_knm_unix_300_oe.log 2>&1 &
model_type=unixCoder
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}
task_name=bm25_knm_unix_300_oe

CUDA_VISIBLE_DEVICES=1 python hybrid_code/run_line_com_bm25_knn.py \
--dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
--model_type ${model_type} \
--pretrain_dir ../transformers/unixcoder-base \
--lit_file ./dataset/intra_scenario_completion/java/literals.json \
--data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
--langs java \
--task_name ${task_name} \
--output_dir ${output_dir} \
--do_generate \
--use_bm25 \
--max_chunk_len ${chunk_len} \
--use_knm \
--only_errors