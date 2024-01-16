# nohup run_bash/bm25_knn_gpt2/block_300_lmbda_05.sh > log_files/bm25_knn_gpt2_300_05.log 2>&1 &
model_type=gpt2
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knn_${chunk_len}
task_name=bm25_knn_gpt2_300_05

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn.py \
--dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
--model_type ${model_type} \
--pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
--lit_file ./dataset/intra_scenario_completion/java/literals.json \
--data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
--langs java \
--task_name ${task_name} \
--output_dir ${output_dir} \
--do_generate \
--use_bm25 \
--max_chunk_len ${chunk_len} \
--use_knn \
--lmbda 0.5 

# --data_process \
# --do_search \