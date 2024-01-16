# nohup run_bash/knn_gpt2/lmbda_04.sh > log_files/knn_gpt2_04.log 2>&1 &
model_type=gpt2
pretrained_dir=../transformers/CodeGPT-small-java-adaptedGPT2
data_dir=./dataset/intra_scenario_completion/java
lit_file=${data_dir}/literals.json
output_dir=./save/intra_scenario/java/line/Android/${model_type}
dstore_dir=${output_dir}/knn_lm/db
temp_output_dir=${output_dir}/knn_lm
log_file=knn_gpt2_04.log
task_name=knn_gpt2_04_new

# 1. build the database (same as token completion for Android.)
scenario_dir=${data_dir}/token_completion/Android

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
    --data_dir=${scenario_dir} \
    --lit_file=${lit_file} \
    --langs=java \
    --output_dir=${temp_output_dir} \
    --pretrain_dir=${pretrained_dir} \
    --log_file=${log_file} \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=4 \
    --logging_steps=100 \
    --seed=42 \
    --build_index \
    --with_knn \
    --need_knn_train \
    --dstore_dir=${dstore_dir} 

# step 2. inference next line.
scenario_dir=${data_dir}/line_completion/Android

CUDA_VISIBLE_DEVICES=0 python ./code/run_lm.py \
    --task_name=${task_name} \
    --data_dir=${scenario_dir} \
    --lit_file=${lit_file} \
    --langs=java \
    --output_dir=${temp_output_dir} \
    --pretrain_dir=${pretrained_dir} \
    --log_file=${log_file} \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=4 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_line \
    --with_knn \
    --dstore_dir=${dstore_dir} \
    --lmbda=0.4