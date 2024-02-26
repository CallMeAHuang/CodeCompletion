# model_type=unixCoder
model_type=gpt2
# pretrained_dir=../transformers/unixcoder-base
pretrained_dir=../transformers/CodeGPT-small-java-adaptedGPT2
data_dir=./dataset/intra_scenario_completion/java
lit_file=${data_dir}/literals.json
output_dir=./save/intra_scenario/java/line/Android/${model_type}

dstore_dir=${output_dir}/knm_lm/db

# 1. build the database (same as token completion for Android.)
# scenario_dir=${data_dir}/token_completion/Android

# CUDA_VISIBLE_DEVICES=1 python ./code/run_lm.py \
#     --data_dir=${scenario_dir} \
#     --lit_file=${lit_file} \
#     --langs=java \
#     --output_dir=${output_dir}/knn_lm \
#     --pretrain_dir=${pretrained_dir} \
#     --log_file=log.log \
#     --model_type=${model_type} \
#     --block_size=1024 \
#     --per_gpu_eval_batch_size=4 \
#     --logging_steps=100 \
#     --seed=42 \
#     --build_index \
#     --with_knn \
#     --dstore_dir=${dstore_dir} \
    # --need_knn_train \
    

# step 2. inference next line.
scenario_dir=${data_dir}/line_completion/Android

CUDA_VISIBLE_DEVICES=1 python ./code/run_lm.py \
    --data_dir=${scenario_dir} \
    --lit_file=${lit_file} \
    --langs=java \
    --output_dir=${output_dir}/knm_lm \
    --pretrain_dir=${pretrained_dir} \
    --log_file=log.log \
    --model_type=${model_type} \
    --block_size=1024 \
    --per_gpu_eval_batch_size=4 \
    --logging_steps=100 \
    --seed=42 \
    --do_eval_line \
    --with_knn \
    --dstore_dir=${dstore_dir} \
    --only_errors \
    --use_bayes