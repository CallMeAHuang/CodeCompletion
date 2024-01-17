CUDA_VISIBLE_DEVICES=1 python reacc/run_line_com_bm25_knn.py \
--dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
--model_type gpt2 \
--pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
--lit_file ./dataset/intra_scenario_completion/java/literals.json \
--data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
--langs java \
--output_dir ./reacc_save/intra_scenario/java/line/Android/gpt2/knn_lm \
--do_generate \
--use_bm25 \
# --do_search \
# --data_process \