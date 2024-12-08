# HyRACC: A Hybrid Retrieval-Augmented Framework for More Efficient Code Completion

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets](#Datasets)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgement](#Acknowledgement)

## Introduction
In this paper, we present HyRACC, a hybrid RAG framework for code completion. This framework improves code completion efficiency by constructing hybrid retrieval databases, performing hybrid retrieval, and employing adaptively inference method. We validate HyRACC on models of different scales and datasets from various domains. The results indicate that compared to existing methods, HyRACC shows improvements in code completion accuracy, response speed, and GPU usage.

## Installation
faiss-gpu==1.7.2

transformers==4.27.2

fuzzywuzzy==0.18.0

torch==1.13.0+cu117

## Datasets

Our data stem from the API-Bench(https://github.com/JohnnyPeng18/APIBench). This dataset includes various code scenarios in two common programming languages, Java and Python.

## Usage
Folder **hybrid_code** contains the code of the project. Folder **run_bash** contains the running scripts of our experiments. All base models are sourced from Hugging Face’s relevant repositories. We use CodeGPT as the example, and the model of UnixCoder is consistent. 

### BASE
```
model_type=gpt2
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}
task_name=gpt2

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn.py \
  --dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
  --model_type ${model_type} \
  --pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
  --lit_file ./dataset/intra_scenario_completion/java/literals.json \
  --data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
  --langs java \
  --task_name ${task_name} \
  --output_dir ${output_dir} \
  --do_generate 
```
### BM25
```
model_type=gpt2
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}
task_name=bm25_gpt2_300

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn.py \
  --dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
  --model_type ${model_type} \
  --pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
  --lit_file ./dataset/intra_scenario_completion/java/literals.json \
  --data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
  --langs java \
  --task_name ${task_name} \
  --output_dir ${output_dir} \
  --data_process \
  --do_search \
  --do_generate \
  --max_chunk_len ${chunk_len} \
  --use_bm25
```
### KNN-LM
```
model_type=gpt2
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}
task_name=knm_gpt2_300

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn.py \
  --dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
  --model_type ${model_type} \
  --pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
  --lit_file ./dataset/intra_scenario_completion/java/literals.json \
  --data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
  --langs java \
  --task_name ${task_name} \
  --output_dir ${output_dir} \
  --data_process \
  --do_search \
  --do_generate \
  --max_chunk_len ${chunk_len} \
  --use_knm
```
### HyRACC
```
model_type=gpt2
chunk_len=300
output_dir=./hybrid_save/intra_scenario/java/line/Android/${model_type}/knm_${chunk_len}
task_name=bm25_knm_gpt2_300

CUDA_VISIBLE_DEVICES=0 python hybrid_code/run_line_com_bm25_knn.py \
  --dstore_file ./dataset/intra_scenario_completion/java/token_completion/Android/train.txt \
  --model_type ${model_type} \
  --pretrain_dir ../transformers/CodeGPT-small-java-adaptedGPT2 \
  --lit_file ./dataset/intra_scenario_completion/java/literals.json \
  --data_dir ./dataset/intra_scenario_completion/java/line_completion/Android/ \
  --langs java \
  --task_name ${task_name} \
  --output_dir ${output_dir} \
  --data_process \
  --do_search \
  --do_generate \
  --max_chunk_len ${chunk_len} \
  --use_bm25
  --use_knm
```

## Contributing
If you would like to contribute to the project, please open an issue or submit a pull request.

## Acknowledgement
This repository is inspired by codes from these repository: https://github.com/neulab/knn-transformers and https://github.com/zetang94/ASE2023_kNM-LM. We greatly appreciate the authors for providing their code.
