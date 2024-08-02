#!/bin/bash

vars_gigachat=("OPENAI_API_KEY" "GIGACHAT_AUTHORIZATION_DATA")
vars_yandex=("OPENAI_API_KEY" "FOLDER_ID" "IAM_TOKEN")
vars_hf=("OPENAI_API_KEY" "HF_TOKEN")

source automation/utils.sh
original_model_name=$(read_original_model_name)

original_model_name_lowercase=$(echo "$original_model_name" | awk '{print tolower($0)}')
model_type=$(resolve_model_type "$original_model_name_lowercase")
check_export_vars "vars_${model_type}[@]"

var_values=$(cat "automation/default_hyperparameters.txt")
var_values=$(get_var_values)

model_alias="${original_model_name////-}-$(date '+%d-%m-%y')"
make_yaml_configs

prepare_env "$model_type"

gen_answer
gen_judgment
show_result "$var_values"

clean_env "$model_type"
