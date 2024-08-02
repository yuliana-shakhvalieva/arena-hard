#!/bin/bash

function get_param_from_string {
  echo "$1" | grep "$2" | cut -d '=' -f 2 | sed 's/ //g'
}

function read_original_model_name {
  read -p "Enter model name.

Examples:
  - deepvk/gemma-2b-sft
  - yandexgpt-lite
  - GigaChat

" original_model_name;

  echo "${original_model_name//["'\""]/}"
}

function resolve_model_type {
  if [[ "$1" = 'yandex' ]];
  then
    echo 'yandex'
  elif [[ "$1" = 'gigachat' ]];
  then
    echo 'gigachat'
  else
    echo 'hf'
  fi
}

function check_export_vars {
  vars_count=0

  for var_name in ${!1};
  do
    if ! [[ -n "${!var_name}" ]];
    then
      vars_count+=0
      echo "You should export $var_name"
    else
      echo "OK $var_name"
    fi
  done

  if ! [[ ($vars_count == 0) ]];
  then
    exit 1
  fi
}

function get_var_values {
  read -r -p "----------------------------------------
$var_values
----------------------------------------
Would you like to use this hyperparameters? [y/N] " response

  case "$response" in
  [yY][eE][sS]|[yY])
    echo "$var_values"
    ;;
  *)
    read -p "
Enter hyperparameters you would like to change:
<var_name_1> <var_value_1> <var_name_2> <var_value_2> ...
" new_values

    array=($new_values)
    for ((i=0;i< ${#array[@]} ;i+=2));
    do
      var_name=${array[i]}
      var_value=${array[i+1]}
      var_values=$(echo "$var_values" | sed "s/\($var_name = \([^=]*\)\)/$var_name = $var_value/g")
    done
    var_values=$(get_var_values)
    echo "$var_values"
    ;;
  esac
}

function make_yaml_configs {
  hostname="$(hostname -f)"

  python automation/make_yaml_config.py $var_values  \
         "original_model_name" "$original_model_name"  \
         "model_alias" "$model_alias" \
         "model_type" "$model_type" \
         "hostname" "$hostname"
}

function get_gigachat_token {
  cat "automation/.get_token_ascii.txt"

  response=$(curl -k -s -L -X POST 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -H 'Accept: application/json' \
  -H 'RqUID: '"$(uuidgen)"'' \
  -H 'Authorization: Basic '"$1"'' \
  --data-urlencode 'scope=GIGACHAT_API_PERS')
   echo "$response" | sed -E 's/.*"access_token":"(.*)",.*/\1/'
}

function run_vllm {
  cat "automation/.run_vllm_ascii.txt"

  clean_env "$model_type"

  vllm_port=$(get_param_from_string "$var_values" "vllm_port")
  hf_cache="/nfs/$(whoami)/hf_cache"

  export TOKEN="$HF_TOKEN"
  export MODEL="$original_model_name"
  export PORT="$vllm_port"
  export HF_CACHE="$hf_cache"

  docker run --runtime nvidia --gpus '"device=0,1,2,3,4,5,6,7"' \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -dit \
  --name "script_vllm_container" \
  --env "HUGGING_FACE_HUB_TOKEN=${TOKEN}" \
  --env VLLM_ATTENTION_BACKEND=FLASHINFER \
  --ipc=host \
  --net=host \
  --uts=host \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --api-key default-token \
  --dtype auto \
  --port "$PORT"

  while docker logs "script_vllm_container" | grep -q "Application startup complete";
  do
    sleep 0.1;
  done
}

function clean_env {
  if [[ "$1" == 'hf' ]];
  then
    container_id=$(docker ps -a | grep "script_vllm_container" | cut -d " " -f1)

    if ! [ -z "$container_id" ];
    then
      docker kill "$container_id"
      docker rm -f "$container_id"
    fi

  fi
}

function prepare_env {
  if [[ "$1" == 'gigachat' ]];
  then
    GIGACHAT_TOKEN=$(get_gigachat_token "$GIGACHAT_AUTHORIZATION_DATA")
    export GIGACHAT_TOKEN="$GIGACHAT_TOKEN"

  elif [[ "$1" == 'hf' ]];
  then
    run_vllm
  fi
}

function gen_answer {
  cat "automation/.gen_answer_ascii.txt"
  python gen_answer.py
}

function gen_judgment {
  cat "automation/.gen_judgment_ascii.txt"
  python gen_judgment.py
}

function show_result {
  cat "automation/.results_ascii.txt"
  judge_model=$(get_param_from_string "$var_values" "judge_model")
  python show_result.py --judge-name "$judge_model"
}



