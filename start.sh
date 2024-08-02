#!/bin/bash

#function get_keypress {
#  local REPLY IFS=
#  >/dev/tty printf '%s' "$*"
#  [[ $ZSH_VERSION ]] && read -rk1  # Use -u0 to read from STDIN
#  # See https://unix.stackexchange.com/q/383197/143394 regarding '\n' -> ''
#  [[ $BASH_VERSION ]] && </dev/tty read -rn1
#  printf '%s' "$REPLY"
#}
#
## Get a y/n from the user, return yes=0, no=1 enter=$2
## Prompt using $1.
## If set, return $2 on pressing enter, useful for cancel or defualting
#function get_yes_keypress {
#  local prompt="${1:-Are you sure} [y/n]? "
#  local enter_return=$2
#  local REPLY
#  # [[ ! $prompt ]] && prompt="[y/n]? "
#  while REPLY=$(get_keypress "$prompt"); do
#    [[ $REPLY ]] && printf '\n' # $REPLY blank if user presses enter
#    case "$REPLY" in
#      Y|y)  return 0;;
#      N|n)  return 1;;
#      '')   [[ $enter_return ]] && return "$enter_return"
#    esac
#  done
#}
#
## Credit: http://unix.stackexchange.com/a/14444/143394
## Prompt to confirm, defaulting to NO on <enter>
## Usage: confirm "Dangerous. Are you sure?" && rm *
#function confirm {
#  local prompt="${*:-Are you sure} [y/N]? "
#  get_yes_keypress "$prompt" 1
#}
#
## Prompt to confirm, defaulting to YES on <enter>
#function confirm_yes {
#  local prompt="${*:-Are you sure} [Y/n]? "
#  get_yes_keypress "$prompt" 0
#}
#
#confirm_yes "Show dangerous command" && echo "rm *"
#
#exit 0

vars_gigachat=("OPENAI_API_KEY" "GIGACHAT_AUTHORIZATION_DATA")
vars_yandex=("OPENAI_API_KEY" "FOLDER_ID" "IAM_TOKEN")
vars_hf=("OPENAI_API_KEY" "HF_TOKEN")

#sudo apt-get install uuid-runtime
source automation/utils.sh
# todo: прошлый раз запцскалась эта модель, еще раз запустим или поменяем имя модели?
original_model_name=$(read_original_model_name)

#original_model_name='deepvk/gemma-2b-sft'
#original_model_name='yandexgpt-lite'
#original_model_name='GigaChat'

#if test -f "data/arena-hard-v0.1/model_answer/$model_alias";
#then
#  echo "File exists $model_alias. Would you like to restart?"
#fi

original_model_name_lowercase=$(echo "$original_model_name" | awk '{print tolower($0)}')
model_type=$(resolve_model_type "$original_model_name_lowercase")
check_export_vars "vars_${model_type}[@]"

var_values=$(cat "automation/default_hyperparameters.txt")
var_values=$(get_var_values) # todo while

model_alias="${original_model_name////-}-$(date '+%d-%m-%y')"
make_yaml_configs

prepare_env "$model_type"

#sleep 30

gen_answer
gen_judgment
show_result "$var_values"

clean_env "$model_type"

# добавить поддержку моделей сбер яндекс опенай и тех что лежат на диске
# для хф моделей парал 250
# много моделей прогонять список моделй и списко бенчей bench_name: arena-hard-v0.1
#arena-general
# check different models
#  уже запускали эту модель, продолжить заполнять или начать заново