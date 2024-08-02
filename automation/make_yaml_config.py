# -*- coding: utf-8 -*-
import sys
import yaml
import config_constants


class QuotedString(str):
    pass


class CustomDumper(yaml.Dumper):
    def represent_data(self, data):
        if type(data) == QuotedString:
            return self.represent_scalar('tag:yaml.org,2002:str', data, style='"')

        return super(CustomDumper, self).represent_data(data)


def save_yaml(data_name, data):
    file_name = 'config/' + data_name + '.yaml'

    with open(file_name, 'w', encoding='utf-8') as outfile:
        yaml.dump(data, outfile,
                  default_flow_style=False,
                  encoding='utf-8',
                  width=2000,
                  allow_unicode=True,
                  Dumper=CustomDumper)


def save_api_config(hyperparameters):
    api_config = {
        'gpt-4o': {
            'model_name': 'gpt-4o',
            'endpoints': None,
            'api_type': 'openai',
            'parallel': config_constants.gpt_4o_parallel,
        },
        'gpt-4-0613': {
            'model_name': 'gpt-4-0613',
            'endpoints': None,
            'api_type': 'openai',
            'parallel': config_constants.gpt_4_0613_parallel,
        }
    }

    if hyperparameters['model_type'] == 'yandex':
        model_config = {
            'model_name': hyperparameters['original_model_name'],
            'system_prompt': QuotedString('Ты полезный AI-ассистент.'),
            'endpoints': None,
            'api_type': 'yandex',
            'parallel': config_constants.yandex_parallel,
        }
    elif hyperparameters['model_type'] == 'gigachat':
        model_config = {
            'model_name': hyperparameters['original_model_name'],
            'system_prompt': QuotedString('Ты полезный AI-ассистент.'),
            'endpoints': None,
            'api_type': 'sber',
            'parallel': config_constants.gigachat_parallel,
        }
    elif hyperparameters['model_type'] == 'hf':
        model_config = {
            'model_name': hyperparameters['original_model_name'],
            'endpoints': [
                {
                    'api_base': f"http://{hyperparameters['hostname']}:{hyperparameters['vllm_port']}/v1",
                    'api_key': 'default-token',
                },
            ],
            'api_type': 'openai',
            'parallel': hyperparameters['hf_parallel'],
        }
    else:
        raise ValueError('Incorrect model type')

    api_config[hyperparameters['model_alias']] = model_config
    save_yaml('api_config', api_config)


def save_gen_answer_config(hyperparameters):
    gen_answer_config = {
        'name': f'config of answer generation for {hyperparameters["bench_name"]}',
        'bench_name': hyperparameters['bench_name'],
        'temperature': hyperparameters['gen_answer_temperature'],
        'max_tokens': hyperparameters['gen_answer_max_tokens'],
        'num_choices': hyperparameters['gen_answer_num_choices'],
        'question_file': QuotedString(config_constants.question_file_name),
        'model_list': [hyperparameters['baseline_model'], hyperparameters['model_alias']],
    }

    save_yaml('gen_answer_config', gen_answer_config)


def save_judge_config(hyperparameters):
    judge_config = {
        'name': f'judgment config file for {hyperparameters["bench_name"]}',
        'bench_name': hyperparameters['bench_name'],
        'judge_model': hyperparameters['judge_model'],
        'reference': config_constants.reference,
        'ref_model': config_constants.ref_model,
        'baseline': config_constants.baseline,
        'baseline_model': hyperparameters['baseline_model'],
        'pairwise': config_constants.pairwise,
        'temperature': hyperparameters['judge_config_temperature'],
        'max_tokens': hyperparameters['judge_config_max_tokens'],
        'regex_pattern': config_constants.regex_pattern,
        'system_prompt': QuotedString(config_constants.system_prompt),
        'prompt_template': config_constants.prompt_template,
        'question_file': QuotedString(config_constants.question_file_name),
        'model_list': [hyperparameters['model_alias']],
    }

    save_yaml('judge_config', judge_config)


def correct_type(arg):
    try:
        return int(arg)
    except ValueError:
        try:
            return float(arg)
        except ValueError:
            return arg


def main(args):
    data = [correct_type(arg) for arg in args[1:] if arg != "="]
    hyperparameters = dict()

    for i in range(0, len(data), 2):
        var_name = data[i]
        var_value = data[i + 1]
        hyperparameters[var_name] = var_value

    save_api_config(hyperparameters)
    save_gen_answer_config(hyperparameters)
    save_judge_config(hyperparameters)


if __name__ == '__main__':
    main(sys.argv)
