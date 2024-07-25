import os
import json
import re
import time
import yaml
import random

from glob import glob

import fasttext
from huggingface_hub import hf_hub_download

# API setting constants
API_MAX_RETRY = 2
API_RETRY_SLEEP = 10
REPETITION_OUTPUT = "$REPETITION$"
REPETITION_ERROR_MESSAGE = ("Sorry! We've encountered an issue with repetitive patterns in your prompt. "
                            "Please try again with a different prompt.")
API_ERROR_OUTPUT = "$ERROR$"
CODE_SNIPPET_PATTERN = re.compile("```[^\S\r\n]*[a-z]*\n.*?\n```", re.DOTALL)


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


IAM_TOKEN = os.environ.get("IAM_TOKEN")
FOLDER_ID = os.environ.get("FOLDER_ID")
GIGACHAT_TOKEN = os.environ.get("GIGACHAT_TOKEN")

YANDEX_POST_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
SBER_POST_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None) -> (str, int, int):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = API_ERROR_OUTPUT
    prompt_tokens = 0
    completion_tokens = 0
    for _ in range(API_MAX_RETRY):
        try:
            # print(messages)
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
                )
            output = completion.choices[0].message.content
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            if REPETITION_ERROR_MESSAGE in e.message:
                output = REPETITION_OUTPUT
                print(f"GOT REPETITION ERROR. Set output to: {REPETITION_OUTPUT}")
                break
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), repr(e))
    
    return output, prompt_tokens, completion_tokens


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # print(sys_msg)
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def chat_completion_gemini(model, messages, temperature, max_tokens):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    # Set up the model
    generation_config = {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": max_tokens,
    }

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            gemini = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings)

            convo = gemini.start_chat(history=[])
            
            convo.send_message(messages)
            output = convo.last.text
            break
        except genai.types.generation_types.StopCandidateException as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    
    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def chat_completion_yandex(model, messages, temperature, max_tokens, api_dict=None):
    import requests

    # The prompt.json content
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/{model}/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens
        },
        "messages": messages
    }

    # Set up the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {IAM_TOKEN}",
    }

    output: str = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = requests.post(YANDEX_POST_URL, headers=headers, json=data)

            response.raise_for_status()
            response_json = response.json()

            output = response_json["result"]["alternatives"][0]["message"]["text"]

            total_tokens = response_json["result"]["usage"]["totalTokens"]
            prompt_tokens = response_json["result"]["usage"]["inputTextTokens"]
            completion_tokens = response_json["result"]["usage"]["completionTokens"]

            time.sleep(1)

            break
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {repr(http_err)}, code: {http_err.response.status_code}")
            if http_err.response.status_code == 429:
                print(f"Sleep for {API_RETRY_SLEEP} seconds...")
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(type(e), repr(e))

    return output


def chat_completion_sber(model, messages, temperature, max_tokens, api_dict=None):
    import requests
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.01 if temperature == 0 else temperature,
            "stream": False,
            "max_tokens": max_tokens,
        },
        #ensure_ascii=False
    )

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {GIGACHAT_TOKEN}'
    }

    output: str = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = requests.request("POST", SBER_POST_URL, headers=headers, data=payload, verify=False)

            response.raise_for_status()
            response_json = response.json()

            output = response_json["choices"][0]["message"]["content"]

            # total_tokens = response_json["result"]["usage"]["totalTokens"]
            # prompt_tokens = response_json["result"]["usage"]["inputTextTokens"]
            # completion_tokens = response_json["result"]["usage"]["completionTokens"]

            time.sleep(1)

            break
        except requests.exceptions.HTTPError as http_err:
            print("Error", http_err)
            print("TEXT", response.text)
            print(f"HTTP error occurred: {repr(http_err)}, code: {http_err.response.status_code}")
            if http_err.response.status_code == 429:
                print(f"Sleep for {API_RETRY_SLEEP} seconds...")
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(type(e), repr(e))

    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def detect_language(answer_file: str, lang_detect_model="facebook/fasttext-language-identification"):
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            l = json.loads(l)
            qid = l["question_id"]
            answers[qid] = l

    model_path = hf_hub_download(repo_id=lang_detect_model, filename="model.bin")
    model = fasttext.load_model(model_path)

    for qid in answers.keys():
        for i in range(len(answers[qid]["choices"])):
            for j in range(len(answers[qid]["choices"][i]["turns"])):
                text = answers[qid]["choices"][i]["turns"][j]["content"]
                text = re.sub(CODE_SNIPPET_PATTERN, "", text)
                text = text.replace("\n", " ")

                label, _ = model.predict(text)

                answers[qid]["choices"][i]["turns"][j]["lang"] = label[0]

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")


def check_for_repetition(text: str, min_size: int = 20, repetition_rate: int = 10) -> bool:
    for end_pos in range(len(text) - min_size, len(text) - len(text) // repetition_rate, -1):
        if text.count(text[end_pos:]) > repetition_rate:
            return True
    return False


def detect_repetitions(answer_file: str):
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            l = json.loads(l)
            qid = l["question_id"]
            answers[qid] = l

    for qid in answers.keys():
        for i in range(len(answers[qid]["choices"])):
            for j in range(len(answers[qid]["choices"][i]["turns"])):
                text = answers[qid]["choices"][i]["turns"][j]["content"]

                answers[qid]["choices"][i]["turns"][j]["repetition"] = check_for_repetition(text)

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(json.dumps(answers[qid], ensure_ascii=False) + "\n")