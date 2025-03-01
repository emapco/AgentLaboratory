import json
import os
import time
from collections.abc import Iterable

import anthropic
import ollama
import openai
import tiktoken
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

TOKENS_IN = dict()
TOKENS_OUT = dict()
MODEL_CLIENTS = dict()


def ollama_cost():
    # power
    system_power_usage = 0.2  # kwh
    local_power_cost = 0.52  # $/kwh - weighted average (2-tier cost - Bay Area)
    ollama_power_per_hour_cost = system_power_usage * local_power_cost
    # tokens
    ollama_tokens_per_second = 28
    ollama_tokens_per_hour = 3600 * ollama_tokens_per_second
    # cost
    ollama_token_cost = ollama_power_per_hour_cost / ollama_tokens_per_hour
    return ollama_token_cost


def curr_cost_est():
    ollama_token_cost = ollama_cost()
    num_tokens = 1000000
    costmap_in = {
        "gpt-4o": 2.50 / num_tokens,
        "gpt-4o-mini": 0.150 / num_tokens,
        "o1-preview": 15.00 / num_tokens,
        "o1-mini": 3.00 / num_tokens,
        "claude-3-5-sonnet": 3.00 / num_tokens,
        "deepseek-chat": 1.00 / num_tokens,
        "o1": 15.00 / num_tokens,
        "ollama": ollama_token_cost,
        "fireworks": 3.00 / num_tokens,
    }
    costmap_out = {
        "gpt-4o": 10.00 / num_tokens,
        "gpt-4o-mini": 0.6 / num_tokens,
        "o1-preview": 60.00 / num_tokens,
        "o1-mini": 12.00 / num_tokens,
        "claude-3-5-sonnet": 12.00 / num_tokens,
        "deepseek-chat": 5.00 / num_tokens,
        "o1": 60.00 / num_tokens,
        "ollama": ollama_token_cost,
        "fireworks": 8.00 / num_tokens,
    }
    return sum([costmap_in[_] * TOKENS_IN[_] for _ in TOKENS_IN]) + sum(
        [costmap_out[_] * TOKENS_OUT[_] for _ in TOKENS_OUT]
    )


def compute_tokens(
    model_str: str, prompt: str, system_prompt: str, answer: str, print_cost: bool
):
    try:
        anthropic_models, openai_reasoning_models, openai_gpt_models = model_names()
        if model_str.startswith("ollama:"):
            model_str = "ollama"
        if model_str.startswith("fireworks:"):
            model_str = "fireworks"

        if model_str in [
            *anthropic_models,
            *openai_reasoning_models,
        ]:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model_str in openai_gpt_models:
            encoding = tiktoken.encoding_for_model(model_str)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model_str not in TOKENS_IN:
            TOKENS_IN[model_str] = 0
            TOKENS_OUT[model_str] = 0
        TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
        TOKENS_OUT[model_str] += len(encoding.encode(answer))
        if print_cost:
            print(
                f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost"
            )
    except Exception as e:
        if print_cost:
            print(f"Cost approximation has an error? {e}")


def model_names():
    anthropic_models = [
        "claude-3-7-sonnet-latest",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-latest",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-latest",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
        "claude-2.0",
    ]
    openai_reasoning_models = [
        "o3-mini",
        "o3-mini-2025-01-31",
        "o1",
        "o1-2024-12-17",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
    ]
    openai_gpt_models = [
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-audio-preview-2024-12-17",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
    ]
    return anthropic_models, openai_reasoning_models, openai_gpt_models


def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    openai_api_key=None,
    tries=5,
    timeout=5.0,
    temp=None,
    print_cost=True,
):
    preloaded_api = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

    anthropic_models, openai_reasoning_models, openai_gpt_models = model_names()

    answer: str | None = None
    for _ in range(tries):
        try:
            if model_str in openai_gpt_models:
                messages: Iterable[ChatCompletionMessageParam] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                client: OpenAI = MODEL_CLIENTS.get(model_str, OpenAI())
                MODEL_CLIENTS[model_str] = client
                completion = client.chat.completions.create(
                    model=model_str,
                    messages=messages,
                    temperature=temp,
                )
                answer = completion.choices[0].message.content
            elif model_str in openai_reasoning_models:
                messages = [{"role": "user", "content": system_prompt + prompt}]
                client: OpenAI = MODEL_CLIENTS.get(model_str, OpenAI())
                MODEL_CLIENTS[model_str] = client
                completion = client.chat.completions.create(
                    model=model_str, messages=messages
                )
                answer = completion.choices[0].message.content
            elif model_str in anthropic_models:
                anthropic_client: anthropic.Anthropic = MODEL_CLIENTS.get(
                    model_str,
                    anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
                )
                MODEL_CLIENTS[model_str] = anthropic_client
                message = anthropic_client.messages.create(
                    model=model_str,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                )  # type: ignore
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "deepseek-chat":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                client: OpenAI = MODEL_CLIENTS.get(
                    model_str,
                    OpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url="https://api.deepseek.com/v1",
                    ),
                )
                MODEL_CLIENTS[model_str] = client
                completion = client.chat.completions.create(
                    model=model_str, messages=messages, temperature=temp
                )
                answer = completion.choices[0].message.content
            elif model_str.startswith("fireworks:"):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                client: OpenAI = MODEL_CLIENTS.get(
                    model_str,
                    OpenAI(
                        api_key=os.environ["FIREWORKS_API_KEY"],
                        base_url="https://api.fireworks.ai/inference/v1",
                    ),
                )
                MODEL_CLIENTS[model_str] = client
                completion = client.chat.completions.create(
                    model=f"accounts/fireworks/models/{model_str[10:]}",
                    messages=messages,
                    temperature=temp,
                )
                answer = completion.choices[0].message.content
            elif model_str.startswith("ollama:"):
                ollama_client: ollama.Client = MODEL_CLIENTS.get(
                    model_str,
                    ollama.Client(
                        os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                    ),
                )
                MODEL_CLIENTS[model_str] = ollama_client
                response: ollama.ChatResponse = ollama_client.chat(
                    model=model_str[7:],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = response.message.content
            else:
                possible_models = (
                    anthropic_models
                    + openai_reasoning_models
                    + openai_gpt_models
                    + [
                        "deepseek-chat",
                        "an ollama model prefixed with 'ollama:'",
                        "a fireworks model prefixed with 'fireworks:'",
                    ]
                )
                raise ValueError(
                    f"Unknown model: {model_str} - possible models: {possible_models}"
                )

            answer = answer or ""
            compute_tokens(model_str, prompt, system_prompt, answer, print_cost)
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
