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

encoding = tiktoken.get_encoding("cl100k_base")


def ollama_cost():
    # power
    system_power_usage = 0.2  # kwh
    local_power_cost = 0.52  # $/kwh - weighted average (2-tier cost - Bay Area)
    ollama_power_per_hour_cost = system_power_usage * local_power_cost
    # tokens
    ollama_tokens_per_second = 20
    ollama_tokens_per_hour = 3600 * ollama_tokens_per_second
    # cost
    ollama_token_cost = ollama_power_per_hour_cost / ollama_tokens_per_hour
    return ollama_token_cost


def curr_cost_est():
    ollama_token_cost = ollama_cost()
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "ollama": ollama_token_cost,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "ollama": ollama_token_cost,
    }
    return sum([costmap_in[_] * TOKENS_IN[_] for _ in TOKENS_IN]) + sum(
        [costmap_out[_] * TOKENS_OUT[_] for _ in TOKENS_OUT]
    )


def compute_tokens(model_str, prompt, system_prompt, answer, print_cost):
    try:
        if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif model_str in ["deepseek-chat"]:
            encoding = tiktoken.encoding_for_model("cl100k_base")
        else:
            encoding = tiktoken.encoding_for_model(model_str)

        if model_str.startswith("ollama:"):
            model_str = "ollama"
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


def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    openai_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=5.0,
    temp=None,
    print_cost=True,
):
    preloaded_api = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    answer: str | None = None
    for _ in range(tries):
        try:
            if (
                model_str == "gpt-4o-mini"
                or model_str == "gpt4omini"
                or model_str == "gpt-4omini"
                or model_str == "gpt4o-mini"
            ):
                model_str = "gpt-4o-mini"
                messages: Iterable[ChatCompletionMessageParam] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                client = OpenAI()
                if temp is None:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                    )
                else:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                        temperature=temp,
                    )
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )  # type: ignore
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                client = OpenAI()
                if temp is None:
                    completion = client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=messages,
                    )
                else:
                    completion = client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=messages,
                        temperature=temp,
                    )
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                deepseek_client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com/v1",
                )
                if temp is None:
                    completion = deepseek_client.chat.completions.create(
                        model="deepseek-chat", messages=messages
                    )
                else:
                    completion = deepseek_client.chat.completions.create(
                        model="deepseek-chat", messages=messages, temperature=temp
                    )
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [{"role": "user", "content": system_prompt + prompt}]
                client = OpenAI()
                completion = client.chat.completions.create(
                    model="o1-mini-2024-09-12", messages=messages
                )
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [{"role": "user", "content": system_prompt + prompt}]
                client = OpenAI()
                completion = client.chat.completions.create(
                    model="o1-2024-12-17", messages=messages
                )
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [{"role": "user", "content": system_prompt + prompt}]
                client = OpenAI()
                completion = client.chat.completions.create(
                    model="o1-preview", messages=messages
                )
                answer = completion.choices[0].message.content
            elif model_str.startswith("ollama:"):
                ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                ollama_client = ollama.Client(ollama_host)
                response: ollama.ChatResponse = ollama_client.chat(
                    model=model_str[7:],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = response.message.content

            compute_tokens(model_str, prompt, system_prompt, answer, print_cost)

            return answer or ""
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")
