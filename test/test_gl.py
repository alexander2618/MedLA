import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "deepseek-ai/DeepSeek-R1",
    "stream": False,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1
}
headers = {
    "Authorization": "sk-wqczdxpmqmhyywptmkchcdruhdhrdlwszomrxtissmjtlpix",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)