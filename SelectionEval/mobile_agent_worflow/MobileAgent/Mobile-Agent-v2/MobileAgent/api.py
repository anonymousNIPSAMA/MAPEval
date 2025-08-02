import base64
import requests
import json
import hashlib
import os
import threading
from functools import wraps
import aiohttp
import asyncio

# 帮我写cache装饰器
def cache(func):
    cache_file = f"cache_{func.__name__}.jsonl"
    lock = threading.Lock()  # 线程锁确保多线程安全
    
    def load_cache():
        cache_data = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            cache_data[entry['key']] = entry['value']
            except:
                pass
        return cache_data
    
    def append_cache(key, value):
        try:
            with lock:  # 确保线程安全
                entry = {'key': key, 'value': value}
                with open(cache_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except:
            pass
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建包含函数名和参数的唯一键
        key_data = {
            'function': func.__name__,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key = hashlib.md5(str(key_data).encode('utf-8')).hexdigest()
        
        # 每次都从文件读取缓存
        cache_data = load_cache()
        
        if key not in cache_data:
            # 计算结果并追加到缓存文件
            result = func(*args, **kwargs)
            append_cache(key, result)
            return result
        else:
            # 返回缓存的结果
            return cache_data[key]

    return wrapper

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# @cache
def inference_chat(chat, model, api_url, token,temperature=0.0):    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': temperature,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    i = 0
    while i < 2:
        i += 1
        try:
            res = requests.post(api_url, headers=headers, json=data)
            res_json = res.json()
            res_content = res_json['choices'][0]['message']['content']
        except:
            print("Network Error:")
            try:
                print(res.json())
            except:
                print("Request Failed")
        else:
            break
    
    return res_content

# @cache
async def inference_achat(chat, model, api_url, token, temperature=0.0):
    if 'qwen' in model:
        api_url = "http://localhost:9999/v1/chat/completions"
    
    """
    异步版本的inference_chat函数
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    data = {
        "model": model,
        "messages": [],
        "max_tokens": 2048,
        'temperature': temperature,
        "seed": 1234
    }

    for role, content in chat:
        data["messages"].append({"role": role, "content": content})

    i = 0
    while i < 2:
        i += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=data) as response:
                    res_json = await response.json()
                    res_content = res_json['choices'][0]['message']['content']
                    return res_content
        except Exception as e:
            print(f"Network Error: {e}")
            if i >= 2:
                print("Request Failed after retries")
                raise
            await asyncio.sleep(1)  # 等待1秒后重试
    
    return res_content
