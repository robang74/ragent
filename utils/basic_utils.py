import streamlit as st
from streamlit import cache_resource

import os
import json
import uuid
from typing import List, Dict

from llm.ollama.completion import get_ollama_model_list


def model_selector(model_type):
    if model_type == "OpenAI" or model_type == "AOAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        try:
           model_list = get_ollama_model_list() 
           return model_list
        except:
            return ["qwen:7b-chat"]
    elif model_type == "Groq":
        return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    elif model_type == "Llamafile":
        return ["Noneed"]
    else:
        return None
    
    
def split_list_by_key_value(dict_list, key, value):
    result = []
    temp_list = []
    count = 0

    for d in dict_list:
        # 检查字典是否有指定的key，并且该key的值是否等于指定的value
        if d.get(key) == value:
            count += 1
            temp_list.append(d)
            # 如果指定值的出现次数为2，则分割列表
            if count == 2:
                result.append(temp_list)
                temp_list = []
                count = 0
        else:
            # 如果当前字典的key的值不是指定的value，则直接添加到当前轮次的列表
            temp_list.append(d)

    # 将剩余的临时列表（如果有）添加到结果列表
    if temp_list:
        result.append(temp_list)

    return result


class Meta(type):
    def __new__(cls, name, bases, attrs):
        for name, value in attrs.items():
            if callable(value) and not name.startswith('__') and not name.startswith('_'):  # 跳过特殊方法和私有方法
                attrs[name] = cache_resource(value)
        return super().__new__(cls, name, bases, attrs)
    

def save_basic_chat_history(
        chat_name: str,
        chat_history: List[Dict[str, str]], 
        chat_history_file: str = 'chat_history.json'):
    """
    保存一般 LLM Chat 聊天记录
    
    Args:
        user_id (str): 用户id
        chat_history (List[Tuple[str, str]]): 聊天记录
        chat_history_file (str, optional): 聊天记录文件. Defaults to 'chat_history.json'.
    """
    # TODO: 添加重名检测，如果重名则添加时间戳
    # 如果聊天历史记录文件不存在，则创建一个空的文件
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    
    # 打开聊天历史记录文件，读取数据
    with open(chat_history_file, 'r', encoding='utf-8') as f:
        data:dict = json.load(f)
        
    # 如果聊天室名字不在数据中，则添加聊天的名字和完整聊天历史记录
    if chat_name not in data:
        data.update(
            {
                chat_name: {
                    "chat_history":chat_history,
                    "id": str(uuid.uuid4())
                }
            }
        )
        
    # 打开聊天历史记录文件，写入更新后的数据
    with open(chat_history_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)