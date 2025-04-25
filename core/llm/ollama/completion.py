import requests
import json
import os
import streamlit as st
from openai import OpenAI

from typing import List, Dict, Union, Optional


def ollama_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的llamafile Completion Client配置

    Args:
        kwargs (dict): 配置参数
            model (str): 模型名称
            api_key (str): API Key
            base_url (str): Base URL
            params (dict): 其他请求参数
                temperature (float): 温度
                top_p (float): Top P
                stream (bool): 是否流式输出
        
    Returns:
        config (list): 配置列表
    '''
    config = {
        "model": kwargs.get("model", "noneed"),
        "api_key": kwargs.get("api_key", "noneed"),
        "base_url": kwargs.get("base_url","http://localhost:11434/v1"),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.5),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": kwargs.get("stream", False),
        },
        "model_client_cls": "OllamaClient",
    }
    return [config]

def get_ollama_model_list(url:str):
    """
    获取模型标签列表
    
    Returns:
        model_list(list): 所有模型的名称
    """
    default_url = "http://localhost:11434/api/"
    if url == "" or url is None:
        url = os.getenv("OLLAMA_API_ENDPOINT", default_url)
    model_tags_url = f"{url}tags"
    response = requests.get(model_tags_url)

    if response.status_code != 200:
        raise ValueError("无法获取模型标签列表")
    
    if response == None:
        response = {}

    tags = response.json()
    
    # 获取所有模型的名称
    model_list = [model['name'] for model in tags['models']]
    return model_list
