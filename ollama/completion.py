import requests
import json
import streamlit as st

from typing import List, Dict, Union, Optional

def process_chat_response(response):
    """
    处理聊天API的流式输出

    Args:
        response (requests.Response): 聊天API的响应对象

    Yields:
        dict: 每个聊天消息的结果
    """
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data


def process_api_params(is_enable:bool=False,
                       choose_list:list=["max_tokens","frequency_penalty"],
                       **kwargs):
    '''
    处理 Ollama API 传入参数
     OpenAI 的参数和 Ollama 的参数在设置时，合适的数值大小不同
    此外，某些参数的名称也不同，详见 https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    Args:
        is_enable (bool): 是否启用参数处理
        choose_list (list): 需要保留到 API 请求的参数列表
        kwargs (dict): 传入的参数
    '''
    if is_enable:
        # 对option_kwargs进行处理，使其符合API的要求
        num_ctx = kwargs.pop('max_tokens', None)
        if num_ctx is not None:
            kwargs['num_ctx'] = num_ctx
        
        repeat_penalty = kwargs.pop('frequency_penalty', None)
        if repeat_penalty is not None:
            kwargs['repeat_penalty'] = repeat_penalty

        kwargs = {k: v for k, v in kwargs.items() if k in choose_list}

        return kwargs
    

@st.cache_data(show_spinner=False)
def get_ollama_model_list(url:str="http://localhost:11434/api/"):
    """
    获取模型标签列表
    
    Returns:
        model_list(list): 所有模型的名称
    """
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


class OllamaResponse():
    '''
    用于处理 Ollama API 返回结果的类
    '''
    def __init__(self,response:dict):
        self.response:dict = response
        self.cost:float = 0


class OllamaCompletionClient:
    '''用于生成 Ollama 聊天补全的基本类'''
    def __init__(self):
        self.url:str = "http://localhost:11434/api/"
        self.cost:float = 0

    def create_completion(self,
            chat_history:list,
            model:str,
            **option_kwargs
        ):
        """
        发送聊天请求

        Args:
            messages (list): 聊天消息列表，每个消息包括角色和内容,与 OpenAI message 格式相同

        Yields:
            dict: 服务器返回的聊天结果
        """
        # url:str="http://localhost:11434/api/chat"
        chat_url = f"{self.url}chat"

        kwargs = process_api_params(is_enable=True, **option_kwargs)

        data = {
            "model": model,
            "messages": chat_history,
            "stream": False, # 流式输出则注释掉此行
            "options": kwargs
        }

        try:
            response = requests.post(chat_url, json=data, stream=True, timeout=10)
            response_ollama = OllamaResponse(response.json())
        except:
            response = {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": ""
                }
            }
            response_ollama = OllamaResponse(response)
            st.error("请求超时")

        return response_ollama


    def get_ollama_model_list(self):
        """
        获取模型标签列表
        
        Returns:
            model_list(list): 所有模型的名称
        """
        model_tags_url = f"{self.url}tags"
        response = requests.get(model_tags_url)

        if response.status_code != 200:
            raise ValueError("无法获取模型标签列表")
        
        if response == None:
            response = {}

        tags = response.json()

        # 获取所有模型的名称
        model_list = [model['name'] for model in tags['models']]
        return model_list
    
    def extract_text_or_completion_object(self,response:Union[dict,OllamaResponse]):
        """
        从服务器返回的聊天结果中提取文本或 OllamaResponse 对象
        
        Args:
            response (dict,OllamaResponse): 服务器返回的聊天结果
            
        Returns:
            response_list: 聊天结果的文本列表，一般只有一个消息有文本
        """
        if isinstance(response,dict):
            response_list = [response['message']['content']]
        elif isinstance(response,OllamaResponse):
            response_list = [response.response['message']['content']]
        else:
            raise ValueError("response 参数必须是 dict 或 OllamaResponse 类型")
        return response_list

