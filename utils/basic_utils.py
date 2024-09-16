import streamlit as st
import pyperclip
from streamlit import cache_resource

import os
import json
import uuid
import copy
import base64
from datetime import datetime, timezone
from loguru import logger
from functools import lru_cache
from typing import List, Dict, Optional, Union
from io import BytesIO
from dotenv import load_dotenv
load_dotenv(override=True)

from llm.ollama.completion import get_ollama_model_list
from llm.groq.completion import get_groq_models
from configs.basic_config import I18nAuto, SUPPORTED_LANGUAGES
from configs.chat_config import OAILikeConfigProcessor

i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

@lru_cache(maxsize=10)
def model_selector(model_type):
    if model_type == "OpenAI" or model_type == "AOAI":
        return ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        try:
           model_list = get_ollama_model_list() 
           return model_list
        except:
            return ["qwen:7b-chat"]
    elif model_type == "Groq":
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_list = get_groq_models(api_key=groq_api_key,only_id=True)

            # exclude tts model
            model_list_exclude_tts = [model for model in model_list if "whisper" not in model]
            excluded_models = [model for model in model_list if model not in model_list_exclude_tts]

            logger.info(f"Groq model list: {model_list}, excluded models:{excluded_models}")
            return model_list_exclude_tts
        except:
            logger.info("Failed to get Groq model list, using default model list")
            return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    elif model_type == "Llamafile":
        return ["Noneed"]
    elif model_type == "LiteLLM":
        return ["Noneed"]
    else:
        return None


def oai_model_config_selector(oai_model_config:Dict):
    config_processor = OAILikeConfigProcessor()
    model_name = list(oai_model_config.keys())[0]
    config_dict = config_processor.get_config()

    if model_name in config_dict:
        return model_name, config_dict[model_name]["base_url"], config_dict[model_name]["api_key"]
    else:
        return "noneed", "http://127.0.0.1:8080/v1", "noneed"


# Display chat messages from history on app rerun
@st.cache_data
def write_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> None:
    if chat_history:
        for message in chat_history:
            try:
                if message["role"] == "system":
                    continue
            except:
                pass
            with st.chat_message(message["role"]):
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                elif isinstance(message["content"], List):
                    for content in message["content"]:
                        if content["type"] == "text":
                            st.markdown(content["text"])
                        elif content["type"] == "image_url":
                            # 如果开头为data:image/jpeg;base64，则解码为BytesIO对象
                            if content["image_url"]["url"].startswith("data:image/jpeg;base64"):
                                image_data = base64.b64decode(content["image_url"]["url"].split(",")[1])
                                st.image(image_data)
                            else:
                                st.image(content["image_url"])


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


def list_length_transform(n, lst) -> List:
    '''
    聊天上下文限制函数
    
    Args:
        n (int): 限制列表lst的长度为n
        lst (list): 需要限制长度的列表
        
    Returns:
        list: 限制后的列表
    '''
    # 如果列表lst的长度大于n，则返回lst的最后n个元素
    if len(lst) > n:
        return lst[-n:]
    # 如果列表lst的长度小于等于n，则返回lst本身
    else:
        return lst


def detect_and_decode(data_bytes):
    """
    尝试使用不同的编码格式来解码bytes对象。

    参数:
    data_bytes (bytes): 需要进行编码检测和解码的bytes对象。

    返回:
    tuple: 包含解码后的字符串和使用的编码格式。
           如果解码失败，返回错误信息。
    """
    # 定义常见的编码格式列表
    encodings = ['utf-8', 'ascii', 'gbk', 'iso-8859-1']

    # 遍历编码格式，尝试解码
    for encoding in encodings:
        try:
            # 尝试使用当前编码解码
            decoded_data = data_bytes.decode(encoding)
            # 如果解码成功，返回解码后的数据和编码格式
            return decoded_data, encoding
        except UnicodeDecodeError:
            # 如果当前编码解码失败，继续尝试下一个编码
            continue

    # 如果所有编码格式都解码失败，返回错误信息
    return "无法解码，未知的编码格式。", None


def config_list_postprocess(config_list: List[Dict]):
    """将config_list中，每个config的params字段合并到各个config中。"""
    config_list = copy.deepcopy(config_list)
    for config in config_list:
        if "params" in config:
            params = config["params"]
            del config["params"]
            config.update(**params)
    return config_list


def dict_filter(
        dict_data: Dict, 
        filter_keys: List[str] = None,
        filter_values: List[str] = None
    ) -> Dict:
    """
    过滤字典中的键值对，只保留指定的键或值。
    
    Args:
        dict_data (Dict): 要过滤的字典。
        filter_keys (List[str], optional): 要保留的键列表。默认值为None。
        filter_values (List[str], optional): 要保留的值列表。默认值为None。
        
    Returns:
        Dict: 过滤后的字典。
    """
    if filter_keys is None and filter_values is None:
        return dict_data
    
    filtered_dict = {}
    for key, value in dict_data.items():
        if (filter_keys is None or key in filter_keys) and (filter_values is None or value in filter_values):
            filtered_dict[key] = value
            
    return filtered_dict


def reverse_traversal(lst: List) -> Dict[str, str]:
    '''
    反向遍历列表，直到找到非空且不为'TERMINATE'的元素为止。
    用于处理 Tool Use Agent 的 chat_history 列表。
    '''
    # 遍历列表中的每一个元素
    for item in reversed(lst):
        # 如果元素中的内容不为空且不为'TERMINATE'，则打印元素
        if item.get('content', '') not in ('', 'TERMINATE'):
            return item


def copy_to_clipboard(content: str):
    '''
    将内容复制到剪贴板,并提供streamlit提醒
    '''
    pyperclip.copy(content)
    st.toast(i18n("The content has been copied to the clipboard"), icon="✂️")


def current_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


def current_datetime_utc_str() -> str:
    return current_datetime_utc().strftime("%Y-%m-%dT%H:%M:%S")


def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def encode_image(image: BytesIO) -> str:
    """
    将 BytesIO 对象编码为 base64 字符串.
    
    Args:
        image (BytesIO): BytesIO 对象
    Returns:
        str: base64 编码的字符串
    """
    image_data = image.getvalue()
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded


def user_input_constructor(
    prompt: str, 
    images: Optional[Union[BytesIO, List[BytesIO]]] = None, 
) -> Dict:
    """
    构造用户输入的字典。
    """
    base_input = {
        "role": "user"
    }

    if images is None:
        base_input["content"] = prompt
    elif isinstance(images, (BytesIO, list)):
        text_input = {
            "type": "text",
            "text": prompt
        }
        if isinstance(images, BytesIO):
            images = [images]
        
        image_inputs = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image)}"
                }
            } for image in images
        ]
        base_input["content"] = [text_input, *image_inputs]
    else:
        raise TypeError("images must be a BytesIO object or a list of BytesIO objects")

    return base_input