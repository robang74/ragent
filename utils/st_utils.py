import streamlit as st
import streamlit.components.v1 as components
import os
import time
import whisper
from streamlit_float import *
from audiorecorder import audiorecorder

from configs.basic_config import I18nAuto, SUPPORTED_LANGUAGES
from utils.basic_utils import copy_to_clipboard
from tools.toolkits import TO_TOOLS

# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])


def back_to_top(script_container = st.empty(), buttom_container = st.empty()):
    """
    Scroll the page to the top.
    
    Args:
        script_container (streamlit.empty, optional): The temporary container to hold the script. Defaults to st.empty().
        buttom_container (streamlit.empty, optional): The container to hold the button. Defaults to st.empty().
    """
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = 0;
    </script>
    '''
    top_container = buttom_container.container()
    top_css = float_css_helper(width="2.2rem", right="10rem", bottom="13rem")
    with top_container:
        up_button = st.button("⭱", key="up_button")
        if up_button:
            with script_container:
                components.html(js)
                time.sleep(.5) # To make sure the script can execute before being deleted
            script_container.empty()
    top_container.float(top_css)


def back_to_bottom(script_container = st.empty(), buttom_container = st.empty()):
    """
    Scroll the page to the bottom.
    
    Args:
        script_container (streamlit.empty, optional): The temporary container to hold the script. Defaults to st.empty().
        buttom_container (streamlit.empty, optional): The container to hold the button. Defaults to st.empty().
    """
    js = '''
    <script>
        var body = window.parent.document.querySelector(".main");
        console.log(body);
        body.scrollTop = body.scrollHeight;
    </script>
    '''

    bottom_container = buttom_container.container()
    bottom_css = float_css_helper(width="2.2rem", right="10rem", bottom="10rem")
    with bottom_container:
        bottom_button = st.button("⭳", key="bottom_button")
        if bottom_button:
            with script_container:
                components.html(js)
                time.sleep(.5) # To make sure the script can execute before being deleted
            script_container.empty()
    bottom_container.float(bottom_css)


def float_chat_input_with_audio_recorder(if_tools_call: str = False, prompt_disabled: bool = False) -> str:
    """
    Create a container with a floating chat input and an audio recorder.

    Returns:
        str: The text input from the user.
    """        
    # Create a container with a floating chat input and an audio recorder
    chat_input_container = st.container()
    with chat_input_container:
        # divide_context_column, character_input_column, voice_input_column = st.columns([0.1,0.9,0.1])
        if if_tools_call:
            tools_popover = st.popover(label="🔧")
            tools_popover.multiselect(
                label=i18n("Functions"),
                options=TO_TOOLS.keys(),
                default=list(TO_TOOLS.keys())[:2],
                help=i18n("Select functions you want to use."),
                # format_func 将所有名称开头的"tool_"去除
                format_func=lambda x: x.replace("tool_",""),
                key="tools_popover"
            )
        character_input_column, voice_input_column = st.columns([0.9,0.1])
        # divide_context_placeholder = divide_context_column.empty()
        # divide_context_button = divide_context_placeholder.button(
        #     label="✂️",
        # )
        # if divide_context_button:
        #     storage.upsert()

        # the chat input in the middle
        character_input_placeholder = character_input_column.empty()
        prompt = character_input_placeholder.chat_input("What is up?", disabled=prompt_disabled)

        # the button (actually popover) on the right side of the chat input is to record audio
        voice_input_popover = voice_input_column.popover(
            label="🎤"
        )
        voice_input_model_name = voice_input_popover.selectbox(
            label=i18n("Voice input model"),
            options=whisper.available_models(),
            index=3,
            key="voice_input_model"   
        )
        audio_recorder_container =  voice_input_popover.container(border=True)
        with audio_recorder_container:
            # TODO:没有麦克风可能无法录音
            # audio_recorded = audiorecorder(start_prompt='',stop_prompt='',pause_prompt='')
            audio_recorded = audiorecorder(pause_prompt='pause')
            audio_placeholder = st.empty()
            transcribe_button_placeholder = st.empty()
            if len(audio_recorded) > 0:
                # To play audio in frontend:
                audio = audio_recorded.export().read()
                audio_placeholder.audio(audio)
                transcribe_button = transcribe_button_placeholder.button(
                    label=i18n("Transcribe"),
                    use_container_width=True
                )
                # 临时存储音频文件
                with open("dynamic_configs/temp.wav", "wb") as f:
                    f.write(audio)
                # TODO：按下识别按钮后，才能识别语音
                # 加载语音识别模型
                if transcribe_button:
                    with st.status(i18n("Transcribing...")):
                        st.write(i18n("Loading model"))
                        voice_input_model = whisper.load_model(
                            name=voice_input_model_name,
                            download_root="./tts_models"
                        )
                        st.write(i18n("Model loaded"))
                        # 识别语音
                        st.write(i18n("Transcribing"))
                        transcribe_result = voice_input_model.transcribe(audio="dynamic_configs/temp.wav",word_timestamps=True,verbose=True)
                        st.write(i18n("Transcribed"))
                    content = transcribe_result.get("text","No result.")
                    copy_to_clipboard(content)
                    st.code(content)
                    # 删除临时文件
                    os.remove("dynamic_configs/temp.wav")

    chat_input_css = float_css_helper(bottom="6rem", display="flex", justify_content="center", margin="0 auto")
    chat_input_container.float(chat_input_css)
    return prompt