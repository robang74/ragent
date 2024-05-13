from typing import Any, Dict, Iterator, List, Mapping, Optional

from pydantic import Extra

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from openai import OpenAI


class OpenAILikeLLM(LLM):
    """
    OpenAI-like LLM wrapper. Use OpenAI-Python SDK to call the API.

    Attention:
        llm_config(Dict) can only passed in when use "invoke" method.
        Like:
            llm.invoke(prompt,**llm_config)
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def __init__(
            self,
            api_key: str,
            base_url: str,
            model: str,
            **kwargs: Any,
    ) -> None:
        """Initialize the OpenAILikeLLM instance."""
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # self.params 中有两个键，分别是"params"和"model_client_cls"(可能)，
        self.params = kwargs

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the OpenAI API to get a completion for the prompt.
        
        Args:
            prompt (str): The prompt to pass into the API.
            stop (Optional[List[str]]): The stop sequence to use.
            run_manager (Optional[CallbackManagerForLLMRun]): A callback manager
                instance that can be used to track events.
            **kwargs: Additional keyword arguments to pass into the API.
                such as api_key, base_url, temperature, top_p, max_tokens, etc.
        """
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        model_params = kwargs.get("params", {})
        response = client.chat.completions.create(
            **model_params,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.model,
        )
        return response.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "openai-like"