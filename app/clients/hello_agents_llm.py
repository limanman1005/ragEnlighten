"""OpenAI-compatible LLM client (Hello Agents book style) using project .env settings."""

from __future__ import annotations

import os
from openai import OpenAI

from app.core.config import settings


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的 LLM 客户端。
    调用兼容 OpenAI 接口的服务，默认使用流式响应。
    配置优先使用构造参数，否则从项目 Settings / .env 读取。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = (
            model
            or settings.llm_model
            or os.getenv("LLM_MODEL")
            or os.getenv("LLM_MODEL_ID")
        )
        api_key = api_key or settings.llm_api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or settings.llm_base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all([self.model, api_key, base_url]):
            raise ValueError(
                "模型 ID、API 密钥和服务地址必须传入，或在 .env 中配置 "
                "LLM_MODEL、LLM_API_KEY、LLM_BASE_URL。"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
    ) -> str | None:
        """调用大语言模型进行思考，并返回完整响应文本。"""
        print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("大语言模型响应成功:")
            collected_content: list[str] = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            return None
