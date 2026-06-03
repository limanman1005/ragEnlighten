"""本地快速验证 HelloAgentsLLM（读取项目根目录 .env）。"""

from __future__ import annotations

import sys
from pathlib import Path

# 从 scripts/ 运行时，将项目根目录加入 path，以便 import app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.hello_agents_llm import HelloAgentsLLM


def main() -> None:
    try:
        llm_client = HelloAgentsLLM()

        example_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        print("--- 调用 LLM ---")
        response_text = llm_client.think(example_messages)
        if response_text:
            print("\n--- 完整模型响应 ---")
            print(response_text)

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
