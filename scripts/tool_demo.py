"""本地验证 ToolExecutor + SerpApi 搜索工具（读取项目根目录 .env）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.clients.search_api import search
from app.clients.tool_executor import ToolExecutor


def main() -> None:
    tool_executor = ToolExecutor()

    search_description = (
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中"
        "找不到的信息时，应使用此工具。"
    )
    tool_executor.registerTool("Search", search_description, search)

    print("\n--- 可用的工具 ---")
    print(tool_executor.getAvailableTools())

    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = tool_executor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")


if __name__ == "__main__":
    main()
