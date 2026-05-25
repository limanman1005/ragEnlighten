import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.services.react_agent import _build_agent_runtime


class ReactAgentMcpRuntimeTests(unittest.TestCase):
    def test_local_mode_remains_default(self):
        original_enabled = settings.mcp_enabled
        settings.mcp_enabled = False
        try:
            with patch("app.services.react_agent.create_agent") as mock_create_agent:
                mock_create_agent.return_value = object()
                agent, documents, tool_names, mcp_load_error = asyncio.run(
                    _build_agent_runtime("resume_bank")
                )
        finally:
            settings.mcp_enabled = original_enabled

        self.assertIsNotNone(agent)
        self.assertEqual(documents, [])
        self.assertEqual(
            tool_names,
            ["knowledge_base_search", "collection_overview", "web_search"],
        )
        self.assertIsNone(mcp_load_error)
        passed_tools = mock_create_agent.call_args.kwargs["tools"]
        self.assertEqual(len(passed_tools), 3)

    def test_mcp_mode_passes_loaded_tools_to_runtime(self):
        original_enabled = settings.mcp_enabled
        original_fallback = settings.mcp_fallback_to_local
        settings.mcp_enabled = True
        settings.mcp_fallback_to_local = True
        try:
            fake_tools = [object(), object(), object()]
            with patch(
                "app.services.react_agent.load_mcp_tools",
                new=AsyncMock(return_value=(fake_tools, None)),
            ), patch("app.services.react_agent.create_agent") as mock_create_agent:
                mock_create_agent.return_value = object()
                _, _, tool_names, mcp_load_error = asyncio.run(_build_agent_runtime())
        finally:
            settings.mcp_enabled = original_enabled
            settings.mcp_fallback_to_local = original_fallback

        self.assertIsNone(mcp_load_error)
        self.assertEqual(len(tool_names), 3)
        passed_tools = mock_create_agent.call_args.kwargs["tools"]
        self.assertIs(passed_tools, fake_tools)

    def test_mcp_load_failure_falls_back_to_local_tools(self):
        original_enabled = settings.mcp_enabled
        original_fallback = settings.mcp_fallback_to_local
        settings.mcp_enabled = True
        settings.mcp_fallback_to_local = True
        try:
            with patch(
                "app.services.react_agent.load_mcp_tools",
                new=AsyncMock(return_value=([], "connection refused")),
            ), patch("app.services.react_agent.create_agent") as mock_create_agent:
                mock_create_agent.return_value = object()
                _, _, tool_names, mcp_load_error = asyncio.run(_build_agent_runtime())
        finally:
            settings.mcp_enabled = original_enabled
            settings.mcp_fallback_to_local = original_fallback

        self.assertEqual(mcp_load_error, "connection refused")
        self.assertEqual(
            tool_names,
            ["knowledge_base_search", "collection_overview", "web_search"],
        )


if __name__ == "__main__":
    unittest.main()
