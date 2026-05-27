import os
import unittest
from unittest.mock import patch

from app.core import config, tracing
from app.core.config import Settings


class TracingHelperTests(unittest.TestCase):
    def _patch_settings(self, settings: Settings):
        return patch.object(config, "settings", settings)

    def test_is_tracing_enabled_requires_api_key(self):
        settings = Settings(_env_file=None, langsmith_tracing_enabled=True)
        settings.langsmith_api_key = ""
        with self._patch_settings(settings):
            self.assertFalse(tracing.is_tracing_enabled())

        settings.langsmith_api_key = "test-key"
        with self._patch_settings(settings):
            self.assertTrue(tracing.is_tracing_enabled())

    def test_build_run_config_disabled_returns_only_recursion_limit(self):
        settings = Settings(_env_file=None, langsmith_tracing_enabled=False)
        with self._patch_settings(settings):
            run_config = tracing.build_run_config(
                "react-agent-query",
                recursion_limit=32,
                route="react_agent",
                endpoint=tracing.REACT_AGENT_QUERY_ENDPOINT,
                streaming=False,
            )
        self.assertEqual(run_config, {"recursion_limit": 32})

    def test_build_run_config_enabled_includes_metadata(self):
        settings = Settings(
            _env_file=None,
            langsmith_tracing_enabled=True,
            langsmith_api_key="test-key",
            langsmith_project="test-project",
            chroma_collection_name="demo",
        )
        with self._patch_settings(settings):
            run_config = tracing.build_run_config(
                "react-agent-query",
                route="react_agent",
                endpoint=tracing.REACT_AGENT_QUERY_ENDPOINT,
                streaming=False,
                collection_name="resume_bank",
                history_count=2,
                phase="planning",
            )
        self.assertEqual(run_config["run_name"], "react-agent-query")
        self.assertEqual(run_config["metadata"]["route"], "react_agent")
        self.assertEqual(run_config["metadata"]["phase"], "planning")

    def test_build_run_config_uses_trace_context_metadata(self):
        settings = Settings(
            _env_file=None,
            langsmith_tracing_enabled=True,
            langsmith_api_key="test-key",
            chroma_collection_name="demo",
        )
        trace_ctx = tracing.build_agent_trace_context(
            route="plan_execute_agent",
            endpoint=tracing.PLAN_EXECUTE_QUERY_ENDPOINT,
            streaming=False,
            collection_name="resume_bank",
            history_count=3,
        )
        with self._patch_settings(settings):
            run_config = tracing.build_run_config(
                "plan-execute-planning",
                trace_context=trace_ctx,
                phase="planning",
            )
        self.assertEqual(run_config["metadata"]["route"], "plan_execute_agent")
        self.assertEqual(run_config["metadata"]["endpoint"], tracing.PLAN_EXECUTE_QUERY_ENDPOINT)
        self.assertEqual(run_config["metadata"]["collection_name"], "resume_bank")
        self.assertEqual(run_config["metadata"]["history_count"], 3)
        self.assertEqual(run_config["metadata"]["phase"], "planning")

    def test_merge_run_configs_combines_metadata_and_tags(self):
        merged = tracing.merge_run_configs(
            {"recursion_limit": 64, "tags": ["a"], "metadata": {"route": "react_agent"}},
            {"tags": ["b"], "metadata": {"streaming": True}},
        )
        self.assertEqual(merged["recursion_limit"], 64)
        self.assertEqual(merged["tags"], ["a", "b"])
        self.assertEqual(
            merged["metadata"],
            {"route": "react_agent", "streaming": True},
        )

    def test_configure_langsmith_tracing_sets_env_when_enabled(self):
        tracing._CONFIGURED = False
        settings = Settings(
            _env_file=None,
            langsmith_tracing_enabled=True,
            langsmith_api_key="test-key",
            langsmith_project="ragEnlighten-dev",
            langsmith_endpoint="https://api.smith.langchain.com",
        )
        with self._patch_settings(settings):
            tracing.configure_langsmith_tracing()
        self.assertEqual(os.environ.get("LANGCHAIN_TRACING_V2"), "true")
        self.assertEqual(os.environ.get("LANGCHAIN_API_KEY"), "test-key")
        self.assertEqual(os.environ.get("LANGCHAIN_PROJECT"), "ragEnlighten-dev")
        self.assertEqual(os.environ.get("LANGCHAIN_ENDPOINT"), "https://api.smith.langchain.com")

    def test_configure_langsmith_tracing_skips_export_without_api_key(self):
        tracing._CONFIGURED = False
        env_snapshot = {
            "LANGCHAIN_TRACING_V2": os.environ.pop("LANGCHAIN_TRACING_V2", None),
            "LANGCHAIN_API_KEY": os.environ.pop("LANGCHAIN_API_KEY", None),
        }
        settings = Settings(
            _env_file=None,
            langsmith_tracing_enabled=True,
            langsmith_api_key="",
        )
        try:
            with self._patch_settings(settings):
                tracing.configure_langsmith_tracing()
            self.assertNotIn("LANGCHAIN_TRACING_V2", os.environ)
        finally:
            for key, value in env_snapshot.items():
                if value is not None:
                    os.environ[key] = value
            tracing._CONFIGURED = False

    def test_configure_langsmith_tracing_skips_env_when_disabled(self):
        tracing._CONFIGURED = False
        env_snapshot = {
            "LANGCHAIN_TRACING_V2": os.environ.pop("LANGCHAIN_TRACING_V2", None),
            "LANGCHAIN_API_KEY": os.environ.pop("LANGCHAIN_API_KEY", None),
        }
        settings = Settings(_env_file=None, langsmith_tracing_enabled=False)
        try:
            with self._patch_settings(settings):
                tracing.configure_langsmith_tracing()
            self.assertNotIn("LANGCHAIN_TRACING_V2", os.environ)
        finally:
            for key, value in env_snapshot.items():
                if value is not None:
                    os.environ[key] = value
            tracing._CONFIGURED = False


if __name__ == "__main__":
    unittest.main()
