"""Offline checks of the real-KG boundary; fixtures never access external services."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from kg_interface import KnowledgeGraph, load_knowledge_graph


class RecordingBackend(KnowledgeGraph):
    """Isolated lookup fixture for validating argument and result forwarding."""

    def __init__(self, source="test-fixture"):
        self.source = source
        self.calls = []

    def lookup(self, entity, question="", relation="", k=10):
        self.calls.append((entity, question, relation, k))
        return entity, [(entity, "relation", "Neighbor")]


class InterfaceTests(unittest.TestCase):
    def test_interface_cannot_be_used_as_a_backend(self):
        with self.assertRaises(TypeError):
            KnowledgeGraph()

    def test_external_backend_receives_constructor_configuration(self):
        module = ModuleType("_kg_boundary_test")
        module.RealKG = RecordingBackend
        with patch.dict(sys.modules, {module.__name__: module}):
            backend = load_knowledge_graph("_kg_boundary_test:RealKG", {"source": "configured-source"})
        self.assertEqual(backend.source, "configured-source")

    def test_invalid_backend_and_configuration_are_rejected(self):
        for spec in ("", "module", ":Class", "module:", "module:Class:extra"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                load_knowledge_graph(spec)
        with self.assertRaises(TypeError):
            load_knowledge_graph("unused:Class", [])
        with self.assertRaises(TypeError):
            load_knowledge_graph("builtins:dict")
        with self.assertRaises(TypeError):
            load_knowledge_graph("kg_interface:KnowledgeGraph")

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(HERE / "run_agent_kbqa.py"), *args],
            capture_output=True, text=True, check=False,
        )

    def test_help_requires_no_backend_or_model(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--kg_backend", result.stdout)

    def test_missing_backend_stops_before_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            result = self.run_cli("--out_dir", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertIn("--kg_backend", result.stderr)
            self.assertFalse(output.exists())

    def test_invalid_configuration_stops_before_backend_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "kg.json"
            config.write_text(json.dumps(None))
            output = Path(tmp) / "results"
            result = self.run_cli("--kg_backend", "not_installed:RealKG", "--kg_config", str(config),
                                  "--out_dir", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertIn("JSON object", result.stderr)
            self.assertFalse(output.exists())

    def test_unknown_backend_stops_before_creating_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            result = self.run_cli("--kg_backend", "_missing_kg_backend_:RealKG", "--out_dir", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertIn("Cannot initialize the real KG backend", result.stderr)
            self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("qwen_agent"), "Qwen-Agent is required for tool integration checks")
class ToolTests(unittest.TestCase):
    def setUp(self):
        from kg_tool import KGSearch

        self.tool_class = KGSearch

    def test_lookup_forwards_context_and_records_graph_evidence(self):
        backend = RecordingBackend()
        tool = self.tool_class(backend, "Question?", top_k=3)
        result = tool.call('{"entity": "Entity", "relation": "relation"}')
        self.assertEqual(backend.calls, [("Entity", "Question?", "relation", 3)])
        self.assertEqual(result, "(Entity, relation, Neighbor)")
        self.assertEqual(tool.trace[0]["triples"], [("Entity", "relation", "Neighbor")])

    def test_empty_evidence_and_connection_failure_remain_distinct(self):
        backend = RecordingBackend()
        tool = self.tool_class(backend, "Question?")
        with patch.object(backend, "lookup", return_value=("Entity", [])):
            self.assertIn("No facts found", tool.call({"entity": "Entity"}))
        with patch.object(backend, "lookup", side_effect=ConnectionError("KG unavailable")):
            with self.assertRaisesRegex(ConnectionError, "KG unavailable"):
                tool.call({"entity": "Entity"})
        self.assertEqual(len(tool.trace), 1)

    def test_invalid_retrieval_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tool_class(RecordingBackend(), "Question?", top_k=0)


if __name__ == "__main__":
    unittest.main()
