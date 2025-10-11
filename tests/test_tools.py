import unittest

from ptis.tools import PythonTool, PythonToolError, RAGTool


class RAGToolTests(unittest.TestCase):
    def test_query_returns_ranked_docs(self) -> None:
        tool = RAGTool()
        tool.index([
            ("doc1", "通义千问支持多模型路由"),
            ("doc2", "数学模型擅长计算"),
        ])
        results = tool.query("数学推理")
        self.assertTrue(results)
        self.assertEqual(results[0].doc_id, "doc2")


class PythonToolTests(unittest.TestCase):
    def test_run_simple_expression(self) -> None:
        tool = PythonTool()
        output = tool.run("result = abs(-5)")
        self.assertEqual(output, "5")

    def test_disallow_imports(self) -> None:
        tool = PythonTool()
        with self.assertRaises(PythonToolError):
            tool.run("import os\nresult = 1")


if __name__ == "__main__":
    unittest.main()
