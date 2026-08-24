"""Tests for Agent4SR modules: corpus, io, prompts, function_call, tools, agent, combine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent4sr.corpus import generate_corpus, save_corpus
from src.agent4sr.function_call import (
    corpus_codes_path,
    corpus_methods_path,
    find_class,
    find_method,
    fuzzy_search,
    get_classes,
    get_code_snippet,
    get_methods,
    get_paths,
    load_corpus_codes,
    load_corpus_methods,
    split4search,
)
from src.agent4sr.io import BugInputs, load_bug_inputs
from src.agent4sr.prompts import (
    sr_finisher_user_prompt,
    sr_initial_user_prompt,
    sr_retry_user_prompt,
    sr_system_prompt,
    sr_tool_call_user_prompt,
)
from src.agent4sr.tools import ToolContext, execute_tool, normalize_method_name, tool_schemas
from src.common.java_parser import MethodInfo
from src.common.method_entity import method_info_to_corpus_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CORPUS_METHODS = [
    "org.example$SampleClass.add(int,int)",
    "org.example$SampleClass.format(String,Object)",
    "org.example.sub$Helper.run()",
    "org.example.sub$Helper.stop(int)",
]

SAMPLE_CORPUS_CODES = [
    "public int add(int a, int b) { return a + b; }",
    "public String format(String template, Object value) { return String.format(template, value); }",
    'public void run() { System.out.println("running"); }',
    "public void stop(int code) { System.exit(code); }",
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a temporary Agent4SR directory with corpus files."""
    a4sr_dir = tmp_path / "data" / "D4J" / "processed" / "TestProj" / "1" / "Agent4SR"
    a4sr_dir.mkdir(parents=True)
    (a4sr_dir / "corpus_methods.txt").write_text(
        "\n".join(SAMPLE_CORPUS_METHODS) + "\n", encoding="utf-8"
    )
    (a4sr_dir / "corpus_codes.txt").write_text(
        "\n".join(SAMPLE_CORPUS_CODES) + "\n", encoding="utf-8"
    )
    return a4sr_dir


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Create a temporary processed directory with trigger_test_clean.txt and bug_report.json."""
    proc = tmp_path / "data" / "D4J" / "processed" / "TestProj" / "1"
    proc.mkdir(parents=True)
    (proc / "trigger_test_clean.txt").write_text(
        "public void testAdd() {\n    assertEquals(2, add(1, 1));\n}\n"
        "---\njava.lang.AssertionError\n\tat org.example.SampleClass.add(SampleClass.java:10)\n"
    )
    # A stale raw dump that must NOT win over the clean artifact.
    (proc / "trigger_tests").write_text(
        "org.example.TestClass::testAdd\norg.example.TestClass::testFormat\n"
    )
    (proc / "bug_report.json").write_text(
        json.dumps({"title": "NPE in add", "description": "Null pointer when adding"})
    )
    return proc


# =====================================================================
# Tests for corpus.py
# =====================================================================


class TestCorpusIdFormatting:
    """Test the canonical Java method_info_to_corpus_id formatting."""

    def test_basic_method(self) -> None:
        m = MethodInfo(
            method_id="org.example.SampleClass.add(int,int).9.11",
            class_fqn="org.example.SampleClass",
            method_name="add",
            param_types=["int", "int"],
            start_line=9,
            end_line=11,
            content="public int add(int a, int b) { return a + b; }",
        )
        result = method_info_to_corpus_id(m)
        assert result == "org.example$SampleClass.add(int,int)"

    def test_no_package(self) -> None:
        m = MethodInfo(
            method_id="Foo.bar().1.5",
            class_fqn="Foo",
            method_name="bar",
            param_types=[],
            start_line=1,
            end_line=5,
            content="void bar() {}",
        )
        result = method_info_to_corpus_id(m)
        # No package, so the original signature is returned without $
        assert result == "Foo.bar()"

    def test_nested_package(self) -> None:
        m = MethodInfo(
            method_id="org.apache.commons.lang3.StringUtils.isEmpty(CharSequence).42.50",
            class_fqn="org.apache.commons.lang3.StringUtils",
            method_name="isEmpty",
            param_types=["CharSequence"],
            start_line=42,
            end_line=50,
            content="public static boolean isEmpty(CharSequence cs) { ... }",
        )
        result = method_info_to_corpus_id(m)
        assert result == "org.apache.commons.lang3$StringUtils.isEmpty(CharSequence)"

    def test_no_params(self) -> None:
        m = MethodInfo(
            method_id="com.foo.Bar.baz().10.15",
            class_fqn="com.foo.Bar",
            method_name="baz",
            param_types=[],
            start_line=10,
            end_line=15,
            content="void baz() {}",
        )
        result = method_info_to_corpus_id(m)
        assert result == "com.foo$Bar.baz()"


class TestCorpusGeneration:
    """Test generate_corpus and save_corpus."""

    @staticmethod
    def _sample_only_src(tmp_path: Path, fixtures_dir: Path) -> Path:
        src_dir = tmp_path / "sample_src"
        src_dir.mkdir(parents=True)
        (src_dir / "SampleClass.java").write_text((fixtures_dir / "SampleClass.java").read_text())
        return src_dir

    def test_generate_corpus_from_fixture(self, tmp_path: Path, fixtures_dir: Path) -> None:
        """Generate corpus from the SampleClass.java fixture."""
        src_dir = self._sample_only_src(tmp_path, fixtures_dir)
        with patch("src.agent4sr.corpus.get_src_dir", return_value=src_dir):
            corpus = generate_corpus("TestProj", "1")

        assert len(corpus.method_ids) == 2
        assert len(corpus.raw_codes) == 2
        # Should contain the add and format methods
        ids = corpus.method_ids
        assert any("add" in mid for mid in ids)
        assert any("format" in mid for mid in ids)

    def test_save_corpus_creates_files(self, tmp_path: Path, fixtures_dir: Path) -> None:
        """save_corpus should create corpus_methods.txt and corpus_codes.txt."""
        src_dir = self._sample_only_src(tmp_path, fixtures_dir)
        out_dir = tmp_path / "Agent4SR"
        out_dir.mkdir(parents=True)
        with (
            patch("src.agent4sr.corpus.get_src_dir", return_value=src_dir),
            patch("src.agent4sr.corpus.get_sr_dir", return_value=out_dir),
        ):
            save_corpus("TestProj", "1")

        assert (out_dir / "corpus_methods.txt").exists()
        assert (out_dir / "corpus_codes.txt").exists()
        methods = (out_dir / "corpus_methods.txt").read_text().strip().splitlines()
        codes = (out_dir / "corpus_codes.txt").read_text().strip().splitlines()
        assert len(methods) == 2
        assert len(codes) == 2

    def test_save_corpus_skip_existing(self, tmp_path: Path) -> None:
        """save_corpus with skip_existing should not regenerate when corpus files exist."""
        out_dir = tmp_path / "Agent4SR"
        out_dir.mkdir(parents=True)
        (out_dir / "corpus_methods.txt").write_text("existing\n")
        (out_dir / "corpus_codes.txt").write_text("existing\n")
        with patch("src.agent4sr.corpus.get_sr_dir", return_value=out_dir):
            save_corpus("TestProj", "1", skip_existing=True)
        # Should not have changed
        assert (out_dir / "corpus_methods.txt").read_text() == "existing\n"


# =====================================================================
# Tests for io.py
# =====================================================================


class TestBugInputs:
    """Test BugInputs dataclass."""

    def test_bug_report_text_block_with_report(self) -> None:
        inputs = BugInputs(
            project="Chart",
            bug_id="1",
            trigger_test="test code",
            bug_report_title="NPE in add",
            bug_report_description="Null pointer when adding",
        )
        block = inputs.bug_report_text_block()
        assert block is not None
        assert "NPE in add" in block
        assert "Null pointer when adding" in block
        assert "```" in block

    def test_bug_report_text_block_without_report(self) -> None:
        inputs = BugInputs(
            project="Chart",
            bug_id="1",
            trigger_test="test code",
            bug_report_title=None,
            bug_report_description=None,
        )
        assert inputs.bug_report_text_block() is None

    def test_trigger_test_text_block(self) -> None:
        inputs = BugInputs(
            project="Chart",
            bug_id="1",
            trigger_test="org.example.Test::testFoo",
            bug_report_title=None,
            bug_report_description=None,
        )
        block = inputs.trigger_test_text_block()
        assert "org.example.Test::testFoo" in block
        assert "```" in block

    def test_load_bug_inputs(self, processed_dir: Path) -> None:
        """load_bug_inputs should prefer trigger_test_clean.txt over the raw dump."""
        with patch("src.agent4sr.io.get_processed_dir", return_value=processed_dir):
            inputs = load_bug_inputs("TestProj", "1")

        assert inputs.project == "TestProj"
        assert inputs.bug_id == "1"
        # Clean artifact carries test source + trimmed trace; raw dump must lose.
        assert "assertEquals(2, add(1, 1))" in inputs.trigger_test
        assert "AssertionError" in inputs.trigger_test
        assert "testFormat" not in inputs.trigger_test
        assert inputs.bug_report_title == "NPE in add"
        assert inputs.bug_report_description == "Null pointer when adding"

    def test_load_bug_inputs_falls_back_to_raw_trigger(self, tmp_path: Path) -> None:
        """load_bug_inputs should fall back to trigger_tests when the clean file is absent."""
        proc = tmp_path / "proc"
        proc.mkdir()
        (proc / "trigger_tests").write_text("org.example.TestClass::testAdd\n")
        with patch("src.agent4sr.io.get_processed_dir", return_value=proc):
            inputs = load_bug_inputs("TestProj", "1")

        assert inputs.trigger_test == "org.example.TestClass::testAdd"

    def test_load_bug_inputs_no_bug_report(self, tmp_path: Path) -> None:
        """load_bug_inputs should work without a bug_report.json."""
        proc = tmp_path / "proc"
        proc.mkdir()
        (proc / "trigger_test_clean.txt").write_text("test source\n")
        with patch("src.agent4sr.io.get_processed_dir", return_value=proc):
            inputs = load_bug_inputs("TestProj", "1")

        assert inputs.trigger_test == "test source"
        assert inputs.bug_report_title is None
        assert inputs.bug_report_description is None

    def test_load_bug_inputs_missing_trigger(self, tmp_path: Path) -> None:
        """load_bug_inputs should degrade to an empty trigger when no artifact exists."""
        proc = tmp_path / "proc"
        proc.mkdir()
        with patch("src.agent4sr.io.get_processed_dir", return_value=proc):
            inputs = load_bug_inputs("TestProj", "1")

        assert inputs.trigger_test == ""
        assert inputs.bug_report_title is None
        assert inputs.bug_report_description is None


# =====================================================================
# Tests for prompts.py
# =====================================================================


class TestPrompts:
    """Test prompt generation functions."""

    def test_system_prompt_contains_functions(self) -> None:
        prompt = sr_system_prompt(max_tool_calls=10)
        assert "get_paths" in prompt
        assert "find_method" in prompt
        assert "10 chances" in prompt

    def test_initial_user_prompt_with_bug_report(self) -> None:
        inputs = BugInputs(
            project="Chart",
            bug_id="1",
            trigger_test="test code here",
            bug_report_title="Title",
            bug_report_description="Desc",
        )
        prompt = sr_initial_user_prompt(inputs)
        assert "Title" in prompt
        assert "test code here" in prompt
        assert "locate the faulty method" in prompt

    def test_initial_user_prompt_without_bug_report(self) -> None:
        inputs = BugInputs(
            project="Chart",
            bug_id="1",
            trigger_test="test code",
            bug_report_title=None,
            bug_report_description=None,
        )
        prompt = sr_initial_user_prompt(inputs)
        assert "test code" in prompt
        assert "Title" not in prompt

    def test_tool_call_prompt(self) -> None:
        prompt = sr_tool_call_user_prompt()
        assert "call a function" in prompt.lower()

    def test_retry_prompt(self) -> None:
        prompt = sr_retry_user_prompt()
        assert "right format" in prompt

    def test_finisher_prompt(self) -> None:
        prompt = sr_finisher_user_prompt()
        assert "Top_1" in prompt
        assert "Top_5" in prompt

    # --- D4J byte-identity snapshots (regression guard: the D4J results were
    #     tuned on this exact wording — any drift must be deliberate) -----------

    def test_system_prompt_d4j_byte_identical(self) -> None:
        """The Defects4J system prompt must stay byte-for-byte stable."""
        assert sr_system_prompt(max_tool_calls=10) == (
            "You are a debugging assistant of our Java software. You will be presented with a "
            "bug report, a trigger test and tools (functions) to access the source code of the "
            "system under test (SUT). Your task is to locate the top-5 most likely culprit "
            "methods based on the bug report, the trigger test and the information you retrieve "
            "using given functions. "
            "\nFunction calls you can use are as follows.\n"
            "* get_paths() -> Get all path names of Java source files in the repository. *\n"
            "* get_classes_of_path(path_name) -> Get all classes under a given path. *\n"
            "* get_methods_of_class(class_name) -> Get all methods of a given class. *\n"
            "* get_code_snippet_of_method(method_name) -> Get the code snippet of a Java "
            "method. *\n"
            "* find_class(class_name) -> Fuzzy search classes in the repository. *\n"
            "* find_method(method_name) -> Fuzzy search methods in the repository. *\n"
            "* exit() -> Exit function calling to give your final answer when you are "
            "confident of the answer. *\n"
            "You have 10 chances to call function."
        )
        # The default (no dataset) must equal the explicit defects4j form.
        assert sr_system_prompt(max_tool_calls=10) == sr_system_prompt(
            max_tool_calls=10, dataset="defects4j"
        )

    def test_finisher_prompt_d4j_byte_identical(self) -> None:
        """The Defects4J finisher prompt must stay byte-for-byte stable."""
        assert sr_finisher_user_prompt() == (
            "Based on the available information, provide complete name of the "
            "top-5 most likely culprit methods for the bug please. "
            "Since your answer will be processed automatically, please give your answer "
            "in the format as follows.\n"
            "Top_1 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
            "Top_2 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
            "Top_3 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
            "Top_4 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
            "Top_5 : PathName.ClassName.MethodName(ArgType1, ArgType2)\n"
        )
        assert sr_finisher_user_prompt() == sr_finisher_user_prompt(dataset="defects4j")

    # --- BugsInPy (Python) variants -----------------------------------------

    def test_system_prompt_bugsinpy_says_python(self) -> None:
        out = sr_system_prompt(max_tool_calls=10, dataset="bugsinpy")
        assert "Python software" in out
        assert "Python source files" in out
        assert "Java" not in out

    def test_finisher_prompt_bugsinpy_python_template_and_module_level(self) -> None:
        out = sr_finisher_user_prompt(dataset="bugsinpy")
        assert "Top_1" in out and "Top_5" in out
        # Python answer template (param names, dotted module path)
        assert "ModuleName.ClassName.method_name(arg1, arg2)" in out
        # The module-level-function (no class) case is spelled out
        assert "ModuleName.function_name(arg1, arg2)" in out
        # No Java type-style template leaks in
        assert "ArgType1" not in out


# =====================================================================
# Tests for function_call.py (fuzzy search + tool-backing functions)
# =====================================================================


class TestSplit4Search:
    """Test the split4search tokenizer."""

    def test_simple_dotted(self) -> None:
        assert split4search("StringUtils.isEmpty") == ["StringUtils", "isEmpty"]

    def test_with_params(self) -> None:
        result = split4search("StringUtils.isEmpty(CharSequence)")
        assert result == ["StringUtils", "isEmpty", "CharSequence"]

    def test_with_fq_params(self) -> None:
        result = split4search("Foo.bar(java.lang.String)")
        assert result == ["Foo", "bar", "String"]

    def test_multiple_params(self) -> None:
        result = split4search("Foo.bar(int,String)")
        assert result == ["Foo", "bar", "int", "String"]


class TestFuzzySearch:
    """Test fuzzy_search function."""

    def test_exact_token_match(self) -> None:
        choices = ["org.example.Foo.bar(int)", "org.example.Baz.qux()"]
        result = fuzzy_search("Foo.bar(int)", choices)
        assert "org.example.Foo.bar(int)" in result

    def test_levenshtein_fallback(self) -> None:
        choices = ["org.example.Foo.bar(int)", "org.example.Baz.qux()"]
        result = fuzzy_search("Foo.baz(int)", choices)
        assert len(result) >= 1

    def test_returns_close_matches(self) -> None:
        choices = ["abc", "abd", "xyz"]
        result = fuzzy_search("abc", choices)
        assert "abc" in result


class TestCorpusLoading:
    """Test corpus loading and tool-backing functions with temp corpus files."""

    def test_load_corpus_methods(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            methods = load_corpus_methods("TestProj", "1")
        assert len(methods) == 4
        assert methods[0] == "org.example$SampleClass.add(int,int)"

    def test_load_corpus_codes(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            codes = load_corpus_codes("TestProj", "1")
        assert len(codes) == 4
        assert "return a + b" in codes[0]

    def test_get_paths(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_paths("TestProj", "1")
        paths = result.strip().splitlines()
        assert "org.example" in paths
        assert "org.example.sub" in paths

    def test_get_classes(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_classes("TestProj", "1", "org.example")
        assert "SampleClass" in result

    def test_get_classes_wrong_path(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_classes("TestProj", "1", "com.wrong")
        assert "wrong path name" in result.lower() or "Do you mean" in result

    def test_get_methods(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_methods("TestProj", "1", "org.example.SampleClass")
        assert "add(int,int)" in result
        assert "format(String,Object)" in result

    def test_get_methods_wrong_class(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_methods("TestProj", "1", "org.example.WrongClass")
        assert "wrong class name" in result.lower() or "Do you mean" in result

    def test_get_code_snippet_exact(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_code_snippet("TestProj", "1", "org.example.SampleClass.add(int,int)")
        assert "return a + b" in result

    def test_get_code_snippet_fuzzy(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = get_code_snippet("TestProj", "1", "org.example.SampleClass.ad(int,int)")
        # Should suggest the correct method
        assert "Do you mean" in result or "return a + b" in result

    def test_find_class(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = find_class("TestProj", "1", "SampleClass")
        assert "SampleClass" in result

    def test_find_method(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = find_method("TestProj", "1", "add")
        assert "add" in result

    def test_corpus_methods_path(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            p = corpus_methods_path("TestProj", "1")
        assert p.name == "corpus_methods.txt"

    def test_corpus_codes_path(self, corpus_dir: Path) -> None:
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            p = corpus_codes_path("TestProj", "1")
        assert p.name == "corpus_codes.txt"


# =====================================================================
# Tests for tools.py
# =====================================================================


class TestToolSchemas:
    """Test tool_schemas and execute_tool."""

    def test_tool_schemas_count(self) -> None:
        schemas = tool_schemas()
        assert len(schemas) == 7

    def test_tool_schemas_names(self) -> None:
        schemas = tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "get_paths" in names
        assert "get_classes_of_path" in names
        assert "get_methods_of_class" in names
        assert "get_code_snippet_of_method" in names
        assert "find_class" in names
        assert "find_method" in names
        assert "exit" in names

    def test_tool_schemas_d4j_descriptions_byte_identical(self) -> None:
        """D4J tool descriptions (fed to the LLM) must stay stable; default == defects4j."""
        d = {s["function"]["name"]: s["function"]["description"] for s in tool_schemas()}
        assert d["get_paths"] == "Get all path names of Java source files in the repository."
        assert d["get_code_snippet_of_method"] == "Get the code snippet of a Java method."
        assert tool_schemas() == tool_schemas("defects4j")

    def test_tool_schemas_bugsinpy_says_python(self) -> None:
        d = {s["function"]["name"]: s["function"]["description"] for s in tool_schemas("bugsinpy")}
        assert d["get_paths"] == "Get all path names of Python source files in the repository."
        assert d["get_code_snippet_of_method"] == "Get the code snippet of a Python method."

    def test_execute_tool_exit(self) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1")
        result = execute_tool(ctx=ctx, name="exit", args={})
        assert result == "exit"

    def test_execute_tool_unknown(self) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1")
        result = execute_tool(ctx=ctx, name="unknown_tool", args={})
        assert "Unknown tool" in result

    def test_execute_tool_get_paths(self, corpus_dir: Path) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = execute_tool(ctx=ctx, name="get_paths", args={})
        assert "org.example" in result

    def test_execute_tool_bad_args(self, corpus_dir: Path) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1")
        result = execute_tool(ctx=ctx, name="get_classes_of_path", args={})
        assert "error" in result.lower()

    def test_normalize_method_name_exact(self, corpus_dir: Path) -> None:
        """Dotted LLM input resolves to the canonical $-bearing corpus ID."""
        ctx = ToolContext(project="TestProj", bug_id="1")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = normalize_method_name(
                ctx=ctx,
                method_name="org.example.SampleClass.add(int,int)",
            )
        assert result == "org.example$SampleClass.add(int,int)"

    def test_normalize_method_name_already_canonical(self, corpus_dir: Path) -> None:
        """A $-bearing input is returned unchanged."""
        ctx = ToolContext(project="TestProj", bug_id="1")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            result = normalize_method_name(
                ctx=ctx,
                method_name="org.example$SampleClass.add(int,int)",
            )
        assert result == "org.example$SampleClass.add(int,int)"


class TestBugsInPyToolRouting:
    """ToolContext(dataset="bugsinpy") must route the tools to the BIP corpus
    and JSON-decode corpus_codes.txt (Python source is multi-line)."""

    @pytest.fixture
    def bip_corpus_dir(self, tmp_path: Path) -> Path:
        """A BIP SR corpus dir: Python ids + json.dumps-encoded multi-line codes."""
        d = tmp_path / "data" / "BIP" / "processed" / "TestProj" / "1" / "FlexFL" / "SR"
        d.mkdir(parents=True)
        methods = [
            "pkg.mod$Cls.method(self,x)",
            "pkg.mod$module_level_fn(a,b)",
        ]
        # multi-line Python source, json-encoded one per physical line
        codes = [
            "def method(self, x):\n    return x + 1",
            "def module_level_fn(a, b):\n    return a + b",
        ]
        (d / "corpus_methods.txt").write_text("\n".join(methods) + "\n", encoding="utf-8")
        (d / "corpus_codes.txt").write_text(
            "\n".join(json.dumps(c) for c in codes) + "\n", encoding="utf-8"
        )
        return d

    def test_get_paths_routes_to_bip_corpus(self, bip_corpus_dir: Path) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1", dataset="bugsinpy")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=bip_corpus_dir):
            out = execute_tool(ctx=ctx, name="get_paths", args={})
        assert out.strip() == "pkg.mod"
        assert "$" not in out

    def test_get_classes_filters_module_level_empty_class(self, bip_corpus_dir: Path) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1", dataset="bugsinpy")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=bip_corpus_dir):
            out = execute_tool(ctx=ctx, name="get_classes_of_path", args={"path_name": "pkg.mod"})
        # The module-level function contributes no blank class.
        assert "" not in out.splitlines()
        assert "Cls" in out

    def test_get_code_snippet_decodes_multiline_python(self, bip_corpus_dir: Path) -> None:
        ctx = ToolContext(project="TestProj", bug_id="1", dataset="bugsinpy")
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=bip_corpus_dir):
            out = execute_tool(
                ctx=ctx,
                name="get_code_snippet_of_method",
                args={"method_name": "pkg.mod.Cls.method(self,x)"},
            )
        # Faithful multi-line Python (json-decoded), not a single escaped line.
        assert out == "def method(self, x):\n    return x + 1"

    def test_default_dataset_stays_d4j(self, corpus_dir: Path) -> None:
        """An unset dataset still reads the D4J corpus verbatim (single-line codes)."""
        ctx = ToolContext(project="TestProj", bug_id="1")
        assert ctx.dataset == "defects4j"
        with patch("src.agent4sr.function_call.get_sr_dir", return_value=corpus_dir):
            out = execute_tool(
                ctx=ctx,
                name="get_code_snippet_of_method",
                args={"method_name": "org.example.SampleClass.add(int,int)"},
            )
        assert out == "public int add(int a, int b) { return a + b; }"


# =====================================================================
# Tests for agent.py (config + parsing, no actual LLM calls)
# =====================================================================


class TestSRConfig:
    """Test SRConfig creation."""

    def test_from_cli(self) -> None:
        from src.agent4sr.agent import SRConfig

        cfg = SRConfig.from_cli(
            model="llama3.1:8b",
            iterations=10,
            temperature=0.0,
            base_url="http://localhost:11434",
            verify=True,
            bugs=["1", "2", "3"],
        )
        assert cfg.model == "llama3.1:8b"
        assert cfg.bugs == ("1", "2", "3")

    def test_from_cli_strips_whitespace(self) -> None:
        from src.agent4sr.agent import SRConfig

        cfg = SRConfig.from_cli(
            model="m",
            iterations=5,
            temperature=0.0,
            base_url="http://localhost:11434",
            verify=True,
            bugs=[" 1 ", "", " 2 "],
        )
        assert cfg.bugs == ("1", "2")


class TestParseTop5:
    """Test _parse_top5 extraction."""

    def test_parse_all_5(self) -> None:
        from src.agent4sr.agent import _parse_top5

        content = (
            "Top_1 : org.example.Foo.bar()\n"
            "Top_2 : org.example.Foo.baz(int)\n"
            "Top_3 : org.example.Qux.run()\n"
            "Top_4 : org.example.Qux.stop()\n"
            "Top_5 : org.example.Qux.start()\n"
        )
        result = _parse_top5(content)
        assert len(result) == 5
        assert result[0] == "org.example.Foo.bar()"
        assert result[4] == "org.example.Qux.start()"

    def test_parse_partial(self) -> None:
        from src.agent4sr.agent import _parse_top5

        content = "Top_1 : org.example.Foo.bar()\nTop_3 : org.example.Baz.qux()\n"
        result = _parse_top5(content)
        assert len(result) == 2

    def test_parse_empty(self) -> None:
        from src.agent4sr.agent import _parse_top5

        result = _parse_top5("No top methods here.")
        assert result == []


# =====================================================================
# Tests for combine.py
# =====================================================================


class TestCombine:
    """Test combine module functions."""

    def test_read_fl_csv(self, tmp_path: Path) -> None:
        from src.agent4sr.combine import _read_fl_csv

        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "Signature,Suspiciousness\norg.example.Foo.bar(),0.9\norg.example.Foo.baz(),0.8\n"
        )
        result = _read_fl_csv(csv_path, key_column="Signature")
        assert result == ["org.example.Foo.bar()", "org.example.Foo.baz()"]

    def test_read_fl_csv_missing_file(self, tmp_path: Path) -> None:
        from src.agent4sr.combine import _read_fl_csv

        with pytest.raises(FileNotFoundError):
            _read_fl_csv(tmp_path / "nonexistent.csv")

    def test_read_fl_csv_statement_column(self, tmp_path: Path) -> None:
        from src.agent4sr.combine import _read_fl_csv

        csv_path = tmp_path / "stmts.csv"
        csv_path.write_text(
            "Statement,Suspiciousness\norg.example.Foo#42,0.9\norg.example.Foo#43,0.8\n"
        )
        result = _read_fl_csv(csv_path, key_column="Statement")
        assert result == ["org.example.Foo#42", "org.example.Foo#43"]


# =====================================================================
# Tests for model_id plumbing
# =====================================================================


def test_write_candidates_uses_model_id_for_dir_name(tmp_path: Path) -> None:
    """``write_candidates`` should name the candidates dir after ``model_id``."""
    from src.agent4sr.combine import write_candidates

    model_dir = tmp_path / "Agent4SR" / "llama3_1_8b__1"
    model_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_combine(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = kwargs.get("model_id")
        return ["m1", "m2"]

    with (
        patch("src.agent4sr.combine.combine_candidates_for_bug", side_effect=fake_combine),
        patch("src.agent4sr.combine.get_sr_model_dir", return_value=model_dir),
    ):
        out = write_candidates("Lang", 1, "llama3.1:8b", model_id="llama3_1_8b__1")

    assert captured["model_id"] == "llama3_1_8b__1"
    assert out.parent.name == "llama3_1_8b__1_All"


def test_combine_candidates_passes_model_id_to_loader(tmp_path: Path) -> None:
    """``combine_candidates_for_bug`` must forward ``model_id`` when loading SR top5."""
    from src.agent4sr.combine import combine_candidates_for_bug

    captured: dict[str, object] = {}

    def fake_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["model_id"] = kwargs.get("model_id")
        return []

    # Empty dirs trigger the inner FileNotFoundError catch -> suspicious=[]
    # but still allow the SR loader (which we're testing) to be invoked.
    with (
        patch("src.agent4sr.combine.get_sbir_dir", return_value=tmp_path),
        patch("src.agent4sr.combine.get_ochiai_dir", return_value=tmp_path),
        patch("src.agent4sr.combine.get_boostn_dir", return_value=tmp_path),
        patch("src.agent4sr.combine._load_sr_top5", side_effect=fake_load),
    ):
        combine_candidates_for_bug("Lang", 1, "llama3.1:8b", model_id="alt_id")

    assert captured["model_id"] == "alt_id"


def test_generate_top20_uses_model_id_for_filename(tmp_path: Path) -> None:
    """``generate_top20`` should name top20 outputs after ``model_id``."""
    from src.common import rankings as rankings_mod

    rankings_dir = tmp_path / "rankings"
    model_dir = tmp_path / "modeldir"
    rankings_dir.mkdir()
    model_dir.mkdir()

    with (
        patch("src.common.rankings.get_rankings_dir", return_value=rankings_dir),
        patch("src.common.rankings.get_sr_model_dir", return_value=model_dir),
        patch("src.common.rankings.generate_all_rankings", return_value={}),
        patch("src.common.rankings.generate_top15"),
        patch("src.common.rankings._build_top15_rows", return_value=[]),
        patch("src.common.rankings.load_method_entities", side_effect=FileNotFoundError),
    ):
        out = rankings_mod.generate_top20("Lang", 1, "llama3.1:8b", model_id="custom__2")

    assert out == rankings_dir / "top20" / "custom__2.txt"
    assert (rankings_dir / "top20" / "custom__2.csv").exists()


def _top20_ent(cid: str):
    from src.common.method_entity import MethodEntity

    return MethodEntity(
        corpus_id=cid,
        class_fqn_dotted=cid.split("(")[0].replace("$", ".").rsplit(".", 1)[0],
        path="pkg/mod.py",
        start_line=1,
        end_line=2,
    )


def _run_top20(tmp_path: Path, rankings_by_method, agent_lines: list[str]) -> tuple[Path, Path]:
    """Drive ``generate_top20`` with fake per-method rankings and agent lines."""
    from src.common import rankings as rankings_mod

    rankings_dir = tmp_path / "rankings"
    model_dir = tmp_path / "modeldir"
    rankings_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    if agent_lines:
        (model_dir / "top5.txt").write_text("\n".join(agent_lines) + "\n", encoding="utf-8")

    all_entities = list(
        {e.corpus_id: e for ranked in rankings_by_method.values() for e, _, _ in ranked}.values()
    ) + [_top20_ent(line) for line in agent_lines]

    with (
        patch("src.common.rankings.get_rankings_dir", return_value=rankings_dir),
        patch("src.common.rankings.get_sr_model_dir", return_value=model_dir),
        patch("src.common.rankings.generate_all_rankings", return_value=rankings_by_method),
        patch("src.common.rankings.generate_top15"),
        patch("src.common.rankings.load_method_entities", return_value=all_entities),
    ):
        txt_out = rankings_mod.generate_top20("P", 1, "llama3.1:8b", model_id="m")
    return txt_out, rankings_dir / "top20" / "m.csv"


def test_generate_top20_flexfl_exactly_20_with_duplicates(tmp_path: Path) -> None:
    """FlexFL rule: SBIR[:5] + Ochiai[:5] + BoostN[:5] + Agent4SR[:5], no dedup.

    Sources sharing methods must still yield exactly 20 lines, duplicates kept,
    in SBIR -> Ochiai -> BoostN -> Agent4SR order."""
    shared = [_top20_ent(f"pkg.mod$Cls.m{i}(self)") for i in range(5)]
    ranked = [(e, 1.0 - 0.1 * i, i + 1) for i, e in enumerate(shared)]
    # All three FL methods rank the SAME five methods; the agent repeats one.
    agent_lines = [f"pkg.mod$Cls.m{i}(self)" for i in (0, 0, 1, 2, 3)]

    txt_out, csv_out = _run_top20(
        tmp_path,
        {"sbir": ranked, "ochiai": ranked, "boostn": ranked},
        agent_lines,
    )

    txt_lines = [ln for ln in txt_out.read_text().splitlines() if ln.strip()]
    assert len(txt_lines) == 20
    expected_block = [f"pkg.mod.Cls.m{i}(self)" for i in range(5)]
    assert txt_lines[0:5] == expected_block  # SBIR
    assert txt_lines[5:10] == expected_block  # Ochiai (duplicates of SBIR kept)
    assert txt_lines[10:15] == expected_block  # BoostN
    # Agent tail preserved verbatim, including its intra-list duplicate.
    assert txt_lines[15:20] == [f"pkg.mod.Cls.m{i}(self)" for i in (0, 0, 1, 2, 3)]

    csv_lines = csv_out.read_text().splitlines()
    assert len(csv_lines) - 1 == 20
    methods = [ln.split(",")[1] for ln in csv_lines[1:]]
    assert methods == ["SBIR"] * 5 + ["Ochiai"] * 5 + ["BoostN"] * 5 + ["Agent4SR"] * 5


def test_generate_top20_short_source_yields_fewer_than_20(tmp_path: Path) -> None:
    """A source with < 5 entries produces a < 20-line list (no padding/fallback)."""
    ents = [_top20_ent(f"pkg.mod$Cls.m{i}(self)") for i in range(6)]
    full = [(e, 1.0, i + 1) for i, e in enumerate(ents[:5])]
    short = [(ents[5], 1.0, 1)]  # BoostN has only one method

    txt_out, _ = _run_top20(
        tmp_path,
        {"sbir": full, "ochiai": full, "boostn": short},
        [f"pkg.mod$Cls.m{i}(self)" for i in range(5)],
    )

    txt_lines = [ln for ln in txt_out.read_text().splitlines() if ln.strip()]
    assert len(txt_lines) == 16  # 5 + 5 + 1 + 5


def test_generate_top20_missing_method_and_empty_agent(tmp_path: Path) -> None:
    """Missing FL methods and an absent top5.txt shrink the list accordingly."""
    ents = [_top20_ent(f"pkg.mod$Cls.m{i}(self)") for i in range(5)]
    full = [(e, 1.0, i + 1) for i, e in enumerate(ents)]

    txt_out, _ = _run_top20(tmp_path, {"ochiai": full}, [])

    txt_lines = [ln for ln in txt_out.read_text().splitlines() if ln.strip()]
    assert txt_lines == [f"pkg.mod.Cls.m{i}(self)" for i in range(5)]  # Ochiai only


def test_generate_top20_agent_list_capped_at_five(tmp_path: Path) -> None:
    """Agent lines beyond the first five are ignored (SR_top5[:5])."""
    txt_out, _ = _run_top20(
        tmp_path,
        {},
        [f"pkg.mod$Cls.m{i}(self)" for i in range(7)],
    )

    txt_lines = [ln for ln in txt_out.read_text().splitlines() if ln.strip()]
    assert txt_lines == [f"pkg.mod.Cls.m{i}(self)" for i in range(5)]


def test_run_agent4sr_records_base_url_and_input_in_result() -> None:
    """``SRRunResult`` defaults should populate base_url + input fields."""
    from src.agent4sr.agent import DEFAULT_SR_INPUT_KEYS, SRRunResult

    r = SRRunResult(
        project="P",
        bug_id="1",
        model="m",
        iterations=5,
        temperature=0.0,
        started_at=0.0,
        finished_at=1.0,
        final_content="",
        top5_raw=[],
        top5=[],
        transcript=[],
        response_dumps=[],
        base_url="http://x",
        input=list(DEFAULT_SR_INPUT_KEYS),
    )
    assert r.base_url == "http://x"
    assert r.input == ["bug_report", "ochiai", "boostn", "sbir"]


# =====================================================================
# Tests for run_agent4sr.py CLI
# =====================================================================


class TestCLI:
    """Test CLI argument parsing."""

    def test_build_parser_corpus(self) -> None:
        from run_agent4sr import build_parser

        parser = build_parser()
        args = parser.parse_args(["corpus", "-p", "Chart"])
        assert args.project == "Chart"
        assert args.versions is None
        assert args.cmd == "corpus"

    def test_build_parser_corpus_with_versions(self) -> None:
        from run_agent4sr import build_parser

        parser = build_parser()
        args = parser.parse_args(["corpus", "-p", "Chart", "-v", "1", "2"])
        assert args.project == "Chart"
        assert args.versions == [1, 2]
        assert args.cmd == "corpus"

    def test_build_parser_run(self) -> None:
        from run_agent4sr import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "-p",
                "Lang",
                "-v",
                "1",
                "--model",
                "llama3.1:70b",
                "--iterations",
                "15",
            ]
        )
        assert args.project == "Lang"
        assert args.versions == [1]
        assert args.model == "llama3.1:70b"
        assert args.iterations == 15

    def test_build_parser_smoke(self) -> None:
        from run_agent4sr import build_parser

        parser = build_parser()
        args = parser.parse_args(["smoke"])
        assert args.cmd == "smoke"
        assert args.model == "llama3.1:8b"

    def test_build_parser_versions_omitted(self) -> None:
        from run_agent4sr import build_parser

        parser = build_parser()
        args = parser.parse_args(["corpus", "-p", "Chart"])
        assert args.project == "Chart"
        assert args.versions is None
        assert args.cmd == "corpus"


class TestBugsInPyRunGate:
    """The BIP run path must gate on the valid-bug set and skip blank triggers
    without ever invoking the LLM."""

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            project="TestProj",
            versions=[1],
            dataset="bugsinpy",
            model="llama3.1:8b",
            iterations=10,
            temperature=0.0,
            base_url="http://localhost:11434",
            no_verify=False,
            force=False,
        )

    def test_blank_trigger_skips_without_calling_llm(self) -> None:
        import run_agent4sr as r

        blank = BugInputs(
            project="TestProj",
            bug_id="1",
            trigger_test="",
            bug_report_title=None,
            bug_report_description=None,
        )
        with (
            patch.object(r, "_bip_run_skip_reason", return_value=None),
            patch.object(r, "load_bug_inputs", return_value=blank),
            patch.object(r, "run_agent4sr_for_bug") as mock_run,
            patch.object(r, "load_tracker") as mock_load,
        ):
            rc = r._cmd_run_bugsinpy(self._args(), parser=None)  # type: ignore[arg-type]

        assert rc == 0
        mock_run.assert_not_called()
        mock_load.assert_not_called()  # gate skip happens before tracker work

    def test_not_corpus_ready_skips_without_calling_llm(self) -> None:
        import run_agent4sr as r

        with (
            patch.object(r, "_bip_run_skip_reason", return_value="SR corpus missing"),
            patch.object(r, "run_agent4sr_for_bug") as mock_run,
            patch.object(r, "load_bug_inputs") as mock_inputs,
        ):
            rc = r._cmd_run_bugsinpy(self._args(), parser=None)  # type: ignore[arg-type]

        assert rc == 0
        mock_run.assert_not_called()
        mock_inputs.assert_not_called()  # gate skip happens before input loading
