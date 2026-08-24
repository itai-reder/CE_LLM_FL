"""Tests for src.common.java_parser module."""

from __future__ import annotations

from pathlib import Path

from src.common.java_parser import (
    _derive_class_fqn,
    _expand_method_start_line,
    _find_method_end,
    extract_methods_from_java,
    extract_statements_from_java,
    find_java_files,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JAVA = FIXTURES_DIR / "SampleClass.java"
SAMPLE_WITH_CTOR_JAVA = FIXTURES_DIR / "SampleWithConstructor.java"
SAMPLE_GENERIC_JAVA = FIXTURES_DIR / "SampleGenericParams.java"


# ---------------------------------------------------------------------------
# MethodInfo extraction
# ---------------------------------------------------------------------------


class TestExtractMethods:
    """Tests for extract_methods_from_java."""

    def test_extracts_two_methods(self) -> None:
        methods = extract_methods_from_java(SAMPLE_JAVA)
        assert len(methods) == 2

    def test_method_ids_format(self) -> None:
        methods = extract_methods_from_java(SAMPLE_JAVA)
        # Method IDs should follow: pkg.Class.method(Params).startLine.endLine
        for m in methods:
            parts = m.method_id.split(".")
            # At minimum: org.example.SampleClass.add(int,int).9.11
            assert len(parts) >= 5, f"Unexpected method_id format: {m.method_id}"

    def test_add_method_details(self) -> None:
        methods = extract_methods_from_java(SAMPLE_JAVA)
        add_method = [m for m in methods if m.method_name == "add"]
        assert len(add_method) == 1
        m = add_method[0]
        assert m.class_fqn == "org.example.SampleClass"
        assert m.param_types == ["int", "int"]
        assert m.start_line == 9
        assert m.end_line == 11
        assert "return a + b" in m.content

    def test_format_method_details(self) -> None:
        methods = extract_methods_from_java(SAMPLE_JAVA)
        fmt = [m for m in methods if m.method_name == "format"]
        assert len(fmt) == 1
        m = fmt[0]
        assert m.class_fqn == "org.example.SampleClass"
        assert m.param_types == ["String", "Object"]
        assert m.start_line == 13
        assert m.end_line == 18

    def test_file_path_set(self) -> None:
        methods = extract_methods_from_java(SAMPLE_JAVA)
        for m in methods:
            assert m.file_path == str(SAMPLE_JAVA)

    def test_nonexistent_file_returns_empty(self) -> None:
        methods = extract_methods_from_java("/tmp/does_not_exist_xyz.java")
        assert methods == []

    def test_malformed_java_returns_empty(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "Bad.java"
        bad_file.write_text("this is not valid java {{{{")
        methods = extract_methods_from_java(bad_file)
        assert methods == []

    def test_extracts_constructor_as_method_entry(self) -> None:
        methods = extract_methods_from_java(SAMPLE_WITH_CTOR_JAVA)
        ctor = [m for m in methods if m.method_name == "SampleWithConstructor"]
        assert len(ctor) == 1
        m = ctor[0]
        assert m.class_fqn == "org.example.SampleWithConstructor"
        assert m.param_types == ["int"]
        assert m.start_line == 6
        assert m.end_line == 8

    def test_extracts_constructor_and_method(self) -> None:
        methods = extract_methods_from_java(SAMPLE_WITH_CTOR_JAVA)
        names = sorted(m.method_name for m in methods)
        assert names == ["SampleWithConstructor", "getX"]

    def test_preserves_generic_parameter_shapes(self) -> None:
        methods = extract_methods_from_java(SAMPLE_GENERIC_JAVA)
        generic = [m for m in methods if m.method_name == "genericMethod"]
        assert len(generic) == 1
        assert generic[0].param_types == ["Class<?>", "Map<TypeVariable<?>,Type>"]


# ---------------------------------------------------------------------------
# Statement extraction
# ---------------------------------------------------------------------------


class TestExtractStatements:
    """Tests for extract_statements_from_java."""

    def test_extracts_statements(self) -> None:
        stmts = extract_statements_from_java(SAMPLE_JAVA)
        assert len(stmts) > 0

    def test_stmt_id_format(self) -> None:
        stmts = extract_statements_from_java(SAMPLE_JAVA)
        for s in stmts:
            assert "#" in s.stmt_id, f"stmt_id missing '#': {s.stmt_id}"
            _fqn, line_str = s.stmt_id.rsplit("#", 1)
            assert line_str.isdigit()

    def test_skips_comments_and_blanks(self) -> None:
        stmts = extract_statements_from_java(SAMPLE_JAVA)
        contents = [s.content for s in stmts]
        # No blank lines, comments, or annotations should appear
        for c in contents:
            assert c.strip() != ""
            assert not c.startswith("//")
            assert not c.startswith("/*")
            assert not c.startswith("*")
            assert not c.startswith("@")

    def test_skips_package_and_import(self) -> None:
        stmts = extract_statements_from_java(SAMPLE_JAVA)
        contents = [s.content for s in stmts]
        for c in contents:
            assert not c.startswith("package ")
            assert not c.startswith("import ")

    def test_class_fqn(self) -> None:
        stmts = extract_statements_from_java(SAMPLE_JAVA)
        for s in stmts:
            assert s.class_fqn == "org.example.SampleClass"

    def test_nonexistent_file_returns_empty(self) -> None:
        stmts = extract_statements_from_java("/tmp/does_not_exist_xyz.java")
        assert stmts == []

    def test_with_source_root(self, tmp_path: Path) -> None:
        """When source_root is given, FQN is derived from relative path."""
        src_root = tmp_path / "src"
        pkg_dir = src_root / "com" / "foo"
        pkg_dir.mkdir(parents=True)
        java_file = pkg_dir / "Bar.java"
        java_file.write_text("public class Bar {\n    int x = 1;\n}\n")
        stmts = extract_statements_from_java(java_file, source_root=src_root)
        assert len(stmts) > 0
        assert stmts[0].class_fqn == "com.foo.Bar"


# ---------------------------------------------------------------------------
# _find_method_end
# ---------------------------------------------------------------------------


class TestFindMethodEnd:
    """Tests for the brace-matching helper."""

    def test_simple_method(self) -> None:
        lines = [
            "public void foo() {",
            "    int x = 1;",
            "}",
        ]
        assert _find_method_end(lines, 0) == 3  # 1-based

    def test_nested_braces(self) -> None:
        lines = [
            "public void foo() {",
            "    if (true) {",
            "        bar();",
            "    }",
            "}",
        ]
        assert _find_method_end(lines, 0) == 5

    def test_no_closing_brace_returns_file_length(self) -> None:
        lines = [
            "public void foo() {",
            "    int x = 1;",
        ]
        assert _find_method_end(lines, 0) == 2


class TestExpandMethodStartLine:
    """Tests for method-start expansion helper."""

    def test_keeps_declaration_when_only_line_comment_above(self) -> None:
        lines = [
            "// helper",
            "public void foo() {",
            "}",
        ]
        assert _expand_method_start_line(lines, 2) == 2

    def test_includes_contiguous_javadoc_block(self) -> None:
        lines = [
            "/**",
            " * docs",
            " */",
            "public void foo() {",
            "}",
        ]
        assert _expand_method_start_line(lines, 4) == 1

    def test_includes_annotation_and_javadoc(self) -> None:
        lines = [
            "/**",
            " * docs",
            " */",
            "@Override",
            "public String toString() {",
            '    return "x";',
            "}",
        ]
        assert _expand_method_start_line(lines, 5) == 1


# ---------------------------------------------------------------------------
# _derive_class_fqn
# ---------------------------------------------------------------------------


class TestDeriveClassFqn:
    """Tests for FQN derivation."""

    def test_from_package_declaration(self) -> None:
        lines = ["package org.example;", "", "public class Foo {"]
        fqn = _derive_class_fqn(Path("Foo.java"), lines, None)
        assert fqn == "org.example.Foo"

    def test_no_package(self) -> None:
        lines = ["public class Foo {"]
        fqn = _derive_class_fqn(Path("Foo.java"), lines, None)
        assert fqn == "Foo"

    def test_from_source_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        pkg = src / "com" / "bar"
        pkg.mkdir(parents=True)
        f = pkg / "Baz.java"
        f.write_text("")
        fqn = _derive_class_fqn(f, [], str(src))
        assert fqn == "com.bar.Baz"


# ---------------------------------------------------------------------------
# find_java_files
# ---------------------------------------------------------------------------


class TestFindJavaFiles:
    """Tests for recursive Java file discovery."""

    def test_finds_java_files(self, tmp_path: Path) -> None:
        (tmp_path / "A.java").write_text("class A {}")
        (tmp_path / "B.java").write_text("class B {}")
        (tmp_path / "readme.txt").write_text("hello")
        files = find_java_files(tmp_path, exclude_tests=False)
        assert len(files) == 2
        assert all(f.suffix == ".java" for f in files)

    def test_excludes_test_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "Main.java").write_text("class Main {}")
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "MainTest.java").write_text("class MainTest {}")
        files = find_java_files(tmp_path, exclude_tests=True)
        assert len(files) == 1
        assert files[0].name == "Main.java"

    def test_empty_dir(self, tmp_path: Path) -> None:
        files = find_java_files(tmp_path)
        assert files == []
