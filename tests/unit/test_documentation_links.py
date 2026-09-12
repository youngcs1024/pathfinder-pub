from pathlib import Path

import pytest

from scripts.check_documentation_links import anchors, check_documentation, links, main


def _write(root: Path, name: str, content: str = "") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_local_links_cover_root_nested_docs_images_and_code_targets(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "[guide](docs/guide.md#安装)\n![logo](docs/logo.svg)\n")
    _write(tmp_path, "docs/guide.md", "# 安装\n[root](../README.md)\n[code](../src/app.py#L1)")
    _write(tmp_path, "docs/logo.svg", "<svg/>")
    _write(tmp_path, "src/app.py")
    _write(tmp_path, "src/prompt.md", "[outside documentation scope](missing.md)")
    assert check_documentation(tmp_path) == (2, 4, [])


@pytest.mark.parametrize(
    ("link", "reason"),
    [
        ("[missing](docs/missing.md)", "missing local target"),
        ("![missing](missing.png)", "missing local target"),
        ("[anchor](docs/guide.md#absent)", "missing Markdown anchor"),
        ("[escape](../outside.md)", "local target escapes repository"),
        ("[missing][id]\n\n[id]: missing.md", "missing local target"),
    ],
)
def test_missing_targets_and_anchors_have_source_line_and_destination(
    tmp_path: Path, link: str, reason: str
) -> None:
    _write(tmp_path, "README.md", "# Entry\n\n" + link)
    _write(tmp_path, "docs/guide.md", "# Present")
    _, _, problems = check_documentation(tmp_path)
    assert problems
    assert all(problem.source == Path("README.md") for problem in problems)
    assert all(problem.reason == reason for problem in problems)
    assert any(problem.line == 3 for problem in problems)
    assert all(problem.target in link for problem in problems)


def test_references_support_forward_full_collapsed_shortcut_and_images(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "README.md",
        "[full][GUIDE]\n[guide][]\n[guide]\n![image][guide]\n"
        '\n[guide]: <docs/a b.md#标题> "title"\n',
    )
    _write(tmp_path, "docs/a b.md", "# 标题")
    assert check_documentation(tmp_path) == (2, 5, [])


def test_inline_destinations_support_encoding_titles_and_parentheses(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "README.md",
        '[encoded](docs/a%20b.md#%E6%A0%87%E9%A2%98 "title")\n'
        "[angle](<docs/a b.md#标题>)\n[paren](docs/a(b).md)\n"
        "[escaped](docs/a\\(b\\).md)\n[root](/docs/a%20b.md#标题)\n",
    )
    _write(tmp_path, "docs/a b.md", "# 标题")
    _write(tmp_path, "docs/a(b).md")
    assert check_documentation(tmp_path) == (3, 5, [])


def test_headings_keep_chinese_inline_code_and_duplicate_suffixes() -> None:
    text = (
        "# 2. 能力主张—实现—证据—边界\n## `RunStatus` and **state**\n"
        "## Same\n## Same\n## Same-1\n## Same\n"
        'Setext heading\n===\n<a id="stable-id"></a>\n'
        "<a name='old-id'></a>\n# [Linked](README.md) heading\n"
    )
    assert anchors(text) == {
        "2-能力主张实现证据边界",
        "runstatus-and-state",
        "same",
        "same-1",
        "same-1-1",
        "same-2",
        "setext-heading",
        "stable-id",
        "old-id",
        "linked-heading",
    }


def test_fenced_indented_inline_code_and_comments_are_not_links_or_headings() -> None:
    text = (
        "```md\n[bad](missing.md)\n# Fake\n```\n"
        "~~~~md\n# Fake too\n[bad](missing.md)\n~~~\n[bad](missing.md)\n~~~~\n"
        "    [indented](missing.md)\n\t[tab](missing.md)\n"
        "`[bad](missing.md)` and ``[bad](missing.md) `code` ``\n"
        '<!-- [bad](missing.md)\n<a id="fake"></a> -->\n'
        "[real](README.md)\n"
    )
    assert links(text) == [(16, "README.md")]
    assert anchors(text) == set()
    assert anchors('`<a id="example"></a>`') == set()


def test_external_urls_and_undefined_reference_labels_are_not_local_files(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "README.md",
        "[web](https://example.invalid/doc#anchor)\n[mail](mailto:demo@example.invalid)\n"
        "[web](//example.invalid/a)\n[issue][unresolved]\n[J1][J2]\n"
        "<https://example.invalid/autolink>\n",
    )
    assert check_documentation(tmp_path) == (1, 0, [])


def test_same_document_explicit_anchor_and_empty_destination(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", '<a id="stable"></a>\n[here](#stable)\n[top]()')
    assert check_documentation(tmp_path) == (1, 2, [])


def test_cli_reports_failure_and_recovers_after_target_is_added(tmp_path: Path, capsys) -> None:
    source = _write(tmp_path, "README.md", "# Entry\n[bad](docs/guide.md#heading)\n")
    before = source.read_bytes()
    assert main(["--repository", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "README.md:2: missing local target: docs/guide.md#heading" in output.err
    assert "1 files, 1 local links, 1 errors" in output.out
    _write(tmp_path, "docs/guide.md", "# Heading")
    assert main(["--repository", str(tmp_path)]) == 0
    assert source.read_bytes() == before


def test_empty_documentation_scope_fails_closed(tmp_path: Path, capsys) -> None:
    assert main(["--repository", str(tmp_path)]) == 1
    assert "no documents found" in capsys.readouterr().err


def test_unreadable_markdown_fails_closed(tmp_path: Path, capsys) -> None:
    (tmp_path / "README.md").write_bytes(b"\xff")
    assert main(["--repository", str(tmp_path)]) == 1
    assert "cannot complete check" in capsys.readouterr().err
