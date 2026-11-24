import pytest

from pooria_bench import scorers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Some reasoning\nAnswer: 391", "391"),
        ("answer is **Carol**.", "Carol**."),
        ("first line\nlast line", "last line"),
        ("Answer: 1\nwait, recheck\nFinal answer: 2", "2"),
        ("", ""),
    ],
)
def test_extract_answer(text, expected):
    assert scorers.extract_answer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Answer: 1,048,575", 1048575.0),
        ("The price is $36.72\nAnswer: 36.72", 36.72),
        (" 391", 391.0),
        ("Answer: -3.5e2", -350.0),
        ("Answer: none", None),
    ],
)
def test_extract_number(text, expected):
    assert scorers.extract_number(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Answer: (B)", "B"),
        ("The answer is c.", "C"),
        ("I think it's D", "D"),
        ("Answer: E", None),
    ],
)
def test_extract_choice(text, expected):
    assert scorers.extract_choice(text, "ABCD") == expected


def test_exact_match_normalizes_case_and_markup():
    assert scorers.exact_match("Reasoning...\nAnswer: **canberra**.", "Canberra").correct
    assert not scorers.exact_match("Answer: Sydney", "Canberra").correct


def test_numeric_match_tolerance():
    assert scorers.numeric_match("Answer: 1353.13", 1353.125, rel_tol=1e-3).correct
    assert not scorers.numeric_match("Answer: 1300", 1353.125, rel_tol=1e-3).correct
    assert not scorers.numeric_match("no number here", 5).correct


def test_invisible_characters_do_not_break_scoring():
    # Seen from Claude Opus 5.5: a U+E007F (CANCEL TAG) prefix on an otherwise normal reply.
    assert scorers.numeric_match("\U000e007f391", 391).correct
    assert scorers.exact_match("Answer: ​Canberra‍", "Canberra").correct
    assert scorers.strip_invisible("⁠​‌‍") == ""


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"so the answer is \boxed{70}.", "70"),
        (r"\boxed{1} then later \boxed{259}", "259"),
        (r"\boxed{\frac{288}{5}}", r"\frac{288}{5}"),
        (r"\fbox{ 81 }", "81"),
        ("no box here", None),
        (r"\boxed{unclosed", None),
    ],
)
def test_extract_boxed(text, expected):
    assert scorers.extract_boxed(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("070", 70), (r"\text{259}", 259), (r"336^\circ", 336), (r"336^{\circ}", 336),
        ("m+n=293", 293), ("1{,}024", 1024), ("259.", 259), (r"\$ 504", 504),
        (r"\frac{1}{2}", None), ("12.5", None), ("", None), (None, None),
    ],
)
def test_parse_integer(text, expected):
    assert scorers.parse_integer(text) == expected


def test_boxed_integer_match():
    ok = scorers.boxed_integer_match(r"...thus \boxed{070}", 70)
    assert ok.correct and ok.details["format_ok"] and ok.extracted == "70"
    wrong = scorers.boxed_integer_match(r"\boxed{71}", 70)
    assert not wrong.correct and wrong.details["format_ok"]
    fallback = scorers.boxed_integer_match("Answer: 70", 70)
    assert fallback.correct and fallback.details["format_ok"] is False
    bare = scorers.boxed_integer_match("\U000e007f**70**", 70)
    assert bare.correct and bare.details["format_ok"] is False
    missing = scorers.boxed_integer_match("I think it is 70", 70)
    assert not missing.correct and missing.details["format_ok"] is False
    assert not scorers.boxed_integer_match("", 70).correct
