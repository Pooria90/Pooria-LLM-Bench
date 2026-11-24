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
