from pooria_bench.tasks import get_task
from pooria_bench.tasks.aime import INSTRUCTIONS


def test_aime2025_dataset_shape():
    samples = get_task("aime2025").load()
    assert len(samples) == 30
    assert samples[0].id == "I-01" and samples[-1].id == "II-15"
    assert all(isinstance(s.target, int) and 0 <= s.target <= 999 for s in samples)
    figures = {s.id: s.metadata["figure"] for s in samples if s.metadata["category"] == "figure"}
    assert figures == {"I-02": "tikz", "I-10": "latex-array", "I-11": "asy",
                       "II-03": "tikz", "II-05": "tikz", "II-06": "tikz"}
    # spot-check answers against the official key
    by_id = {s.id: s.target for s in samples}
    assert (by_id["I-01"], by_id["I-15"], by_id["II-01"]) == (70, 735, 468)


def test_aime2025_prompt_and_scoring():
    task = get_task("aime2025")
    s = task.load(1)[0]
    (prompt,), _ = task.make_input(s)
    assert prompt.startswith(s.input) and prompt.endswith(INSTRUCTIONS)
    assert task.score(s, r"... \boxed{70}").correct
    assert not task.score(s, None).correct


def test_aime2026_dataset_shape():
    samples = get_task("aime2026").load()
    assert len(samples) == 30
    assert samples[0].id == "I-01" and samples[-1].id == "II-15"
    assert all(isinstance(s.target, int) and 0 <= s.target <= 999 for s in samples)
    figures = {s.id: s.metadata["figure"] for s in samples if s.metadata["category"] == "figure"}
    assert figures == {"I-15": "ascii-grid", "II-02": "asy"}
    by_id = {s.id: s.target for s in samples}
    # II-01 and II-10 follow MathArena's wording (the AoPS wiki has another version: 196 / 340)
    assert (by_id["I-01"], by_id["II-01"], by_id["II-10"]) == (277, 178, 850)
    assert not any("aime" in s.input.lower() for s in samples)
