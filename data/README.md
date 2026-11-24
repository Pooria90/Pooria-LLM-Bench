# Datasets

## aime2025.jsonl

All 30 problems of AIME 2025 (I and II), built by `scripts/build_aime2025.py`.

- **Source:** [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025). Answers
  were cross-checked against `yentinglin/aime_2025` and `math-ai/aime25`; all 30 agree.
- **Figures as text:** six problems include a figure, which MathArena already encodes as text,
  and that text is passed to the model unchanged:
  - TikZ: I-02, II-03, II-05, II-06
  - Asymptote: I-11
  - LaTeX array: I-10

  Two of them (II-05, II-06) say "as shown" and rely on the figure for the order or layout of
  points. The other four can be solved from the words alone.
- **Fields:** `id` (`I-01`..`II-15`), `section`, `number`, `problem` (LaTeX), `answer`
  (an integer from 0 to 999), `problem_type`, `figure` (`tikz` / `asy` / `latex-array` / null),
  `source`, `aops_url`.

AIME problems are © Mathematical Association of America; included here for research evaluation.
