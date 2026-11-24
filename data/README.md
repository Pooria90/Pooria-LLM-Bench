# Datasets

Both AIME files are built by `scripts/build_aime.py <year>`. Fields: `id` (`I-01`..`II-15`),
`section`, `number`, `problem` (LaTeX), `answer` (integer 0-999), `problem_type` (2025 only),
`figure` (`tikz` / `asy` / `latex-array` / `ascii-grid` / null), `source`, `aops_url`.

## aime2026.jsonl

All 30 problems of AIME 2026 (I and II).

- **Source:** [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026). Answers
  were cross-checked against `math-ai/aime26` and `zhenghaoxu/aime-2026`; all 30 agree.
- **Figures as text:** I-15 (an example partition drawn as an ASCII grid) and II-02 (Asymptote source).
- **Wording differs from the AoPS wiki on two problems.** II-01 here asks for the 10th terms (answer
  178; the wiki asks for 11th terms, answer 196). II-10 asks for the *sum* of all possible values
  (850; the wiki asks for the greatest value, 340). The wiki notes a separate international version.
  The answer key matches the wording we send.

## aime2025.jsonl

All 30 problems of AIME 2025 (I and II).

- **Source:** [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025). Answers
  were cross-checked against `yentinglin/aime_2025` and `math-ai/aime25`; all 30 agree.
- **Figures as text:** six problems include a figure, which MathArena already encodes as text,
  and that text is passed to the model unchanged:
  - TikZ: I-02, II-03, II-05, II-06
  - Asymptote: I-11
  - LaTeX array: I-10

  Two of them (II-05, II-06) say "as shown" and rely on the figure for the order or layout of
  points. The other four can be solved from the words alone.

AIME problems are © Mathematical Association of America; included here for research evaluation.
