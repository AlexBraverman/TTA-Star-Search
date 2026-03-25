# TTA*: Test-Time A* Search

Clean re-implementation of **TTA\*** from the paper
*"Test-Time A* Search for Multistep Reasoning in Small Language Models"*

TTA* improves LLM reasoning at inference time by framing multi-step problem solving as a goal-directed tree search — no fine-tuning, no external reward models.

---

## How it works

Each node in the search tree represents a candidate solution. The search expands nodes according to an A\*-style cost function:

```
f(n) = g(n) + h(n)
     = w · depth(n) + (100 − Reward(n))
```

- **g(n)**: path cost (penalises deep, long-winded reasoning chains)
- **h(n)**: heuristic (estimated remaining "distance" to a correct solution)
- **Reward(n)**: median of multiple self-evaluations — averaging reduces noise from unreliable SLM critiques

At each step the node with the **lowest f** is expanded: the model critiques its own answer and generates `num_children` refined candidates. This balances exploration (low h, promising answers) with exploitation (low g, shallow paths).

---

## Quick start

```bash
pip install -r requirements.txt

python experiments/run_gsm8k.py \
    --model Qwen/Qwen3-4B-Instruct \
    --num_problems 10 \
    --max_iter 4 \
    --output results/gsm8k_tta_run.json
```

Results are saved as JSON with per-problem answers, rewards, node histories, and overall accuracy.

---

## Key arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B-Instruct` | HuggingFace model ID |
| `--num_problems` | 10 | Problems to evaluate |
| `--max_iter` | 4 | TTA* expansion steps |
| `--g_weight` | 1.0 | Weight `w` on path cost |
| `--num_children` | 2 | Candidates generated per expansion |
| `--num_reward_evals` | 3 | Self-critiques averaged per node |
| `--split` | test | Dataset split |

---

## Project structure

```
tta/
  model.py      # LLMWrapper — model loading & generation
  node.py       # Node — critique, reward, f-score
  search.py     # TTAStar — the A* search loop
  evaluate.py   # Answer extraction & accuracy for GSM8K
experiments/
  run_gsm8k.py  # Main experiment script
results/        # Output JSONs (gitignored)
```

---

## Datasets

- **GSM8K**: `gsm8k` on HuggingFace Datasets
- **MATH500 / MATH401 / AIME**: coming soon
