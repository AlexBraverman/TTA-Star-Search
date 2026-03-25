"""
Experiment: TTA* on 10 GSM8K problems (4 iterations each).

Usage:
    python experiments/run_gsm8k.py \
        --model Qwen/Qwen3-4B-Instruct \
        --num_problems 10 \
        --max_iter 4 \
        --output results/gsm8k_tta_run.json
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta import LLMWrapper, TTAStar, extract_gsm8k_answer, is_correct


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct", help="HuggingFace model ID")
    p.add_argument("--num_problems", type=int, default=10, help="Number of GSM8K problems to run")
    p.add_argument("--max_iter", type=int, default=4, help="TTA* max iterations")
    p.add_argument("--g_weight", type=float, default=1.0, help="A* g(n) weight")
    p.add_argument("--num_children", type=int, default=2, help="Children per expansion")
    p.add_argument("--num_reward_evals", type=int, default=3, help="Critique calls per node")
    p.add_argument("--split", default="test", help="Dataset split (test or train)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for problem selection")
    p.add_argument("--output", default="results/gsm8k_tta_run.json", help="Output JSON path")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    model = LLMWrapper(args.model)

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    # Deterministic subset
    import random
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), args.num_problems)
    problems = [dataset[i] for i in indices]

    # Setup TTA*
    searcher = TTAStar(
        model=model,
        max_iter=args.max_iter,
        g_weight=args.g_weight,
        num_children=args.num_children,
        num_reward_evals=args.num_reward_evals,
    )

    results = []
    num_correct = 0
    start = time.time()

    for idx, problem in enumerate(tqdm(problems, desc="TTA* on GSM8K")):
        question = problem["question"]
        ground_truth = problem["answer"]

        print(f"\n{'='*60}")
        print(f"Problem {idx+1}/{args.num_problems}")
        print(f"Q: {question[:120]}...")

        try:
            best_node, history = searcher.run(question)
            predicted = extract_gsm8k_answer(best_node.answer)
            correct = is_correct(predicted, ground_truth)
        except Exception as e:
            print(f"  [ERROR] {e}")
            best_node = None
            predicted = None
            correct = False
            history = []

        if correct:
            num_correct += 1

        print(f"  Predicted: {predicted}  |  Correct: {correct}  |  Reward: {best_node.reward:.1f if best_node else 'N/A'}")

        results.append({
            "idx": indices[idx],
            "question": question,
            "ground_truth": ground_truth,
            "best_answer": best_node.answer if best_node else None,
            "predicted": predicted,
            "correct": correct,
            "reward": best_node.reward if best_node else None,
            "depth_reached": best_node.depth if best_node else None,
            "nodes_expanded": len(history),
            "history": [
                {"depth": n.depth, "reward": n.reward, "f": n.f, "answer": n.answer}
                for n in history
            ],
        })

    elapsed = time.time() - start
    accuracy = num_correct / len(problems) * 100

    summary = {
        "config": vars(args),
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_problems": len(problems),
        "elapsed_seconds": elapsed,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Accuracy: {num_correct}/{len(problems)} ({accuracy:.1f}%)")
    print(f"Time: {elapsed:.1f}s")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
