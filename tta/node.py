import re
from .model import LLMWrapper


CRITIQUE_PROMPT = (
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Carefully evaluate the answer above. Point out any errors in reasoning or "
    "arithmetic, then assign a grade out of 100 using the format 'Grade: XX'."
)

REFINE_PROMPT = (
    "Question:\n{question}\n\n"
    "Previous Answer:\n{answer}\n\n"
    "Critique:\n{critique}\n\n"
    "Using the feedback above, solve the problem again step by step. "
    "End your answer with '#### <number>' where <number> is your final numerical answer."
)

SOLVE_PROMPT = (
    "Solve the following math problem step by step. "
    "End your answer with '#### <number>' where <number> is your final numerical answer.\n\n"
    "Problem: {question}"
)


class Node:
    def __init__(
        self,
        question: str,
        answer: str,
        model: LLMWrapper,
        depth: int = 0,
        num_reward_evals: int = 3,
    ):
        self.question = question
        self.answer = answer
        self.model = model
        self.depth = depth
        self.f: float | None = None

        # One critique text (used for child prompting) + averaged reward for stability
        self.critique: str = self._generate_critique()
        self.reward: float = self._compute_reward(num_reward_evals)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_critique(self) -> str:
        prompt = CRITIQUE_PROMPT.format(question=self.question, answer=self.answer)
        return self.model.chat(
            [{"role": "user", "content": prompt}], temperature=0.3
        )

    def _parse_score(self, text: str) -> float:
        match = re.search(r"Grade:\s*(\d{1,3})", text)
        return max(0.0, min(float(match.group(1)), 100.0)) if match else 50.0

    def _compute_reward(self, num_evals: int) -> float:
        """Median grade over `num_evals` independent critique calls.

        Median (not mean) is used per the paper to stabilize noisy SLM evaluations.
        """
        import statistics
        scores = [self._parse_score(self._generate_critique()) for _ in range(num_evals)]
        return statistics.median(scores)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_f(self, g_weight: float = 1.0) -> float:
        """f(n) = g(n) + h(n)  where g = g_weight * depth, h = 100 - reward."""
        g = g_weight * self.depth
        h = 100.0 - self.reward
        self.f = g + h
        return self.f

    def refine_prompt(self) -> str:
        return REFINE_PROMPT.format(
            question=self.question,
            answer=self.answer,
            critique=self.critique,
        )
