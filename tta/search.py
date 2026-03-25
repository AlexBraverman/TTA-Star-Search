from .model import LLMWrapper
from .node import Node, SOLVE_PROMPT


class TTAStar:
    """
    TTA* search loop.

    Parameters
    ----------
    model        : LLMWrapper
    max_iter     : maximum number of A* expansion steps
    g_weight     : weight on path cost g(n) = g_weight * depth
    num_children : candidate answers generated per expansion
    num_reward_evals : critique calls averaged for each node's reward
    early_stop_reward: exit early when a node reaches this reward
    """

    def __init__(
        self,
        model: LLMWrapper,
        max_iter: int = 4,
        g_weight: float = 1.0,
        num_children: int = 2,
        num_reward_evals: int = 3,
        early_stop_reward: float = 95.0,
    ):
        self.model = model
        self.max_iter = max_iter
        self.g_weight = g_weight
        self.num_children = num_children
        self.num_reward_evals = num_reward_evals
        self.early_stop_reward = early_stop_reward

    def run(self, question: str) -> tuple[Node, list[Node]]:
        """
        Run TTA* on a single question.

        Returns
        -------
        best_node : Node with highest reward found
        history   : all nodes visited (root first)
        """
        # --- Root ---
        root_answer = self.model.chat(
            [{"role": "user", "content": SOLVE_PROMPT.format(question=question)}],
            temperature=0.3,
        )
        root = Node(question, root_answer, self.model, depth=0, num_reward_evals=self.num_reward_evals)
        root.compute_f(self.g_weight)

        open_set: list[Node] = [root]
        history: list[Node] = [root]
        best_node = root

        for iteration in range(self.max_iter):
            if not open_set:
                break

            # Pick node with lowest f
            current = min(open_set, key=lambda n: n.f)
            open_set.remove(current)

            if current.reward >= self.early_stop_reward:
                best_node = current
                break

            # Expand: generate num_children refined answers
            for _ in range(self.num_children):
                child_answer = self.model.chat(
                    [{"role": "user", "content": current.refine_prompt()}],
                    temperature=0.7,
                )
                child = Node(
                    question,
                    child_answer,
                    self.model,
                    depth=current.depth + 1,
                    num_reward_evals=self.num_reward_evals,
                )
                child.compute_f(self.g_weight)
                open_set.append(child)
                history.append(child)

                if child.reward > best_node.reward:
                    best_node = child

        return best_node, history
