"""
Differentiable Logic Operations for Neuro-Symbolic Reasoning

This module implements differentiable versions of logical operations
that can be used in neural networks for symbolic reasoning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tensor, Tuple, List, Optional


class DifferentiableLogic(nn.Module):
    """
    A module containing differentiable implementations of logical operations.

    All operations work on tensors with values in [0, 1] where:
    - 0.0 represents False
    - 1.0 represents True
    - Values in between represent uncertainty/soft truth values
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def logical_and(self, a: Tensor, b: Tensor) -> Tensor:
        """Differentiable AND operation: AND(a, b) = a * b"""
        return a * b

    def logical_or(self, a: Tensor, b: Tensor) -> Tensor:
        """Differentiable OR operation: OR(a, b) = a + b - a * b"""
        return a + b - a * b

    def logical_not(self, a: Tensor) -> Tensor:
        """Differentiable NOT operation: NOT(a) = 1 - a"""
        return 1.0 - a

    def logical_implies(self, a: Tensor, b: Tensor) -> Tensor:
        """Differentiable IMPLIES operation: IMPLIES(a, b) = 1 - a + a * b"""
        return 1.0 - a + a * b

    def logical_xor(self, a: Tensor, b: Tensor) -> Tensor:
        """Differentiable XOR operation: XOR(a, b) = a + b - 2 * a * b"""
        return a + b - 2 * a * b

    def logical_equiv(self, a: Tensor, b: Tensor) -> Tensor:
        """Differentiable EQUIVALENCE: EQUIV(a, b) = 1 - |a - b|"""
        return 1.0 - torch.abs(a - b)


class SoftUnification(nn.Module):
    """
    Soft unification module for matching symbolic patterns.

    This implements a differentiable version of unification from logic programming,
    allowing neural networks to perform pattern matching operations.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pattern_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.term_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.unify_score = nn.Linear(hidden_dim, 1)

    def forward(self, pattern: Tensor, term: Tensor) -> Tensor:
        """
        Compute soft unification score between pattern and term.

        Args:
            pattern: Pattern representation [batch_size, hidden_dim]
            term: Term representation [batch_size, hidden_dim]

        Returns:
            Unification score in [0, 1]
        """
        pattern_emb = F.relu(self.pattern_encoder(pattern))
        term_emb = F.relu(self.term_encoder(term))

        # Compute interaction
        interaction = pattern_emb * term_emb
        score = torch.sigmoid(self.unify_score(interaction))

        return score.squeeze(-1)


class LogicalReasoner(nn.Module):
    """
    Neural module for performing multi-step logical reasoning.

    This module can chain together logical operations to perform
    complex reasoning tasks in a differentiable manner.
    """

    def __init__(self, num_predicates: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.logic_ops = DifferentiableLogic()
        self.unification = SoftUnification(hidden_dim)

        # Predicate embeddings
        self.predicate_embeddings = nn.Embedding(num_predicates, hidden_dim)

        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, facts: List[Tensor], query: Tensor) -> Tensor:
        """
        Perform logical reasoning over facts to answer query.

        Args:
            facts: List of fact tensors [batch_size, hidden_dim]
            query: Query tensor [batch_size, hidden_dim]

        Returns:
            Answer probability [batch_size]
        """
        batch_size = query.size(0)

        # Initialize reasoning state with query
        reasoning_state = query

        # Iteratively apply reasoning layers
        for layer in self.reasoning_layers:
            # Update reasoning state
            reasoning_state = F.relu(layer(reasoning_state))

            # Check against all facts
            fact_scores = []
            for fact in facts:
                score = self.unification(reasoning_state, fact)
                fact_scores.append(score)

            if fact_scores:
                # Aggregate fact evidence
                fact_tensor = torch.stack(fact_scores, dim=1)  # [batch_size, num_facts]
                max_score, _ = torch.max(fact_tensor, dim=1)

                # Use max score to update reasoning state
                reasoning_state = reasoning_state * max_score.unsqueeze(-1)

        # Final answer prediction
        answer_logit = self.output_proj(reasoning_state)
        answer_prob = torch.sigmoid(answer_logit).squeeze(-1)

        return answer_prob


def test_differentiable_logic():
    """Test basic differentiable logic operations."""
    logic = DifferentiableLogic()

    # Test cases
    a = torch.tensor([0.0, 0.3, 0.7, 1.0])
    b = torch.tensor([0.0, 0.8, 0.2, 1.0])

    print("Testing Differentiable Logic Operations:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"AND(a, b) = {logic.logical_and(a, b)}")
    print(f"OR(a, b) = {logic.logical_or(a, b)}")
    print(f"NOT(a) = {logic.logical_not(a)}")
    print(f"IMPLIES(a, b) = {logic.logical_implies(a, b)}")
    print(f"XOR(a, b) = {logic.logical_xor(a, b)}")
    print(f"EQUIV(a, b) = {logic.logical_equiv(a, b)}")


def test_logical_reasoner():
    """Test the logical reasoner module."""
    batch_size = 4
    hidden_dim = 64
    num_predicates = 10

    reasoner = LogicalReasoner(num_predicates, hidden_dim)

    # Create dummy facts and query
    facts = [torch.randn(batch_size, hidden_dim) for _ in range(3)]
    query = torch.randn(batch_size, hidden_dim)

    # Forward pass
    answer_prob = reasoner(facts, query)

    print(f"\nTesting Logical Reasoner:")
    print(f"Input facts: {len(facts)} facts of shape {facts[0].shape}")
    print(f"Query shape: {query.shape}")
    print(f"Answer probabilities: {answer_prob}")
    print(f"Answer probabilities shape: {answer_prob.shape}")


if __name__ == "__main__":
    test_differentiable_logic()
    test_logical_reasoner()