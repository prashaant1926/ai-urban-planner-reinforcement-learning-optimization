"""
Integration of SAT solvers with neural networks for neuro-symbolic reasoning.

This module provides tools for converting neural network outputs to SAT problems
and integrating SAT solver results back into neural computation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass

try:
    from pysat.solvers import Glucose3
    from pysat.formula import CNF
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False
    print("Warning: PySAT not available. Install with 'pip install python-sat'")


@dataclass
class SATClause:
    """Represents a SAT clause as a list of literals."""
    literals: List[int]  # Positive for variable, negative for negation

    def __str__(self):
        return " ∨ ".join([f"x_{abs(lit)}" if lit > 0 else f"¬x_{abs(lit)}" for lit in self.literals])


class NeuralSATInterface(nn.Module):
    """
    Interface between neural networks and SAT solvers.

    This module converts soft neural outputs to hard SAT constraints
    and can backpropagate through discrete SAT solutions using
    straight-through estimators.
    """

    def __init__(self, num_variables: int, threshold: float = 0.5):
        super().__init__()
        self.num_variables = num_variables
        self.threshold = threshold

    def soft_to_sat(self, soft_assignment: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
        """
        Convert soft probability assignment to hard SAT assignment.

        Args:
            soft_assignment: Tensor of shape [batch_size, num_variables] with values in [0, 1]
            temperature: Temperature for Gumbel-Softmax sampling

        Returns:
            Hard assignment tensor of shape [batch_size, num_variables] with values in {0, 1}
        """
        if self.training:
            # Use Gumbel-Softmax for differentiable sampling during training
            logits = torch.stack([1 - soft_assignment, soft_assignment], dim=-1)
            hard_assignment = F.gumbel_softmax(logits, tau=temperature, hard=True)[..., 1]
        else:
            # Use threshold for deterministic assignment during inference
            hard_assignment = (soft_assignment > self.threshold).float()

        return hard_assignment

    def sat_to_soft(self, sat_assignment: torch.Tensor) -> torch.Tensor:
        """
        Convert hard SAT assignment back to soft probabilities.

        Uses straight-through estimator for gradient flow.
        """
        # Straight-through estimator: forward is hard, backward is soft
        return sat_assignment

    def check_satisfaction(self, assignment: torch.Tensor,
                          clauses: List[SATClause]) -> torch.Tensor:
        """
        Check if assignment satisfies all clauses.

        Args:
            assignment: Boolean assignment [batch_size, num_variables]
            clauses: List of SAT clauses

        Returns:
            Satisfaction tensor [batch_size] with 1.0 if satisfied, 0.0 otherwise
        """
        batch_size = assignment.size(0)
        satisfaction_scores = torch.ones(batch_size, device=assignment.device)

        for clause in clauses:
            clause_satisfied = torch.zeros(batch_size, device=assignment.device)

            for literal in clause.literals:
                var_idx = abs(literal) - 1  # Convert to 0-indexed
                if literal > 0:
                    clause_satisfied = torch.max(clause_satisfied, assignment[:, var_idx])
                else:
                    clause_satisfied = torch.max(clause_satisfied, 1 - assignment[:, var_idx])

            satisfaction_scores = satisfaction_scores * clause_satisfied

        return satisfaction_scores


class DifferentiableSATSolver(nn.Module):
    """
    A differentiable SAT solver that can be integrated into neural networks.

    This uses relaxation techniques to make SAT solving approximately differentiable.
    """

    def __init__(self, num_variables: int, max_iterations: int = 100):
        super().__init__()
        self.num_variables = num_variables
        self.max_iterations = max_iterations

        # Learnable parameters for variable selection heuristics
        self.variable_weights = nn.Parameter(torch.randn(num_variables))
        self.clause_weights = nn.Parameter(torch.randn(1))  # Will be expanded based on clauses

    def forward(self, clauses: List[SATClause],
                initial_assignment: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve SAT problem approximately using differentiable relaxation.

        Args:
            clauses: List of SAT clauses
            initial_assignment: Optional initial variable assignment

        Returns:
            assignment: Variable assignment [batch_size, num_variables]
            satisfaction: Satisfaction score [batch_size]
        """
        batch_size = initial_assignment.size(0) if initial_assignment is not None else 1

        if initial_assignment is None:
            # Random initialization
            assignment = torch.rand(batch_size, self.num_variables)
        else:
            assignment = initial_assignment.clone()

        # Iterative improvement using gradient-based optimization
        for _ in range(self.max_iterations):
            assignment = self._update_assignment(assignment, clauses)

        # Check final satisfaction
        interface = NeuralSATInterface(self.num_variables)
        satisfaction = interface.check_satisfaction(assignment, clauses)

        return assignment, satisfaction

    def _update_assignment(self, assignment: torch.Tensor,
                          clauses: List[SATClause]) -> torch.Tensor:
        """Update variable assignment to better satisfy clauses."""
        batch_size = assignment.size(0)
        new_assignment = assignment.clone()

        # Compute clause violations
        for clause in clauses:
            clause_violation = self._compute_clause_violation(assignment, clause)

            # Update variables involved in violated clauses
            for literal in clause.literals:
                var_idx = abs(literal) - 1
                if literal > 0:
                    # Increase probability for positive literal
                    new_assignment[:, var_idx] += 0.1 * clause_violation * self.variable_weights[var_idx]
                else:
                    # Decrease probability for negative literal
                    new_assignment[:, var_idx] -= 0.1 * clause_violation * self.variable_weights[var_idx]

        # Clamp to [0, 1]
        new_assignment = torch.clamp(new_assignment, 0.0, 1.0)

        return new_assignment

    def _compute_clause_violation(self, assignment: torch.Tensor,
                                 clause: SATClause) -> torch.Tensor:
        """Compute how much a clause is violated by current assignment."""
        batch_size = assignment.size(0)
        clause_satisfaction = torch.zeros(batch_size, device=assignment.device)

        for literal in clause.literals:
            var_idx = abs(literal) - 1
            if literal > 0:
                clause_satisfaction = torch.max(clause_satisfaction, assignment[:, var_idx])
            else:
                clause_satisfaction = torch.max(clause_satisfaction, 1 - assignment[:, var_idx])

        # Violation is 1 - satisfaction
        return 1.0 - clause_satisfaction


class HybridSATNeuralNetwork(nn.Module):
    """
    A hybrid neural network that integrates SAT solving into the computation graph.

    This network can learn to generate SAT problems from input data and solve them
    to produce structured outputs.
    """

    def __init__(self, input_dim: int, num_variables: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim

        # Neural components
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        self.variable_predictor = nn.Linear(hidden_dim, num_variables)
        self.clause_generator = nn.Linear(hidden_dim, hidden_dim)

        # SAT solving components
        self.sat_solver = DifferentiableSATSolver(num_variables)
        self.sat_interface = NeuralSATInterface(num_variables)

        # Output decoder
        self.output_decoder = nn.Linear(num_variables, 1)

    def forward(self, x: torch.Tensor,
                target_clauses: Optional[List[SATClause]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid SAT-neural network.

        Args:
            x: Input tensor [batch_size, input_dim]
            target_clauses: Optional target clauses for supervised learning

        Returns:
            Dictionary with 'assignment', 'satisfaction', and 'output' tensors
        """
        batch_size = x.size(0)

        # Encode input
        hidden = torch.relu(self.input_encoder(x))

        # Predict initial variable assignment
        initial_assignment = torch.sigmoid(self.variable_predictor(hidden))

        # Generate or use provided clauses
        if target_clauses is None:
            # For now, use simple default clauses
            # In practice, this would be learned from data
            target_clauses = [
                SATClause([1, 2]),  # x1 ∨ x2
                SATClause([-1, 3]), # ¬x1 ∨ x3
            ]

        # Solve SAT problem
        assignment, satisfaction = self.sat_solver(target_clauses, initial_assignment)

        # Convert to hard assignment for output
        hard_assignment = self.sat_interface.soft_to_sat(assignment)

        # Decode output
        output = torch.sigmoid(self.output_decoder(hard_assignment))

        return {
            'assignment': assignment,
            'hard_assignment': hard_assignment,
            'satisfaction': satisfaction,
            'output': output.squeeze(-1)
        }


def test_sat_integration():
    """Test the SAT solver integration."""
    if not PYSAT_AVAILABLE:
        print("PySAT not available, skipping SAT integration test")
        return

    print("Testing SAT Integration:")

    # Create simple SAT problem: (x1 ∨ x2) ∧ (¬x1 ∨ x3) ∧ (¬x2 ∨ ¬x3)
    clauses = [
        SATClause([1, 2]),    # x1 ∨ x2
        SATClause([-1, 3]),   # ¬x1 ∨ x3
        SATClause([-2, -3])   # ¬x2 ∨ ¬x3
    ]

    print("SAT Problem:")
    for i, clause in enumerate(clauses):
        print(f"  Clause {i+1}: {clause}")

    # Test with hybrid network
    batch_size = 4
    input_dim = 10
    num_variables = 3

    network = HybridSATNeuralNetwork(input_dim, num_variables)
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    results = network(x, clauses)

    print(f"\nResults:")
    print(f"Soft assignments: {results['assignment']}")
    print(f"Hard assignments: {results['hard_assignment']}")
    print(f"Satisfaction scores: {results['satisfaction']}")
    print(f"Network outputs: {results['output']}")


if __name__ == "__main__":
    test_sat_integration()