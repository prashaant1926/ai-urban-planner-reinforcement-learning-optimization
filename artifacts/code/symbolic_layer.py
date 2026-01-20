"""
Differentiable Symbolic Layer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLogicGate(nn.Module):
    """Differentiable logic gate with temperature-controlled sharpness."""

    def __init__(self, operation='and', temperature=1.0):
        super().__init__()
        self.operation = operation
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x, y):
        """
        Args:
            x, y: Tensors of shape (batch_size, ...) with values in [0, 1]
        Returns:
            Tensor of same shape representing logical operation
        """
        if self.operation == 'and':
            # Soft AND: min approximation using log-sum-exp trick
            return torch.sigmoid(-self.temperature * torch.log(
                torch.exp(-self.temperature * x) + torch.exp(-self.temperature * y)
            ))
        elif self.operation == 'or':
            # Soft OR: max approximation
            return torch.sigmoid(self.temperature * torch.log(
                torch.exp(self.temperature * x) + torch.exp(self.temperature * y)
            ))
        elif self.operation == 'not':
            # Soft NOT
            return torch.sigmoid(self.temperature * (1 - 2*x))
        else:
            raise ValueError(f"Unknown operation: {self.operation}")


class SymbolicLayer(nn.Module):
    """Layer that applies symbolic rules to neural representations."""

    def __init__(self, input_dim, num_rules, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules

        # Learnable rule weights
        self.rule_weights = nn.Parameter(torch.randn(num_rules, input_dim))
        self.rule_bias = nn.Parameter(torch.zeros(num_rules))

        # Logic gates for rule combination
        self.and_gate = SoftLogicGate('and', temperature)
        self.or_gate = SoftLogicGate('or', temperature)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Tensor of shape (batch_size, num_rules)
        """
        # Compute rule activations
        rule_logits = torch.matmul(x, self.rule_weights.T) + self.rule_bias
        rule_probs = torch.sigmoid(rule_logits)

        # Apply logical combinations (simplified example)
        # In practice, this would be more complex rule structure
        combined = rule_probs
        for i in range(1, self.num_rules):
            combined = self.and_gate(combined[:, :i], rule_probs[:, i:i+1])

        return combined


def test_symbolic_layer():
    """Test the symbolic layer implementation."""
    batch_size, input_dim, num_rules = 32, 10, 5

    layer = SymbolicLayer(input_dim, num_rules)
    x = torch.randn(batch_size, input_dim)

    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"Gradients computed successfully: {layer.rule_weights.grad is not None}")


if __name__ == "__main__":
    test_symbolic_layer()