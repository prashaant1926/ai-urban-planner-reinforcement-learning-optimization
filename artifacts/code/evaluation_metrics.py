"""
Evaluation metrics for neural-symbolic reasoning systems
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Any


class CompositionalityMetrics:
    """Metrics for evaluating compositional generalization."""

    @staticmethod
    def compositional_accuracy(predictions: torch.Tensor,
                             targets: torch.Tensor,
                             composition_mask: torch.Tensor) -> float:
        """
        Compute accuracy on compositional test cases.

        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            composition_mask: Boolean mask indicating compositional examples
        """
        comp_preds = predictions[composition_mask]
        comp_targets = targets[composition_mask]

        if len(comp_targets) == 0:
            return 0.0

        correct = (comp_preds.argmax(dim=1) == comp_targets).float()
        return correct.mean().item()

    @staticmethod
    def length_generalization_curve(predictions: List[torch.Tensor],
                                   targets: List[torch.Tensor],
                                   lengths: List[int]) -> Dict[int, float]:
        """
        Compute accuracy as a function of sequence length.

        Args:
            predictions: List of prediction tensors for each length
            targets: List of target tensors for each length
            lengths: List of sequence lengths
        """
        results = {}
        for pred, tgt, length in zip(predictions, targets, lengths):
            if len(tgt) > 0:
                acc = (pred.argmax(dim=1) == tgt).float().mean().item()
                results[length] = acc
            else:
                results[length] = 0.0
        return results

    @staticmethod
    def systematic_generalization_score(train_compositions: List[str],
                                      test_compositions: List[str],
                                      test_accuracies: List[float]) -> float:
        """
        Measure how well the model generalizes to unseen compositions.

        Args:
            train_compositions: List of operator sequences seen during training
            test_compositions: List of operator sequences in test set
            test_accuracies: Accuracy for each test composition
        """
        train_set = set(train_compositions)
        novel_indices = [i for i, comp in enumerate(test_compositions)
                        if comp not in train_set]

        if not novel_indices:
            return 0.0

        novel_accuracies = [test_accuracies[i] for i in novel_indices]
        return np.mean(novel_accuracies)


class InterpretabilityMetrics:
    """Metrics for evaluating interpretability of neural-symbolic systems."""

    @staticmethod
    def rule_extraction_accuracy(extracted_rules: List[str],
                               ground_truth_rules: List[str]) -> float:
        """
        Measure how accurately the system can extract logical rules.

        Args:
            extracted_rules: Rules extracted from the learned model
            ground_truth_rules: True logical rules used to generate data
        """
        # Simple string matching - could be made more sophisticated
        matches = 0
        for rule in ground_truth_rules:
            if rule in extracted_rules:
                matches += 1

        if len(ground_truth_rules) == 0:
            return 1.0 if len(extracted_rules) == 0 else 0.0

        return matches / len(ground_truth_rules)

    @staticmethod
    def explanation_consistency(explanations: List[str],
                              predictions: torch.Tensor) -> float:
        """
        Measure consistency between explanations and model predictions.

        Args:
            explanations: Natural language explanations
            predictions: Model predictions
        """
        # Placeholder - would need actual explanation evaluation
        # Could use semantic similarity, entailment checking, etc.
        return 0.85  # Dummy value

    @staticmethod
    def symbolic_faithfulness(symbolic_output: torch.Tensor,
                            neural_output: torch.Tensor,
                            threshold: float = 0.1) -> float:
        """
        Measure how faithfully symbolic reasoning matches neural computation.

        Args:
            symbolic_output: Output from symbolic reasoning component
            neural_output: Output from neural component
            threshold: Maximum allowed difference
        """
        diff = torch.abs(symbolic_output - neural_output)
        faithful = (diff < threshold).float()
        return faithful.mean().item()


class BenchmarkEvaluator:
    """Comprehensive evaluator for neural-symbolic reasoning benchmarks."""

    def __init__(self):
        self.comp_metrics = CompositionalityMetrics()
        self.interp_metrics = InterpretabilityMetrics()

    def evaluate_model(self, model, test_loader, device='cpu') -> Dict[str, Any]:
        """
        Comprehensive evaluation of a neural-symbolic model.

        Args:
            model: The model to evaluate
            test_loader: DataLoader with test data
            device: Computing device
        """
        model.eval()
        results = {
            'accuracy': [],
            'predictions': [],
            'targets': [],
            'explanations': [],
            'composition_masks': []
        }

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets, comp_mask = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                predictions = outputs['predictions'] if isinstance(outputs, dict) else outputs

                results['predictions'].append(predictions)
                results['targets'].append(targets)
                results['composition_masks'].append(comp_mask)

                # Compute batch accuracy
                acc = (predictions.argmax(dim=1) == targets).float().mean()
                results['accuracy'].append(acc.item())

        # Aggregate results
        all_preds = torch.cat(results['predictions'])
        all_targets = torch.cat(results['targets'])
        all_masks = torch.cat(results['composition_masks'])

        metrics = {
            'overall_accuracy': np.mean(results['accuracy']),
            'compositional_accuracy': self.comp_metrics.compositional_accuracy(
                all_preds, all_targets, all_masks
            ),
            'standard_deviation': np.std(results['accuracy'])
        }

        return metrics

    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Compare multiple models across different metrics.

        Args:
            model_results: Dict mapping model names to their evaluation results
        """
        comparison = {}

        for metric in ['overall_accuracy', 'compositional_accuracy']:
            metric_values = {name: results[metric] for name, results in model_results.items()}

            best_model = max(metric_values.items(), key=lambda x: x[1])
            worst_model = min(metric_values.items(), key=lambda x: x[1])

            comparison[metric] = {
                'best': best_model,
                'worst': worst_model,
                'mean': np.mean(list(metric_values.values())),
                'std': np.std(list(metric_values.values()))
            }

        return comparison


def example_evaluation():
    """Example usage of evaluation metrics."""

    # Simulate some results
    batch_size, num_classes = 100, 3
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    composition_mask = torch.rand(batch_size) > 0.5

    # Compute metrics
    comp_metrics = CompositionalityMetrics()
    comp_acc = comp_metrics.compositional_accuracy(predictions, targets, composition_mask)

    print(f"Overall accuracy: {(predictions.argmax(1) == targets).float().mean():.3f}")
    print(f"Compositional accuracy: {comp_acc:.3f}")

    # Example rule extraction evaluation
    interp_metrics = InterpretabilityMetrics()
    extracted = ["A AND B", "B OR C"]
    ground_truth = ["A AND B", "NOT C", "B OR C"]
    rule_acc = interp_metrics.rule_extraction_accuracy(extracted, ground_truth)
    print(f"Rule extraction accuracy: {rule_acc:.3f}")


if __name__ == "__main__":
    example_evaluation()