"""
Experimental setup for evaluating neuro-symbolic reasoning systems.

This module provides tools for:
1. Dataset generation and loading
2. Model evaluation and benchmarking
3. Analysis and visualization of results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import random
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for neuro-symbolic experiments."""
    model_type: str = "hybrid"
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    hidden_dim: int = 128
    num_variables: int = 10
    max_clauses: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "experiments"


class LogicalReasoningDataset(Dataset):
    """Dataset for logical reasoning tasks."""

    def __init__(self, num_samples: int = 1000, num_variables: int = 5,
                 max_clauses: int = 3, difficulty: str = "medium"):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.max_clauses = max_clauses
        self.difficulty = difficulty

        # Generate dataset
        self.data = self._generate_dataset()

    def _generate_dataset(self) -> List[Dict]:
        """Generate logical reasoning problems."""
        data = []

        for _ in range(self.num_samples):
            # Generate random logical problem
            problem = self._generate_problem()
            data.append(problem)

        return data

    def _generate_problem(self) -> Dict:
        """Generate a single logical reasoning problem."""
        # Generate random clauses
        num_clauses = random.randint(1, self.max_clauses)
        clauses = []

        for _ in range(num_clauses):
            # Random clause length (2-3 literals)
            clause_length = random.randint(2, 3)
            clause = []

            for _ in range(clause_length):
                var_id = random.randint(1, self.num_variables)
                # Random negation
                if random.random() < 0.5:
                    var_id = -var_id
                clause.append(var_id)

            clauses.append(clause)

        # Generate target assignment
        assignment = [random.choice([0, 1]) for _ in range(self.num_variables)]

        # Check if assignment satisfies clauses
        satisfies = self._check_satisfaction(assignment, clauses)

        # Generate query (ask if a specific assignment works)
        query_assignment = assignment.copy()
        if random.random() < 0.3:  # Sometimes modify assignment
            idx = random.randint(0, self.num_variables - 1)
            query_assignment[idx] = 1 - query_assignment[idx]

        query_satisfies = self._check_satisfaction(query_assignment, clauses)

        return {
            'clauses': clauses,
            'assignment': assignment,
            'satisfies': satisfies,
            'query_assignment': query_assignment,
            'query_satisfies': query_satisfies
        }

    def _check_satisfaction(self, assignment: List[int], clauses: List[List[int]]) -> bool:
        """Check if assignment satisfies all clauses."""
        for clause in clauses:
            clause_satisfied = False
            for literal in clause:
                var_idx = abs(literal) - 1
                if literal > 0:
                    if assignment[var_idx] == 1:
                        clause_satisfied = True
                        break
                else:
                    if assignment[var_idx] == 0:
                        clause_satisfied = True
                        break

            if not clause_satisfied:
                return False

        return True

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Convert to tensors
        assignment = torch.tensor(item['assignment'], dtype=torch.float32)
        query_assignment = torch.tensor(item['query_assignment'], dtype=torch.float32)
        satisfies = torch.tensor(float(item['satisfies']), dtype=torch.float32)
        query_satisfies = torch.tensor(float(item['query_satisfies']), dtype=torch.float32)

        # Encode clauses as feature vector (simplified)
        clause_features = self._encode_clauses(item['clauses'])

        return {
            'clause_features': clause_features,
            'assignment': assignment,
            'query_assignment': query_assignment,
            'satisfies': satisfies,
            'query_satisfies': query_satisfies
        }

    def _encode_clauses(self, clauses: List[List[int]]) -> torch.Tensor:
        """Encode clauses as feature vector."""
        # Simple encoding: for each variable, count positive and negative occurrences
        features = torch.zeros(self.num_variables * 2)  # pos and neg for each var

        for clause in clauses:
            for literal in clause:
                var_idx = abs(literal) - 1
                if literal > 0:
                    features[var_idx * 2] += 1  # positive occurrence
                else:
                    features[var_idx * 2 + 1] += 1  # negative occurrence

        return features


class ExperimentRunner:
    """Runs and manages neuro-symbolic experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize results storage
        self.results = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'test_results': {}
        }

    def run_experiment(self, model: nn.Module, train_loader: DataLoader,
                      val_loader: DataLoader, test_loader: DataLoader) -> Dict:
        """Run complete experiment with training and evaluation."""
        print(f"Starting experiment with config: {self.config}")
        print(f"Using device: {self.device}")

        # Move model to device
        model = model.to(self.device)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)

            # Store results
            self.results['train_losses'].append(train_loss)
            self.results['val_losses'].append(val_loss)
            self.results['train_accuracies'].append(train_acc)
            self.results['val_accuracies'].append(val_acc)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Load best model for testing
        model.load_state_dict(best_model_state)

        # Test phase
        test_results = self._test_model(model, test_loader, criterion)
        self.results['test_results'] = test_results

        # Save results
        self._save_results()

        print(f"Experiment completed. Test accuracy: {test_results['accuracy']:.4f}")
        return self.results

    def _train_epoch(self, model: nn.Module, data_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            predictions = outputs['output']

            # Compute loss
            loss = criterion(predictions, batch['query_satisfies'])

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct_predictions += (predicted_labels == batch['query_satisfies']).sum().item()
            total_samples += predictions.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def _validate_epoch(self, model: nn.Module, data_loader: DataLoader,
                       criterion: nn.Module) -> Tuple[float, float]:
        """Run one validation epoch."""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(batch)
                predictions = outputs['output']

                loss = criterion(predictions, batch['query_satisfies'])

                total_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                correct_predictions += (predicted_labels == batch['query_satisfies']).sum().item()
                total_samples += predictions.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def _test_model(self, model: nn.Module, data_loader: DataLoader,
                   criterion: nn.Module) -> Dict[str, float]:
        """Test the model and return detailed results."""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(batch)
                predictions = outputs['output']

                loss = criterion(predictions, batch['query_satisfies'])

                total_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                correct_predictions += (predicted_labels == batch['query_satisfies']).sum().item()
                total_samples += predictions.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['query_satisfies'].cpu().numpy())

        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(data_loader)

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def _save_results(self):
        """Save experiment results to file."""
        results_file = self.save_dir / f"results_{self.config.model_type}_{self.config.seed}.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (list, np.ndarray)):
                        serializable_results[key][sub_key] = list(sub_value)
                    else:
                        serializable_results[key][sub_key] = sub_value
            else:
                serializable_results[key] = value

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")

    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curves
        ax1.plot(self.results['train_losses'], label='Train Loss')
        ax1.plot(self.results['val_losses'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.results['train_accuracies'], label='Train Accuracy')
        ax2.plot(self.results['val_accuracies'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_file = self.save_dir / f"training_curves_{self.config.model_type}_{self.config.seed}.png"
        plt.savefig(plot_file)
        print(f"Training curves saved to {plot_file}")
        plt.close()


def create_data_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    # Generate datasets
    train_dataset = LogicalReasoningDataset(
        num_samples=8000,
        num_variables=config.num_variables,
        max_clauses=config.max_clauses,
        difficulty="medium"
    )

    val_dataset = LogicalReasoningDataset(
        num_samples=1000,
        num_variables=config.num_variables,
        max_clauses=config.max_clauses,
        difficulty="medium"
    )

    test_dataset = LogicalReasoningDataset(
        num_samples=2000,
        num_variables=config.num_variables,
        max_clauses=config.max_clauses,
        difficulty="hard"
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Example model for testing
class SimpleReasoningModel(nn.Module):
    """Simple baseline model for logical reasoning."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Simple concatenation of features
        features = torch.cat([batch['clause_features'], batch['query_assignment']], dim=1)
        output = self.layers(features).squeeze(-1)

        return {'output': output}


def run_example_experiment():
    """Run an example experiment."""
    # Configuration
    config = ExperimentConfig(
        model_type="simple_baseline",
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        hidden_dim=128,
        num_variables=5,
        max_clauses=3,
        seed=42
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Create model
    input_dim = config.num_variables * 2 + config.num_variables  # clause features + assignment
    model = SimpleReasoningModel(input_dim, config.hidden_dim)

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run_experiment(model, train_loader, val_loader, test_loader)

    # Plot results
    runner.plot_training_curves()

    return results


if __name__ == "__main__":
    results = run_example_experiment()