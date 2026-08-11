"""
Training Logger + Curve Plotter
=================================

Logs per-epoch metrics during AE training and generates
publication-ready training curve plots.

Plots generated:
  1. loss_curves.png — recon + behaviour + total loss convergence
  2. lr_schedule.png — cosine annealing learning rate
  3. gradient_norms.png — per-track and per-component gradient flow
  4. silhouette_evolution.png — latent space quality over training
  5. training_summary.png — single-page 2x2 summary for paper
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class TrainingLogger:
    def __init__(self, save_dir='checkpoints/training_curves'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.epochs = []
        self.total_loss = []
        self.recon_loss = []
        self.bhv_loss = []
        self.learning_rates = []
        self.grad_norms = {'feature': [], 'context': [], 'behaviour': [],
                           'encoder': [], 'decoder': [], 'bhv_head': []}
        self.silhouette_checkpoints = []

    def log_epoch(self, epoch, total, recon, bhv, lr, grad_norms=None):
        self.epochs.append(epoch)
        self.total_loss.append(total)
        self.recon_loss.append(recon)
        self.bhv_loss.append(bhv)
        self.learning_rates.append(lr)
        if grad_norms:
            for key in self.grad_norms:
                if key in grad_norms:
                    self.grad_norms[key].append(grad_norms[key])

    def log_silhouette(self, epoch, Z, labels, sample_size=3000):
        n = min(sample_size, len(Z))
        idx = np.random.choice(len(Z), n, replace=False)
        try:
            score = silhouette_score(Z[idx], labels[idx])
            self.silhouette_checkpoints.append((epoch, score))
            return score
        except Exception:
            return None

    def plot_all(self):
        self._plot_loss_curves()
        self._plot_lr_schedule()
        self._plot_grad_norms()
        self._plot_silhouette_evolution()
        self._plot_summary()
        print(f"  Training curves saved: {self.save_dir}/")

    def _plot_loss_curves(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.epochs, self.total_loss, 'b-', linewidth=1.5)
        axes[0].set_title('Total Loss'); axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.epochs, self.recon_loss, 'r-', label='Reconstruction', linewidth=1.5)
        axes[1].plot(self.epochs, self.bhv_loss, 'g-', label='Behaviour', linewidth=1.5)
        axes[1].set_title('Loss Components'); axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(self.epochs, self.total_loss, 'b-', label='Total', linewidth=1.5)
        axes[2].semilogy(self.epochs, self.recon_loss, 'r--', label='Recon', linewidth=1)
        axes[2].semilogy(self.epochs, self.bhv_loss, 'g--', label='Behaviour', linewidth=1)
        axes[2].set_title('Loss (log scale)'); axes[2].set_xlabel('Epoch')
        axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Training Loss Convergence', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_lr_schedule(self):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.epochs, self.learning_rates, 'purple', linewidth=1.5)
        ax.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=11)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'lr_schedule.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_grad_norms(self):
        has_data = any(len(v) > 0 for v in self.grad_norms.values())
        if not has_data: return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for track in ['feature', 'context', 'behaviour']:
            if self.grad_norms[track]:
                axes[0].plot(self.epochs[:len(self.grad_norms[track])],
                             self.grad_norms[track], label=track, linewidth=1.5)
        axes[0].set_title('Encoder Track Gradient Norms'); axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('L2 Norm'); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
        for comp in ['encoder', 'decoder', 'bhv_head']:
            if self.grad_norms[comp]:
                axes[1].plot(self.epochs[:len(self.grad_norms[comp])],
                             self.grad_norms[comp], label=comp, linewidth=1.5)
        axes[1].set_title('AE Component Gradient Norms'); axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('L2 Norm'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
        fig.suptitle('Gradient Flow Analysis', fontsize=13); fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'gradient_norms.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_silhouette_evolution(self):
        if not self.silhouette_checkpoints: return
        epochs, scores = zip(*self.silhouette_checkpoints)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, scores, 'go-', markersize=8, linewidth=2)
        ax.fill_between(epochs, 0, scores, alpha=0.1, color='green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Latent Space Quality During Training', fontsize=12)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Silhouette Score')
        ax.grid(True, alpha=0.3)
        ax.annotate(f'{scores[0]:.3f}', (epochs[0], scores[0]),
                    textcoords="offset points", xytext=(10, 10), fontsize=10)
        ax.annotate(f'{scores[-1]:.3f}', (epochs[-1], scores[-1]),
                    textcoords="offset points", xytext=(10, -15), fontsize=10, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'silhouette_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_summary(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot(self.epochs, self.recon_loss, 'r-', label='Reconstruction', linewidth=1.5)
        ax.plot(self.epochs, self.bhv_loss, 'g-', label='Behaviour', linewidth=1.5)
        ax.plot(self.epochs, self.total_loss, 'b-', label='Total', linewidth=1.5, alpha=0.7)
        ax.set_title('Training Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(self.epochs, self.learning_rates, 'purple', linewidth=1.5)
        ax.set_title('Learning Rate'); ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        for track in ['feature', 'context', 'behaviour']:
            if self.grad_norms[track]:
                ax.plot(self.epochs[:len(self.grad_norms[track])],
                        self.grad_norms[track], label=track, linewidth=1.5)
        ax.set_title('Track Gradient Norms'); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        if self.silhouette_checkpoints:
            epochs, scores = zip(*self.silhouette_checkpoints)
            ax.plot(epochs, scores, 'go-', markersize=6, linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Latent Space Quality'); ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
        fig.suptitle('Training Summary', fontsize=14); fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, 'training_summary.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def compute_grad_norms(model):
    """Compute gradient L2 norms for each component."""
    norms = {}
    components = {
        'feature': model.feature_encoder,
        'context': model.context_encoder,
        'behaviour': model.behaviour_encoder,
        'encoder': model.encoder,
        'decoder': model.decoder,
    }
    if hasattr(model, 'behaviour_head'):
        components['bhv_head'] = model.behaviour_head
    for name, module in components.items():
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms[name] = total_norm ** 0.5
    return norms