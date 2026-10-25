"""
Comprehensive Training Script for Model WITH Extensive EMCC
Full comprehensive model architecture with ALL EMCC (Grouped Depthwise Convolution) blocks

This script trains the model WITH:
- EMCC blocks on ALL encoder skip connections
- EMCC blocks on ALL decoder skip connections
- All grouped depthwise convolution-based feature refinement

Purpose: Train the most comprehensive EMCC-enhanced model for maximum performance
Total EMCC blocks: 10 (2 bottleneck + 4 encoder + 4 decoder)

Data split update:
- Loads trainImage/trainMask and valImage/valMask from preprocessing output.
- Does not randomly split 2D slices, avoiding patient leakage when preprocessing split is patient-level.
- Test volumes are not loaded during training.
"""

from glob import glob
import argparse
import os
import random
import logging
import numpy as np
import time
from dataset import Dataset
import torch
import torch.optim as optim
from models.model import BayesianEMCCUncertaintyModel
from losses import BCEDiceLoss
from metrics import iou_score, dice_coef, sensitivity, specificity, hausdorff_distance_95
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import json
import csv

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='yk', type=str)
parser.add_argument('--experiment', default='comprehensive_EMCC', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description', default='Bayesian Uncertainty-Aware EMCC Model', type=str)

# Data paths
parser.add_argument('--root', default='path to training set', type=str)
parser.add_argument('--img_paths_root', default='processed_data/trainImage', type=str)
parser.add_argument('--mask_paths_root', default='processed_data/trainMask', type=str)
parser.add_argument('--val_img_paths_root', default='processed_data/valImage', type=str,
                    help='Precomputed validation images from patient-level split')
parser.add_argument('--val_mask_paths_root', default='processed_data/valMask', type=str,
                    help='Precomputed validation masks from patient-level split')

# Model parameters
parser.add_argument('--model_name', default='BayesianEMCCUncertaintyModel', type=str)
parser.add_argument('--drop_rate', default=0.20, type=float, help='Dropout rate for encoder, bottleneck, and decoder')
#parser.add_argument('--l1_reg', default=0.0001, type=float, help='L1 regularization coefficient (weight decay)')
#parser.add_argument('--l2_reg', default=0.0001, type=float, help='L2 regularization coefficient (additional to optimizer weight_decay)')
# Training metrics are reported as measured; no artificial scaling.

# Training parameters [[memory:2550481]]
parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--kl_weight', default=1e-4, type=float, help='Weight for normalized Bayesian KL divergence')
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--criterion', default='bce_dice', type=str)
parser.add_argument('--num_class', default=3, type=int, help='3 classes for WT, TC, ET')
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=60, type=int)
parser.add_argument('--save_freq', default=5, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--roi_height', default=160, type=int, help='ROI crop height')
parser.add_argument('--roi_width', default=160, type=int, help='ROI crop width')

# Enhanced logging for ablation study
parser.add_argument('--log_freq', default=1, type=int, help='Logging frequency')
parser.add_argument('--eval_freq', default=1, type=int, help='Detailed evaluation frequency')
parser.add_argument('--uncertainty_plot_freq', default=1, type=int, help='Generate reliability/risk-coverage plots every N epochs; 0 disables')
parser.add_argument('--uq_max_pixels_per_batch', default=5000, type=int, help='Maximum sampled pixels per class per validation batch for UQ plots')

args = parser.parse_args()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(output, target, threshold=0.5):
    """Calculate pixel-wise accuracy for segmentation"""
    with torch.no_grad():
        if isinstance(output, dict):
            mean_pred = output['final_mean']
        else:
            mean_pred = output
        probs = torch.sigmoid(mean_pred)
        preds = (probs > threshold).float()
        correct = (preds == target).float()
        accuracy = correct.mean().item()
        return accuracy

# ==============================
# UPDATED LOSS (replaces uncertainty_loss)
# ==============================
def calculate_model_kl(model):
    """Non-silent variational KL; invoke after the model's forward pass.

    Reuses the model's block-wise Blitz KL implementation so loss bookkeeping
    cannot accidentally omit every Bayesian layer.
    """
    kl_fn = getattr(model, "kl_divergence", None)
    if not callable(kl_fn):
        raise RuntimeError("Model must implement kl_divergence()")
    kl = kl_fn()
    if not torch.is_tensor(kl) or not kl.requires_grad or not torch.isfinite(kl).all():
        raise RuntimeError("KL must be a finite differentiable tensor. Check Bayesian blocks.")
    return kl


def hetero_ce_with_temp(mean_pred, var_pred, target, alpha=0.5, eps=1e-6):
    """
    Heteroscedastic, temperature-scaled BCE:
      - temp = sqrt(1 + var)  -> softens logits where predicted variance is high
      - BCE on softened logits
      - + small regularizer on log(1+var) to discourage infinite variance
    """
    temp = torch.sqrt(1.0 + var_pred.clamp_min(eps))
    mean_t = mean_pred / temp
    ce = F.binary_cross_entropy_with_logits(mean_t, target)
    reg = 0.5 * torch.log1p(var_pred).mean()
    return ce + alpha * reg


def calculate_volume_overlap_error(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def calculate_relative_volume_difference(pred, target, smooth=1e-5):
    pred_vol = pred.sum()
    target_vol = target.sum()
    if target_vol < smooth:
        return 0.0
    return abs(pred_vol - target_vol) / target_vol

def calculate_average_symmetric_surface_distance(pred, target):
    try:
        if isinstance(pred, torch.Tensor):
            pred_np = pred.cpu().numpy()
        else:
            pred_np = pred
        if isinstance(target, torch.Tensor):
            target_np = target.cpu().numpy()
        else:
            target_np = target
        pred_binary = (pred_np > 0.5).astype(np.uint8)
        target_binary = (target_np > 0.5).astype(np.uint8)
        from scipy.ndimage import distance_transform_edt
        pred_surface = pred_binary - distance_transform_edt(pred_binary) > -0.5
        target_surface = target_binary - distance_transform_edt(target_binary) > -0.5
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return 0.0
        dist_pred_to_target = distance_transform_edt(~target_binary)[pred_surface]
        dist_target_to_pred = distance_transform_edt(~pred_binary)[target_surface]
        assd = (dist_pred_to_target.mean() + dist_target_to_pred.mean()) / 2.0
        return float(assd)
    except Exception:
        return 0.0

def calculate_predictive_entropy(mean_pred, var_pred):
    try:
        probs = torch.sigmoid(mean_pred)
        eps = 1e-8
        entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
        return entropy.mean().item()
    except Exception:
        return 0.0

def calculate_aleatoric_uncertainty(var_pred):
    try:
        return var_pred.mean().item()
    except Exception:
        return 0.0

def calculate_mutual_information(mean_pred, var_pred):
    """Not identifiable from one forward pass; use repeated stochastic predictions."""
    return float('nan')

def calculate_predictive_variance(var_pred):
    try:
        return var_pred.var().item()
    except Exception:
        return 0.0

def calculate_uncertainty_error_correlation(mean_pred, var_pred, target):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        error = torch.abs(pred_probs - target)
        error_np = error.flatten().detach().cpu().numpy()
        uncertainty_np = var_pred.flatten().detach().cpu().numpy()
        correlation = np.corrcoef(error_np, uncertainty_np)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation)
    except Exception:
        return 0.0

def calculate_auroc_error_detection(mean_pred, var_pred, target, threshold=0.5):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        pred_binary = (pred_probs > threshold).float()
        error_binary = (pred_binary != target).float()
        uncertainty_np = var_pred.flatten().detach().cpu().numpy()
        error_np = error_binary.flatten().detach().cpu().numpy()
        if len(np.unique(error_np)) < 2:
            return 0.5
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(error_np, uncertainty_np)
        if np.isnan(auroc):
            return 0.5
        return float(auroc)
    except Exception:
        return 0.5

def calculate_expected_calibration_error(mean_pred, var_pred, target, n_bins=15):
    """Binary confidence calibration: confidence vs whether thresholded prediction is correct."""
    try:
        probs = torch.sigmoid(mean_pred).detach().cpu().numpy().reshape(-1)
        y = target.detach().cpu().numpy().reshape(-1).astype(np.uint8)
        pred = (probs >= 0.5).astype(np.uint8)
        confidence = np.maximum(probs, 1.0 - probs)
        correct = (pred == y).astype(np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidence >= edges[i]) & ((confidence < edges[i+1]) if i < n_bins-1 else (confidence <= edges[i+1]))
            if mask.any():
                ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
        return float(ece)
    except Exception:
        return float('nan')

def calculate_maximum_calibration_error(mean_pred, var_pred, target, n_bins=10):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        probs_np = pred_probs.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        probs_flat = probs_np.flatten()
        targets_flat = targets_np.flatten()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = targets_flat[in_bin].mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        return float(max_error)
    except Exception:
        return 0.0

def calculate_brier_score(mean_pred, var_pred, target):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        probs_np = pred_probs.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        probs_flat = probs_np.flatten()
        targets_flat = targets_np.flatten()
        brier_score = np.mean((probs_flat - targets_flat) ** 2)
        return float(brier_score)
    except Exception:
        return 0.0

def calculate_negative_log_likelihood(mean_pred, var_pred, target):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        probs_np = pred_probs.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        probs_flat = np.clip(probs_np.flatten(), 1e-8, 1 - 1e-8)
        targets_flat = targets_np.flatten()
        nll = -np.mean(targets_flat * np.log(probs_flat) + (1 - targets_flat) * np.log(1 - probs_flat))
        return float(nll)
    except Exception:
        return 0.0

def calculate_psnr(pred, target):
    """Calculate Peak Signal-to-Noise Ratio"""
    try:
        pred_probs = torch.sigmoid(pred)
        mse = F.mse_loss(pred_probs, target.float())
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(torch.tensor(max_pixel) / torch.sqrt(mse))
        return psnr.item()
    except Exception:
        return 0.0

def calculate_mse(pred, target):
    """Calculate Mean Squared Error"""
    try:
        pred_probs = torch.sigmoid(pred)
        mse = F.mse_loss(pred_probs, target.float())
        return mse.item()
    except Exception:
        return 0.0

def calculate_p_value(pred, target):
    """Not applicable: compare patient-level paired metrics across models instead."""
    return float('nan')

def create_reliability_diagram_data(mean_pred, var_pred, target, n_bins=10):
    try:
        pred_probs = torch.sigmoid(mean_pred)
        probs_np = pred_probs.detach().cpu().numpy()
        targets_np = target.detach().cpu().numpy()
        probs_flat = probs_np.flatten()
        targets_flat = targets_np.flatten()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        reliabilities, confidences, counts = [], [], []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = targets_flat[in_bin].mean()
                avg_confidence_in_bin = probs_flat[in_bin].mean()
                reliabilities.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                counts.append(in_bin.sum())
            else:
                reliabilities.append(0)
                confidences.append(bin_centers[i])
                counts.append(0)
        return {'reliabilities': reliabilities, 'confidences': confidences,
                'counts': counts, 'bin_centers': bin_centers}
    except Exception:
        return {'reliabilities': [0]*n_bins, 'confidences': [0]*n_bins,
                'counts': [0]*n_bins, 'bin_centers': np.linspace(0,1,n_bins)}

def _surface_metrics_2d(pred_mask, true_mask):
    """Slice-level surface distances in pixels; these are NOT 3D mm metrics."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    if not pred_mask.any() and not true_mask.any():
        return 0.0, 0.0
    if not pred_mask.any() or not true_mask.any():
        return float('nan'), float('nan')
    ps = pred_mask ^ binary_erosion(pred_mask)
    ts = true_mask ^ binary_erosion(true_mask)
    d_to_t = distance_transform_edt(~ts)[ps]
    d_to_p = distance_transform_edt(~ps)[ts]
    all_d = np.concatenate([d_to_t, d_to_p])
    return float(np.percentile(all_d, 95)), float((d_to_t.mean()+d_to_p.mean())/2.0)


def calculate_comprehensive_metrics(output, target):
    """Calculate consistent hard-mask segmentation metrics for a 2D slice batch.

    HD95/ASSD are slice-level pixel distances here. Use reconstructed patient
    volumes and physical spacing for manuscript 3D HD95/ASSD.
    """
    with torch.no_grad():
        mean_pred = output['final_mean'] if isinstance(output, dict) else output
        probs = torch.sigmoid(mean_pred).detach().cpu().numpy()
        yall = (target.detach().cpu().numpy() >= 0.5).astype(np.uint8)
        labels = ['WT', 'TC', 'ET']
        metrics = {}
        keys = ['dice','sensitivity','specificity','auc','precision','recall','f1',
                'hd95','voe','rvd','assd','pe','au','mi','pv','ucc','auroc_error',
                'ece','mce','brier','nll','psnr','mse','pvalue']
        collected = {k: [] for k in keys}
        for c, label in enumerate(labels):
            p = probs[:, c]
            y = yall[:, c]
            pred = (p >= 0.5).astype(np.uint8)
            pf, yf = p.reshape(-1), y.reshape(-1)
            pred_f = pred.reshape(-1)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            eps = 1e-8
            dice = float((2*tp + eps)/(2*tp+fp+fn+eps))
            sens = float(tp/(tp+fn+eps))
            spec = float(tn/(tn+fp+eps))
            precision = float(tp/(tp+fp+eps))
            f1 = float(2*tp/(2*tp+fp+fn+eps))
            try:
                auc = float(roc_auc_score(yf, pf)) if np.unique(yf).size == 2 else float('nan')
            except Exception:
                auc = float('nan')
            hds, assds = [], []
            for j in range(y.shape[0]):
                hd, assd = _surface_metrics_2d(pred[j], y[j])
                if np.isfinite(hd): hds.append(hd)
                if np.isfinite(assd): assds.append(assd)
            hd95 = float(np.mean(hds)) if hds else float('nan')
            assd = float(np.mean(assds)) if assds else float('nan')
            true_vol, pred_vol = float(y.sum()), float(pred.sum())
            rvd = float((pred_vol-true_vol)/true_vol) if true_vol > 0 else float('nan')
            voe = float(1.0-dice)
            # Confidence calibration: max(p, 1-p) vs correctness of thresholded class.
            confidence = np.maximum(pf, 1-pf)
            correct = (pred_f == yf).astype(np.float64)
            edges = np.linspace(0, 1, 16)
            ece, mce = 0.0, 0.0
            for bi in range(15):
                mask = (confidence >= edges[bi]) & ((confidence < edges[bi+1]) if bi < 14 else (confidence <= edges[bi+1]))
                if mask.any():
                    gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
                    ece += float(mask.mean()) * gap
                    mce = max(mce, gap)
            brier = float(np.mean((pf-yf)**2))
            epsp = 1e-7
            nll = float(-np.mean(yf*np.log(np.clip(pf,epsp,1-epsp)) + (1-yf)*np.log(np.clip(1-pf,epsp,1-epsp))))
            mse = brier
            psnr = float(-10*np.log10(max(mse, 1e-12)))
            pe = float(np.mean(-pf*np.log(np.clip(pf,epsp,1))-(1-pf)*np.log(np.clip(1-pf,epsp,1))))
            if isinstance(output, dict) and 'final_var' in output:
                var = output['final_var'][:, c].detach().float().cpu().numpy().reshape(-1)
                error = (pred_f != yf).astype(np.uint8)
                try:
                    auroc_error = float(roc_auc_score(error, var)) if np.unique(error).size == 2 else float('nan')
                except Exception:
                    auroc_error = float('nan')
                au = float(np.mean(var))
                pv = float(np.var(var))
                ucc = float(np.corrcoef(var, error)[0,1]) if np.std(var)>0 and np.std(error)>0 else float('nan')
            else:
                au = pv = ucc = auroc_error = float('nan')
            # One mean/variance prediction is insufficient to estimate MI reliably.
            mi = float('nan')
            pvalue = float('nan')  # Pixel-level significance test is not appropriate.
            row = {'dice':dice,'sensitivity':sens,'specificity':spec,'auc':auc,
                   'precision':precision,'recall':sens,'f1':f1,'hd95':hd95,
                   'voe':voe,'rvd':rvd,'assd':assd,'pe':pe,'au':au,'mi':mi,
                   'pv':pv,'ucc':ucc,'auroc_error':auroc_error,'ece':float(ece),
                   'mce':float(mce),'brier':brier,'nll':nll,'psnr':psnr,
                   'mse':mse,'pvalue':pvalue}
            metrics[label] = row
            for k in keys: collected[k].append(row[k])
        metrics['average'] = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float('nan') for k,v in collected.items()}
        return metrics


def create_basic_training_plots(training_history, save_dir):
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    train_ious = training_history.get('train_ious', [])
    val_ious = training_history.get('val_ious', [])
    train_accuracies = training_history.get('train_accuracies', [])
    val_accuracies = training_history.get('val_accuracies', [])
    if not (train_losses and val_losses):
        print("[WARNING] No basic training metrics found for plotting")
        return
    epochs = list(range(len(train_losses)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Basic Training Progress', fontsize=16, fontweight='bold')
    axes[0, 0].plot(epochs, train_losses, color='#e74c3c', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, color='#f39c12', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, train_ious, color='#2ecc71', label='Train IoU', linewidth=2)
    axes[0, 1].plot(epochs, val_ious, color='#8e44ad', label='Validation IoU', linewidth=2)
    axes[0, 1].set_title('Training and Validation IoU', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    if train_accuracies and val_accuracies:
        axes[1, 0].plot(epochs, train_accuracies, color='#3498db', label='Train Accuracy', linewidth=2)
        axes[1, 0].plot(epochs, val_accuracies, color='#e67e22', label='Validation Accuracy', linewidth=2)
        axes[1, 0].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Accuracy Data Available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Accuracy (No Data)', fontweight='bold')
    axes[1, 1].plot(epochs, train_losses, color='#e74c3c', alpha=0.7, label='Train Loss')
    axes[1, 1].plot(epochs, val_losses, color='#f39c12', alpha=0.7, label='Val Loss')
    ax_twin = axes[1, 1].twinx()
    ax_twin.plot(epochs, train_ious, color='#27ae60', linestyle='--', alpha=0.7, label='Train IoU')
    ax_twin.plot(epochs, val_ious, color='#9b59b6', linestyle='--', alpha=0.7, label='Val IoU')
    axes[1, 1].set_title('Combined Training Progress', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss'); ax_twin.set_ylabel('IoU')
    axes[1, 1].legend(loc='upper left'); ax_twin.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'basic_training_progress.png'), dpi=800, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'basic_training_progress.pdf'), bbox_inches='tight')
    plt.close()
    print(f"[PLOTS] Basic training plots saved to: {save_dir}")

def create_validation_specific_plots(training_history, save_dir):
    detailed_metrics = training_history.get('detailed_metrics', [])
    if not detailed_metrics:
        print("[WARNING] No detailed metrics available for validation plots")
        return
    epochs = [m['epoch'] for m in detailed_metrics]
    labels = ['WT', 'TC', 'ET', 'average']
    colors = ["#4B0E0694", "#BBA339", "#12D370", "#39398A"]
    fig, axes = plt.subplots(7, 3, figsize=(20, 35))
    fig.suptitle('Validation Performance Analysis - All Metrics', fontsize=14, fontweight='bold')
    validation_metrics = [
        ('dice', 'Dice Coefficient', 'Segmentation Overlap Quality'),
        ('sensitivity', 'Sensitivity (Recall)', 'True Positive Detection Rate'),
        ('specificity', 'Specificity', 'True Negative Detection Rate'),
        ('auc', 'AUC-ROC', 'Classification Performance'),
        ('precision', 'Precision', 'Positive Prediction Accuracy'),
        ('f1', 'F1-Score', 'Balanced Precision-Recall'),
        ('recall', 'Recall', 'Same as Sensitivity'),
        ('hd95', 'Hausdorff Distance 95%', 'Surface Distance Quality'),
        ('voe', 'Volume Overlap Error', 'Volume Segmentation Error'),
        ('rvd', 'Relative Volume Difference', 'Volume Estimation Error'),
        ('assd', 'Average Symmetric Surface Distance', 'Surface Accuracy'),
        ('pe', 'Predictive Entropy', 'Total Uncertainty'),
        ('au', 'Aleatoric Uncertainty', 'Data Uncertainty'),
        ('mi', 'Mutual Information', 'Epistemic Uncertainty'),
        ('pv', 'Predictive Variance', 'Prediction Variance'),
        ('ucc', 'Uncertainty-Error Correlation', 'Uncertainty Quality'),
        ('auroc_error', 'AUROC for Error Detection', 'Error Detection Performance'),
        ('ece', 'Expected Calibration Error', 'Calibration Quality'),
        ('mce', 'Maximum Calibration Error', 'Worst Case Calibration'),
        ('brier', 'Brier Score', 'Probability Prediction Quality'),
        ('nll', 'Negative Log-Likelihood', 'Proper Scoring Rule'),
    ]
    for idx, (metric, title, description) in enumerate(validation_metrics):
        if idx >= 21:
            break
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        for label, color in zip(labels, colors):
            values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if isinstance(m, dict) and 'train' in m and 'val' in m:
                    # Use validation data for validation plots
                    data = m.get('val', {})
                else:
                    # Fallback to old format
                    data = m
                    
                if label in data and metric in data[label]:
                    values.append(data[label][metric])
                else:
                    values.append(0)
            if values and any(v > 0 for v in values):
                line_style = '--' if label == 'average' else '-'
                line_width = 3 if label == 'average' else 2
                marker_size = 8 if label == 'average' else 6
                alpha = 0.9 if label == 'average' else 0.7
                ax.plot(epochs, values, marker='o', linewidth=line_width,
                        markersize=marker_size, label=label, color=color,
                        alpha=alpha, linestyle=line_style)
        ax.set_title(f'{title}\n{description}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        handles, labels_list = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
        if len(epochs) > 2:
            avg_values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if isinstance(m, dict) and 'train' in m and 'val' in m:
                    data = m.get('val', {})
                else:
                    data = m
                    
                if 'average' in data and metric in data['average']:
                    avg_values.append(data['average'][metric])
            if len(avg_values) == len(epochs):
                z = np.polyfit(epochs, avg_values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "gray", linestyle=':', alpha=0.5, linewidth=1)
    ax_loss = axes[2, 1]
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    if train_losses and val_losses:
        loss_epochs = list(range(len(train_losses)))
        ax_loss.plot(loss_epochs, train_losses, 'b-', label='Training Loss', alpha=0.7, linewidth=2)
        ax_loss.plot(loss_epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7, linewidth=2)
        ax_loss.set_title('Training vs Validation Loss', fontweight='bold')
        ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss')
        ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_iou = axes[2, 2]
    train_ious = training_history.get('train_ious', [])
    val_ious = training_history.get('val_ious', [])
    if train_ious and val_ious:
        iou_epochs = list(range(len(train_ious)))
        ax_iou.plot(iou_epochs, train_ious, color='#2ecc71', label='Training IoU', alpha=0.7, linewidth=2)
        ax_iou.plot(iou_epochs, val_ious, color='#8e44ad', label='Validation IoU', alpha=0.7, linewidth=2)
        ax_iou.set_title('Training vs Validation IoU', fontweight='bold')
        ax_iou.set_xlabel('Epoch'); ax_iou.set_ylabel('IoU')
        ax_iou.legend(); ax_iou.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'validation_analysis.png'), dpi=800, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'validation_analysis.pdf'), bbox_inches='tight')
    plt.close()
    create_class_specific_validation_plots(detailed_metrics, epochs, save_dir)
    print(f"[PLOTS] Validation analysis plots saved to: {save_dir}")

def create_class_specific_validation_plots(detailed_metrics, epochs, save_dir):
    labels = ['WT', 'TC', 'ET']
    class_names = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    colors = ["#4B0E0694", "#BBA339", "#12D370"]
    metrics_info = [
        ('dice', 'Dice Coefficient'),
        ('sensitivity', 'Sensitivity'),
        ('specificity', 'Specificity'),
        ('auc', 'AUC-ROC'),
        ('precision', 'Precision'),
        ('f1', 'F1-Score'),
        ('hd95', 'Hausdorff Distance 95%'),
        ('voe', 'Volume Overlap Error'),
        ('rvd', 'Relative Volume Difference'),
        ('assd', 'Average Symmetric Surface Distance'),
        ('pe', 'Predictive Entropy'),
        ('au', 'Aleatoric Uncertainty'),
        ('mi', 'Mutual Information'),
        ('pv', 'Predictive Variance'),
        ('ucc', 'Uncertainty-Error Correlation'),
        ('auroc_error', 'AUROC for Error Detection'),
        ('ece', 'Expected Calibration Error'),
        ('mce', 'Maximum Calibration Error'),
        ('brier', 'Brier Score'),
        ('nll', 'Negative Log-Likelihood')
    ]
    for label, class_name, color in zip(labels, class_names, colors):
        fig, axes = plt.subplots(7, 3, figsize=(18, 42))
        fig.suptitle(f'{class_name} ({label}) - Validation Performance Analysis', fontsize=16, fontweight='bold')
        for idx, (metric, metric_name) in enumerate(metrics_info):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if isinstance(m, dict) and 'train' in m and 'val' in m:
                    # Use validation data for validation plots
                    data = m.get('val', {})
                else:
                    # Fallback to old format
                    data = m
                    
                if label in data and metric in data[label]:
                    values.append(data[label][metric])
                else:
                    values.append(0)
            if values and any(v > 0 for v in values):
                ax.plot(epochs, values, marker='o', linewidth=3, markersize=6,
                        color=color, alpha=0.8, label=f'{label} {metric_name}')
                if len(epochs) > 2:
                    z = np.polyfit(epochs, values, 1)
                    p = np.poly1d(z)
                    ax.plot(epochs, p(epochs), color=color, linestyle='--', alpha=0.5, linewidth=2)
                    improvement = values[-1] - values[0] if len(values) > 1 else 0
                    improvement_text = f"Delta: {improvement:+.3f}"
                    ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_title(f'{metric_name}', fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.5); ax.set_ylim(0, 1.05)
            if metric in ['dice', 'sensitivity', 'specificity', 'precision', 'f1']:
                ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Excellent (>=0.8)')
                ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Good (>=0.7)')
            elif metric == 'auc':
                ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Excellent (>=0.9)')
                ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Good (>=0.8)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{label.lower()}_validation_detailed.png'), dpi=800, bbox_inches='tight')
        plt.close()

def create_separate_train_val_metric_plots(training_history, save_dir):
    """Create separate plots for training and validation metrics"""
    detailed_metrics = training_history.get('detailed_metrics', [])
    if not detailed_metrics:
        print("[WARNING] No detailed metrics available for separate train/val plots")
        return
    
    epochs = [m['epoch'] for m in detailed_metrics]
    labels = ['WT', 'TC', 'ET', 'average']
    colors = ["#4B0E0694", "#BBA339", "#12D370", "#39398A"]
    
    # All metrics to plot
    all_metrics = [
        ('dice', 'Dice Coefficient'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('specificity', 'Specificity'),
        ('auc', 'AUC-ROC'),
        ('precision', 'Precision'),
        ('f1', 'F1-Score'),
        ('recall', 'Recall'),
        ('hd95', 'Hausdorff Distance 95%'),
        ('voe', 'Volume Overlap Error'),
        ('rvd', 'Relative Volume Difference'),
        ('assd', 'Average Symmetric Surface Distance'),
        ('pe', 'Predictive Entropy'),
        ('au', 'Aleatoric Uncertainty'),
        ('mi', 'Mutual Information'),
        ('pv', 'Predictive Variance'),
        ('ucc', 'Uncertainty-Error Correlation'),
        ('auroc_error', 'AUROC for Error Detection'),
        ('ece', 'Expected Calibration Error'),
        ('mce', 'Maximum Calibration Error'),
        ('brier', 'Brier Score'),
        ('nll', 'Negative Log-Likelihood'),
        ('psnr', 'Peak Signal-to-Noise Ratio'),
        ('mse', 'Mean Squared Error'),
        ('pvalue', 'P-Value (Statistical Significance)'),
    ]
    
    # Create separate training plot with training data only
    create_metric_plot(detailed_metrics, epochs, labels, colors, all_metrics, 
                       'training', 'Training Metrics', save_dir, data_type='train')
    
    # Create separate validation plot with validation data only
    create_metric_plot(detailed_metrics, epochs, labels, colors, all_metrics, 
                       'validation', 'Validation Metrics', save_dir, data_type='val')

def create_metric_plot(detailed_metrics, epochs, labels, colors, all_metrics, plot_type, title, save_dir, data_type='train'):
    """Helper function to create separate plots for each metric"""
    os.makedirs(save_dir, exist_ok=True)
    
    for metric, title_name in all_metrics:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for label, color in zip(labels, colors):
            values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if 'train' in m and 'val' in m:
                    # Use train or val data based on data_type
                    data = m[data_type] if data_type in m else m.get('train', {})
                else:
                    # Fallback to old format
                    data = m
                    
                if label in data and metric in data[label]:
                    values.append(data[label][metric])
                else:
                    values.append(0)
            
            if values and any(v > 0 for v in values):
                line_style = '--' if label == 'average' else '-'
                line_width = 3 if label == 'average' else 2
                marker_size = 8 if label == 'average' else 6
                alpha = 0.9 if label == 'average' else 0.7
                ax.plot(epochs, values, marker='o', linewidth=line_width,
                        markersize=marker_size, label=label, color=color,
                        alpha=alpha, linestyle=line_style)
        
        # Add trend line for average
        if len(epochs) > 2:
            avg_values = []
            for m in detailed_metrics:
                if 'train' in m and 'val' in m:
                    data = m[data_type] if data_type in m else m.get('train', {})
                else:
                    data = m
                    
                if 'average' in data and metric in data['average']:
                    avg_values.append(data['average'][metric])
            if len(avg_values) == len(epochs) and any(v > 0 for v in avg_values):
                z = np.polyfit(epochs, avg_values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "gray", linestyle=':', alpha=0.5, linewidth=2, label='Trend')
        
        ax.set_title(f'{title} - {title_name}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title_name, fontsize=12)
        # Only show legend if there are plotted lines
        handles, labels_list = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=11)
        ax.grid(True, alpha=0.5)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        # Clean filename
        metric_name = metric.replace(' ', '_').lower()
        plt.savefig(os.path.join(save_dir, f'{plot_type}_{metric_name}.png'), dpi=800, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{plot_type}_{metric_name}.pdf'), bbox_inches='tight')
        plt.close()
    
    print(f"[PLOTS] {title} plots saved to: {save_dir} ({len(all_metrics)} separate plots)")

def create_comprehensive_plots(training_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    detailed_metrics = training_history.get('detailed_metrics', [])
    create_basic_training_plots(training_history, save_dir)
    create_validation_specific_plots(training_history, save_dir)
    create_separate_train_val_metric_plots(training_history, save_dir)
    if not detailed_metrics:
        print("[WARNING] No detailed metrics found for comprehensive plotting, but basic and validation plots created")
        return
    epochs = [m['epoch'] for m in detailed_metrics]
    labels = ['WT', 'TC', 'ET', 'average']
    colors = ["#4B0E0694", "#BBA339", "#12D370", "#39398A"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Comprehensive Model Performance Metrics', fontsize=16, fontweight='bold')
    metrics_to_plot = ['dice', 'sensitivity', 'specificity', 'auc', 'precision', 'recall', 'f1']
    metric_titles = ['Dice Coefficient', 'Sensitivity (Recall)', 'Specificity',
                    'AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        for label, color in zip(labels, colors):
            values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if isinstance(m, dict) and 'train' in m and 'val' in m:
                    # Use validation data for validation plots
                    data = m.get('val', {})
                else:
                    # Fallback to old format
                    data = m
                    
                if label in data and metric in data[label]:
                    values.append(data[label][metric])
                else:
                    values.append(0)
            if values and any(v > 0 for v in values):
                ax.plot(epochs, values, marker='o', linewidth=2, markersize=4,
                        label=label, color=color, alpha=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        handles, labels_list = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    ax_loss = axes[2, 2]
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    train_ious = training_history.get('train_ious', [])
    val_ious = training_history.get('val_ious', [])
    if train_losses and val_losses:
        loss_epochs = list(range(len(train_losses)))
        ax_loss.plot(loss_epochs, train_losses, color='#e74c3c', label='Train Loss', alpha=0.7)
        ax_loss.plot(loss_epochs, val_losses, color='#f39c12', label='Val Loss', alpha=0.7)
        ax_loss_twin = ax_loss.twinx()
        ax_loss_twin.plot(loss_epochs, train_ious, color='#27ae60', linestyle='--', label='Train IoU', alpha=0.7)
        ax_loss_twin.plot(loss_epochs, val_ious, color='#9b59b6', linestyle='--', label='Val IoU', alpha=0.7)
        ax_loss.set_title('Training Progress', fontweight='bold')
        ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('Loss'); ax_loss_twin.set_ylabel('IoU')
        ax_loss.legend(loc='upper left'); ax_loss_twin.legend(loc='upper right')
        ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_metrics.png'), dpi=800, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'comprehensive_metrics.pdf'), bbox_inches='tight')
    plt.close()
    create_individual_metric_plots(detailed_metrics, epochs, labels, colors, save_dir)
    print(f"[PLOTS] Comprehensive plots saved to: {save_dir}")

def format_training_results_table(training_history, best_iou, total_params, total_time):
    def format_value(mean, std=None):
        if std is not None:
            return f"{mean:.3f} +/- {std:.3f}"
        else:
            return f"{mean:.3f}"
    final_train_loss = training_history['train_losses'][-1] if training_history['train_losses'] else 0
    final_val_loss = training_history['val_losses'][-1] if training_history['val_losses'] else 0
    final_train_iou = training_history['train_ious'][-1] if training_history['train_ious'] else 0
    final_val_iou = training_history['val_ious'][-1] if training_history['val_ious'] else 0
    final_train_acc = training_history['train_accuracies'][-1] if training_history['train_accuracies'] else 0
    final_val_acc = training_history['val_accuracies'][-1] if training_history['val_accuracies'] else 0
    train_loss_mean = np.mean(training_history['train_losses']) if training_history['train_losses'] else 0
    train_loss_std = np.std(training_history['train_losses']) if training_history['train_losses'] else 0
    val_loss_mean = np.mean(training_history['val_losses']) if training_history['val_losses'] else 0
    val_loss_std = np.std(training_history['val_losses']) if training_history['val_losses'] else 0
    train_iou_mean = np.mean(training_history['train_ious']) if training_history['train_ious'] else 0
    train_iou_std = np.std(training_history['train_ious']) if training_history['train_ious'] else 0
    val_iou_mean = np.mean(training_history['val_ious']) if training_history['val_ious'] else 0
    val_iou_std = np.std(training_history['val_ious']) if training_history['val_ious'] else 0
    train_acc_mean = np.mean(training_history['train_accuracies']) if training_history['train_accuracies'] else 0
    train_acc_std = np.std(training_history['train_accuracies']) if training_history['train_accuracies'] else 0
    val_acc_mean = np.mean(training_history['val_accuracies']) if training_history['val_accuracies'] else 0
    val_acc_std = np.std(training_history['val_accuracies']) if training_history['val_accuracies'] else 0

    print("\n" + "="*140)
    print("BAYESIAN EMCC UNCERTAINTY MODEL TRAINING & VALIDATION RESULTS")
    print("="*140)
    print(f"{'Metric':<15} {'Train Mean':<12} {'Train Std':<12} {'Train Final':<12} {'Val Mean':<12} {'Val Std':<12} {'Val Final':<12} {'Best/Peak':<12}")
    print("-"*140)
    metrics_data = [
        ('Loss', train_loss_mean, train_loss_std, final_train_loss, val_loss_mean, val_loss_std, final_val_loss, f"{min(training_history['val_losses']):.3f}" if training_history['val_losses'] else "0.000"),
        ('IoU', train_iou_mean, train_iou_std, final_train_iou, val_iou_mean, val_iou_std, final_val_iou, f"{max(training_history['val_ious']):.3f}" if training_history['val_ious'] else "0.000"),
        ('Accuracy', train_acc_mean, train_acc_std, final_train_acc, val_acc_mean, val_acc_std, final_val_acc, f"{max(training_history['val_accuracies']):.3f}" if training_history['val_accuracies'] else "0.000")
    ]
    for metric_name, train_mean, train_std, train_final, val_mean, val_std, val_final, best_val in metrics_data:
        print(f"{metric_name:<15} {train_mean:<12.3f} {train_std:<12.3f} {train_final:<12.3f} "
              f"{val_mean:<12.3f} {val_std:<12.3f} {val_final:<12.3f} {best_val:<12}")
    print("-"*140)
    epochs_trained = len(training_history['train_losses'])
    print(f"{'Model Info':<15} {'Value':<45} {'Details':<40}")
    print("-"*100)
    print(f"{'Parameters':<15} {'' :<45} {'Bayesian EMCC Architecture':<40}")
    print(f"{'Epochs Trained':<15} {str(epochs_trained):<45} {'With Bayesian uncertainty':<40}")
    print(f"{'Training Time':<15} {'' :<45} {'Simple decoder + EMCC blocks':<40}")
    print(f"{'Architecture':<15} {'Bayesian + EMCC + Simple Decoder':<45} {'Epistemic + Aleatoric uncertainty':<40}")
    detailed_metrics = training_history.get('detailed_metrics', [])
    if detailed_metrics:
        print("\n" + "="*140)
        print("COMPREHENSIVE DETAILED METRICS - TRAINING & VALIDATION")
        print("="*140)
        labels = ['WT', 'TC', 'ET', 'average']
        metrics_names = ['dice', 'sensitivity', 'specificity', 'auc', 'precision', 'recall', 'f1']

        last_entry = detailed_metrics[-1]
        has_split = isinstance(last_entry, dict) and ('train' in last_entry and 'val' in last_entry)

        if has_split:
            # Build per-label/metric trajectories across epochs for train and val
            print("\n" + "COMPREHENSIVE DETAILED METRICS - TRAINING & VALIDATION (MEAN/STD/FINAL)")
            print("-"*180)
            print(f"{'Label':<8} {'Metric':<12} {'Train Mean':<12} {'Train Std':<12} {'Train Final':<12} {'Val Mean':<12} {'Val Std':<12} {'Val Final':<12}")
            print("-"*180)
            for label in labels:
                label_display = label.upper() if label != 'average' else 'AVG'
                for i, metric in enumerate(metrics_names):
                    train_series = []
                    val_series = []
                    for entry in detailed_metrics:
                        if isinstance(entry, dict) and ('train' in entry and 'val' in entry):
                            tr = entry.get('train', {})
                            va = entry.get('val', {})
                            # Append values if present
                            tv = tr.get(label, {}).get(metric, None)
                            vv = va.get(label, {}).get(metric, None)
                            if tv is not None:
                                train_series.append(tv)
                            if vv is not None:
                                val_series.append(vv)
                    # Compute stats safely
                    if train_series:
                        t_mean = float(np.mean(train_series))
                        t_std  = float(np.std(train_series)) if len(train_series) > 1 else 0.0
                        t_final = float(train_series[-1])
                    else:
                        t_mean = t_std = t_final = 0.0
                    if val_series:
                        v_mean = float(np.mean(val_series))
                        v_std  = float(np.std(val_series)) if len(val_series) > 1 else 0.0
                        v_final = float(val_series[-1])
                    else:
                        v_mean = v_std = v_final = 0.0
                    label_col = label_display if i == 0 else ""
                    print(f"{label_col:<8} {metric:<12} {t_mean:<12.3f} {t_std:<12.3f} {t_final:<12.3f} {v_mean:<12.3f} {v_std:<12.3f} {v_final:<12.3f}")
            print("-"*180)
        else:
            # Fallback for legacy structure (no explicit train/val split per epoch)
            print(f"{'Label':<8} {'Metric':<12} {'Mean':<12} {'Std':<12} {'Final':<12}")
            print("-"*120)
            for label in labels:
                label_display = label.upper() if label != 'average' else 'AVG'
                for i, metric in enumerate(metrics_names):
                    values = []
                    for eval_data in detailed_metrics:
                        if label in eval_data and metric in eval_data[label]:
                            values.append(eval_data[label][metric])
                    if values:
                        mean_v = np.mean(values)
                        std_v = np.std(values) if len(values) > 1 else 0.0
                        final_v = values[-1]
                    else:
                        mean_v = std_v = final_v = 0.0
                    label_col = label_display if i == 0 else ""
                    print(f"{label_col:<8} {metric:<12} {mean_v:<12.3f} {std_v:<12.3f} {final_v:<12.3f}")
            print("-"*120)
        print("\n" + "="*180)
        print("DETAILED METRICS PROGRESSION - TRAINING & VALIDATION")
        print("="*180)
        print(f"{'Epoch':<8} {'Train':<50} {'Validation':<50}")
        print(f"{'':8} {'WT Dice':<10} {'TC Dice':<10} {'ET Dice':<10} {'Avg Dice':<10} {'Avg F1':<10} {'WT Dice':<10} {'TC Dice':<10} {'ET Dice':<10} {'Avg Dice':<10} {'Avg F1':<10}")
        print("-"*180)
        for detailed_eval in detailed_metrics:
            epoch_num = detailed_eval.get('epoch', 0)
            val_wt_dice = detailed_eval.get('WT', {}).get('dice', 0)
            val_tc_dice = detailed_eval.get('TC', {}).get('dice', 0)
            val_et_dice = detailed_eval.get('ET', {}).get('dice', 0)
            val_avg_dice = detailed_eval.get('average', {}).get('dice', 0)
            val_avg_f1 = detailed_eval.get('average', {}).get('f1', 0)
            train_wt_dice = val_wt_dice
            train_tc_dice = val_tc_dice
            train_et_dice = val_et_dice
            train_avg_dice = val_avg_dice
            train_avg_f1 = val_avg_f1
            print(f"{epoch_num:<8} {train_wt_dice:<10.3f} {train_tc_dice:<10.3f} {train_et_dice:<10.3f} {train_avg_dice:<10.3f} {train_avg_f1:<10.3f} "
                  f"{val_wt_dice:<10.3f} {val_tc_dice:<10.3f} {val_et_dice:<10.3f} {val_avg_dice:<10.3f} {val_avg_f1:<10.3f}")
    else:
        print("\n" + "="*100)
        print("NO DETAILED METRICS AVAILABLE")
        print("="*100)
        print("Detailed metrics were not collected during training.")
        print("This could be due to:")
        print("- Training stopped before first detailed evaluation")
        print("- Errors in detailed metrics calculation")
        print("- Model evaluation issues")
        print("="*100)
    print("="*100)
    print("")

def create_csv_training_summary(training_history, best_iou, total_params, total_time, save_path):
    final_train_loss = training_history['train_losses'][-1] if training_history['train_losses'] else 0
    final_val_loss = training_history['val_losses'][-1] if training_history['val_losses'] else 0
    final_train_iou = training_history['train_ious'][-1] if training_history['train_ious'] else 0
    final_val_iou = training_history['val_ious'][-1] if training_history['val_ious'] else 0
    train_loss_mean = np.mean(training_history['train_losses']) if training_history['train_losses'] else 0
    val_loss_mean = np.mean(training_history['val_losses']) if training_history['val_losses'] else 0
    train_iou_mean = np.mean(training_history['train_ious']) if training_history['train_ious'] else 0
    val_iou_mean = np.mean(training_history['val_ious']) if training_history['val_ious'] else 0
    epochs_trained = len(training_history['train_losses'])
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value', 'Description'])
        writer.writerow(['Model', 'Bayesian EMCC Uncertainty Model', 'Architecture with Bayesian encoder, EMCC blocks, and simple decoder'])
        writer.writerow(['Total Parameters', f'{total_params:,}', 'Number of trainable parameters'])
        writer.writerow(['Epochs Trained', epochs_trained, 'Total epochs completed'])
        writer.writerow(['Training Time (hours)', f'{total_time/3600:.2f}', 'Total training duration'])
        writer.writerow(['Best Validation IoU', f'{max(training_history["val_ious"]) if training_history["val_ious"] else 0:.4f}', 'Highest validation IoU achieved'])
        writer.writerow([])
        writer.writerow(['Training Metrics', '', ''])
        writer.writerow(['Final Train Loss', f'{final_train_loss:.4f}', 'Loss at final epoch'])
        writer.writerow(['Final Val Loss', f'{final_val_loss:.4f}', 'Validation loss at final epoch'])
        writer.writerow(['Mean Train Loss', f'{train_loss_mean:.4f}', 'Average training loss'])
        writer.writerow(['Mean Val Loss', f'{val_loss_mean:.4f}', 'Average validation loss'])
        writer.writerow(['Final Train IoU', f'{final_train_iou:.4f}', 'IoU at final epoch'])
        writer.writerow(['Final Val IoU', f'{final_val_iou:.4f}', 'Validation IoU at final epoch'])
        writer.writerow(['Mean Train IoU', f'{train_iou_mean:.4f}', 'Average training IoU'])
        writer.writerow(['Mean Val IoU', f'{val_iou_mean:.4f}', 'Average validation IoU'])
        detailed_metrics = training_history.get('detailed_metrics', [])
        if detailed_metrics:
            writer.writerow([])
            writer.writerow(['Final Detailed Metrics', '', ''])
            final_detailed = detailed_metrics[-1]
            for label in ['WT', 'TC', 'ET', 'average']:
                if label in final_detailed:
                    metrics = final_detailed[label]
                    label_display = label.upper() if label != 'average' else 'Average'
                    writer.writerow([f'{label_display} Dice', f"{metrics.get('dice', 0):.3f}", f'Dice coefficient for {label_display}'])
                    writer.writerow([f'{label_display} Sensitivity', f"{metrics.get('sensitivity', 0):.3f}", f'Sensitivity for {label_display}'])
                    writer.writerow([f'{label_display} Specificity', f"{metrics.get('specificity', 0):.3f}", f'Specificity for {label_display}'])
                    writer.writerow([f'{label_display} AUC', f"{metrics.get('auc', 0):.3f}", f'AUC-ROC for {label_display}'])
                    writer.writerow([f'{label_display} Precision', f"{metrics.get('precision', 0):.3f}", f'Precision for {label_display}'])
                    writer.writerow([f'{label_display} F1', f"{metrics.get('f1', 0):.3f}", f'F1-score for {label_display}'])
                    writer.writerow([f'{label_display} HD95', f"{metrics.get('hd95', 0):.3f}", f'Hausdorff Distance 95% for {label_display}'])
                    writer.writerow([f'{label_display} VOE', f"{metrics.get('voe', 0):.3f}", f'Volume Overlap Error for {label_display}'])
                    writer.writerow([f'{label_display} RVD', f"{metrics.get('rvd', 0):.3f}", f'Relative Volume Difference for {label_display}'])
                    writer.writerow([f'{label_display} ASSD', f"{metrics.get('assd', 0):.3f}", f'Average Symmetric Surface Distance for {label_display}'])
                    writer.writerow([f'{label_display} PE', f"{metrics.get('pe', 0):.3f}", f'Predictive Entropy for {label_display}'])
                    writer.writerow([f'{label_display} AU', f"{metrics.get('au', 0):.3f}", f'Aleatoric Uncertainty for {label_display}'])
                    writer.writerow([f'{label_display} MI', f"{metrics.get('mi', 0):.3f}", f'Mutual Information for {label_display}'])
                    writer.writerow([f'{label_display} PV', f"{metrics.get('pv', 0):.3f}", f'Predictive Variance for {label_display}'])
                    writer.writerow([f'{label_display} UCC', f"{metrics.get('ucc', 0):.3f}", f'Uncertainty-Error Correlation for {label_display}'])
                    writer.writerow([f'{label_display} AUROC_Error', f"{metrics.get('auroc_error', 0):.3f}", f'AUROC for Error Detection for {label_display}'])
                    writer.writerow([f'{label_display} ECE', f"{metrics.get('ece', 0):.3f}", f'Expected Calibration Error for {label_display}'])
                    writer.writerow([f'{label_display} MCE', f"{metrics.get('mce', 0):.3f}", f'Maximum Calibration Error for {label_display}'])
                    writer.writerow([f'{label_display} Brier', f"{metrics.get('brier', 0):.3f}", f'Brier Score for {label_display}'])
                    writer.writerow([f'{label_display} NLL', f"{metrics.get('nll', 0):.3f}", f'Negative Log-Likelihood for {label_display}'])

def create_individual_metric_plots(detailed_metrics, epochs, labels, colors, save_dir):
    metrics_info = [
        ('dice', 'Dice Coefficient', 'Higher is Better'),
        ('auc', 'AUC-ROC', 'Higher is Better (0.5 = Random)'),
        ('precision', 'Precision', 'Higher is Better'),
        ('recall', 'Recall (Sensitivity)', 'Higher is Better'),
        ('f1', 'F1-Score', 'Higher is Better'),
        ('specificity', 'Specificity', 'Higher is Better'),
        ('hd95', 'Hausdorff Distance 95%', 'Lower is Better'),
        ('voe', 'Volume Overlap Error', 'Lower is Better'),
        ('rvd', 'Relative Volume Difference', 'Lower is Better'),
        ('assd', 'Average Symmetric Surface Distance', 'Lower is Better'),
        ('pe', 'Predictive Entropy', 'Higher is Better'),
        ('au', 'Aleatoric Uncertainty', 'Higher is Better'),
        ('mi', 'Mutual Information', 'Higher is Better'),
        ('pv', 'Predictive Variance', 'Higher is Better'),
        ('ucc', 'Uncertainty-Error Correlation', 'Higher is Better'),
        ('auroc_error', 'AUROC for Error Detection', 'Higher is Better (0.5 = Random)'),
        ('ece', 'Expected Calibration Error', 'Lower is Better'),
        ('mce', 'Maximum Calibration Error', 'Lower is Better'),
        ('brier', 'Brier Score', 'Lower is Better'),
        ('nll', 'Negative Log-Likelihood', 'Lower is Better')
    ]
    for metric, title, description in metrics_info:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for label, color in zip(labels, colors):
            values = []
            for m in detailed_metrics:
                # Check if this metric entry has separate train/val data
                if isinstance(m, dict) and 'train' in m and 'val' in m:
                    # Use validation data for validation plots
                    data = m.get('val', {})
                else:
                    # Fallback to old format
                    data = m
                    
                if label in data and metric in data[label]:
                    values.append(data[label][metric])
                else:
                    values.append(0)
            if values and any(v > 0 for v in values):
                line_style = '--' if label == 'average' else '-'
                line_width = 3 if label == 'average' else 2
                ax.plot(epochs, values, marker='o', linewidth=line_width,
                        markersize=6, label=label, color=color, alpha=0.8, linestyle=line_style)
        ax.set_title(f'{title} Over Training Epochs\n{description}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel(title, fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
        if metric == 'auc':
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric}_progress.png'), dpi=800, bbox_inches='tight')
        plt.close()

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

# ==============================
# UPDATED TRAIN EPOCH
# ==============================
def train_epoch(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    ious = AverageMeter()
    accuracies = AverageMeter()
    uncertainty_losses = AverageMeter()
    segmentation_losses = AverageMeter()

    model.train()

    for i, data in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

        input, target = data
        input = input.cuda()
        target = target.cuda()

        output = model(input)

        if isinstance(output, dict):
            mean_pred = output['final_mean']
            var_pred = output['final_var']

            # Separate segmentation loss (BCEDiceLoss)
            segmentation_loss = criterion(mean_pred, target)

            # Uncertainty Loss (Heteroscedastic BCE with temperature scaling)
            uncertainty_loss = hetero_ce_with_temp(mean_pred, var_pred, target, alpha=0.5)

            # L1/L2 Regularization
            #reg_loss = calculate_l1_l2_regularization(model, l1_coef=args.l1_reg, l2_coef=args.l2_reg)

            # Total loss is the sum of all losses
            kl_raw = calculate_model_kl(model)
            kl_loss = kl_raw / max(1, len(train_loader.dataset))
            total_loss = segmentation_loss + uncertainty_loss + args.kl_weight * kl_loss

            # Compute IoU and accuracy
            iou = iou_score(mean_pred, target)
            accuracy = calculate_accuracy(output, target)

            # Update losses separately
            segmentation_losses.update(segmentation_loss.item(), input.size(0))
            uncertainty_losses.update(uncertainty_loss.item(), input.size(0))

        else:
            # For cases where the output is not a dictionary (no uncertainty predictions)
            total_loss = criterion(output, target)
            iou = iou_score(output, target)
            accuracy = calculate_accuracy(output, target)

        losses.update(total_loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        accuracies.update(accuracy, input.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log the progress
        if i % (args.log_freq * 10) == 0:
            if isinstance(output, dict):
                logging.info('Epoch: {}_Iter:{}  Seg Loss: {:.4f}, Uncertainty Loss: {:.4f}, Total Loss: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.format(
                    epoch, i, segmentation_loss.item(), uncertainty_loss.item(), total_loss.item(), iou, accuracy))
            else:
                logging.info('Epoch: {}_Iter:{}  Loss: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.format(
                    epoch, i, total_loss.item(), iou, accuracy))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('accuracy', accuracies.avg)
    ])

    # Log uncertainty and segmentation losses separately if they were computed
    if uncertainty_losses.count > 0:
        log['uncertainty_loss'] = uncertainty_losses.avg
    if segmentation_losses.count > 0:
        log['segmentation_loss'] = segmentation_losses.avg

    return log

# ==============================
# UPDATED VALIDATE EPOCH
# ==============================
def validate_epoch(val_loader, model, criterion, epoch, train_size):
    losses = AverageMeter()
    ious = AverageMeter()
    accuracies = AverageMeter()
    uncertainty_losses = AverageMeter()
    segmentation_losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input, target = data
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            if isinstance(output, dict):
                mean_pred = output['final_mean']
                var_pred = output['final_var']

                # Separate segmentation loss (BCEDiceLoss)
                segmentation_loss = criterion(mean_pred, target)

                # Uncertainty Loss (Heteroscedastic BCE with temperature scaling)
                uncertainty_loss = hetero_ce_with_temp(mean_pred, var_pred, target, alpha=0.5)

                # Total loss is the sum of both losses
                kl_raw = calculate_model_kl(model)
                kl_loss = kl_raw / max(1, train_size)
                total_loss = segmentation_loss + uncertainty_loss + args.kl_weight * kl_loss

                iou = iou_score(mean_pred, target)
                accuracy = calculate_accuracy(output, target)

                # Update losses separately
                segmentation_losses.update(segmentation_loss.item(), input.size(0))
                uncertainty_losses.update(uncertainty_loss.item(), input.size(0))
            else:
                total_loss = criterion(output, target)
                iou = iou_score(output, target)
                accuracy = calculate_accuracy(output, target)

            losses.update(total_loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            accuracies.update(accuracy, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('accuracy', accuracies.avg)
    ])

    # Log uncertainty and segmentation losses separately if they were computed
    if uncertainty_losses.count > 0:
        log['uncertainty_loss'] = uncertainty_losses.avg
    if segmentation_losses.count > 0:
        log['segmentation_loss'] = segmentation_losses.avg

    return log

@torch.no_grad()
def generate_reliability_and_risk_coverage(val_loader, model, epoch, output_dir,
                                          device, max_pixels_per_batch=5000):
    """Create validation reliability and risk-coverage plots for WT/TC/ET.

    Reliability: confidence=max(p,1-p) against binary correctness.
    Risk-coverage: error rate after retaining the least-uncertain pixels.
    Uncertainty uses final_var if available; otherwise predictive entropy is used.
    To bound RAM, a reproducible random subset is sampled per batch/class.
    These are pixel-level validation diagnostics, not patient-level 3D metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = ['WT', 'TC', 'ET']
    colors = {'WT': '#8C564B', 'TC': '#E69F00', 'ET': '#2CA02C',
              'Average': '#243B8A', 'reference': '#808080'}
    sampled = {n: {'p': [], 'y': [], 'u': []} for n in names}
    was_training = model.training
    model.eval()
    for batch in val_loader:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, dict):
            logits = outputs.get('final_mean')
            variance = outputs.get('final_var')
        else:
            logits, variance = outputs, None
        if logits is None:
            raise ValueError("Model output dict has no 'final_mean' tensor.")
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        ys = (targets.detach().float().cpu().numpy() >= 0.5).astype(np.uint8)
        if variance is not None:
            unc = variance.detach().float().cpu().numpy()
            # Variance head may have a singleton channel or 3 class channels.
            if unc.shape[1] == 1 and probs.shape[1] > 1:
                unc = np.repeat(unc, probs.shape[1], axis=1)
        else:
            eps = 1e-7
            unc = -(probs*np.log(np.clip(probs, eps, 1.0)) +
                    (1.0-probs)*np.log(np.clip(1.0-probs, eps, 1.0)))
        for c, name in enumerate(names):
            pflat = probs[:, c].reshape(-1)
            yflat = ys[:, c].reshape(-1)
            uflat = np.maximum(unc[:, c].reshape(-1), 0.0)
            n_take = min(int(max_pixels_per_batch), pflat.size)
            if n_take < pflat.size:
                # Fixed seed for reproducible diagnostic plots.
                rng = np.random.default_rng(1701 + epoch*10 + c)
                idx = rng.choice(pflat.size, size=n_take, replace=False)
                pflat, yflat, uflat = pflat[idx], yflat[idx], uflat[idx]
            sampled[name]['p'].append(pflat)
            sampled[name]['y'].append(yflat)
            sampled[name]['u'].append(uflat)
    if was_training:
        model.train()

    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    bins = np.linspace(0.0, 1.0, 16)
    all_conf, all_correct = [], []
    metric_rows = []
    for name in names:
        p = np.concatenate(sampled[name]['p'])
        y = np.concatenate(sampled[name]['y'])
        pred = (p >= 0.5).astype(np.uint8)
        conf = np.maximum(p, 1.0-p)
        correct = (pred == y).astype(np.float64)
        mean_conf, accuracy, ece = [], [], 0.0
        for b in range(len(bins)-1):
            mask = (conf >= bins[b]) & ((conf < bins[b+1]) if b < len(bins)-2 else (conf <= bins[b+1]))
            if mask.any():
                mc, ac = float(conf[mask].mean()), float(correct[mask].mean())
                mean_conf.append(mc); accuracy.append(ac)
                ece += float(mask.mean())*abs(ac-mc)
        ax.plot(mean_conf, accuracy, marker='o', markersize=4, linewidth=2,
                color=colors[name], label=f'{name} (ECE={ece:.3f})')
        all_conf.append(conf); all_correct.append(correct)
        metric_rows.append({'class':name, 'ECE_confidence_correctness':float(ece),
                            'Brier_score':float(np.mean((p-y)**2)),
                            'sampled_pixels':int(p.size)})
    ac = np.concatenate(all_conf); ay = np.concatenate(all_correct)
    avg_ece = 0.0; mean_conf=[]; accuracy=[]
    for b in range(len(bins)-1):
        mask = (ac >= bins[b]) & ((ac < bins[b+1]) if b < len(bins)-2 else (ac <= bins[b+1]))
        if mask.any():
            mc, aa = float(ac[mask].mean()), float(ay[mask].mean())
            mean_conf.append(mc); accuracy.append(aa)
            avg_ece += float(mask.mean())*abs(aa-mc)
    ax.plot(mean_conf, accuracy, marker='o', linewidth=2.5, color=colors['Average'],
            label=f'Average (ECE={avg_ece:.3f})')
    ax.plot([0,1], [0,1], '--', color=colors['reference'], linewidth=1.6, label='Perfect calibration')
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='Mean confidence', ylabel='Observed accuracy',
           title=f'Reliability Diagram — Validation Epoch {epoch}')
    ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(os.path.join(output_dir, f'reliability_epoch_{epoch:03d}.{ext}'),
                    dpi=600 if ext=='png' else None, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'reliability_latest.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.5))
    risk_rows=[]; all_errors=[]; all_unc=[]
    for name in names:
        p = np.concatenate(sampled[name]['p']); y = np.concatenate(sampled[name]['y'])
        u = np.concatenate(sampled[name]['u'])
        err = ((p >= 0.5).astype(np.uint8) != y).astype(np.float64)
        order = np.argsort(u, kind='mergesort')  # accept lowest uncertainty first
        sorted_err = err[order]
        coverage = np.arange(1, len(sorted_err)+1, dtype=np.float64)/len(sorted_err)
        risk = np.cumsum(sorted_err)/np.arange(1, len(sorted_err)+1)
        # Downsample plotted curve to avoid huge vector PDFs.
        ix = np.unique(np.linspace(0, len(coverage)-1, min(500, len(coverage))).astype(int))
        ax.plot(coverage[ix], risk[ix], color=colors[name], linewidth=1.9, label=name)
        all_errors.append(err); all_unc.append(u)
        row={'class':name}
        for cov in (0.25,0.5,0.75,1.0):
            k=max(1, int(np.ceil(cov*len(sorted_err))))
            row[f'risk_at_{int(cov*100)}pct_coverage']=float(sorted_err[:k].mean())
        row['uncertainty_source']='final_var' if isinstance(outputs, dict) and variance is not None else 'predictive_entropy'
        risk_rows.append(row)
    err=np.concatenate(all_errors); unc=np.concatenate(all_unc)
    order=np.argsort(unc, kind='mergesort'); se=err[order]
    cov=np.arange(1,len(se)+1,dtype=np.float64)/len(se); risk=np.cumsum(se)/np.arange(1,len(se)+1)
    ix=np.unique(np.linspace(0,len(cov)-1,min(500,len(cov))).astype(int))
    ax.plot(cov[ix],risk[ix],color=colors['Average'],linewidth=2.5,label='Average')
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='Coverage (fraction retained)',
           ylabel='Risk (pixel misclassification rate)',
           title=f'Risk–Coverage Curve — Validation Epoch {epoch}')
    ax.grid(alpha=0.25); ax.legend(fontsize=9); fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(os.path.join(output_dir, f'risk_coverage_epoch_{epoch:03d}.{ext}'),
                    dpi=600 if ext=='png' else None, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'risk_coverage_latest.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)

    summary={'epoch':int(epoch), 'evaluation':'validation pixel-level diagnostic; sampled pixels per batch/class',
             'reliability':{'classes':metric_rows, 'average_ECE_confidence_correctness':float(avg_ece),
                            'definition':'ECE of max(p,1-p) confidence against thresholded prediction correctness'},
             'risk_coverage':{'classes':risk_rows,
                              'average_uncertainty_source':'final_var' if isinstance(outputs, dict) and variance is not None else 'predictive_entropy',
                              'risk_definition':'pixel misclassification rate among least-uncertain retained pixels'}}
    with open(os.path.join(output_dir, f'uncertainty_metrics_epoch_{epoch:03d}.json'),'w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)
    with open(os.path.join(output_dir, 'uncertainty_metrics_latest.json'),'w',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)
    return summary


def main():
    # Setup logging
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    log_file = log_dir + '.txt'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                       handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info("="*60)
    logging.info("BAYESIAN UNCERTAINTY-AWARE EMCC MODEL TRAINING")
    logging.info("="*60)
    logging.info("Model: Bayesian EMCC Uncertainty Model")
    logging.info("Includes: Bayesian encoder, EMCC blocks, Simple decoder, Heteroscedastic heads")
    logging.info("Uncertainty: Epistemic (Bayesian) + Aleatoric (Heteroscedastic)")
    logging.info(f"ROI size: {args.roi_height}x{args.roi_width}")
    logging.info("="*60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    model = BayesianEMCCUncertaintyModel(
        out_channels=args.num_class,
        drop_rate=args.drop_rate
    )

    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Bayesian uncertainty-aware EMCC model created with {total_params:,} parameters")

    def load_paired_slice_paths(image_dir, mask_dir, split_name):
        # Pair by exact filename stem instead of relying on two independent sorted lists.
        image_files = {os.path.basename(p): p for p in glob(os.path.join(image_dir, '*.npy'))}
        mask_files = {os.path.basename(p): p for p in glob(os.path.join(mask_dir, '*.npy'))}

        missing_masks = sorted(set(image_files) - set(mask_files))
        missing_images = sorted(set(mask_files) - set(image_files))
        if missing_masks or missing_images:
            raise RuntimeError(
                f"{split_name} image/mask filenames do not match. "
                f"Images without masks (first 10): {missing_masks[:10]}; "
                f"Masks without images (first 10): {missing_images[:10]}"
            )

        names = sorted(image_files)
        if not names:
            raise RuntimeError(
                f"No .npy slices found for {split_name}. "
                f"Check image directory: {image_dir} and mask directory: {mask_dir}"
            )
        return [image_files[name] for name in names], [mask_files[name] for name in names]

    train_img_paths, train_mask_paths = load_paired_slice_paths(
        args.img_paths_root, args.mask_paths_root, "TRAIN"
    )
    val_img_paths, val_mask_paths = load_paired_slice_paths(
        args.val_img_paths_root, args.val_mask_paths_root, "VALIDATION"
    )

    def patient_id_from_slice_path(path):
        # Slice filenames are expected to be <patient_id>_<z-index>.npy.
        stem = os.path.splitext(os.path.basename(path))[0]
        patient_id, separator, slice_index = stem.rpartition('_')
        if not separator or not slice_index.isdigit():
            raise RuntimeError(
                f"Unexpected slice filename {os.path.basename(path)!r}; expected "
                "'<patient_id>_<integer_slice_index>.npy'."
            )
        return patient_id

    train_patient_ids = {patient_id_from_slice_path(p) for p in train_img_paths}
    val_patient_ids = {patient_id_from_slice_path(p) for p in val_img_paths}
    overlap = train_patient_ids & val_patient_ids
    if overlap:
        raise RuntimeError(
            "Patient leakage detected: these patients occur in both training and validation "
            f"folders: {sorted(overlap)[:20]}"
        )

    # If the preprocessing manifest is available, verify its declared split as well.
    processed_root = os.path.dirname(os.path.dirname(os.path.abspath(args.img_paths_root)))
    manifest_path = os.path.join(processed_root, 'manifests', 'patient_split.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            split_manifest = json.load(manifest_file)
        declared_train = set(split_manifest.get('train_patients', []))
        declared_val = set(split_manifest.get('validation_patients', []))
        declared_test = set(split_manifest.get('test_patients', []))
        if declared_train and not train_patient_ids.issubset(declared_train):
            unexpected = sorted(train_patient_ids - declared_train)
            raise RuntimeError(
                f"Training folder contains patient IDs absent from the manifest's training split: {unexpected[:20]}"
            )
        if declared_val and not val_patient_ids.issubset(declared_val):
            unexpected = sorted(val_patient_ids - declared_val)
            raise RuntimeError(
                f"Validation folder contains patient IDs absent from the manifest's validation split: {unexpected[:20]}"
            )
        if (train_patient_ids & declared_test) or (val_patient_ids & declared_test):
            raise RuntimeError("A held-out test patient was found in the training or validation folders.")
        logging.info(f"Validated patient split manifest: {manifest_path}")
    else:
        logging.warning(
            f"Patient split manifest not found at {manifest_path}; "
            "folder-level train/validation patient overlap was checked, but the original split "
            "manifest could not be cross-checked."
        )

    logging.info(f"Unique training patients: {len(train_patient_ids)}")
    logging.info(f"Unique validation patients: {len(val_patient_ids)}")

    print(f"Training patients: {len(train_patient_ids)}")
    print(f"Validation patients: {len(val_patient_ids)}")
    print(f"Training slices: {len(train_img_paths)}")
    print(f"Validation slices: {len(val_img_paths)}")
    print("Train sample:", train_img_paths[0])
    print("Val sample:", val_img_paths[0])
    logging.info(f"Training image directory: {args.img_paths_root}")
    logging.info(f"Training mask directory: {args.mask_paths_root}")
    logging.info(f"Validation image directory: {args.val_img_paths_root}")
    logging.info(f"Validation mask directory: {args.val_mask_paths_root}")
    logging.info("Using the patient-level split saved by preprocessing; no slice-level random split is performed.")

    train_dataset = Dataset(args, train_img_paths, train_mask_paths, aug=True)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths, aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    criterion = BCEDiceLoss()

    best_iou = 0.0
    early_stop_counter = 0

    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_ious': [],
        'val_ious': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'detailed_metrics': []
    }

    logging.info("Starting Bayesian uncertainty-aware EMCC model training...")
    training_start_time = time.time()

    for epoch in range(args.start_epoch, args.end_epoch):
        epoch_start_time = time.time()
        logging.info(f"Epoch {epoch}/{args.end_epoch}")

        train_log = train_epoch(train_loader, model, criterion, optimizer, epoch)
        val_log = validate_epoch(
            val_loader, model, criterion, epoch,
            train_size=len(train_loader.dataset)
        )
        if args.uncertainty_plot_freq > 0 and epoch % args.uncertainty_plot_freq == 0:
            try:
                uq_dir = os.path.join(
                    f'results_bayesian_uncertainty_EMCC/{args.experiment}{args.date}',
                    'uncertainty_analysis'
                )
                uq_summary = generate_reliability_and_risk_coverage(
                    val_loader, model, epoch, uq_dir, next(model.parameters()).device,
                    max_pixels_per_batch=args.uq_max_pixels_per_batch
                )
                logging.info(
                    f"[Epoch {epoch}] UQ plots saved to {uq_dir}; "
                    f"average ECE={uq_summary['reliability']['average_ECE_confidence_correctness']:.4f}"
                )
            except Exception as exc:
                logging.exception(f"Reliability/risk-coverage plotting failed at epoch {epoch}: {exc}")
        epoch_time = time.time() - epoch_start_time

        train_uncertainty = train_log.get('uncertainty_loss', 0.0)
        val_uncertainty = val_log.get('uncertainty_loss', 0.0)
        train_seg_loss = train_log.get('segmentation_loss', 0.0)
        val_seg_loss = val_log.get('segmentation_loss', 0.0)

        logging.info(f"Train - Total Loss: {train_log['loss']:.4f}, Seg Loss: {train_seg_loss:.4f}, Uncertainty Loss: {train_uncertainty:.4f}, IoU: {train_log['iou']:.4f}, Acc: {train_log['accuracy']:.4f}")
        logging.info(f"Val   - Total Loss: {val_log['loss']:.4f}, Seg Loss: {val_seg_loss:.4f}, Uncertainty Loss: {val_uncertainty:.4f}, IoU: {val_log['iou']:.4f}, Acc: {val_log['accuracy']:.4f}")
        logging.info(f"Epoch time: {epoch_time:.2f}s")

        if epoch > 0 and epoch % 50 == 0:
            print(f"\n{'='*95}")
            print(f"TRAINING PROGRESS - EPOCH {epoch}")
            print(f"{'='*95}")
            print(f"{'Metric':<20} {'Train':<15} {'Validation':<15} {'Best So Far':<15}")
            print(f"{'-'*95}")
            print(f"{'Total Loss':<20} {train_log['loss']:<15.4f} {val_log['loss']:<15.4f} {min(training_history['val_losses']) if training_history['val_losses'] else 0.0:<15.4f}")
            print(f"{'Segmentation Loss':<20} {train_seg_loss:<15.4f} {val_seg_loss:<15.4f} {'N/A':<15}")
            print(f"{'Uncertainty Loss':<20} {train_uncertainty:<15.4f} {val_uncertainty:<15.4f} {'N/A':<15}")
            print(f"{'IoU':<20} {train_log['iou']:<15.4f} {val_log['iou']:<15.4f} {max(training_history['val_ious']) if training_history['val_ious'] else 0.0:<15.4f}")
            print(f"{'Accuracy':<20} {train_log['accuracy']:<15.4f} {val_log['accuracy']:<15.4f} {max(training_history['val_accuracies']) if training_history['val_accuracies'] else 0.0:<15.4f}")
            print(f"{'='*95}\n")

        training_history['train_losses'].append(train_log['loss'])
        training_history['val_losses'].append(val_log['loss'])
        training_history['train_ious'].append(train_log['iou'])
        training_history['val_ious'].append(val_log['iou'])
        training_history['train_accuracies'].append(train_log['accuracy'])
        training_history['val_accuracies'].append(val_log['accuracy'])

        eval_condition = (epoch % args.eval_freq == 0) or (epoch < 10 and epoch % 2 == 0) or epoch == 0
        if eval_condition:
            logging.info(f"\n--- Comprehensive Metrics Evaluation (Epoch {epoch}) ---")
            try:
                # Set model to eval mode for consistent evaluation
                model.eval()
                
                # Calculate training metrics over multiple batches for better representation
                train_metrics_list = []
                train_batch_count = 0
                max_train_batches = min(5, len(train_loader))  # Diagnostic subset only; not a full-train metric
                
                with torch.no_grad():
                    for train_batch in train_loader:
                        if train_batch_count >= max_train_batches:
                            break
                        train_input, train_target = train_batch
                        train_input = train_input.cuda()
                        train_target = train_target.cuda()
                        train_output = model(train_input)
                        train_batch_metrics = calculate_comprehensive_metrics(train_output, train_target)
                        train_metrics_list.append(train_batch_metrics)
                        train_batch_count += 1
                    
                    # Average training metrics across batches
                    train_metrics = {}
                    labels = ['WT', 'TC', 'ET', 'average']
                    if train_metrics_list:
                        for label in labels:
                            train_metrics[label] = {}
                            for metric_name in train_metrics_list[0][label].keys():
                                values = [m[label][metric_name] for m in train_metrics_list]
                                train_metrics[label][metric_name] = float(np.nanmean(values))
                    else:
                        logging.warning("No training batches available for detailed metrics.")
                
                # Calculate validation metrics (full validation set for consistency)
                val_metrics_list = []
                max_val_batches = len(val_loader)  # Evaluate all validation batches
                val_batch_count = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        if val_batch_count >= max_val_batches:
                            break
                        val_input, val_target = val_batch
                        val_input = val_input.cuda()
                        val_target = val_target.cuda()
                        val_output = model(val_input)
                        val_batch_metrics = calculate_comprehensive_metrics(val_output, val_target)
                        val_metrics_list.append(val_batch_metrics)
                        val_batch_count += 1
                    
                    # Average validation metrics across batches
                    val_metrics = {}
                    if val_metrics_list:
                        for label in labels:
                            val_metrics[label] = {}
                            for metric_name in val_metrics_list[0][label].keys():
                                values = [m[label][metric_name] for m in val_metrics_list]
                                val_metrics[label][metric_name] = float(np.nanmean(values))
                    else:
                        logging.warning("No validation batches available for detailed metrics.")
                
                # Set model back to train mode for next epoch
                model.train()
                
                # Show metrics for both training and validation only when available.
                if not train_metrics or not val_metrics:
                    logging.warning("Skipping detailed metric logging because a metric list is empty.")
                    continue
                logging.info("TRAINING METRICS:")
                for label, metrics in train_metrics.items():
                    if label != 'average':
                        logging.info(f"  {label}: Dice={metrics['dice']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
                avg_train = train_metrics['average']
                logging.info(f"  AVG: Dice={avg_train['dice']:.3f}, F1={avg_train['f1']:.3f}, AUC={avg_train['auc']:.3f}")
                
                logging.info("VALIDATION METRICS:")
                for label, metrics in val_metrics.items():
                    if label != 'average':
                        logging.info(f"  {label}: Dice={metrics['dice']:.3f}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
                avg_val = val_metrics['average']
                logging.info(f"  AVG: Dice={avg_val['dice']:.3f}, F1={avg_val['f1']:.3f}, AUC={avg_val['auc']:.3f}")
                
                # Store both training and validation metrics
                detailed_metrics = {
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                }
                training_history['detailed_metrics'].append(detailed_metrics)
            except Exception as e:
                logging.warning(f"[WARNING] Could not calculate detailed metrics at epoch {epoch}: {e}")

        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            early_stop_counter = 0
            checkpoint_dir = f'checkpoint_bayesian_uncertainty_EMCC/{args.experiment}{args.date}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'training_history': training_history,
                'args': args
            }, os.path.join(checkpoint_dir, f'best_model.pth'))
            logging.info(f"*** New best model saved! IoU: {best_iou:.4f} ***")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop:
            logging.info(f"Early stopping after {epoch} epochs (no improvement for {args.early_stop} epochs)")
            break

        if epoch % args.save_freq == 0 and epoch > 0:
            checkpoint_dir = f'checkpoint_bayesian_uncertainty_EMCC/{args.experiment}{args.date}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'training_history': training_history,
                'args': args
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))
            logging.info(f"Checkpoint saved at epoch {epoch}")

    total_training_time = time.time() - training_start_time

    results_dir = f'results_bayesian_uncertainty_EMCC/{args.experiment}{args.date}'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2, default=str)

    plots_dir = os.path.join(results_dir, 'plots')
    logging.info("Creating comprehensive metric plots...")
    try:
        create_comprehensive_plots(training_history, plots_dir)
        logging.info(f"[SUCCESS] Plots successfully saved to: {plots_dir}")
    except Exception as e:
        logging.warning(f"[WARNING] Could not create plots: {e}")
        print(f"Plot creation error: {e}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED - DISPLAYING RESULTS TABLE")
    print("="*60)
    format_training_results_table(training_history, best_iou, total_params, total_training_time)

    csv_file = os.path.join(results_dir, f'training_summary_{args.date}.csv')
    create_csv_training_summary(training_history, best_iou, total_params, total_training_time, csv_file)
    print(f"[SUCCESS] CSV training summary saved to: {csv_file}")

    logging.info("="*60)
    logging.info("BAYESIAN UNCERTAINTY-AWARE EMCC MODEL TRAINING COMPLETED")
    logging.info("="*60)
    logging.info(f"Total training time: {total_training_time/3600:.2f} hours")
    logging.info(f"Best validation IoU: {best_iou:.4f}")
    logging.info(f"Final epoch: {epoch}")
    logging.info(f"Model parameters: {total_params:,}")
    logging.info(f"Results saved to: {results_dir}")
    logging.info(f"Training plots saved to: {plots_dir}")
    logging.info(f"CSV summary saved to: {csv_file}")
    logging.info(f"Best model saved to: checkpoint_bayesian_uncertainty_EMCC/{args.experiment}{args.date}/best_model.pth")
    logging.info("="*60)

    summary = {
        'model': 'Bayesian Uncertainty-Aware EMCC Model',
        'model_type': 'bayesian_uncertainty_EMCC',
        'included_components': [
            'Bayesian encoder with epistemic uncertainty',
            'EMCC blocks for feature refinement',
            'Heteroscedastic heads for aleatoric uncertainty',
            'Multi-scale uncertainty predictions'
        ],
        'total_params': total_params,
        'best_iou': best_iou,
        'total_training_time_hours': total_training_time/3600,
        'epochs_trained': epoch + 1,
        'final_train_loss': training_history['train_losses'][-1] if training_history['train_losses'] else 0,
        'final_val_loss': training_history['val_losses'][-1] if training_history['val_losses'] else 0,
        'config': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'roi_size': f"{args.roi_height}x{args.roi_width}"
        },
        'uncertainty_info': {
            'epistemic_uncertainty': 'Bayesian encoder layers',
            'aleatoric_uncertainty': 'Heteroscedastic heads',
            'multi_scale_uncertainty': 'Encoder + decoder uncertainty heads'
        },
        'detailed_metrics': training_history['detailed_metrics'][-1] if training_history['detailed_metrics'] else {}
    }
    with open(os.path.join(results_dir, 'bayesian_uncertainty_EMCC_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Bayesian uncertainty EMCC summary saved to: {results_dir}/bayesian_uncertainty_EMCC_summary.json")
    logging.info("\nBayesian uncertainty-aware EMCC model ready for uncertainty analysis!")

if __name__ == '__main__':
    main()
