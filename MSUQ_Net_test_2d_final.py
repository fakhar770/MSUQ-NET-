#!/usr/bin/env python3
"""MSUQ-Net: held-out 2D BraTS slice evaluation and Bayesian MC-30 uncertainty.

Run from the repository root:
    python test_2d.py --model_path /path/to/best_model.pth

Requires processed_data/testImage and processed_data/testMask made with the SAME
patient-level split and preprocessing used by the trained checkpoint. No fallback
slice-level random split is permitted. This script computes SLICE-LEVEL metrics.

Model input channels: FLAIR, T1, T1CE, T2. Model outputs: WT, TC, ET.
Model weights in BayesianConv2d are sampled in eval mode (blitz Bayesian layers).
Ordinary dropout and BatchNorm remain in evaluation mode.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from models.model import BayesianEMCCUncertaintyModel

REGIONS = ("WT", "TC", "ET")
MODALITIES = ("FLAIR", "T1", "T1CE", "T2")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testImage_paths_root", default="processed_data/testImage")
    p.add_argument("--testMask_paths_root", default="processed_data/testMask")
    p.add_argument("--model_path", required=True, help="Trusted checkpoint from the matching trained model")
    p.add_argument("--output_dir", default="evaluation_results_2d")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mc_samples", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--drop_rate", type=float, default=0.20, help="Match checkpoint training configuration")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_visuals", action="store_true")
    p.add_argument("--max_visuals", type=int, default=20, help="Number of 2D figures to save; -1 means all")
    p.add_argument("--save_uq_curves", action="store_true")
    p.add_argument("--visual_dpi", type=int, default=180, help="Figure DPI (180 default; 200 for manuscript)")
    p.add_argument("--uq_pixels_per_batch", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1000)
    return p.parse_args()


def paired_files(img_root, mask_root):
    images = sorted(Path(img_root).glob("*.npy"))
    masks = sorted(Path(mask_root).glob("*.npy"))
    if not images or not masks:
        raise FileNotFoundError("Missing held-out testImage/testMask .npy files. Refusing to split training slices for testing.")
    ids_i = {p.stem: p for p in images}
    ids_m = {p.stem: p for p in masks}
    if len(ids_i) != len(images) or len(ids_m) != len(masks):
        raise ValueError("Duplicate image or mask stems.")
    if ids_i.keys() != ids_m.keys():
        raise ValueError(f"Unpaired test files: images only={sorted(ids_i.keys()-ids_m.keys())[:5]}, masks only={sorted(ids_m.keys()-ids_i.keys())[:5]}")
    ids = sorted(ids_i)
    return [str(ids_i[k]) for k in ids], [str(ids_m[k]) for k in ids], ids


def load_model(path, device, drop_rate):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint required, but not found: {path}")
    # Only load checkpoints from a trusted source: weights_only=False can unpickle code.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("Safe checkpoint loading failed. Use a trusted, compatible PyTorch checkpoint, or inspect its serialization separately.") from exc
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state, epoch = checkpoint, "unknown"
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a model state_dict")
    if state and all(str(k).startswith("module.") for k in state):
        state = {k[7:]: value for k, value in state.items()}
    model = BayesianEMCCUncertaintyModel(out_channels=3, drop_rate=drop_rate)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint strictly: {path}, epoch={epoch}")
    return model


@torch.inference_mode()
def mc_inference(model, x, samples):
    """Mean sigmoid probability, logit sample variance (unbiased), and mean predicted variance."""
    if samples < 2:
        raise ValueError("At least two MC passes are required for unbiased epistemic variance")
    model.eval()  # keep regular dropout OFF and BatchNorm on stored statistics
    sum_prob = sum_mu = sum_mu2 = sum_var = None
    for _ in range(samples):
        output = model(x)
        if not isinstance(output, dict) or not {'final_mean', 'final_var'} <= output.keys():
            raise RuntimeError("Model must return dict with final_mean and final_var")
        mu = output['final_mean'].float()
        var = output['final_var'].float()
        if mu.shape != var.shape or mu.shape[1] != 3:
            raise RuntimeError(f"Unexpected uncertainty-head shapes {mu.shape}, {var.shape}")
        prob = torch.sigmoid(mu)
        if sum_prob is None:
            sum_prob, sum_mu, sum_mu2, sum_var = prob, mu, mu.square(), var
        else:
            sum_prob = sum_prob + prob
            sum_mu = sum_mu + mu
            sum_mu2 = sum_mu2 + mu.square()
            sum_var = sum_var + var
    mean_mu = sum_mu / samples
    epi = ((sum_mu2 - samples * mean_mu.square()) / (samples - 1)).clamp_min(0)
    return {'probs': sum_prob / samples, 'epistemic': epi, 'aleatoric': sum_var / samples}


def surface_metrics_2d(pred, gt):
    """(HD95, ASSD) in pixels; undefined when exactly one mask is empty."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return float('nan'), float('nan')
    conn = ndi.generate_binary_structure(2, 1)
    ps = pred ^ ndi.binary_erosion(pred, structure=conn, border_value=0)
    gs = gt ^ ndi.binary_erosion(gt, structure=conn, border_value=0)
    dp = ndi.distance_transform_edt(~gs)[ps]
    dg = ndi.distance_transform_edt(~ps)[gs]
    # Symmetric pooled 95th percentile and mean directed surface distance.
    return float(np.percentile(np.concatenate((dp, dg)), 95)), float((dp.sum()+dg.sum())/(dp.size+dg.size))


def binary_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    dice = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    iou = tp/(tp+fp+fn) if tp+fp+fn else 1.0
    sens = tp/(tp+fn) if tp+fn else float('nan')
    spec = tn/(tn+fp) if tn+fp else float('nan')
    precision = tp/(tp+fp) if tp+fp else float('nan')
    hd95, assd = surface_metrics_2d(pred, gt)
    rvd = (int(pred.sum())-int(gt.sum()))/int(gt.sum()) if gt.any() else float('nan')
    return dict(Dice=dice, IoU=iou, Sensitivity=sens, Specificity=spec,
                Precision=precision, HD95_px=hd95, ASSD_px=assd,
                VOE=1-iou, RVD=rvd, TP=tp, FP=fp, FN=fn, TN=tn)


def ece_binary(prob, target, bins=15, threshold=0.5):
    """Prediction-confidence ECE using the selected classification threshold."""
    prob = np.asarray(prob).reshape(-1)
    target = np.asarray(target, bool).reshape(-1)
    pred = prob >= threshold
    conf = np.where(pred, prob, 1-prob)
    correct = pred == target
    edges = np.linspace(0, 1, bins+1)
    score = 0.0
    for i in range(bins):
        choose = (conf >= edges[i]) & (conf <= edges[i+1] if i == bins-1 else conf < edges[i+1])
        if choose.any():
            score += float(choose.mean())*abs(float(correct[choose].mean())-float(conf[choose].mean()))
    return score


def colored_regions(mask):
    """Composite WT yellow, TC green, ET blue, with nested-region priority."""
    h, w = mask.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[mask[0] > .5] = (1.0, 1.0, 0.0)
    rgb[mask[1] > .5] = (0.0, 1.0, 0.0)
    rgb[mask[2] > .5] = (0.0, 0.0, 1.0)
    return rgb


def normalized_mri(channel):
    arr = np.asarray(channel, np.float32)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def add_visual(path, inp, gt, pred, epi, ale, probs, sample_label="", dpi=180):
    """Original comprehensive 4x6 publication-style 2D visualization.

    Row 1: 4 MRI sequences.
    Row 2: 3 ground-truth masks, combined mask, legend.
    Row 3: 3 predicted masks, combined prediction, T1CE overlay.
    Row 4: epistemic/aleatoric/total uncertainty, error, error-vs-uncertainty, Dice.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f"Sample {sample_label}: Image, Prediction, Ground Truth, Uncertainty", fontsize=15, fontweight='bold')
    axes = [fig.add_subplot(4, 6, i + 1) for i in range(24)]
    for ax in axes:
        ax.axis('off')
    # Presentation order matches the supplied reference figure. Model input order
    # remains unchanged: FLAIR, T1, T1CE, T2.
    display_indices = (1, 2, 3, 0)  # T1, T1CE, T2, FLAIR
    for panel, channel in enumerate(display_indices):
        axes[panel].imshow(normalized_mri(inp[channel]), cmap='gray', vmin=0, vmax=1)
        axes[panel].set_title(MODALITIES[channel], fontweight='bold')
    titles = ('WT (Whole)', 'TC (Core)', 'ET (Enhancing)')
    gt_cmaps = ('YlOrBr', 'Greens', 'Blues')
    pred_cmaps = ('YlOrBr', 'Reds', 'Blues')
    for c in range(3):
        axes[6+c].imshow(gt[c], cmap=gt_cmaps[c], vmin=0, vmax=1)
        axes[6+c].set_title('GT: ' + titles[c], fontsize=11, fontweight='bold')
        axes[12+c].imshow(pred[c], cmap=pred_cmaps[c], vmin=0, vmax=1)
        axes[12+c].set_title('Pred: ' + titles[c], fontsize=11, fontweight='bold')
    axes[9].imshow(colored_regions(gt))
    axes[9].set_title('GT: Combined', fontweight='bold')
    axes[10].text(.05, .8, '■ WT (Yellow)', color='gold', transform=axes[10].transAxes, fontweight='bold')
    axes[10].text(.05, .5, '■ TC (Green)', color='green', transform=axes[10].transAxes, fontweight='bold')
    axes[10].text(.05, .2, '■ ET (Blue)', color='blue', transform=axes[10].transAxes, fontweight='bold')
    axes[15].imshow(colored_regions(pred))
    axes[15].set_title('Pred: Combined', fontweight='bold')
    base = normalized_mri(inp[2])  # actual modality order FLAIR, T1, T1CE, T2
    base_rgb = np.stack([base]*3, axis=-1)
    color_seg = colored_regions(pred)
    inside = np.any(pred, axis=0)
    base_rgb[inside] = base_rgb[inside]*.5 + color_seg[inside]*.5
    axes[16].imshow(np.clip(base_rgb, 0, 1))
    axes[16].set_title('Overlay (T1CE)', fontweight='bold')
    epi_combined = epi.mean(axis=0)
    ale_combined = ale.mean(axis=0)
    total = epi_combined + ale_combined
    error = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=0)
    for loc, arr, name, cm in [
        (18, epi_combined, 'Epistemic Unc.\n(MC Variance)', 'hot'),
        (19, ale_combined, 'Aleatoric Unc.\n(σ² map)', 'hot'),
        (20, total, 'Total Uncertainty\n(Epi + Ale)', 'hot'),
        (21, error, 'Prediction Error', 'Reds')]:
        im = axes[loc].imshow(arr, cmap=cm)
        axes[loc].set_title(name, fontweight='bold', fontsize=10)
        fig.colorbar(im, ax=axes[loc], fraction=.046, pad=.04)
    norm_u = (total-total.min())/(np.ptp(total)+1e-8)
    norm_e = (error-error.min())/(np.ptp(error)+1e-8)
    axes[22].imshow(np.stack([norm_e, norm_u, np.zeros_like(norm_e)], axis=-1))
    axes[22].set_title('Error(R) vs Unc(G)', fontweight='bold', fontsize=10)
    dice_scores = [binary_metrics(pred[c], gt[c])['Dice'] for c in range(3)]
    axes[23].text(.02,.82,f'Dice WT: {dice_scores[0]:.3f}',transform=axes[23].transAxes,fontweight='bold')
    axes[23].text(.02,.62,f'Dice TC: {dice_scores[1]:.3f}',transform=axes[23].transAxes,fontweight='bold')
    axes[23].text(.02,.42,f'Dice ET: {dice_scores[2]:.3f}',transform=axes[23].transAxes,fontweight='bold')
    axes[23].text(.02,.20,f'Avg: {np.mean(dice_scores):.3f}',transform=axes[23].transAxes,fontweight='bold',color='green')
    fig.tight_layout(rect=(0,0,1,.975), h_pad=1.6, w_pad=1.1)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def error_uncertainty_correlation(p, y, uncertainty):
    """Pearson corr(|predictive-probability error|, epistemic variance)."""
    err = np.abs(np.asarray(p, float).ravel() - np.asarray(y, float).ravel())
    unc = np.asarray(uncertainty, float).ravel()
    if np.std(err) == 0 or np.std(unc) == 0:
        return float('nan')
    return float(np.corrcoef(err, unc)[0,1])


def draw_uq_curves(samples, out_dir, threshold=0.5):
    import matplotlib.pyplot as plt
    for region in REGIONS:
        data = samples[region]
        if not data['p']:
            continue
        p = np.concatenate(data['p']); y = np.concatenate(data['y']); u = np.concatenate(data['u'])
        pred = p >= threshold
        confidence = np.where(pred, p, 1-p)
        correct = pred == y
        fig, ax = plt.subplots(figsize=(6, 5))
        bins = np.linspace(0, 1, 16)
        xx, yy = [], []
        for j in range(15):
            m = (confidence >= bins[j]) & (confidence <= bins[j+1] if j==14 else confidence < bins[j+1])
            if m.any(): xx.append(float(confidence[m].mean())); yy.append(float(correct[m].mean()))
        ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
        ax.plot(xx, yy, 'o-', label=region)
        ax.set(xlabel='Confidence', ylabel='Correctness', xlim=(0, 1), ylim=(0, 1), title=region+' reliability')
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir/f'{region}_reliability.png', dpi=180); plt.close(fig)
        err = (~correct).astype(float)
        order = np.argsort(u)
        risk = np.cumsum(err[order])/np.arange(1,len(err)+1)
        coverage = np.arange(1,len(err)+1)/len(err)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(coverage, risk)
        ax.set(xlabel='Coverage (fraction of lowest-uncertainty pixels)', ylabel='Misclassification risk',
               title=region+' risk–coverage')
        fig.tight_layout(); fig.savefig(out_dir/f'{region}_risk_coverage.png', dpi=180); plt.close(fig)


def main():
    args = parse_args()
    if args.mc_samples < 2 or args.batch_size < 1 or not 0 < args.threshold < 1:
        raise ValueError('Require mc_samples >= 2, batch_size >= 1, 0 < threshold < 1')
    if args.visual_dpi < 1:
        raise ValueError('visual_dpi must be positive')
    if args.save_uq_curves and args.uq_pixels_per_batch < 1:
        raise ValueError('uq_pixels_per_batch must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    imgpaths, maskpaths, ids = paired_files(args.testImage_paths_root, args.testMask_paths_root)
    dataset = Dataset(args, imgpaths, maskpaths, aug=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                        pin_memory=(device.type=='cuda'))
    model = load_model(args.model_path, device, args.drop_rate)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.save_visuals: (out/'visualizations').mkdir(exist_ok=True)
    if args.save_uq_curves: (out/'uncertainty_analysis').mkdir(exist_ok=True)
    cache = {r:{'p': [], 'y': [], 'u': []} for r in REGIONS}
    rows=[]; offset=0; image_count=0
    print(f'Evaluating {len(dataset)} held-out 2D slices, MC samples={args.mc_samples}')
    for inputs, targets in tqdm(loader, desc='2D test'):
        inputs=inputs.to(device).float()
        targets=targets.to(device).float()
        output=mc_inference(model, inputs, args.mc_samples)
        probs=output['probs'].cpu().numpy()
        epi=output['epistemic'].cpu().numpy()
        ale=output['aleatoric'].cpu().numpy()
        gt=targets.cpu().numpy() >= .5
        inp=inputs.cpu().numpy()
        for b in range(len(probs)):
            sid=ids[offset+b]
            for c, region in enumerate(REGIONS):
                p=probs[b,c]; y=gt[b,c]; pred=p >= args.threshold
                row={'slice_id':sid,'region':region, **binary_metrics(pred,y)}
                row.update(EpistemicVar=float(epi[b,c].mean()), AleatoricVar=float(ale[b,c].mean()),
                           PredictiveEntropy=float(np.mean(-p*np.log(np.clip(p,1e-7,1))-(1-p)*np.log(np.clip(1-p,1e-7,1)))),
                           Brier=float(np.mean((p-y)**2)),
                           NLL=float(np.mean(-y*np.log(np.clip(p,1e-7,1-1e-7))-(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))),
                           ECE=ece_binary(p,y),
                           UCC=error_uncertainty_correlation(p,y,epi[b,c]),
                           F1=binary_metrics(pred,y)['Dice'],
                           Recall=binary_metrics(pred,y)['Sensitivity'])
                if np.unique(y).size==2:
                    row['AUROC']=float(roc_auc_score(y.reshape(-1),p.reshape(-1)))
                else: row['AUROC']=float('nan')
                error=(pred!=y).reshape(-1)
                if np.unique(error).size==2:
                    row['ErrorDetectionAUROC']=float(roc_auc_score(error,epi[b,c].reshape(-1)))
                else: row['ErrorDetectionAUROC']=float('nan')
                rows.append(row)
                if args.save_uq_curves:
                    total=p.size
                    selected=rng.choice(total, min(total,args.uq_pixels_per_batch), replace=False)
                    cache[region]['p'].append(p.reshape(-1)[selected]); cache[region]['y'].append(y.reshape(-1)[selected]);
                    cache[region]['u'].append((epi[b,c]+ale[b,c]).reshape(-1)[selected])
            if args.save_visuals and (args.max_visuals < 0 or image_count < args.max_visuals):
                add_visual(out/'visualizations'/f'sample_{image_count:04d}_{sid}.png', inp[b], gt[b], probs[b]>=args.threshold, epi[b], ale[b], probs[b], sample_label=f'{image_count+1} ({sid})', dpi=args.visual_dpi)
                image_count+=1
        offset+=len(probs)
    if not rows: raise RuntimeError('No test slices were evaluated')
    with (out/'per_slice_2d_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics=[k for k in rows[0] if k not in ('slice_id','region','TP','FP','FN','TN')]
    summary=[]
    for region in REGIONS:
        sub=[r for r in rows if r['region']==region]
        for metric in metrics:
            values=np.array([r[metric] for r in sub],float)
            valid=values[np.isfinite(values)]
            summary.append(dict(region=region,metric=metric,mean=float(valid.mean()) if valid.size else float('nan'),
                                std=float(valid.std(ddof=1)) if valid.size>1 else (0.0 if valid.size==1 else float('nan')),
                                n_defined=int(valid.size),n_slices=len(sub)))
    with (out/'summary_2d_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    report={'evaluation':'2D slice-level', 'checkpoint':str(args.model_path), 'n_slices':len(ids),
            'mc_samples':args.mc_samples,'threshold':args.threshold,'modality_order':list(MODALITIES),
            'model': 'BayesianEMCCUncertaintyModel',
            'prediction':'mean(sigmoid(stochastic logits))',
            'epistemic':'unbiased variance of stochastic logits',
            'aleatoric':'mean predicted variance',
            'dropout':'disabled: model.eval()',
            'HD95_ASSD':'2D pixel distances; one-empty undefined and excluded from summary',
            'split_provenance_verified':False,
            'warning':'Check that this checkpoint was trained without exposure to these held-out patients.',
            'UCC':'Pearson correlation of absolute probability error with epistemic variance',
            'NLL':'Bernoulli NLL of posterior-mean predictive probability (not inverse-variance weighted)',
            'VOE':'1 - IoU (true volumetric overlap error)',
            'summary':summary}
    # allow_nan=False ensures portable standards-compliant JSON
    def clean(obj):
        if isinstance(obj,float) and not np.isfinite(obj): return None
        if isinstance(obj,list): return [clean(v) for v in obj]
        if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
        return obj
    (out/'summary_2d_metrics.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding='utf-8')
    if args.save_uq_curves: draw_uq_curves(cache,out/'uncertainty_analysis',threshold=args.threshold)
    print('Done:',out/'per_slice_2d_metrics.csv',out/'summary_2d_metrics.csv')
    for region in REGIONS:
        d=next(r for r in summary if r['region']==region and r['metric']=='Dice')
        print(f"{region} 2D Dice: {d['mean']:.4f} ± {d['std']:.4f}, n={d['n_defined']}")

if __name__=='__main__':
    main()
