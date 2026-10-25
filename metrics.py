import numpy as np

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

# Try to import medpy for HD95 calculation
try:
    from medpy.metric import binary
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("Warning: medpy not available. HD95 calculation will be skipped.")


def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0


def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def threshold_mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)



def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)

def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

def specificity(output, target):
    """Calculate specificity (true negative rate)"""
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    
    # True negatives and false positives
    true_negatives = ((~output_) & (~target_)).sum()
    false_positives = (output_ & (~target_)).sum()
    
    return (true_negatives + smooth) / \
        (true_negatives + false_positives + smooth)

def hausdorff_distance_95(output, target):
    """
    Calculate the 95th percentile of the Hausdorff Distance using medpy
    
    Args:
        output: Predicted segmentation (binary)
        target: Ground truth segmentation (binary)
    
    Returns:
        HD95 value (lower is better)
    """
    if not MEDPY_AVAILABLE:
        print("Warning: medpy not available. Returning 1000.0 for HD95.")
        return 1000.0
    
    try:
        # Ensure numpy arrays
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # Ensure uint8 type
        output = output.astype(np.uint8)
        target = target.astype(np.uint8)
        
        # Convert to binary if needed (in case we get probability values)
        if output.max() > 1:
            output = (output > 0.5).astype(np.uint8)
        if target.max() > 1:
            target = (target > 0.5).astype(np.uint8)
        
        # Check if masks are empty or contain no binary objects
        if output.sum() == 0 or target.sum() == 0:
            return 1000.0  # Large penalty for empty masks
        
        # Additional check: ensure masks contain at least some foreground pixels
        if output.max() == 0 or target.max() == 0:
            return 1000.0
        
        # Use medpy's HD95 calculation
        hd95_value = binary.hd95(output, target)
        return float(hd95_value)
        
    except Exception as e:
        # Return large value for any errors (empty masks, etc.)
        return 1000.0

def compute_brain_tumor_metrics(output, target):
    """
    Compute comprehensive metrics for brain tumor segmentation
    
    Args:
        output: Model predictions (batch_size, 3, H, W) for WT, TC, ET
        target: Ground truth (batch_size, 3, H, W) for WT, TC, ET
    
    Returns:
        Dictionary with metrics for each label
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # Threshold predictions and convert to uint8
    output = (output > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    
    metrics = {}
    labels = ['WT', 'TC', 'ET']
    
    for i, label in enumerate(labels):
        if len(output.shape) == 4:  # Batch dimension
            # Compute metrics for each sample in batch
            dice_scores = []
            sens_scores = []
            spec_scores = []
            hd95_scores = []
            
            for j in range(output.shape[0]):
                pred = output[j, i, :, :].astype(np.uint8)
                gt = target[j, i, :, :].astype(np.uint8)
                
                # Dice coefficient
                dice = dice_coef(pred, gt)
                dice_scores.append(dice)
                
                # Sensitivity
                sens = sensitivity(pred, gt)
                sens_scores.append(sens)
                
                # Specificity
                spec = specificity(pred, gt)
                spec_scores.append(spec)
                
                # HD95 - ensure proper data types
                hd95 = hausdorff_distance_95(pred, gt)
                hd95_scores.append(hd95)
            
            metrics[label] = {
                'dice': np.array(dice_scores),
                'sensitivity': np.array(sens_scores),
                'specificity': np.array(spec_scores),
                'hd95': np.array(hd95_scores)
            }
        else:  # Single sample
            pred = output[i, :, :].astype(np.uint8)
            gt = target[i, :, :].astype(np.uint8)
            
            metrics[label] = {
                'dice': dice_coef(pred, gt),
                'sensitivity': sensitivity(pred, gt),
                'specificity': specificity(pred, gt),
                'hd95': hausdorff_distance_95(pred, gt)
            }
    
    return metrics

def aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple batches"""
    if not metrics_list:
        return {}
    
    labels = ['WT', 'TC', 'ET']
    metric_names = ['dice', 'sensitivity', 'specificity', 'hd95', 'hausdorff']
    
    aggregated = {}
    for label in labels:
        aggregated[label] = {}
        for metric_name in metric_names:
            # Collect all values for this label and metric
            values = []
            for batch_metrics in metrics_list:
                if label in batch_metrics and metric_name in batch_metrics[label]:
                    batch_values = batch_metrics[label][metric_name]
                    if isinstance(batch_values, np.ndarray):
                        values.extend(batch_values.tolist())
                    else:
                        values.append(batch_values)
            
            if values:
                aggregated[label][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            else:
                aggregated[label][metric_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'values': []
                }
    
    return aggregated

def format_results_table(train_metrics, val_metrics, test_metrics):
    """Format results into a table similar to the user's example"""
    
    def format_value(mean, std):
        return f"{mean:.3f} ± {std:.3f}"
    
    print("\n" + "="*100)
    print("COMPREHENSIVE BRAIN TUMOR SEGMENTATION RESULTS")
    print("="*100)
    print(f"{'Label':<5} {'Metric':<8} {'Training Results':<18} {'Validation Results':<20} {'Test Results':<15}")
    print(f"{'':5} {'':8} {'(Mean ± SD)':<18} {'(Mean ± SD)':<20} {'(Mean ± SD)':<15}")
    print("-"*100)
    
    labels = ['WT', 'TC', 'ET']
    metrics = ['dice', 'sensitivity', 'specificity', 'hd95']
    metric_names = ['Dice', 'Sen', 'Spec', 'HD95']
    
    for label in labels:
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            # Get values for each phase
            train_val = train_metrics.get(label, {}).get(metric, {'mean': 0, 'std': 0})
            val_val = val_metrics.get(label, {}).get(metric, {'mean': 0, 'std': 0})
            test_val = test_metrics.get(label, {}).get(metric, {'mean': 0, 'std': 0})
            
            # Format row
            if i == 0:  # First metric for this label
                label_str = label
            else:
                label_str = ""
            
            print(f"{label_str:<5} {metric_name:<8} {format_value(train_val['mean'], train_val['std']):<18} "
                  f"{format_value(val_val['mean'], val_val['std']):<20} "
                  f"{format_value(test_val['mean'], test_val['std']):<15}")
    
    print("="*100)
    print("")