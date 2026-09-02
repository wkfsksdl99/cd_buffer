import math
import os
import random
import types
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_cityscapes import CityscapesDataset
from dataset_kitti import KITTIDataset
from model_frcnn_improved import create_model_improved
from utils.metrics import ap_per_class
class SaveOutputPreHook:

    def __init__(self):
        self.outputs = []

    def __call__(self, module, input):
        if isinstance(input, tuple) and len(input) == 1:
            self.outputs.append(input[0].clone())
        else:
            self.outputs.append(input.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        if not self.outputs:
            return None
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        if not self.outputs:
            return None
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=0)
        return out

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_batch_map50(model, images, targets, device, conf_thres=0.25):
    model.eval()
    iou_thresholds = torch.tensor([0.5], device=device)
    stats = []
    with torch.no_grad():
        if not isinstance(images, list):
            images = [images]
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            labels = target['labels'].cpu().numpy()
            ground_truth_boxes = target['boxes'].cpu().numpy()
            target_count = len(labels)
            target_classes = labels.tolist() if target_count else []
            scores = output['scores']
            keep = scores > conf_thres
            boxes = output['boxes'][keep]
            predicted_classes = output['labels'][keep]
            scores = scores[keep]

            if len(boxes) == 0:
                if target_count:
                    stats.append((
                        torch.zeros(0, 1, dtype=torch.bool),
                        torch.tensor([]),
                        torch.tensor([]),
                        target_classes,
                    ))
                continue

            correct = torch.zeros(
                boxes.shape[0], 1, dtype=torch.bool, device=device
            )
            if target_count:
                detected = set()
                target_class_tensor = torch.as_tensor(
                    labels, dtype=torch.float, device=device
                )
                target_boxes = torch.as_tensor(
                    ground_truth_boxes, dtype=torch.float, device=device
                )
                for class_id in torch.unique(target_class_tensor):
                    target_indices = (
                        class_id == target_class_tensor
                    ).nonzero(as_tuple=False).view(-1)
                    prediction_indices = (
                        class_id == predicted_classes
                    ).nonzero(as_tuple=False).view(-1)
                    if prediction_indices.numel() == 0:
                        continue
                    ious, matches = torchvision.ops.box_iou(
                        boxes[prediction_indices],
                        target_boxes[target_indices],
                    ).max(1)
                    for match_index in (
                        ious > iou_thresholds[0]
                    ).nonzero(as_tuple=False):
                        target_index = target_indices[matches[match_index]]
                        if target_index.item() in detected:
                            continue
                        detected.add(target_index.item())
                        correct[prediction_indices[match_index]] = (
                            ious[match_index] > iou_thresholds
                        )
                        if len(detected) == target_count:
                            break
            stats.append((
                correct.cpu(),
                scores.cpu(),
                predicted_classes.cpu(),
                target_classes,
            ))

    if not stats:
        return 0.0
    stats = [np.concatenate(values, 0) for values in zip(*stats)]
    if not stats[0].any():
        return 0.0
    _, _, average_precision, _, _ = ap_per_class(*stats, plot=False)
    return float(average_precision[:, 0].mean())


class PrunableBN(nn.Module):

    def __init__(self, bn_layer: nn.Module, init='bn_weights'):
        super().__init__()
        self.bn = bn_layer
        nf = getattr(bn_layer, 'num_features', None) or getattr(bn_layer, 'num_channels', None) or bn_layer.weight.shape[0]
        self.register_buffer('mask', torch.ones(1, nf, 1, 1))
        self.eps = 1e-20
        self.register_buffer('threshold', torch.tensor(0.05))
        if init == 'ones':
            init_val = torch.ones(nf)
        elif init == 'zeros':
            init_val = torch.zeros(nf)
        elif init == 'random':
            init_val = torch.rand(nf)
        elif init == 'bn_weights':
            init_val = torch.abs(bn_layer.weight.data.clone())
        else:
            raise ValueError(f'Invalid initialization method: {init}')
        self.mask_scores = nn.Parameter(init_val)
        self.init_mask_scores = init_val.clone()

    def set_threshold(self, t: float):
        self.threshold.fill_(t)

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def num_features(self):
        return getattr(self.bn, 'num_features', None) or getattr(self.bn, 'num_channels', None) or self.bn.weight.shape[0]

    def forward(self, x):
        abs_scores = torch.abs(self.mask_scores)
        hard_mask = (abs_scores >= self.threshold).float()
        soft_mask = torch.sigmoid((abs_scores - self.threshold) * 20.0)
        mask = hard_mask - soft_mask.detach() + soft_mask
        mask = mask.view(1, -1, 1, 1)
        return self.bn(x) * mask

def calculate_sensitivity_weights(feature_maps, detections, source_stats, adaptable_bn_names, conf_thresh=0.5):
    sensitivities = {}
    for name, F_t in feature_maps.items():
        if name not in adaptable_bn_names or F_t is None:
            continue
        img_diff = torch.abs(F_t.mean(dim=[0]) - source_stats[name]['img_mean'].to(F_t.device))
        img_sens = img_diff.mean(dim=[1, 2])
        ins_sens = torch.zeros_like(img_sens)
        if detections:
            boxes_batch = []
            device = F_t.device
            for i, det in enumerate(detections):
                if det and len(det) > 0 and isinstance(det, dict) and ('boxes' in det):
                    boxes = det['boxes']
                    scores = det['scores']
                    valid = scores > conf_thresh
                    if valid.sum() > 0:
                        vb = boxes[valid]
                        batch_idx = torch.full((vb.shape[0], 1), i, device=device)
                        boxes_batch.append(torch.cat([batch_idx, vb], dim=1))
            if boxes_batch:
                all_boxes = torch.cat(boxes_batch, dim=0)
                try:
                    f_t_ins = torchvision.ops.roi_align(F_t, all_boxes, output_size=(7, 7))
                    if f_t_ins.size(0) > 0:
                        ins_diff = torch.abs(f_t_ins.mean(dim=[0]) - source_stats[name]['ins_mean'].to(F_t.device))
                        ins_sens = ins_diff.mean(dim=[1, 2])
                except Exception:
                    pass
        sensitivities[name] = img_sens + ins_sens
    return sensitivities

def _compute_threshold_from_scores(scores: torch.Tensor, target_ratio: float, epsilon: float=1e-06) -> float:
    if scores.numel() == 0:
        return 0.0
    target_ratio = float(max(0.0, min(1.0, target_ratio)))
    if target_ratio <= 0.0:
        return 0.0
    sorted_scores, _ = torch.sort(scores.view(-1))
    n = sorted_scores.numel()
    if target_ratio >= 1.0:
        return float(sorted_scores[-1].item() + epsilon)
    kth = int(math.ceil(target_ratio * n))
    kth = max(1, min(kth, n))
    threshold = float(sorted_scores[kth - 1].item() + epsilon)
    return threshold

def determine_pruning_thresholds(prunable_layers: List[Tuple[str, 'PrunableBN']], target_ratio: float, mode: str='global', epsilon: float=1e-06) -> Dict[str, float]:
    if not prunable_layers:
        return {}
    if mode not in {'global', 'layer'}:
        raise ValueError(f'Unsupported pruning threshold mode: {mode}')
    thresholds: Dict[str, float] = {}
    target_ratio = float(max(0.0, min(1.0, target_ratio)))
    if mode == 'global':
        combined_scores = []
        for _, module in prunable_layers:
            combined_scores.append(torch.abs(module.mask_scores.detach()).view(-1))
        if combined_scores:
            all_scores = torch.cat(combined_scores)
            threshold = _compute_threshold_from_scores(all_scores, target_ratio, epsilon)
        else:
            threshold = 0.0
        for name, _ in prunable_layers:
            thresholds[name] = threshold
        return thresholds
    for name, module in prunable_layers:
        scores = torch.abs(module.mask_scores.detach()).view(-1)
        thresholds[name] = _compute_threshold_from_scores(scores, target_ratio, epsilon)
    return thresholds

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TransformationLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, light: bool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = None if light else nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.scale1 = nn.Parameter(torch.tensor(0.5)) if light else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale1 if self.scale1 is not None else 0.5
        output = scale * self.conv1(x)
        if self.conv3 is not None:
            output = output + 0.5 * self.conv3(x)
        return output

class TTABuffer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, name: str, light: bool=False, device=None, dtype=None, mask_temperature: float=10.0):
        super().__init__()
        parameter_dtype = torch.float32 if dtype is None else dtype
        self.name = name
        self.alpha = nn.Parameter(0.01 * torch.ones(out_channels, dtype=parameter_dtype, device=device))
        self.adapter = TransformationLayer(in_channels, out_channels, light=light)
        if device is not None or dtype is not None:
            self.adapter.to(device=device, dtype=dtype)
        self.mask_temperature = mask_temperature
        self._mask_source: Optional[PrunableBN] = None

    def set_mask_source(self, prunable_bn: PrunableBN, temperature: Optional[float]=None):
        self._mask_source = prunable_bn
        if temperature is not None:
            self.mask_temperature = float(temperature)

    def _compute_mask_scale(self) -> Optional[torch.Tensor]:
        if self._mask_source is None:
            return None
        scores = self._mask_source.mask_scores
        threshold = self._mask_source.threshold
        temperature = max(self.mask_temperature, 1e-06)
        soft_mask = torch.sigmoid((scores.abs() - threshold) * temperature)
        return ((1.0 - soft_mask) * 10.0).clamp_(0.0, 10.0).view(1, -1, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        mask_scale = self._compute_mask_scale()
        if mask_scale is not None:
            alpha = alpha * mask_scale
        return alpha * self.adapter(inputs)

def _make_prunable_bn(module: nn.Module, device: torch.device) -> PrunableBN:
    num_features = getattr(module, 'num_features', None) or getattr(module, 'num_channels', None) or module.weight.shape[0]
    batch_norm = nn.BatchNorm2d(num_features, eps=module.eps).to(device)
    with torch.no_grad():
        batch_norm.weight.copy_(module.weight)
        batch_norm.bias.copy_(module.bias)
        batch_norm.running_mean.copy_(module.running_mean)
        batch_norm.running_var.copy_(module.running_var)
    return PrunableBN(batch_norm, init='bn_weights').to(device)

def insert_buffer_to_model(model: nn.Module, light: bool=False):
    bn_params = []
    mask_params = []
    adapter_params = []
    adapter_param_groups: List[Dict[str, Any]] = []
    replaced_bn_names = []
    for stage_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        stage = getattr(model.backbone.body, stage_name)
        for index, block in enumerate(stage):
            if not isinstance(block, torchvision.models.resnet.Bottleneck):
                continue
            names = [f'backbone.body.{stage_name}.{index}.bn{number}' for number in (1, 2, 3)]
            replaced_bn_names.extend(names)
            device = next(block.parameters()).device
            buffer_name = f'{stage_name}.{index}.tta_buffer_3'
            buffer = TTABuffer(block.conv3.out_channels, block.conv3.out_channels, f'{stage_name}.{index}', light=light).to(device)
            buffer.train()
            buffer_parameters = list(buffer.parameters())
            for parameter in buffer_parameters:
                parameter.requires_grad = True
            adapter_params.extend(buffer_parameters)
            adapter_param_groups.append({'layer_name': names[2], 'params': buffer_parameters, 'buffer_name': buffer_name})
            block.add_module('tta_buffer_3', buffer)
            prunable_modules = []
            for attribute in ('bn1', 'bn2', 'bn3'):
                prunable = _make_prunable_bn(getattr(block, attribute), device)
                setattr(block, attribute, prunable)
                prunable_modules.append(prunable)
                for parameter in prunable.bn.parameters():
                    parameter.requires_grad = True
                    bn_params.append(parameter)
                prunable.mask_scores.requires_grad = True
                mask_params.append(prunable.mask_scores)
            buffer.set_mask_source(prunable_modules[2])
            block.tta_has_buffer = True

            def modified_forward(self, inputs):
                identity = inputs
                output = self.relu(self.bn1(self.conv1(inputs)))
                output = self.relu(self.bn2(self.conv2(output)))
                output = self.bn3(self.conv3(output))
                if self.downsample is not None:
                    identity = self.downsample(inputs)
                output = identity + output + self.tta_buffer_3(identity)
                return self.relu(output)
            block.forward = types.MethodType(modified_forward, block)
    return (adapter_params, bn_params, mask_params, replaced_bn_names, adapter_param_groups)

def run_cd_buffer_tta(args, *, light=False):
    if not light:
        set_seed(629)
    dataroot = args.dataroot
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    weights = args.weights
    dataset = args.dataset
    source_split = args.source_split
    target_split = args.target_split
    target_task = args.target_task
    target_severity = args.target_severity
    save_dir = args.save_dir
    backbone = 'resnet50'
    pruning_ratio_start = max(0.0, min(1.0, args.pruning_ratio_start))
    pruning_ratio_end = max(0.0, min(1.0, args.pruning_ratio_end))
    if pruning_ratio_end < pruning_ratio_start:
        pruning_ratio_start, pruning_ratio_end = (pruning_ratio_end, pruning_ratio_start)
    p0, p1 = (pruning_ratio_start, pruning_ratio_end)
    reactivation_prob = max(0.0, min(1.0, args.pruning_reactivation_prob))
    threshold_mode = args.pruning_threshold_mode.lower()
    if threshold_mode not in {'global', 'layer'}:
        raise ValueError(f'Invalid pruning_threshold_mode: {args.pruning_threshold_mode}')
    threshold_epsilon = max(1e-08, float(args.pruning_threshold_epsilon))
    ratio_schedule = args.pruning_ratio_schedule.lower()
    if ratio_schedule not in {'constant', 'linear'}:
        raise ValueError(f'Invalid pruning_ratio_schedule: {args.pruning_ratio_schedule}')
    lambda_wreg = 0.05 if light else args.lambda_wreg
    conf_thresh = 0.5
    tta_lr = 0.0001 if light else args.tta_lr
    bn_lr = 0.0001 if light else args.bn_lr
    l1_loss = nn.L1Loss(reduction='mean')
    print('=' * 60)
    print('CD Buffer TTA started')
    print(
        f'Configuration: ratio={p0:.3f}->{p1:.3f} ({ratio_schedule}), '
        f'threshold_mode={threshold_mode}, '
        f'react_prob={reactivation_prob:.3f}, '
        f'lambda_wreg={lambda_wreg}, tta_lr={tta_lr}'
    )
    print('=' * 60)
    print('\nPreparing datasets...')
    if dataset == 'kitti':
        class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        source_dataset = KITTIDataset(dataroot, class_names, split=source_split, task='initial', severity='')
        target_dataset = KITTIDataset(dataroot, class_names, split=target_split, task=target_task, severity=target_severity)
    elif dataset == 'cityscapes':
        class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        source_dataset = CityscapesDataset(dataroot, class_names, split=source_split, task='initial', severity='')
        target_dataset = CityscapesDataset(dataroot, class_names, split=target_split, task=target_task, severity=target_severity)
    elif dataset == 'acdc' and (not light):
        from dataset_acdc import ACDCDataset
        dataroot_acdc = args.acdc_dataroot
        class_names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        source_dataset = CityscapesDataset(dataroot, class_names, split=source_split, task='initial', severity='')
        target_dataset = ACDCDataset(dataroot_acdc, class_names, condition=target_task)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    def collate_fn(batch):
        return tuple(zip(*batch))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=light, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    print('\nLoading model...')
    checkpoint = torch.load(weights, map_location=device)
    num_classes = len(class_names) + 1
    model = create_model_improved(num_classes=num_classes, backbone_type=backbone)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    print('\nFinding BN layers...')
    adaptable_bn_names = []
    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        stage = getattr(model.backbone.body, stage_name)
        for i, block in enumerate(stage):
            if isinstance(block, torchvision.models.resnet.Bottleneck):
                bn1_name = f'backbone.body.{stage_name}.{i}.bn1'
                bn2_name = f'backbone.body.{stage_name}.{i}.bn2'
                bn3_name = f'backbone.body.{stage_name}.{i}.bn3'
                adaptable_bn_names.append(bn1_name)
                adaptable_bn_names.append(bn2_name)
                adaptable_bn_names.append(bn3_name)
    print(f'Found {len(adaptable_bn_names)} BN layers')
    print('\nPreparing source statistics...')
    source_stats_path = args.source_stats_path
    source_stats = {}
    if os.path.exists(source_stats_path):
        print('Existing source statistics found; loading...')
        source_stats = torch.load(source_stats_path, map_location='cpu')
        print(f'Source statistics loaded: {len(source_stats)} layers')
    else:
        print('Source statistics not found; extracting...')
        named_modules_dict = dict(model.named_modules())
        chosen_bn_layers = [named_modules_dict[name] for name in adaptable_bn_names]
        nL = len(chosen_bn_layers)
        print(f'  Extracting pre-hook features from {nL} bn1/bn2/bn3 layers')
        save_outputs = [SaveOutputPreHook() for _ in range(nL)]
        mean_img = [AverageMeter() for _ in range(nL)]
        mean_img_var = [AverageMeter() for _ in range(nL)]
        with torch.no_grad():
            for images, targets in tqdm(source_loader, desc='Extracting BN prev features'):
                if not isinstance(images, list):
                    images = [images]
                images = [img.to(device, non_blocking=True) for img in images]
                model.eval()
                hooks = [chosen_bn_layers[i].register_forward_pre_hook(save_outputs[i]) for i in range(nL)]
                _ = model(images)
                for i in range(nL):
                    m = save_outputs[i].get_out_mean()
                    if m is not None:
                        mean_img[i].update(torch.nan_to_num(m, nan=0.0))
                    v = save_outputs[i].get_out_var()
                    if v is not None:
                        mean_img_var[i].update(torch.nan_to_num(v, nan=1.0))
                    save_outputs[i].clear()
                for hook in hooks:
                    hook.remove()
        save_outputs2 = [SaveOutputPreHook() for _ in range(nL)]
        mean_ins = [AverageMeter() for _ in range(nL)]
        with torch.no_grad():
            for images, targets in tqdm(source_loader, desc='Extracting ROI features'):
                if not isinstance(images, list):
                    images = [images]
                images = [img.to(device, non_blocking=True) for img in images]
                model.eval()
                hooks = [chosen_bn_layers[i].register_forward_pre_hook(save_outputs2[i]) for i in range(nL)]
                dets = model(images)
                for i in range(nL):
                    if len(save_outputs2[i].outputs) > 0:
                        fmap = save_outputs2[i].outputs[0]
                        for bi, det in enumerate(dets):
                            if det and len(det) > 0:
                                boxes = det['boxes']
                                scores = det['scores']
                                v = scores > 0.5
                                if v.sum() > 0:
                                    vb = boxes[v]
                                    roi_boxes = torch.cat([torch.full((vb.size(0), 1), bi, device=device), vb], dim=1)
                                    try:
                                        ins_feat = torchvision.ops.roi_align(fmap, roi_boxes, output_size=(7, 7))
                                        if ins_feat.size(0) > 0:
                                            m = torch.nan_to_num(ins_feat.mean(dim=0), nan=0.0)
                                            mean_ins[i].update(m)
                                    except Exception:
                                        pass
                    save_outputs2[i].clear()
                for hook in hooks:
                    hook.remove()
        for i, name in enumerate(adaptable_bn_names):
            source_stats[name] = {'img_mean': mean_img[i].avg.cpu(), 'img_var': mean_img_var[i].avg.cpu(), 'ins_mean': mean_ins[i].avg.cpu()}
        print(f'Saving source statistics to {source_stats_path}...')
        os.makedirs(os.path.dirname(source_stats_path) or '.', exist_ok=True)
        torch.save(source_stats, source_stats_path)
        print('Source statistics saved')
    print('Source statistics ready')
    print('\nInserting Buffer TTA modules (backbone only)...')
    tta_params, bn_params, mask_params, replaced_bn_names, tta_adapter_param_groups = insert_buffer_to_model(model, light=light)
    named_modules_dict = dict(model.named_modules())
    prunable_bn_layers: List[Tuple[str, PrunableBN]] = [(name, named_modules_dict[name]) for name in replaced_bn_names if isinstance(named_modules_dict.get(name), PrunableBN)]
    prunable_layer_names = [name for name, _ in prunable_bn_layers]
    n_wrapped = len(prunable_bn_layers)
    print(f'Total BN layers: {n_wrapped} (including replaced layers)')
    for p in model.parameters():
        p.requires_grad = False
    for p in tta_params:
        p.requires_grad = True
    for name, module in prunable_bn_layers:
        module.bn.weight.requires_grad_(True)
        if module.bn.bias is not None:
            module.bn.bias.requires_grad_(True)
        module.mask_scores.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': tta_params, 'lr': tta_lr}, {'params': bn_params, 'lr': bn_lr}, {'params': mask_params, 'lr': bn_lr}])
    running_loss_sum = 0.0
    running_map50 = 0.0
    running_pruning_ratio = 0.0
    num_batches = 0
    batch_performances = []
    target_loader_len = len(target_loader)
    target_progress = tqdm(target_loader, desc='TTA')
    for batch_idx, (images, targets) in enumerate(target_progress):
        if ratio_schedule == 'linear':
            denom = max(1, target_loader_len - 1)
            progress = min(float(batch_idx) / float(denom), 1.0)
            target_pruning_ratio = p0 + (p1 - p0) * progress
        else:
            target_pruning_ratio = p1
        current_thresholds = determine_pruning_thresholds(prunable_bn_layers, target_pruning_ratio, mode=threshold_mode, epsilon=threshold_epsilon)
        for name, module in prunable_bn_layers:
            module.set_threshold(float(current_thresholds.get(name, 0.0)))
        if not isinstance(images, list):
            images = [images]
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        save_outs = [SaveOutputPreHook() for _ in range(n_wrapped)]
        hooks = [prunable_bn_layers[i][1].register_forward_pre_hook(save_outs[i]) for i in range(n_wrapped)]
        model.eval()
        predictions = model(images)
        batch_mean_tta = [torch.nan_to_num(save_outs[x].get_out_mean(), nan=0.0) for x in range(n_wrapped)]
        batch_var_tta = [torch.nan_to_num(save_outs[x].get_out_var(), nan=1.0) for x in range(n_wrapped)]
        loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(device)
        loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(device)
        for i in range(n_wrapped):
            layer_name = prunable_bn_layers[i][0]
            if layer_name in source_stats:
                loss_mean += l1_loss(batch_mean_tta[i].to(device), source_stats[layer_name]['img_mean'].to(device))
                loss_var += l1_loss(batch_var_tta[i].to(device), source_stats[layer_name]['img_var'].to(device))
        f_loss = loss_mean + loss_var
        feature_maps = {prunable_bn_layers[i][0]: save_outs[i].outputs[0] if len(save_outs[i].outputs) > 0 else None for i in range(n_wrapped)}
        for i in range(n_wrapped):
            save_outs[i].clear()
            hooks[i].remove()
        sens = calculate_sensitivity_weights(feature_maps, predictions, source_stats, prunable_layer_names, conf_thresh)
        lwreg_list = []
        for name, module in prunable_bn_layers:
            if name in sens:
                lwreg = torch.mean(torch.abs(sens[name].detach() * module.mask_scores))
                if torch.isfinite(lwreg).all():
                    lwreg_list.append(lwreg)
        total_lwreg = torch.stack(lwreg_list).mean() if lwreg_list else torch.tensor(0.0, device=device)
        layer_sens: Dict[str, torch.Tensor] = {}
        for name, module in prunable_bn_layers:
            if name in sens:
                channel_sens = torch.abs(sens[name].detach())
                mean_sens = channel_sens.mean()
                layer_sens[name] = torch.clamp(mean_sens, min=0.0, max=10.0)
            else:
                layer_sens[name] = torch.tensor(0.0, device=device)
        pruned, tot = (0, 0)
        for name, module in prunable_bn_layers:
            g = module.mask_scores.data.clone()
            threshold_val = current_thresholds.get(name, 0.0)
            pruned += int((g.abs() < threshold_val).sum().item())
            tot += int(g.numel())
        if tot > 0:
            pruning_ratio = float(pruned) / float(tot)
        else:
            pruning_ratio = 0.0
        if pruning_ratio < target_pruning_ratio * 1.2:
            opt_loss = f_loss + lambda_wreg * total_lwreg
        else:
            opt_loss = f_loss
        opt_loss_val = float(opt_loss.detach().cpu())
        with torch.no_grad():
            for name, module in prunable_bn_layers:
                scores = module.mask_scores.data
                threshold_value = current_thresholds.get(name, 0.0)
                module.mask.data = (
                    scores.abs() >= threshold_value
                ).float().view(1, -1, 1, 1)
                if (
                    pruning_ratio >= target_pruning_ratio * 0.8
                    and reactivation_prob > 0.0
                ):
                    pruned_mask = scores.abs() < threshold_value
                    if pruned_mask.any():
                        random_mask = torch.bernoulli(
                            torch.full_like(scores, reactivation_prob)
                        ).bool()
                        reactivate_mask = pruned_mask & random_mask
                        if reactivate_mask.any():
                            updated_scores = scores.clone()
                            updated_scores[reactivate_mask] = (
                                module.init_mask_scores[reactivate_mask]
                            )
                            module.mask_scores.data.copy_(updated_scores)
                            module.mask.data = (
                                module.mask_scores.data.abs() >= threshold_value
                            ).float().view(1, -1, 1, 1)
        optimizer.zero_grad(set_to_none=True)
        opt_loss.backward()
        if tta_adapter_param_groups:
            for adapter_group in tta_adapter_param_groups:
                layer_name = adapter_group.get('layer_name')
                sens_value = layer_sens.get(layer_name) if layer_name is not None else None
                if sens_value is None:
                    scale_factor = 1.0
                else:
                    sens_scalar = float(torch.clamp(sens_value, min=0.0, max=10.0).item())
                    scale_factor = 1.0 + sens_scalar / 10.0 * 9.0
                for param in adapter_group.get('params', []):
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)
        optimizer.step()
        running_loss_sum += opt_loss_val
        running_pruning_ratio += pruning_ratio
        num_batches += 1
        current_map50 = evaluate_batch_map50(
            model,
            images,
            targets,
            device,
            conf_thres=0.25,
        )
        running_map50 += current_map50
        average_map50 = running_map50 / num_batches
        target_progress.set_postfix_str(
            f'Average mAP50: {average_map50:.4f}',
            refresh=True,
        )
        batch_performances.append({
            'batch_idx': batch_idx,
            'map50': current_map50,
        })

    divisor = max(1, num_batches)
    final_avg_loss = running_loss_sum / divisor
    final_avg_pruning = running_pruning_ratio / divisor
    final_avg_map50 = running_map50 / divisor
    print(f'Final average mAP@0.5: {final_avg_map50:.4f}')
    print('\nSaving model...')
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        save_dir, 'cd_buffer_tta_model.pth'
    )
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'source_stats': source_stats,
            'adaptable_bn_names': adaptable_bn_names,
            'final_loss': final_avg_loss,
            'final_pruning_ratio': final_avg_pruning,
            'final_map50': final_avg_map50,
            'num_batches': num_batches,
            'batch_performances': batch_performances,
            'pruning_threshold_mode': threshold_mode,
            'pruning_threshold_epsilon': threshold_epsilon,
            'pruning_ratio_schedule': ratio_schedule,
            'pruning_ratio_start': p0,
            'pruning_ratio_end': p1,
            'pruning_reactivation_prob': reactivation_prob,
        },
        checkpoint_path,
    )
    summary_path = os.path.join(
        save_dir, 'cd_buffer_tta_model.txt'
    )
    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.write(f'Average loss: {final_avg_loss:.4f}\n')
        summary_file.write(
            f'Average pruning ratio: {final_avg_pruning:.4f}\n'
        )
        summary_file.write(f'Average mAP@0.5: {final_avg_map50:.4f}\n')
        summary_file.write(f'Processed batches: {num_batches}\n')
        summary_file.write(
            f'Pruning ratio schedule: {ratio_schedule}\n'
        )
        summary_file.write(
            f'Pruning threshold mode: {threshold_mode}\n'
        )
        summary_file.write(
            f'Pruning threshold epsilon: {threshold_epsilon}\n'
        )
        summary_file.write(
            'Pruning reactivation probability: '
            f'{reactivation_prob}\n'
        )
    print(f'Model saved: {checkpoint_path}')
    return (
        model,
        source_stats,
        final_avg_loss,
        final_avg_pruning,
        final_avg_map50,
        batch_performances,
    )








