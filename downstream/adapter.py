from __future__ import annotations
import copy
import random
from dataclasses import asdict, dataclass
from typing import Callable, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
GraphLike = object
EncodeFn = Callable[[torch.Tensor, GraphLike], torch.Tensor]
PredictorKind = Literal['prototype', 'linear']
FusionKind = Literal['add', 'concat', 'adapted']
TransformKind = Literal['l2', 'center_l2']
FeatureRegularizerKind = Literal['none', 'decorrelation', 'mcr']
FeatureAdapterKind = Literal['low_rank', 'mlp', 'mul_prompt']
MCRAssignmentKind = Literal['fixed_input', 'shared_epoch', 'periodic_epoch']
MCRTargetKind = Literal['input', 'low_rank', 'adapted_embedding', 'final_embedding']
MCRAssignmentSourceKind = Literal['input', 'original_embedding', 'ensemble', 'original_scores', 'stability', 'support_em']
MCRSoftLabelPolicy = Literal['all', 'confidence_weight', 'margin_weight', 'confidence_threshold', 'class_balanced_topk']
PseudoRefinementKind = Literal['topk', 'accumulate', 'threshold', 'accumulate_threshold']

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class DirectAdapterConfig:
    epochs: int = 100
    patience: int = 30
    feature_rank: int = 16
    feature_gamma: float = 0.1
    feature_lr: float = 0.001
    zero_init: bool = False
    feature_regularizer: FeatureRegularizerKind = 'decorrelation'
    decorrelation_weight: float = 0.1
    decorrelation_sample_size: int = 0
    mcr_weight: float = 0.01
    mcr_eps: float = 0.5
    mcr_assignment_temperature: float = 0.2
    mcr_assignment_mode: MCRAssignmentKind = 'fixed_input'
    mcr_update_interval: int = 1
    mcr_sample_size: int = 0
    mcr_normalize_by_dim: bool = True
    mcr_target: MCRTargetKind = 'input'
    mcr_assignment_source: MCRAssignmentSourceKind = 'input'
    mcr_soft_label_policy: MCRSoftLabelPolicy = 'all'
    mcr_confidence_power: float = 1.0
    mcr_confidence_threshold: float = 0.8
    mcr_topk_per_class: int = 200
    mcr_anchor_support: bool = False
    mcr_soft_em_steps: int = 2
    mcr_soft_em_anchor: float = 5.0
    mcr_soft_em_sharpening: float = 2.0
    mcr_soft_em_blend: float = 1.0
    mcr_stability_dropout: float = 0.1
    mcr_class_balanced: bool = False
    mcr_global_weight: float = 1.0
    mcr_warmup_epochs: int = 0
    mcr_ramp_epochs: int = 0
    mcr_gradient_balance: bool = False
    mcr_gradient_ratio: float = 0.25
    mcr_gradient_min_weight: float = 0.0
    mcr_gradient_max_weight: float = 50.0
    fusion: FusionKind = 'add'
    fusion_beta: float = 10.0
    normalize_branches: bool = True
    predictor: PredictorKind = 'prototype'
    predictor_lr: float = 0.01
    weight_decay: float = 0.0
    temperature: float = 0.2
    embedding_transform: TransformKind = 'center_l2'
    diffusion_alpha: float = 0.8
    diffusion_steps: int = 20
    refinement_steps: int = 1
    pseudo_per_class: int = 200
    pseudo_weight: float = 0.5
    pseudo_refinement_mode: PseudoRefinementKind = 'topk'
    pseudo_confidence_threshold: float = 0.8
    gradient_clip: float = 5.0
    seed: int = 39
    feature_adapter: FeatureAdapterKind = 'low_rank'

@dataclass
class DirectAdapterOutput:
    adapter: 'LowRankFeatureAdapter'
    predictor: nn.Module | None
    original_embedding: torch.Tensor
    adapted_embedding: torch.Tensor
    final_embedding: torch.Tensor
    logits: torch.Tensor
    diagnostics: dict[str, float | int | str | dict]

def make_full_mlp_mcr_config(**overrides: object) -> DirectAdapterConfig:
    config = DirectAdapterConfig(epochs=100, patience=30, feature_adapter='mlp', feature_gamma=0.1, feature_lr=0.001, zero_init=False, feature_regularizer='mcr', decorrelation_weight=0.0, mcr_weight=13.0, mcr_eps=1.0, mcr_assignment_temperature=0.1, mcr_assignment_mode='fixed_input', mcr_update_interval=1, mcr_assignment_source='original_embedding', mcr_target='input', mcr_sample_size=0, mcr_normalize_by_dim=True, mcr_soft_label_policy='all', mcr_confidence_power=1.0, mcr_class_balanced=False, mcr_warmup_epochs=0, mcr_ramp_epochs=0, mcr_gradient_balance=False, fusion='add', fusion_beta=10.0, normalize_branches=True, predictor='prototype', temperature=0.2, embedding_transform='center_l2', diffusion_alpha=0.8, diffusion_steps=20, refinement_steps=1, pseudo_per_class=200, pseudo_weight=0.5, pseudo_refinement_mode='topk', gradient_clip=5.0, seed=39)
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise TypeError(f'unknown DirectAdapterConfig field: {key}')
        setattr(config, key, value)
    return config

class LowRankFeatureAdapter(nn.Module):

    def __init__(self, dimension: int, rank: int, gamma: float, zero_init: bool=False) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError('feature adapter rank must be positive')
        self.gamma = float(gamma)
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        if zero_init:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.xavier_uniform_(self.up.weight)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = F.relu(self.up(self.down(features)))
        return (features + self.gamma * delta, delta)

    def forward_with_bottleneck(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bottleneck = self.down(features)
        delta = F.relu(self.up(bottleneck))
        return (features + self.gamma * delta, delta, bottleneck)

class MLPFeatureAdapter(nn.Module):

    def __init__(self, dimension: int, gamma: float, zero_init: bool=False) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.input_layer = nn.Linear(dimension, dimension)
        self.output_layer = nn.Linear(dimension, dimension)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if zero_init:
            nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adapted, delta, _ = self.forward_with_bottleneck(features)
        return (adapted, delta)

    def forward_with_bottleneck(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.input_layer(features))
        delta = self.output_layer(hidden)
        return (features + self.gamma * delta, delta, hidden)

class MultiplicativePromptFeatureAdapter(nn.Module):

    def __init__(self, dimension: int, gamma: float, zero_init: bool=False) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.prompt = nn.Parameter(torch.zeros(dimension))
        if not zero_init:
            nn.init.normal_(self.prompt, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adapted, delta, _ = self.forward_with_bottleneck(features)
        return (adapted, delta)

    def forward_with_bottleneck(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_scale = torch.tanh(self.prompt)
        delta = features * prompt_scale
        adapted = features + self.gamma * delta
        return (adapted, delta, prompt_scale)

def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError('correlation matrix must be square')
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()

def feature_decorrelation_loss(adapted_features: torch.Tensor, sample_size: int=0) -> torch.Tensor:
    if sample_size > 0 and adapted_features.size(0) > sample_size:
        sample_indices = torch.randperm(adapted_features.size(0), device=adapted_features.device)[:sample_size]
        adapted_features = adapted_features[sample_indices]
    centered = adapted_features - adapted_features.mean(dim=0, keepdim=True)
    scale = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-06)
    standardized = centered / scale
    correlation = standardized.T @ standardized / standardized.size(0)
    return off_diagonal(correlation).square().mean()

def _coding_rate(representations: torch.Tensor, eps: float) -> torch.Tensor:
    if representations.size(0) <= 1:
        return representations.new_zeros(())
    centered = representations - representations.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=1)
    dimension = normalized.size(1)
    covariance = normalized.T @ normalized / float(normalized.size(0))
    identity = torch.eye(dimension, device=normalized.device, dtype=normalized.dtype)
    scale = dimension / max(float(eps), 1e-06)
    sign, logabsdet = torch.linalg.slogdet(identity + scale * covariance)
    return 0.5 * torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))

def _weighted_coding_rate(representations: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    mass = weights.sum()
    if representations.size(0) <= 1 or float(mass.detach()) <= 1e-08:
        return representations.new_zeros(())
    normalized_weights = weights / mass.clamp_min(1e-08)
    mean = (normalized_weights[:, None] * representations).sum(dim=0, keepdim=True)
    centered = representations - mean
    normalized = F.normalize(centered, p=2, dim=1)
    covariance = normalized.T * normalized_weights[None, :] @ normalized
    dimension = normalized.size(1)
    identity = torch.eye(dimension, device=normalized.device, dtype=normalized.dtype)
    scale = dimension / max(float(eps), 1e-06)
    sign, logabsdet = torch.linalg.slogdet(identity + scale * covariance)
    return 0.5 * torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))

def prototype_assignments_from_features(features: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, temperature: float=0.2) -> torch.Tensor:
    normalized = F.normalize(features.detach(), p=2, dim=1)
    prototypes = []
    for class_id in range(num_classes):
        support_features = normalized[support_index[support_labels == class_id]]
        if support_features.numel() == 0:
            raise ValueError(f'support split has no sample for class {class_id}')
        prototypes.append(F.normalize(support_features.mean(dim=0), p=2, dim=0))
    prototype_matrix = torch.stack(prototypes)
    temperature = max(float(temperature), 1e-06)
    assignments = F.softmax(normalized @ prototype_matrix.T / temperature, dim=1)
    return assignments.detach()

def feature_mcr_loss(adapted_features: torch.Tensor, soft_assignments: torch.Tensor, eps: float=0.5, sample_size: int=0, normalize_by_dim: bool=True, class_balanced: bool=False, global_weight: float=1.0) -> torch.Tensor:
    if adapted_features.size(0) != soft_assignments.size(0):
        raise ValueError('features and soft assignments must have the same number of nodes')
    if sample_size > 0 and adapted_features.size(0) > sample_size:
        sample_indices = torch.randperm(adapted_features.size(0), device=adapted_features.device)[:sample_size]
        adapted_features = adapted_features[sample_indices]
        soft_assignments = soft_assignments[sample_indices]
    node_weights = soft_assignments.sum(dim=1)
    global_rate = _weighted_coding_rate(adapted_features, node_weights, eps)
    class_mass = soft_assignments.sum(dim=0)
    total_mass = class_mass.sum().clamp_min(1e-08)
    compact_rate = adapted_features.new_zeros(())
    active_classes = 0
    for class_id in range(soft_assignments.size(1)):
        if float(class_mass[class_id].detach()) <= 1e-08:
            continue
        class_rate = _weighted_coding_rate(adapted_features, soft_assignments[:, class_id], eps)
        if class_balanced:
            compact_rate = compact_rate + class_rate
        else:
            compact_rate = compact_rate + class_mass[class_id] / total_mass * class_rate
        active_classes += 1
    if class_balanced and active_classes > 0:
        compact_rate = compact_rate / active_classes
    loss = compact_rate - float(global_weight) * global_rate
    if normalize_by_dim:
        loss = loss / max(adapted_features.size(1), 1)
    return loss

def assign_mcr_soft_labels(assignments: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, policy: MCRSoftLabelPolicy='all', confidence_power: float=1.0, confidence_threshold: float=0.8, topk_per_class: int=200, anchor_support: bool=False) -> tuple[torch.Tensor, dict[str, float]]:
    if assignments.ndim != 2:
        raise ValueError('MCR assignments must be a rank-2 matrix')
    if assignments.size(0) == 0:
        return (assignments.detach(), {'mcr_reliable_fraction': 0.0})
    q = assignments.detach().clone()
    confidence, predicted = q.max(dim=1)
    if q.size(1) > 1:
        top2 = q.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    else:
        margin = confidence
    power = max(float(confidence_power), 0.0)
    if policy == 'all':
        reliability = torch.ones_like(confidence)
    elif policy == 'confidence_weight':
        reliability = confidence.clamp(0.0, 1.0).pow(power)
    elif policy == 'margin_weight':
        denominator = max(1.0 - 1.0 / max(q.size(1), 1), 1e-06)
        reliability = (margin / denominator).clamp(0.0, 1.0).pow(power)
    elif policy == 'confidence_threshold':
        reliability = (confidence >= float(confidence_threshold)).to(q.dtype)
        reliability = reliability * confidence.clamp(0.0, 1.0).pow(power)
    elif policy == 'class_balanced_topk':
        active_mask = torch.zeros_like(confidence, dtype=torch.bool)
        limit = max(int(topk_per_class), 0)
        for class_id in range(q.size(1)):
            eligible = (predicted == class_id).nonzero(as_tuple=False).reshape(-1)
            if eligible.numel() == 0:
                continue
            if limit > 0 and eligible.numel() > limit:
                keep = confidence[eligible].topk(limit, largest=True).indices
                eligible = eligible[keep]
            active_mask[eligible] = True
        reliability = active_mask.to(q.dtype) * confidence.clamp(0.0, 1.0).pow(power)
    else:
        raise ValueError(f'unsupported MCR soft-label policy: {policy}')
    if anchor_support:
        support_index = support_index.to(device=q.device, dtype=torch.long)
        support_labels = support_labels.to(device=q.device, dtype=torch.long)
        reliability[support_index] = 1.0
        q[support_index] = 0.0
        q[support_index, support_labels] = 1.0
    active_mask = reliability > 0
    filtered = q * reliability[:, None]
    stats = {'mcr_reliable_fraction': float(active_mask.float().mean()), 'mcr_mean_confidence': float(confidence.mean()), 'mcr_mean_reliable_confidence': float(confidence[active_mask].mean()) if bool(active_mask.any()) else 0.0, 'mcr_reliable_nodes': float(active_mask.sum())}
    return (filtered.detach(), stats)

def branch_normalize(embedding: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return embedding
    return F.layer_norm(embedding, (embedding.size(-1),))

def fuse_embeddings(original: torch.Tensor, adapted: torch.Tensor, mode: FusionKind, beta: float, normalize: bool) -> torch.Tensor:
    original = branch_normalize(original, normalize)
    adapted = branch_normalize(adapted, normalize)
    if mode == 'add':
        return original + beta * adapted
    if mode == 'concat':
        return torch.cat((original, beta * adapted), dim=-1)
    if mode == 'adapted':
        return adapted
    raise ValueError(f'unsupported fusion mode: {mode}')

def _transform_embedding(embedding: torch.Tensor, transform: TransformKind) -> torch.Tensor:
    if transform == 'center_l2':
        embedding = embedding - embedding.mean(dim=0, keepdim=True)
    elif transform != 'l2':
        raise ValueError(f'unsupported embedding transform: {transform}')
    return F.normalize(embedding, p=2, dim=-1)

def prototype_assignments_from_embedding(embedding: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, temperature: float=0.2, transform: TransformKind='center_l2') -> torch.Tensor:
    normalized = _transform_embedding(embedding.detach(), transform)
    prototypes = []
    for class_id in range(num_classes):
        support_features = normalized[support_index[support_labels == class_id]]
        if support_features.numel() == 0:
            raise ValueError(f'support split has no sample for class {class_id}')
        prototypes.append(support_features.mean(dim=0))
    prototype_matrix = F.normalize(torch.stack(prototypes), p=2, dim=-1)
    temperature = max(float(temperature), 1e-06)
    assignments = F.softmax(normalized @ prototype_matrix.T / temperature, dim=1)
    return assignments.detach()

def support_anchored_soft_em_assignments(embedding: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, temperature: float=0.2, transform: TransformKind='center_l2', steps: int=2, anchor: float=5.0, sharpening: float=2.0) -> torch.Tensor:
    normalized = _transform_embedding(embedding.detach(), transform)
    support_index = support_index.to(device=normalized.device, dtype=torch.long)
    support_labels = support_labels.to(device=normalized.device, dtype=torch.long)
    support_mask = torch.zeros(normalized.size(0), dtype=torch.bool, device=normalized.device)
    support_mask[support_index] = True
    support_prototypes = []
    for class_id in range(num_classes):
        support_features = normalized[support_index[support_labels == class_id]]
        if support_features.numel() == 0:
            raise ValueError(f'support split has no sample for class {class_id}')
        support_prototypes.append(F.normalize(support_features.mean(dim=0), p=2, dim=0))
    support_prototypes = torch.stack(support_prototypes)
    prototypes = support_prototypes.clone()
    temperature = max(float(temperature), 1e-06)
    steps = max(int(steps), 0)
    sharpening = max(float(sharpening), 1e-06)
    anchor = max(float(anchor), 0.0)
    assignments = F.softmax(normalized @ prototypes.T / temperature, dim=1)
    assignments[support_index] = 0.0
    assignments[support_index, support_labels] = 1.0
    for _ in range(steps):
        responsibilities = assignments.pow(sharpening)
        responsibilities[support_mask] = 0.0
        mass = responsibilities.sum(dim=0).clamp_min(1e-08)
        updated = responsibilities.T @ normalized + anchor * support_prototypes
        updated = updated / (mass[:, None] + anchor).clamp_min(1e-08)
        prototypes = F.normalize(updated, p=2, dim=1)
        assignments = F.softmax(normalized @ prototypes.T / temperature, dim=1)
        assignments[support_index] = 0.0
        assignments[support_index, support_labels] = 1.0
    return assignments.detach()

def prototype_logits(embedding: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, query_index: torch.Tensor, num_classes: int, temperature: float=0.2, transform: TransformKind='center_l2', refinement_steps: int=1, pseudo_per_class: int=200, pseudo_weight: float=0.5, pseudo_refinement_mode: PseudoRefinementKind='topk', pseudo_confidence_threshold: float=0.8, shared_assignments: torch.Tensor | None=None) -> torch.Tensor:
    normalized = _transform_embedding(embedding, transform)
    prototypes = []
    for class_id in range(num_classes):
        support_features = normalized[support_index[support_labels == class_id]]
        if support_features.numel() == 0:
            raise ValueError(f'support split has no sample for class {class_id}')
        prototypes.append(support_features.mean(dim=0))
    support_prototypes = F.normalize(torch.stack(prototypes), p=2, dim=-1)
    prototypes = support_prototypes
    support_mask = torch.zeros(embedding.size(0), dtype=torch.bool, device=embedding.device)
    support_mask[support_index] = True
    if refinement_steps <= 0 or pseudo_per_class <= 0:
        return normalized[query_index] @ prototypes.T / temperature
    if shared_assignments is not None:
        if shared_assignments.shape != (embedding.size(0), num_classes):
            raise ValueError(f'shared assignments must have shape ({embedding.size(0)}, {num_classes}), got {tuple(shared_assignments.shape)}')
        probabilities = shared_assignments.detach().to(device=embedding.device, dtype=normalized.dtype)
    else:
        probabilities = None
    if pseudo_refinement_mode == 'topk':
        for _ in range(refinement_steps):
            if shared_assignments is None:
                probabilities = F.softmax(normalized @ prototypes.T / temperature, dim=-1)
            confidence, assignment = probabilities.max(dim=-1)
            refined = []
            for class_id in range(num_classes):
                eligible_indices = ((assignment == class_id) & ~support_mask).nonzero(as_tuple=False).reshape(-1)
                if eligible_indices.numel() == 0:
                    refined.append(support_prototypes[class_id])
                    continue
                count = min(pseudo_per_class, eligible_indices.numel())
                chosen_indices = eligible_indices[confidence[eligible_indices].topk(count, largest=True).indices]
                weights = probabilities[chosen_indices, class_id]
                pseudo = (normalized[chosen_indices] * weights[:, None]).sum(dim=0)
                pseudo = pseudo / weights.sum().clamp_min(1e-12)
                mixed = (1.0 - pseudo_weight) * support_prototypes[class_id] + pseudo_weight * pseudo
                refined.append(F.normalize(mixed, p=2, dim=0))
            prototypes = torch.stack(refined)
    elif pseudo_refinement_mode == 'threshold':
        for _ in range(refinement_steps):
            if shared_assignments is None:
                probabilities = F.softmax(normalized @ prototypes.T / temperature, dim=-1)
            confidence, assignment = probabilities.max(dim=-1)
            refined = []
            for class_id in range(num_classes):
                eligible_indices = ((assignment == class_id) & ~support_mask & (confidence >= pseudo_confidence_threshold)).nonzero(as_tuple=False).reshape(-1)
                if eligible_indices.numel() == 0:
                    refined.append(support_prototypes[class_id])
                    continue
                weights = probabilities[eligible_indices, class_id]
                pseudo = (normalized[eligible_indices] * weights[:, None]).sum(dim=0)
                pseudo = pseudo / weights.sum().clamp_min(1e-12)
                mixed = (1.0 - pseudo_weight) * support_prototypes[class_id] + pseudo_weight * pseudo
                refined.append(F.normalize(mixed, p=2, dim=0))
            prototypes = torch.stack(refined)
    elif pseudo_refinement_mode == 'accumulate':
        assigned_mask = support_mask.clone()
        pseudo_vectors: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
        pseudo_weights: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
        for _ in range(refinement_steps):
            if shared_assignments is None:
                probabilities = F.softmax(normalized @ prototypes.T / temperature, dim=-1)
            confidence, assignment = probabilities.max(dim=-1)
            for class_id in range(num_classes):
                eligible_indices = ((assignment == class_id) & ~assigned_mask).nonzero(as_tuple=False).reshape(-1)
                if eligible_indices.numel() == 0:
                    continue
                count = min(pseudo_per_class, eligible_indices.numel())
                chosen_indices = eligible_indices[confidence[eligible_indices].topk(count, largest=True).indices]
                assigned_mask[chosen_indices] = True
                pseudo_vectors[class_id].append(normalized[chosen_indices])
                pseudo_weights[class_id].append(probabilities[chosen_indices, class_id])
            refined = []
            for class_id in range(num_classes):
                if not pseudo_vectors[class_id]:
                    refined.append(support_prototypes[class_id])
                    continue
                vectors = torch.cat(pseudo_vectors[class_id], dim=0)
                weights = torch.cat(pseudo_weights[class_id], dim=0)
                pseudo = (vectors * weights[:, None]).sum(dim=0)
                pseudo = pseudo / weights.sum().clamp_min(1e-12)
                mixed = (1.0 - pseudo_weight) * support_prototypes[class_id] + pseudo_weight * pseudo
                refined.append(F.normalize(mixed, p=2, dim=0))
            prototypes = torch.stack(refined)
    elif pseudo_refinement_mode == 'accumulate_threshold':
        assigned_mask = support_mask.clone()
        pseudo_vectors: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
        pseudo_weights: list[list[torch.Tensor]] = [[] for _ in range(num_classes)]
        for _ in range(refinement_steps):
            if shared_assignments is None:
                probabilities = F.softmax(normalized @ prototypes.T / temperature, dim=-1)
            confidence, assignment = probabilities.max(dim=-1)
            assigned_any = False
            for class_id in range(num_classes):
                eligible_indices = ((assignment == class_id) & ~assigned_mask & (confidence >= pseudo_confidence_threshold)).nonzero(as_tuple=False).reshape(-1)
                if eligible_indices.numel() == 0:
                    continue
                assigned_any = True
                assigned_mask[eligible_indices] = True
                pseudo_vectors[class_id].append(normalized[eligible_indices])
                pseudo_weights[class_id].append(probabilities[eligible_indices, class_id])
            refined = []
            for class_id in range(num_classes):
                if not pseudo_vectors[class_id]:
                    refined.append(support_prototypes[class_id])
                    continue
                vectors = torch.cat(pseudo_vectors[class_id], dim=0)
                weights = torch.cat(pseudo_weights[class_id], dim=0)
                pseudo = (vectors * weights[:, None]).sum(dim=0)
                pseudo = pseudo / weights.sum().clamp_min(1e-12)
                mixed = (1.0 - pseudo_weight) * support_prototypes[class_id] + pseudo_weight * pseudo
                refined.append(F.normalize(mixed, p=2, dim=0))
            prototypes = torch.stack(refined)
            if not assigned_any:
                break
    else:
        raise ValueError(f'unsupported pseudo refinement mode: {pseudo_refinement_mode}')
    return normalized[query_index] @ prototypes.T / temperature

def diffuse_scores(scores: torch.Tensor, propagation: torch.Tensor | None, alpha: float, steps: int) -> torch.Tensor:
    if propagation is None or alpha <= 0.0 or steps <= 0:
        return scores
    initial = scores
    current = scores
    for _ in range(steps):
        propagated = torch.sparse.mm(propagation, current) if propagation.is_sparse else propagation @ current
        current = (1.0 - alpha) * initial + alpha * propagated
    return current

def prototype_scores(embedding: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, config: DirectAdapterConfig, propagation: torch.Tensor | None=None, shared_assignments: torch.Tensor | None=None) -> torch.Tensor:
    all_index = torch.arange(embedding.size(0), device=embedding.device)
    scores = prototype_logits(embedding, support_index, support_labels, all_index, num_classes, config.temperature, config.embedding_transform, config.refinement_steps, config.pseudo_per_class, config.pseudo_weight, config.pseudo_refinement_mode, config.pseudo_confidence_threshold, shared_assignments)
    return diffuse_scores(scores, propagation, config.diffusion_alpha, config.diffusion_steps)

def _encode_detached(encode_fn: EncodeFn, features: torch.Tensor, graph: GraphLike) -> torch.Tensor:
    with torch.no_grad():
        return encode_fn(features, graph).detach()

def _make_predictor(dimension: int, num_classes: int, config: DirectAdapterConfig, device: torch.device) -> nn.Module | None:
    if config.predictor == 'prototype':
        return None
    if config.predictor == 'linear':
        return nn.Linear(dimension, num_classes).to(device)
    raise ValueError(f'unsupported predictor: {config.predictor}')

def predict_with_direct_adapter(*, encode_fn: EncodeFn, features: torch.Tensor, original_graph: GraphLike, adapted_graph: GraphLike, adapter: LowRankFeatureAdapter, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, config: DirectAdapterConfig, diffusion_propagation: torch.Tensor | None=None, predictor: nn.Module | None=None, original_embedding: torch.Tensor | None=None) -> DirectAdapterOutput:
    if original_embedding is None:
        original_embedding = _encode_detached(encode_fn, features, original_graph)
    adapted_features, delta = adapter(features)
    adapted_embedding = encode_fn(adapted_features, adapted_graph)
    final_embedding = fuse_embeddings(original_embedding, adapted_embedding, config.fusion, config.fusion_beta, config.normalize_branches)
    if predictor is None:
        shared_assignments = None
        if config.feature_regularizer == 'mcr' and config.mcr_assignment_mode == 'shared_epoch':
            shared_assignments = prototype_assignments_from_embedding(final_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
        logits = prototype_scores(final_embedding, support_index, support_labels, num_classes, config, diffusion_propagation, shared_assignments)
    else:
        logits = predictor(final_embedding)
    diagnostics = {'config': asdict(config)}
    return DirectAdapterOutput(adapter, predictor, original_embedding, adapted_embedding, final_embedding, logits, diagnostics)

def fit_direct_adapter(*, encode_fn: EncodeFn, features: torch.Tensor, labels: torch.Tensor, support_index: torch.Tensor, support_labels: torch.Tensor, num_classes: int, original_graph: GraphLike, adapted_graph: GraphLike | None=None, diffusion_propagation: torch.Tensor | None=None, config: DirectAdapterConfig | None=None) -> DirectAdapterOutput:
    del labels
    config = config or DirectAdapterConfig()
    set_seed(config.seed)
    adapted_graph = original_graph if adapted_graph is None else adapted_graph
    device = features.device
    original_embedding = _encode_detached(encode_fn, features, original_graph)
    adapter_type = {'low_rank': LowRankFeatureAdapter, 'mlp': MLPFeatureAdapter, 'mul_prompt': MultiplicativePromptFeatureAdapter}.get(config.feature_adapter)
    if adapter_type is None:
        raise ValueError(f'unsupported feature adapter: {config.feature_adapter}')
    if config.feature_adapter in {'mlp', 'mul_prompt'}:
        adapter = adapter_type(features.size(1), config.feature_gamma, config.zero_init).to(device)
    else:
        adapter = adapter_type(features.size(1), config.feature_rank, config.feature_gamma, config.zero_init).to(device)
    fusion_dimension = original_embedding.size(1) * 2 if config.fusion == 'concat' else original_embedding.size(1)
    predictor = _make_predictor(fusion_dimension, num_classes, config, device)
    parameter_groups = [{'params': adapter.parameters(), 'lr': config.feature_lr}]
    if predictor is not None:
        parameter_groups.append({'params': predictor.parameters(), 'lr': config.predictor_lr})
    optimizer = torch.optim.Adam(parameter_groups, weight_decay=config.weight_decay)
    best_loss = float('inf')
    best_adapter = copy.deepcopy(adapter.state_dict())
    best_predictor = copy.deepcopy(predictor.state_dict()) if predictor is not None else None
    stale = 0
    mcr_assignments = None
    periodic_shared_assignments = None
    periodic_mcr_assignments = None
    mcr_selection_stats: dict[str, float] = {}
    if config.feature_regularizer == 'mcr' and config.mcr_weight > 0.0 and (config.mcr_assignment_mode == 'fixed_input'):
        if config.mcr_assignment_source == 'input':
            assignment_reference = features
            mcr_assignments = prototype_assignments_from_features(assignment_reference, support_index, support_labels, num_classes, config.mcr_assignment_temperature)
        elif config.mcr_assignment_source == 'original_embedding':
            mcr_assignments = prototype_assignments_from_embedding(original_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
        elif config.mcr_assignment_source == 'ensemble':
            adapted_reference = _encode_detached(encode_fn, features, adapted_graph)
            original_assignments = prototype_assignments_from_embedding(original_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
            adapted_assignments = prototype_assignments_from_embedding(adapted_reference, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
            mcr_assignments = 0.5 * (original_assignments + adapted_assignments)
        elif config.mcr_assignment_source == 'original_scores':
            teacher_scores = prototype_scores(original_embedding, support_index, support_labels, num_classes, config, original_graph, None)
            teacher_temperature = max(float(config.mcr_assignment_temperature), 1e-06)
            mcr_assignments = F.softmax(teacher_scores.detach() / teacher_temperature, dim=-1)
        elif config.mcr_assignment_source == 'stability':
            perturbed_features = F.dropout(features, p=float(config.mcr_stability_dropout), training=True)
            perturbed_embedding = _encode_detached(encode_fn, perturbed_features, original_graph)
            clean_assignments = prototype_assignments_from_embedding(original_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
            perturbed_assignments = prototype_assignments_from_embedding(perturbed_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
            mcr_assignments = 0.5 * (clean_assignments + perturbed_assignments)
        elif config.mcr_assignment_source == 'support_em':
            refined_assignments = support_anchored_soft_em_assignments(original_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform, config.mcr_soft_em_steps, config.mcr_soft_em_anchor, config.mcr_soft_em_sharpening)
            blend = float(config.mcr_soft_em_blend)
            if not 0.0 <= blend <= 1.0:
                raise ValueError('mcr_soft_em_blend must be in [0, 1]')
            if blend == 1.0:
                mcr_assignments = refined_assignments
            else:
                initial_assignments = prototype_assignments_from_embedding(original_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
                mcr_assignments = (1.0 - blend) * initial_assignments + blend * refined_assignments
        else:
            raise ValueError(f'unsupported MCR assignment source: {config.mcr_assignment_source}')
        mcr_assignments, mcr_selection_stats = assign_mcr_soft_labels(mcr_assignments, support_index, support_labels, config.mcr_soft_label_policy, config.mcr_confidence_power, config.mcr_confidence_threshold, config.mcr_topk_per_class, config.mcr_anchor_support)
    for epoch in range(config.epochs):
        adapted_features, _, low_rank = adapter.forward_with_bottleneck(features)
        adapted_embedding = encode_fn(adapted_features, adapted_graph)
        final_embedding = fuse_embeddings(original_embedding, adapted_embedding, config.fusion, config.fusion_beta, config.normalize_branches)
        shared_assignments = None
        epoch_mcr_assignments = mcr_assignments
        if config.feature_regularizer == 'mcr' and config.mcr_weight > 0.0 and (config.mcr_assignment_mode in {'shared_epoch', 'periodic_epoch'}):
            refresh_assignments = config.mcr_assignment_mode == 'shared_epoch'
            if config.mcr_assignment_mode == 'periodic_epoch':
                interval = max(int(config.mcr_update_interval), 1)
                refresh_assignments = periodic_shared_assignments is None or epoch % interval == 0
            if refresh_assignments:
                refreshed_assignments = prototype_assignments_from_embedding(final_embedding, support_index, support_labels, num_classes, config.mcr_assignment_temperature, config.embedding_transform)
                filtered_assignments, mcr_selection_stats = assign_mcr_soft_labels(refreshed_assignments, support_index, support_labels, config.mcr_soft_label_policy, config.mcr_confidence_power, config.mcr_confidence_threshold, config.mcr_topk_per_class, config.mcr_anchor_support)
                if config.mcr_assignment_mode == 'shared_epoch':
                    shared_assignments = refreshed_assignments
                    epoch_mcr_assignments = filtered_assignments
                else:
                    periodic_shared_assignments = refreshed_assignments
                    periodic_mcr_assignments = filtered_assignments
            if config.mcr_assignment_mode == 'periodic_epoch':
                shared_assignments = periodic_shared_assignments
                epoch_mcr_assignments = periodic_mcr_assignments
        if predictor is None:
            logits_all = prototype_scores(final_embedding, support_index, support_labels, num_classes, config, diffusion_propagation, shared_assignments)
            logits = logits_all[support_index]
        else:
            logits = predictor(final_embedding[support_index])
        task_loss = F.cross_entropy(logits, support_labels)
        if config.feature_regularizer == 'mcr':
            if epoch_mcr_assignments is None:
                raise RuntimeError('MCR assignments were not initialized')
            if config.mcr_target == 'input':
                mcr_representations = adapted_features
            elif config.mcr_target == 'low_rank':
                mcr_representations = low_rank
            elif config.mcr_target == 'adapted_embedding':
                mcr_representations = adapted_embedding
            elif config.mcr_target == 'final_embedding':
                mcr_representations = final_embedding
            else:
                raise ValueError(f'unsupported MCR target: {config.mcr_target}')
            regularizer_loss = feature_mcr_loss(mcr_representations, epoch_mcr_assignments, config.mcr_eps, config.mcr_sample_size, config.mcr_normalize_by_dim, config.mcr_class_balanced, config.mcr_global_weight)
            if config.mcr_warmup_epochs > 0 and epoch < config.mcr_warmup_epochs:
                schedule = 0.0
            elif config.mcr_ramp_epochs > 0 and epoch < config.mcr_warmup_epochs + config.mcr_ramp_epochs:
                schedule = (epoch - config.mcr_warmup_epochs + 1) / float(config.mcr_ramp_epochs)
            else:
                schedule = 1.0
            regularizer_weight = config.mcr_weight * min(max(schedule, 0.0), 1.0)
            decorrelation_loss = task_loss.new_zeros(())
            mcr_loss = regularizer_loss
        elif config.feature_regularizer == 'decorrelation':
            if config.decorrelation_weight > 0.0:
                regularizer_loss = feature_decorrelation_loss(adapted_features, config.decorrelation_sample_size)
            else:
                regularizer_loss = task_loss.new_zeros(())
            regularizer_weight = config.decorrelation_weight
            decorrelation_loss = regularizer_loss
            mcr_loss = task_loss.new_zeros(())
        elif config.feature_regularizer == 'none':
            regularizer_loss = task_loss.new_zeros(())
            regularizer_weight = 0.0
            decorrelation_loss = task_loss.new_zeros(())
            mcr_loss = task_loss.new_zeros(())
        else:
            raise ValueError(f'unsupported feature regularizer: {config.feature_regularizer}')
        loss = task_loss + regularizer_weight * regularizer_loss
        optimizer.zero_grad(set_to_none=True)
        adapter_parameters = tuple(adapter.parameters())

        def gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
            squared = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
            if not squared:
                return 0.0
            return float(torch.stack(squared).sum().sqrt())
        task_gradients: tuple[torch.Tensor | None, ...] = ()
        raw_regularizer_gradients: tuple[torch.Tensor | None, ...] = ()
        if config.mcr_gradient_balance and config.feature_regularizer == 'mcr':
            task_gradients = torch.autograd.grad(task_loss, adapter_parameters, retain_graph=True, allow_unused=True)
            raw_regularizer_gradients = torch.autograd.grad(regularizer_loss, adapter_parameters, retain_graph=True, allow_unused=True)
        if config.mcr_gradient_balance and config.feature_regularizer == 'mcr':
            task_norm = gradient_norm(task_gradients)
            mcr_norm = gradient_norm(raw_regularizer_gradients)
            if mcr_norm > 1e-12:
                regularizer_weight = float(config.mcr_gradient_ratio * task_norm / mcr_norm)
                regularizer_weight = min(max(regularizer_weight, config.mcr_gradient_min_weight), config.mcr_gradient_max_weight)
                loss = task_loss + regularizer_weight * regularizer_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), config.gradient_clip)
        if predictor is not None:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), config.gradient_clip)
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            stale = 0
            best_adapter = copy.deepcopy(adapter.state_dict())
            if predictor is not None:
                best_predictor = copy.deepcopy(predictor.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    adapter.load_state_dict(best_adapter)
    if predictor is not None and best_predictor is not None:
        predictor.load_state_dict(best_predictor)
    output = predict_with_direct_adapter(encode_fn=encode_fn, features=features, original_graph=original_graph, adapted_graph=adapted_graph, adapter=adapter, support_index=support_index, support_labels=support_labels, num_classes=num_classes, config=config, diffusion_propagation=diffusion_propagation, predictor=predictor, original_embedding=original_embedding)
    return output
