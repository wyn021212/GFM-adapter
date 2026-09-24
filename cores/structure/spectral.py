import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_adjacency_with_loops(edge_index, edge_weight, num_nodes, coalesce=True):
    device = edge_weight.device
    dtype = edge_weight.dtype
    loops = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    full_index = torch.cat([edge_index, loop_index], dim=1)
    full_weight = torch.cat([edge_weight, torch.ones(num_nodes, device=device, dtype=dtype)])
    source, target = full_index
    degree = torch.zeros(num_nodes, device=device, dtype=dtype)
    degree.index_add_(0, target, full_weight)
    inverse_sqrt = degree.clamp_min(1e-12).rsqrt()
    values = full_weight * inverse_sqrt[source] * inverse_sqrt[target]
    propagation = torch.sparse_coo_tensor(full_index, values, (num_nodes, num_nodes), device=device, dtype=dtype)
    return propagation.coalesce() if coalesce else propagation

def symmetric_edges(pairs: torch.Tensor, weights: torch.Tensor):
    reverse = pairs.flip(0)
    edge_index = torch.cat([pairs, reverse], dim=1)
    edge_weight = torch.cat([weights, weights], dim=0)
    return (edge_index, edge_weight)

def whiten_features(features, eigenvalue_floor=1e-05, epsilon=1e-06, max_rank=None):
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(features.size(0) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    retained = eigenvalues > eigenvalues.max() * eigenvalue_floor
    indices = torch.where(retained)[0]
    if max_rank is not None:
        indices = indices[-min(max_rank, indices.numel()):]
    basis = eigenvectors[:, indices]
    scales = eigenvalues[indices].clamp_min(epsilon).rsqrt()
    return centered @ (basis * scales.unsqueeze(0))

def gaussian_chebyshev_coefficients(num_bands, degree, width, quadrature_points, device, dtype):
    theta = (torch.arange(quadrature_points, device=device, dtype=dtype) + 0.5) * (torch.pi / quadrature_points)
    x = torch.cos(theta)
    lambdas = x + 1.0
    centers = torch.linspace(0.0, 2.0, num_bands, device=device, dtype=dtype)
    windows = torch.exp(-0.5 * ((lambdas.unsqueeze(0) - centers.unsqueeze(1)) / width).square())
    windows = windows / windows.sum(dim=0, keepdim=True).clamp_min(1e-12)
    orders = torch.arange(degree + 1, device=device, dtype=dtype)
    cosine = torch.cos(orders.unsqueeze(1) * theta.unsqueeze(0))
    coefficients = 2.0 / quadrature_points * (windows @ cosine.T)
    coefficients[:, 0] *= 0.5
    return (centers, coefficients)

def apply_spectral_bank(signal, propagation, coefficients):
    outputs = [coefficients[:, 0, None, None] * signal[None, :, :]]
    if coefficients.size(1) == 1:
        return outputs[0]
    previous = signal
    current = -torch.sparse.mm(propagation, signal)
    accumulated = outputs[0] + coefficients[:, 1, None, None] * current[None, :, :]
    for order in range(2, coefficients.size(1)):
        following = -2.0 * torch.sparse.mm(propagation, current) - previous
        accumulated = accumulated + coefficients[:, order, None, None] * following[None, :, :]
        previous, current = (current, following)
    return accumulated

@torch.no_grad()
def blackbox_spectral_probe(encoder, features, edge_index, edge_weight, coefficients, probes, perturbation, confidence_beta, seed):
    num_nodes = features.size(0)
    propagation = normalized_adjacency_with_loops(edge_index, edge_weight, num_nodes)
    baseline = encoder.embed(features, edge_index, edge_weight).detach()
    input_scale = features.std(dim=0, unbiased=False)
    input_floor = input_scale.median().clamp_min(1e-06) * 0.01
    input_scale = input_scale.clamp_min(input_floor)
    output_scale = baseline.std(dim=0, unbiased=False)
    output_floor = output_scale.median().clamp_min(1e-06) * 0.01
    output_scale = output_scale.clamp_min(output_floor)
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    all_gains = []
    all_linearities = []
    for _ in range(probes):
        random_signal = torch.randn(features.shape, generator=generator, device=features.device, dtype=features.dtype)
        filtered = apply_spectral_bank(random_signal, propagation, coefficients)
        gains = []
        linearities = []
        for band_signal in filtered:
            standardized_probe = band_signal / band_signal.square().mean().sqrt().clamp_min(1e-08)
            raw_probe = standardized_probe * input_scale.unsqueeze(0)
            positive = encoder.embed(features + perturbation * raw_probe, edge_index, edge_weight)
            negative = encoder.embed(features - perturbation * raw_probe, edge_index, edge_weight)
            derivative = (positive - negative) / (2.0 * perturbation)
            standardized_derivative = derivative / output_scale.unsqueeze(0)
            gains.append(standardized_derivative.square().mean())
            positive_change = (positive - baseline).flatten()
            negative_change = (baseline - negative).flatten()
            linearities.append(F.cosine_similarity(positive_change.unsqueeze(0), negative_change.unsqueeze(0), dim=1).squeeze(0))
        all_gains.append(torch.stack(gains))
        all_linearities.append(torch.stack(linearities))
    gains = torch.stack(all_gains)
    linearities = torch.stack(all_linearities)
    stable_gains = gains * linearities.clamp(min=0.0, max=1.0)
    mean = stable_gains.mean(dim=0)
    std = stable_gains.std(dim=0, unbiased=False)
    standard_error = std / max(probes, 1) ** 0.5
    conservative = (mean - confidence_beta * standard_error).clamp_min(0.0)
    if float(conservative.max()) <= 0.0:
        conservative = mean.clamp_min(1e-12)
    response = conservative / conservative.max().clamp_min(1e-12)
    return response

class FullEdgeAdapter(nn.Module):

    def __init__(self, pairs, initial_weights):
        super().__init__()
        self.register_buffer('pairs', pairs)
        self.weights = nn.Parameter(initial_weights.clone())

    def edges(self):
        return symmetric_edges(self.pairs, self.weights)

    @torch.no_grad()
    def project(self, total_mass, maximum_weight, preserve_mass, steps=50):
        values = self.weights.data
        if not preserve_mass:
            values.clamp_(min=0.0, max=maximum_weight)
            return
        lower = (values - maximum_weight).min()
        upper = values.max()
        for _ in range(steps):
            threshold = 0.5 * (lower + upper)
            projected = (values - threshold).clamp(min=0.0, max=maximum_weight)
            if projected.sum() > total_mass:
                lower = threshold
            else:
                upper = threshold
        self.weights.data.copy_((values - upper).clamp(min=0.0, max=maximum_weight))

def information_spectrum(signal, propagation, coefficients):
    filtered = apply_spectral_bank(signal, propagation, coefficients)
    energy = filtered.square().sum(dim=(1, 2)).clamp_min(1e-12)
    return energy / energy.sum()

def sinkhorn_wasserstein(first, second, centers, regularization, iterations):
    first = first.clamp_min(1e-12)
    second = second.clamp_min(1e-12)
    first = first / first.sum()
    second = second / second.sum()
    cost = (centers[:, None] - centers[None, :]).square()
    log_first = first.log()
    log_second = second.log()
    dual_first = torch.zeros_like(first)
    dual_second = torch.zeros_like(second)
    for _ in range(iterations):
        dual_first = regularization * (log_first - torch.logsumexp((dual_second[None, :] - cost) / regularization, dim=1))
        dual_second = regularization * (log_second - torch.logsumexp((dual_first[:, None] - cost) / regularization, dim=0))
    transport = torch.exp((dual_first[:, None] + dual_second[None, :] - cost) / regularization)
    return (transport * cost).sum()

def spectral_transport_loss(signal, edge_index, edge_weight, num_nodes, coefficients, centers, response_distribution, regularization, sinkhorn_iterations):
    propagation = normalized_adjacency_with_loops(edge_index, edge_weight, num_nodes)
    spectrum = information_spectrum(signal, propagation, coefficients)
    loss = sinkhorn_wasserstein(spectrum, response_distribution, centers, regularization, sinkhorn_iterations)
    return (loss, spectrum)

def adapt_full_graph(pairs, initial_weights, signal, coefficients, centers, response, args):
    adapter = FullEdgeAdapter(pairs, initial_weights).to(signal.device)
    response_distribution = response.clamp_min(1e-12)
    response_distribution = response_distribution / response_distribution.sum()
    total_mass = initial_weights.sum()
    initial_edge_index, initial_edge_weight = adapter.edges()
    with torch.no_grad():
        initial_loss, _ = spectral_transport_loss(signal, initial_edge_index, initial_edge_weight, signal.size(0), coefficients, centers, response_distribution, args.sinkhorn_regularization, args.sinkhorn_iterations)
    best_loss = float(initial_loss.cpu())
    stopping_loss = best_loss * (1.0 - args.transport_gap_closure)
    best_state = copy.deepcopy(adapter.state_dict())
    optimizer = torch.optim.Adam([adapter.weights], lr=args.adapt_lr)
    for epoch in range(args.adapt_epochs):
        edge_index, edge_weight = adapter.edges()
        loss, _ = spectral_transport_loss(signal, edge_index, edge_weight, signal.size(0), coefficients, centers, response_distribution, args.sinkhorn_regularization, args.sinkhorn_iterations)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([adapter.weights], args.gradient_clip)
        optimizer.step()
        adapter.project(total_mass, args.maximum_edge_weight, args.preserve_edge_mass)
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(adapter.state_dict())
        if value <= stopping_loss:
            break
    adapter.load_state_dict(best_state)
    with torch.no_grad():
        return adapter.edges()
