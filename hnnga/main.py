from __future__ import division

import os
import sys
import time
import torch
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pihnn.geometries as geom
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.bc as bc
import hnnga.main as hnnga

from tqdm import tqdm

matplotlib.use("Agg")  # Use non-interactive backend to avoid Abaqus freeze

# part size
size = [(-1.0, 1.0), (1.0, -1.0)]


def random_crack_dict(
    n_cracks,
    lim_x1=0.75,
    lim_y1=0.75,
    lim_x2=0.75,
    lim_y2=0.75,
    delta=0.1,
    sign=-1,
    limit_lengths=[0.1, 1.0],
    seed=None,
):
    """
    Create a list of n_cracks random cracks. Adjust limits and delta to control. Delta value is defining the size of a square around the limit coordinates. The sign parameter can be set to -1 to expand the search square symmetrically into the negative coordinate direction.
    Example: lim_x1=0.5, lim_y1=0.5, delta=0.1 will create points in the square defined by (-0.6, -0.6) to (0.6, 0.6)
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    crack_dict = {}
    n_cracks_generated = 0
    while n_cracks_generated < n_cracks:
        p1 = (
            round(random.uniform(sign * lim_x1 - delta, lim_x1 + delta), 3),
            round(random.uniform(sign * lim_y1 - delta, lim_y1 + delta), 3),
        )
        p2 = (
            round(random.uniform(sign * lim_x2 - delta, lim_x2 + delta), 3),
            round(random.uniform(sign * lim_y2 - delta, lim_y2 + delta), 3),
        )
        length = euclidean_distance(p1, p2)
        if length > limit_lengths[0] and length < limit_lengths[1]:
            crack_dict[n_cracks_generated] = ((p1, p2),)
            n_cracks_generated += 1
    return crack_dict


def compute_cracks_data(
    crack_dict,
    target,
    target_coords,
    sig_ext_t,
    sig_ext_b,
    history,
    gen,
    n_epochs,
    mode="stress",
    lambda_val=1.0,
    mu_val=1.0,
    optimized_weights={},
    out_dir="/results",
):
    cracks_data = {}

    for i, icrack in enumerate(crack_dict.values()):
        # print("Generation :", gen, "| Calculation :", i + 1, "/", len(crack_dict))

        if icrack[0] in history:
            mse, weights, residual = history[tuple(icrack[0])]
        else:
            z1, z2 = complex(*icrack[0][0]), complex(*icrack[0][1])

            stress, model = compute_stress(
                z1=z1,
                z2=z2,
                n_epochs=n_epochs,
                optimized_weights=optimized_weights,
                out_dir=out_dir,
                point_coords=target_coords,
                sig_ext_t=sig_ext_t,
                sig_ext_b=sig_ext_b,
                lambda_val=lambda_val,
                mu_val=mu_val,
            )

            if mode == "strain":
                E = (
                    mu_val * (3 * lambda_val + 2 * mu_val) / (lambda_val + mu_val)
                )  # Young's modulus
                nu = lambda_val / (2 * (lambda_val + mu_val))
                # Use hookes_law_plane_stress as it yields the same residual values as the calculate_residual function in the Abaqus version of the script.
                strain = inverse_hookes_law_plane_stress(stress=stress, E=E, nu=nu)
                # strain = inverse_hookes_law_plane_strain(stress=stress, E=E, nu=nu)
                mse = float(utils.MSE(strain, target))
                residual = residual_error(strain, target)
            if mode == "stress":
                mse = float(utils.MSE(stress, target))
                residual = residual_error(stress, target)

            weights = collect_weights(model)

            history[tuple(icrack[0])] = (mse, weights, residual)

        cracks_data[i] = (icrack[0], mse, weights, residual)

    return cracks_data, history


def compute_stress(
    z1,
    z2,
    sig_ext_b,
    sig_ext_t,
    n_epochs,
    optimized_weights={},
    learn_rate=1.0e-2,
    scheduler_apply=[],
    units=[1, 10, 10, 10, 1],
    np_train=300,
    np_test=40,
    beta=0.5,
    gauss=3,
    out_dir="results/",
    point_coords=[
        [-0.8000, 0.8000],
        [-0.4000, 0.8000],
        [0.0000, 0.8000],
        [0.4000, 0.8000],
        [0.8000, 0.8000],
        [-0.8000, -0.8000],
        [-0.4000, -0.8000],
        [0.0000, -0.8000],
        [0.4000, -0.8000],
        [0.8000, -0.8000],
    ],
    lambda_val=1.0,
    mu_val=1.0,
    h=1.0,
    l=1.0,
):
    line1 = geom.line(
        P1=[-l, -h], P2=[l, -h], bc_type=bc.stress_bc(), bc_value=sig_ext_b
    )
    line2 = geom.line(P1=[l, -h], P2=[l, h], bc_type=bc.stress_bc(), bc_value=0)
    line3 = geom.line(P1=[l, h], P2=[-l, h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
    line4 = geom.line(P1=[-l, h], P2=[-l, -h], bc_type=bc.stress_bc(), bc_value=0)
    crack = geom.line(
        P1=z1,
        P2=z2,
        bc_type=bc.stress_bc(),
    )
    crack.add_crack_tip(tip_side=0)
    crack.add_crack_tip(tip_side=1)
    boundary = geom.boundary(
        [line1, line2, line3, line4, crack], np_train, np_test, enrichment="rice"
    )
    model = nn.enriched_PIHNN(
        "km",
        units,
        boundary,
        material={"lambda": lambda_val, "mu": mu_val},
    )

    if optimized_weights == {}:
        model.initialize_weights(
            "exp", beta, boundary.extract_points(10 * np_train)[0], gauss
        )
    else:
        with torch.no_grad():
            for i, layer in enumerate(model.layers):
                if f"Layer_{i}_W" in optimized_weights:
                    layer.W.copy_(optimized_weights[f"Layer_{i}_W"].to(layer.W.device))
                if layer.has_bias and f"Layer_{i}_B" in optimized_weights:
                    layer.B.copy_(optimized_weights[f"Layer_{i}_B"].to(layer.B.device))

    loss_train, loss_test = utils.train(
        boundary,
        model,
        n_epochs,
        learn_rate,
        scheduler_apply,
        scheduler_gamma=0.5,
        dir=out_dir,
        save_model=False,
    )

    z_data = torch.view_as_complex(
        torch.tensor(point_coords, dtype=torch.float32)
    ).requires_grad_(True)

    sig_xx, sig_yy, sig_xy, _, _ = model(z_data, real_output=True)

    return torch.stack((sig_xx, sig_yy, sig_xy), dim=1), model


def residual_error(strain, strain_target):
    strain = strain.unsqueeze(0) if strain.dim() == 1 else strain
    strain_target = (
        strain_target.unsqueeze(0) if strain_target.dim() == 1 else strain_target
    )

    num = torch.sum(torch.norm(strain - strain_target, dim=-1) ** 2)
    den = torch.sum(torch.norm(strain_target, dim=-1) ** 2)

    if den == 0:
        raise ValueError(
            "Target strain norm is zero. Cannot compute residual (division by zero)."
        )

    residual = num / den
    return float(residual)


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def crossover(cracks_dict, target_size, max_distance_try=10):
    cracks_dict_crossover = {}
    parent_ids = list(cracks_dict.keys())

    if len(parent_ids) < 2:
        print("Not enough parents to initiate a crossover.")
        return cracks_dict

    for i, (k, v) in enumerate(cracks_dict.items()):
        if i >= target_size:
            break
        cracks_dict_crossover[i] = v

    base_index = len(cracks_dict_crossover)
    offspring_needed = target_size - base_index
    attempts_max = target_size * max_distance_try * 2
    offspring_created = 0
    attempts = 0
    random.seed()

    while offspring_created < offspring_needed and attempts < attempts_max:
        id1, id2 = random.sample(parent_ids, 2)
        if id1 == id2:
            continue

        p1 = cracks_dict[id1][0]
        p2 = cracks_dict[id2][0]

        for _ in range(max_distance_try):
            child = (
                (
                    round(random.uniform(p1[0][0], p2[0][0]), 3),
                    round(random.uniform(p1[0][1], p2[0][1]), 3),
                ),
                (
                    round(random.uniform(p1[1][0], p2[1][0]), 3),
                    round(random.uniform(p1[1][1], p2[1][1]), 3),
                ),
            )
            coords_valid = all(-1.0 < x < 1.0 for pt in child for x in pt)
            crack_length = euclidean_distance(child[0], child[1])

            if coords_valid and crack_length >= 0.05 and crack_length <= 0.8:
                cracks_dict_crossover[base_index + offspring_created] = ((child),)
                offspring_created += 1
                break
        attempts += 1
    return cracks_dict_crossover


def mutation(
    cracks_dict,
    mutation_rate=0.5,
    mutation_strength=0.1,
    bounds=(-0.7, 0.7),
    limit_lengths=[0.1, 1.0],
):
    cracks_dict_mutation = {}
    lower, upper = bounds
    epsilon = 1e-1
    sorted_keys = sorted(cracks_dict.keys())
    n = len(sorted_keys)
    midpoint = n // 2
    random.seed()

    # Keep the elites
    for i, k in enumerate(sorted_keys[:midpoint]):
        cracks_dict_mutation[i] = cracks_dict[k]

    index = midpoint
    for k in sorted_keys[midpoint:]:
        coords = cracks_dict[k][0]
        for attempt in range(10):
            coords_mutated = []
            for x, y in coords:
                if random.random() < mutation_rate:
                    x += random.uniform(-mutation_strength, mutation_strength)
                if random.random() < mutation_rate:
                    y += random.uniform(-mutation_strength, mutation_strength)
                x = max(min(x, upper - epsilon), lower + epsilon)
                y = max(min(y, upper - epsilon), lower + epsilon)
                coords_mutated.append((round(x, 3), round(y, 3)))

            distance = euclidean_distance(coords_mutated[0], coords_mutated[1])
            if distance >= limit_lengths[0] and distance < limit_lengths[1]:
                cracks_dict_mutation[index] = (tuple(coords_mutated),)
                index += 1
                break

    return cracks_dict_mutation


# def sort_cracks_dict(cracks_dict, n):
#     cracks_sorted = sorted(cracks_dict, key=lambda k: cracks_dict[k][3])
#     selected_items = [cracks_dict[k] for k in cracks_sorted[:n]]
#     return {i: selected_items[i] for i in range(n)}


def sort_cracks_dict(cracks_dict, n=None):
    # Sort items by residual (value[3])
    sorted_items = sorted(cracks_dict.items(), key=lambda item: item[1][3])

    # If n is specified, take only first n items
    if n is not None:
        sorted_items = sorted_items[:n]

    # Create new dictionary with keys 0, 1, 2, ...
    return {i: value for i, (_, value) in enumerate(sorted_items)}


def sort_cracks_dict_distance(cracks_dict, target_crack):
    target_p1, target_p2 = target_crack
    distances = {}

    for key, crack in cracks_dict.items():
        p1, p2 = crack[0]

        # Compute distances for both orientations
        dist_1 = math.hypot(target_p1[0] - p1[0], target_p1[1] - p1[1]) + math.hypot(
            target_p2[0] - p2[0], target_p2[1] - p2[1]
        )
        dist_2 = math.hypot(target_p1[0] - p2[0], target_p1[1] - p2[1]) + math.hypot(
            target_p2[0] - p1[0], target_p2[1] - p1[1]
        )

        # Take the smaller of the two orientations
        distances[key] = min(dist_1, dist_2)

    # Sort by distance
    sorted_keys = sorted(distances, key=lambda k: distances[k])
    return {i: cracks_dict[k] for i, k in enumerate(sorted_keys)}


def crack_target_dist_crit(cracks_dict, target_crack, target_dist):
    target_p1, target_p2 = target_crack
    for i, crack in enumerate(cracks_dict.values()):
        p1, p2 = crack[0]
        criterion = (
            math.hypot((target_p1[0] - p1[0]), (target_p1[1] - p1[1])) <= target_dist
            and math.hypot((target_p2[0] - p2[0]), (target_p2[1] - p2[1]))
            <= target_dist
        ) or (
            math.hypot((target_p1[0] - p2[0]), (target_p1[1] - p2[1])) <= target_dist
            and math.hypot((target_p2[0] - p1[0]), (target_p2[1] - p1[1]))
            <= target_dist
        )
        # Select crack that fulfils criterion
        if criterion:
            break
    return crack, i


def hookes_law_plane_stress(strain: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """
    Compute stress from strain using Hooke's law under plane stress conditions.

    Parameters:
    strain : torch.Tensor of shape [n_points, 3]
        Each row contains [eps11, eps22, gamma12] (engineering strain)
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns:
    stress : torch.Tensor of shape [n_points, 3]
        Each row contains [sigma11, sigma22, sigma12]
    """
    # Compute stiffness matrix for plane stress
    factor = E / (1 - nu**2)
    D = (
        torch.tensor(
            [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]],
            dtype=strain.dtype,
            device=strain.device,
        )
        * factor
    )

    # Matrix multiplication
    stress = torch.matmul(strain, D.T)
    return stress


def hookes_law_plane_strain(strain: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """
    Compute stress from strain using Hooke's law under plane strain conditions.

    Parameters:
    strain : torch.Tensor of shape [n_points, 3]
        Each row contains [eps11, eps22, gamma12] (engineering strain)
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns:
    stress : torch.Tensor of shape [n_points, 3]
        Each row contains [sigma11, sigma22, sigma12]
    """
    # Compute stiffness matrix for plane strain
    factor = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    D = (
        torch.tensor(
            [
                [1, nu / (1 - nu), 0],
                [nu / (1 - nu), 1, 0],
                [0, 0, (1 - 2 * nu) / (2 * (1 - nu))],
            ],
            dtype=strain.dtype,
            device=strain.device,
        )
        * factor
    )

    # Matrix multiplication
    stress = torch.matmul(strain, D.T)
    return stress


def inverse_hookes_law_plane_stress(
    stress: torch.Tensor, E: float, nu: float
) -> torch.Tensor:
    """
    Compute engineering strain from stress using inverse Hooke's law under plane stress conditions.

    Parameters:
    stress : torch.Tensor of shape [n_points, 3]
        Each row contains [sigma11, sigma22, sigma12]
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns:
    strain : torch.Tensor of shape [n_points, 3]
        Each row contains [eps11, eps22, gamma12] (engineering strain)
    """
    # Create compliance matrix on same device and dtype as stress
    S = torch.tensor(
        [
            [1 / E, -nu / E, 0],
            [-nu / E, 1 / E, 0],
            [0, 0, 2 * (1 + nu) / E],  # engineering shear strain gamma12
        ],
        dtype=stress.dtype,
        device=stress.device,
    )

    # Matrix multiplication
    strain = torch.matmul(stress, S.T)
    return strain


def inverse_hookes_law_plane_strain(
    stress: torch.Tensor, E: float, nu: float
) -> torch.Tensor:
    """
    Compute engineering strain from stress using inverse Hooke's law under plane strain conditions.

    Parameters:
    stress : torch.Tensor of shape [n_points, 3]
        Each row contains [sigma11, sigma22, sigma12]
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns:
    strain : torch.Tensor of shape [n_points, 3]
        Each row contains [eps11, eps22, gamma12] (engineering strain)
    """
    # Compliance matrix for plane strain
    S = torch.tensor(
        [
            [(1 - nu**2) / E, -nu * (1 + nu) / E, 0],
            [-nu * (1 + nu) / E, (1 - nu**2) / E, 0],
            [0, 0, 2 * (1 + nu) / E],  # engineering shear strain gamma12
        ],
        dtype=stress.dtype,
        device=stress.device,
    )

    # Matrix multiplication
    strain = torch.matmul(stress, S.T)
    return strain


def collect_weights(model):
    weights = {}

    for i, layer in enumerate(model.layers):
        W = layer.W.detach().cpu().clone()
        weights[f"Layer_{i}_W"] = W
        if layer.has_bias:
            B = layer.B.detach().cpu().clone()
            weights[f"Layer_{i}_B"] = B

    return weights
