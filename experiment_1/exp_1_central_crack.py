import os
import time
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import hnnga.main as hnnga
import hnnga.plot as plot
import hnnga.io as io
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # set environment variable to avoid error message
)


def main(
    sig_ext_t=1j,
    sig_ext_b=-1j,
    target_dir="./fe_solution/",
    out_dir="./hnnga_solution/",
    random_seed=42,
    save_plots=True,
    standard_initial_population=True,
    fitness_long_range=7.0e-3,
    n_new_cracks_short_range=2,
    stop_criterion_short_range=5.0e-2,
    gen_max=60,
    n_epochs_sawtooth=200,
    n_epochs_short_range=50,
    sawtooth_nmin=3,
    sawtooth_T=5,
    sawtooth_D=3,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    point_coords, strain_target = io.read_strain_file(
        os.path.join(target_dir, "strain_reference.txt")
    )  # Skip header row
    strain_target = torch.Tensor(strain_target)
    point_coords = torch.Tensor(point_coords)

    crack_target = np.loadtxt(
        os.path.join(target_dir, "crack_reference.txt"), skiprows=1
    )
    x1, y1 = crack_target[0]
    x2, y2 = crack_target[1]

    sawtooth_cycle = 1
    gen = 0
    history = {}
    crack_best_by_gen = []
    mse_best_by_gen = []
    res_best_by_gen = []
    duration_by_gen = []

    n_cracks_init = sawtooth_nmin + 2 * sawtooth_D
    n_remove = int((n_cracks_init - sawtooth_nmin) / sawtooth_T)

    if standard_initial_population:
        if random_seed is None:
            seed = 42
        else:
            seed = random_seed
    else:
        seed = None

    cracks = hnnga.random_crack_dict(
        n_cracks=n_cracks_init,
        lim_x1=0.8,
        lim_y1=0.8,
        lim_x2=0.8,
        lim_y2=0.8,
        delta=0.0,
        sign=-1,
        limit_lengths=[0.1, 1.0],
        seed=seed,
    )

    # Initial evaluation
    start_time = time.time()
    cracks_data, history = hnnga.compute_cracks_data(
        crack_dict=cracks,
        history=history,
        gen=gen,
        target=strain_target,
        target_coords=point_coords,
        sig_ext_t=sig_ext_t,
        sig_ext_b=sig_ext_b,
        n_epochs=n_epochs_sawtooth,
        out_dir=out_dir,
        mode="strain",
    )

    cracks_data = hnnga.sort_cracks_dict(
        cracks_dict=cracks_data, n=len(cracks_data) - n_remove
    )

    end_time = time.time()
    duration_by_gen.append((gen, round(end_time - start_time, 3), len(cracks_data)))

    for k in cracks_data:
        history[tuple(cracks_data[k][0])] = (
            cracks_data[k][1],
            cracks_data[k][2],
            cracks_data[k][3],
        )

    crack_best_by_gen.append(cracks_data[0][0])
    mse_best_by_gen.append(cracks_data[0][1])
    res_best_by_gen.append(cracks_data[0][3])

    # Save initial figure
    if save_plots:
        plot.plot_crack_vs_target(
            crack=cracks_data[0][0],
            target=crack_target,
            residual=cracks_data[0][3],
            generation=gen,
            out_path=os.path.join(out_dir, "crack_gen_0.png"),
        )
        plot.plot_figure_population(
            population_dict=cracks_data,
            target=crack_target,
            generation=gen,
        )

    io.export_population(
        cracks_data=cracks_data,
        gen=gen,
        target=crack_target,
        file_path=os.path.join(out_dir),
        file_name=f"population_{gen}.csv",
    )
    io.export_dict(
        cracks_data=cracks_data,
        file_path=os.path.join(out_dir, f"population_{gen}.dict"),
    )

    gen += 1

    # Long-range search
    ref_crack_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    r = stop_criterion_short_range * ref_crack_length

    while (
        all(
            (val[3] > fitness_long_range or np.isnan(val[3]))
            for val in cracks_data.values()
        )
        and not any(
            (
                math.hypot((crack[0][0][0] - x1), (crack[0][0][1] - y1)) <= r
                and math.hypot((crack[0][1][0] - x2), (crack[0][1][1] - y2)) <= r
            )
            or (
                math.hypot((crack[0][1][0] - x1), (crack[0][1][1] - y1)) <= r
                and math.hypot((crack[0][0][0] - x2), (crack[0][0][1] - y2)) <= r
            )
            for crack in cracks_data.values()
        )
        and gen <= gen_max
    ):
        for i in range(sawtooth_T):
            # print("Sawtooth cycle:", sawtooth_cycle, "| Generation:", gen)
            start_time = time.time()

            cracks_crossover = hnnga.crossover(
                hnnga.sort_cracks_dict(
                    cracks_dict=cracks_data, n=int(np.floor(0.5 * len(cracks_data)))
                ),
                len(cracks_data),
            )

            cracks_mutation = hnnga.mutation(
                cracks_dict=cracks_crossover,
                mutation_rate=0.5,
                mutation_strength=0.2,
                bounds=(-0.8, 0.8),
                limit_lengths=[0.1, 1.0],
            )
            cracks_data, history = hnnga.compute_cracks_data(
                crack_dict=cracks_mutation,
                target=strain_target,
                target_coords=point_coords,
                sig_ext_t=sig_ext_t,
                sig_ext_b=sig_ext_b,
                mode="strain",
                history=history,
                gen=gen,
                n_epochs=n_epochs_sawtooth,
                out_dir=out_dir,
            )

            cracks_data = hnnga.sort_cracks_dict(
                cracks_dict=cracks_data, n=len(cracks_data) - n_remove
            )

            end_time = time.time()
            duration_by_gen.append(
                (gen, round(end_time - start_time, 3), len(cracks_data))
            )

            for k in cracks_data:
                history[tuple(cracks_data[k][0])] = (
                    cracks_data[k][1],
                    cracks_data[k][2],
                    cracks_data[k][3],
                )

            crack_best_by_gen.append(cracks_data[0][0])
            mse_best_by_gen.append(cracks_data[0][1])
            res_best_by_gen.append(cracks_data[0][3])

            # Save figure
            if save_plots:
                plot.plot_crack_vs_target(
                    crack=cracks_data[0][0],
                    target=crack_target,
                    residual=cracks_data[0][3],
                    generation=gen,
                    out_path=os.path.join(out_dir, f"crack_gen_{gen}.png"),
                )
                plot.plot_figure_population(
                    population_dict=cracks_data,
                    target=crack_target,
                    generation=gen,
                )

            io.export_population(
                cracks_data=cracks_data,
                gen=gen,
                target=crack_target,
                file_path=os.path.join(out_dir),
                file_name=f"population_{gen}.csv",
            )
            io.export_dict(
                cracks_data=cracks_data,
                file_path=os.path.join(out_dir, f"population_{gen}.dict"),
            )

            gen += 1
            if cracks_data[0][3] <= fitness_long_range:
                break

        # sawtooth cycle complete: increase population
        if gen < gen_max:
            sawtooth_cycle += 1
            cracks = hnnga.random_crack_dict(
                n_cracks=n_cracks_init - sawtooth_nmin,
                lim_x1=0.8,
                lim_y1=0.8,
                lim_x2=0.8,
                lim_y2=0.8,
                delta=0.0,
                sign=-1,
                limit_lengths=[0.1, 1.0],
            )
            cracks_data_add, history = hnnga.compute_cracks_data(
                crack_dict=cracks,
                target=strain_target,
                target_coords=point_coords,
                sig_ext_t=sig_ext_t,
                sig_ext_b=sig_ext_b,
                mode="strain",
                history=history,
                gen=gen,
                n_epochs=n_epochs_sawtooth,
                out_dir=out_dir,
            )

            cracks_data_add_offset = {}
            for i in range(len(cracks_data_add)):
                cracks_data_add_offset[i + sawtooth_nmin] = cracks_data_add[i]

            for key, val in cracks_data_add_offset.items():
                cracks_data[key] = val

            for k in cracks_data:
                history[tuple(cracks_data[k][0])] = (
                    cracks_data[k][1],
                    cracks_data[k][2],
                    cracks_data[k][3],
                )

    # Short range search
    while (
        not any(
            (
                math.hypot((crack[0][0][0] - x1), (crack[0][0][1] - y1)) <= r
                and math.hypot((crack[0][1][0] - x2), (crack[0][1][1] - y2)) <= r
            )
            or (
                math.hypot((crack[0][1][0] - x1), (crack[0][1][1] - y1)) <= r
                and math.hypot((crack[0][0][0] - x2), (crack[0][0][1] - y2)) <= r
            )
            for crack in cracks_data.values()
        )
        and gen <= gen_max
    ):
        start_time = time.time()
        for i in range(3):
            for k in range(n_new_cracks_short_range):
                crack_transfer_learning = {}
                crack_data_transfer_learning = {}
                crack_transfer_learning[i] = cracks_data[i]
                crack_transfer_learning = hnnga.mutation(
                    cracks_dict=crack_transfer_learning,
                    mutation_rate=1.0,
                    mutation_strength=r,
                    limit_lengths=[0.1, 1.0],
                )
                crack_data_transfer_learning, history = hnnga.compute_cracks_data(
                    crack_dict=crack_transfer_learning,
                    target=strain_target,
                    target_coords=point_coords,
                    sig_ext_t=sig_ext_t,
                    sig_ext_b=sig_ext_b,
                    mode="strain",
                    history=history,
                    gen=gen,
                    n_epochs=n_epochs_short_range,
                    optimized_weights=cracks_data[i][2],
                    out_dir=out_dir,
                )
                cracks_data[i + 3 * (k + 1)] = crack_data_transfer_learning[0]

        cracks_data = hnnga.hnnga.sort_cracks_dict(
            cracks_dict=cracks_data, n=len(cracks_data)
        )

        end_time = time.time()
        duration_by_gen.append((gen, round(end_time - start_time, 3), len(cracks_data)))

        for k, kcrack in enumerate(cracks_data.values()):
            history[tuple(kcrack[0])] = (
                kcrack[1],
                kcrack[2],
                kcrack[3],
            )

        crack_best_by_gen.append(cracks_data[0][0])
        mse_best_by_gen.append(cracks_data[0][1])
        res_best_by_gen.append(cracks_data[0][3])

        if save_plots:
            plot.plot_crack_vs_target(
                crack=cracks_data[0][0],
                target=crack_target,
                residual=cracks_data[0][3],
                generation=gen,
                out_path=os.path.join(out_dir, f"crack_gen_{gen}.png"),
            )
            plot.plot_figure_population(
                population_dict=cracks_data,
                target=crack_target,
                generation=gen,
            )
        io.export_population(
            cracks_data=cracks_data,
            gen=gen,
            target=crack_target,
            file_path=os.path.join(out_dir),
            file_name=f"population_{gen}.csv",
        )
        io.export_dict(
            cracks_data=cracks_data,
            file_path=os.path.join(out_dir, f"population_{gen}.dict"),
        )
        gen += 1

    final_crack, key = hnnga.crack_target_dist_crit(
        cracks_dict=cracks_data, target_crack=crack_target, target_dist=r
    )

    if save_plots:
        plot.plot_final_result(
            crack=final_crack[0],
            target=crack_target,
            residual=final_crack[3],
            generation=gen - 1,
            out_path=os.path.join(out_dir, "crack_final.png"),
        )

    io.export_population(
        cracks_data={key: final_crack},
        target=crack_target,
        gen=gen - 1,
        file_path=os.path.join(out_dir),
        file_name="population_final.csv",
    )
    io.export_dict(
        cracks_data={key: final_crack},
        file_path=os.path.join(out_dir, "population_final.dict"),
    )

    # Residual plot
    y_values = [
        val.detach().cpu().item() if torch.is_tensor(val) else val
        for val in res_best_by_gen
    ]
    x_values = list(range(len(y_values)))

    fig, axes = plt.subplots()
    axes.plot(x_values, y_values, label="Residual")
    axes.grid(True)
    axes.legend()
    axes.set_title("Residual evolution")
    fig.savefig(os.path.join(out_dir, "residual_evolution.png"))
    plt.close(fig)

    # Calculate total duration of computations
    total_duration = sum(duration for _, duration, _ in duration_by_gen)

    # Get first and last generation numbers
    first_gen = duration_by_gen[0][0]
    last_gen = duration_by_gen[-1][0]

    # Calculate number of generations
    if first_gen == 0:
        num_generations = last_gen
    else:
        num_generations = last_gen - first_gen + 1

    # Avoid division by zero
    average_duration = (
        total_duration / num_generations
        if num_generations != 0
        else duration_by_gen[0][1]
    )

    with open(os.path.join(out_dir, "computation_time.txt"), mode="w") as f:
        f.write(f"{'Generation':<15}{'Duration (s)':<15}{'No. of Individuals':<20}\n")
        f.write("-" * 50 + "\n")
        for i, duration, population_size in duration_by_gen:
            f.write(f"{i:<15}{duration:<15.5f}{population_size:<20}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total duration: {total_duration} s\n")
        f.write(f"Average duration per generation: {average_duration} s\n")

    # Export corresponding MSE and residual values to file
    with open(os.path.join(out_dir, "mse_vs_residual.txt"), mode="w") as f:
        f.write(f"{'Generation':<15}{'MSE':<15}{'Residual':<15}\n")
        f.write("-" * 40 + "\n")
        for i in range(len(mse_best_by_gen)):
            f.write(f"{i:<15}{mse_best_by_gen[i]:<15.8f}{res_best_by_gen[i]:<15.8f}\n")


if __name__ == "__main__":
    main()
