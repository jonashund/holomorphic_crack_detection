import os
import csv
import pickle
import copy
import numpy as np


def read_strain_file(file_path):
    coords = []
    strain = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) != 6:
                continue  # Skip malformed lines

            coords.append([float(parts[1]), float(parts[2])])
            strain.append([float(parts[3]), float(parts[4]), float(parts[5])])

    coords_array = np.array(coords)  # Shape: [n_point_ids, 2]
    strain_array = np.array(strain)  # Shape: [n_point_ids, 3]

    return coords_array, strain_array


def export_crack(
    crack, target, residual, gen, file_path=os.path.join("crack_tips.txt")
):
    with open(file_path, mode="w") as f:
        f.write(f"Generation: {gen}\n")
        f.write(
            f"Crack: ({round(crack[0][0], 3)}, {round(crack[0][1], 3)}), ({round(crack[1][0], 3)}, {round(crack[1][1], 3)})\n"
        )
        f.write(
            f"Target: ({round(target[0][0], 3)}, {round(target[0][1], 3)}), ({round(target[1][0], 3)}, {round(target[1][1], 3)})\n"
        )
        f.write(f"Residual: {residual}")
    return


def export_population(
    cracks_data, gen, target, file_path=".", file_name="population.csv"
):
    file_path = os.path.join(file_path, file_name)
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Generation", gen])
        writer.writerow(["#", "crack", "target", "residual"])

        # Write data rows
        for key, value in cracks_data.items():
            writer.writerow(
                [key, (value[0][0], value[0][1]), (target[0], target[1]), value[3]]
            )
    return


def export_final_result(
    crack,
    target,
    residual,
    gen,
    file_path="final_result.txt",
):
    with open(file_path, mode="w") as f:
        f.write("No. of generations: {0}\n".format(gen))
        f.write("Residual: {0}\n".format(residual))
        f.write(
            "Crack: ({0},{1}), ({2},{3})\n".format(
                round(crack[0][0], 2),
                round(crack[0][1], 2),
                round(crack[1][0], 2),
                round(crack[1][1], 2),
            )
        )
        f.write(
            "Target crack: ({0},{1}), ({2},{3})".format(
                round(target[0][0], 2),
                round(target[0][1], 2),
                round(target[1][0], 2),
                round(target[1][1], 2),
            )
        )
    return


def export_dict(
    cracks_data, target=((-0.3, 0.0), (0.3, 0.0)), file_path="./cracks_data.dict"
):
    copy_dict = copy.deepcopy(cracks_data)
    new_dict = {}
    for key, value in copy_dict.items():
        new_dict[key] = [value[0], value[1], target, value[3]]
    with open(file_path, "wb") as f:
        pickle.dump(new_dict, f)
    return


def load_dict(file_path="./cracks_data.dict"):
    with open(file_path, "rb") as f:
        return pickle.load(f)
