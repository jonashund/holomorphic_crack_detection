from experiment_1.exp_1_central_crack import main
from pathlib import Path

base_path = Path(__file__).parent
target_path = base_path.joinpath("..", "experiment_1", "fe_solution")


def test_experiment_1():
    main(
        target_dir=target_path,
        gen_max=2,
    )
