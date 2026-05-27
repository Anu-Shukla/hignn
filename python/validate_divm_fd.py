import hignn
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
EPS = 0.005
MODEL_PATH = "nn/two_body_unbounded"
USE_SYMMETRY = True


def make_positions():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.5, 0.3, 0.1],
            [0.4, 2.7, 0.2],
            [0.2, 0.5, 3.0],
        ],
        dtype=np.float32,
    )


def compute_mobility_column(model, position, particle, component):
    position = np.ascontiguousarray(position, dtype=np.float32)
    force = np.zeros_like(position, dtype=np.float32)
    velocity = np.zeros_like(position, dtype=np.float32)
    divm = np.zeros_like(position, dtype=np.float32)

    force[particle, component] = 1.0
    model.update_coord(position)
    model.dot(velocity, force, divm)

    return velocity.astype(np.float64)


def compute_autograd_divm(model, position):
    position = np.ascontiguousarray(position, dtype=np.float32)
    force = np.zeros_like(position, dtype=np.float32)
    velocity = np.zeros_like(position, dtype=np.float32)
    divm = np.zeros_like(position, dtype=np.float32)

    model.update_coord(position)
    model.dot(velocity, force, divm)

    return divm.astype(np.float64)


def compute_fd_divm(model, position, eps):
    position = np.ascontiguousarray(position, dtype=np.float32)
    fd_divm = np.zeros_like(position, dtype=np.float64)

    for particle in range(position.shape[0]):
        for component in range(position.shape[1]):
            if rank == 0:
                print(f"FD column particle={particle}, component={component}")

            plus_position = position.copy()
            minus_position = position.copy()
            plus_position[particle, component] += eps
            minus_position[particle, component] -= eps

            plus_column = compute_mobility_column(
                model, plus_position, particle, component
            )
            minus_column = compute_mobility_column(
                model, minus_position, particle, component
            )

            fd_divm += (plus_column - minus_column) / (2.0 * eps)

    return fd_divm


def print_comparison(fd_divm, autograd_divm):
    abs_error = np.abs(fd_divm - autograd_divm)
    rel_error = abs_error / np.maximum(np.abs(fd_divm), 1e-12)

    if rank != 0:
        return

    print("\nFD divM:")
    print(fd_divm)
    print("\nAutograd divM:")
    print(autograd_divm)
    print("\nError summary:")
    print(f"max abs error:  {np.max(abs_error):.6e}")
    print(f"mean abs error: {np.mean(abs_error):.6e}")
    print(f"max rel error:  {np.max(rel_error):.6e}")
    print(f"mean rel error: {np.mean(rel_error):.6e}")


def configure_model(model, model_path, use_symmetry):
    model.load_two_body_model(model_path)
    model.set_epsilon(0.01)
    model.set_max_iter(50)
    model.set_mat_pool_size_factor(200)
    model.set_post_check_flag(False)
    model.set_use_symmetry_flag(use_symmetry)
    model.set_max_far_dot_work_node_size(10000)
    model.set_max_relative_coord(100000)


def main():
    position = make_positions()

    if rank == 0:
        print(f"Running divM finite-difference check with N={position.shape[0]}")
        print(f"eps={EPS}")
        print(f"use_symmetry={USE_SYMMETRY}")

    hignn.Init()
    model = None
    try:
        model = hignn.HignnModel(position, 2)
        configure_model(model, MODEL_PATH, USE_SYMMETRY)

        autograd_divm = compute_autograd_divm(model, position)
        fd_divm = compute_fd_divm(model, position, EPS)

        print_comparison(fd_divm, autograd_divm)
    finally:
        del model
        hignn.Finalize()


if __name__ == "__main__":
    main()
