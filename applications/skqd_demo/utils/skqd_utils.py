import json
import matplotlib.pyplot as plt


def load_json_data(filename):
    with open(filename) as f:
        return json.load(f)


def plot_quality_and_determinants(
    haiqu_result,
    qiskit_result,
    dmrg_energy,
    ibm_tutorial_energy,
    ibm_num_determinant,
):
    """Plot quality (1 - |error|) convergence and subspace dimension vs SQD iterations."""
    qiskit_energies = qiskit_result.best_energy_per_iteration()
    qiskit_dimensions = qiskit_result.best_subspace_dimension_per_iteration()
    haiqu_energies = haiqu_result.best_energy_per_iteration()
    haiqu_dimensions = haiqu_result.best_subspace_dimension_per_iteration()

    x_qiskit = range(len(qiskit_energies))
    x_haiqu = range(len(haiqu_energies))

    quality_haiqu = [1.0 - abs((e - dmrg_energy) / dmrg_energy) for e in haiqu_energies]
    quality_qiskit = [1.0 - abs((e - dmrg_energy) / dmrg_energy) for e in qiskit_energies]
    quality_ibm = 1.0 - abs((ibm_tutorial_energy - dmrg_energy) / dmrg_energy)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Quality convergence
    axs[0].plot(x_qiskit, quality_qiskit, label="Baseline (no compression)", marker="o", color="#CDE9F7")
    axs[0].plot(x_haiqu, quality_haiqu, label="Haiqu (compressed)", marker="o", color="#236CE6")
    axs[0].axhline(y=quality_ibm, color="#333333", linestyle="--", label="IBM tutorial")
    axs[0].set_title("Quality vs SQD Iteration")
    axs[0].set_xlabel("Iteration Index", fontdict={"fontsize": 12})
    axs[0].set_ylabel("Quality  (1 \u2212 |E \u2212 E_DMRG| / |E_DMRG|)", fontdict={"fontsize": 12})
    axs[0].legend()

    # Subspace dimension
    axs[1].plot(x_qiskit, qiskit_dimensions, label="Baseline (no compression)", marker="o", color="#CDE9F7")
    axs[1].plot(x_haiqu, haiqu_dimensions, label="Haiqu (compressed)", marker="o", color="#236CE6")
    axs[1].axhline(y=ibm_num_determinant, color="#333333", linestyle="--", label="IBM tutorial")
    axs[1].set_title("Determinants Used vs SQD Iteration")
    axs[1].set_xlabel("Iteration Index", fontdict={"fontsize": 12})
    axs[1].set_ylabel("Subspace Dimension (# determinants)", fontdict={"fontsize": 12})
    axs[1].legend()

    plt.tight_layout()
    return fig


def plot_pareto_quality_vs_time(
    haiqu_result,
    qiskit_result,
    dmrg_energy,
    ibm_tutorial_energy,
):
    """Pareto plot: quality (0-1) on y-axis vs cumulative postprocessing time on x-axis."""
    haiqu_energies = haiqu_result.best_energy_per_iteration()
    qiskit_energies = qiskit_result.best_energy_per_iteration()

    time_haiqu = haiqu_result.runtimes_per_iteration()
    time_qiskit = qiskit_result.runtimes_per_iteration()

    cumulative_haiqu = [sum(time_haiqu[: i + 1]) / 60 for i in range(len(time_haiqu))]
    cumulative_qiskit = [sum(time_qiskit[: i + 1]) / 60 for i in range(len(time_qiskit))]

    quality_haiqu = [1.0 - abs((e - dmrg_energy) / dmrg_energy) for e in haiqu_energies]
    quality_qiskit = [1.0 - abs((e - dmrg_energy) / dmrg_energy) for e in qiskit_energies]
    quality_ibm = 1.0 - abs((ibm_tutorial_energy - dmrg_energy) / dmrg_energy)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(cumulative_qiskit, quality_qiskit, marker="o", label="Baseline (no compression)", color="#CDE9F7", linewidth=2)
    ax.plot(cumulative_haiqu, quality_haiqu, marker="o", label="Haiqu (compressed)", color="#236CE6", linewidth=2)
    ax.axhline(y=quality_ibm, color="#333333", linestyle="--", label="IBM tutorial")

    ax.set_xlabel("Cumulative Postprocessing Time (minutes)", fontdict={"fontsize": 12})
    ax.set_ylabel("Quality  (1 \u2212 |E \u2212 E_DMRG| / |E_DMRG|)", fontdict={"fontsize": 12})
    ax.set_title("Pareto Front: Quality vs Postprocessing Cost")
    ax.legend()

    plt.tight_layout()
    return fig
