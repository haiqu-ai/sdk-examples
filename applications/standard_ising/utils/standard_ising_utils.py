import json
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def make_ising_evolution_circuit(
    num_qubits: int,
    num_steps: int,
    backend_coupling_map: list,
    theta_J: float = -np.pi / 2,
):
    """
    Create an Ising evolution circuit on ``num_qubits`` qubits with ``num_steps`` steps.

    The coupling strength ``J`` and transverse field strength ``h`` from the Hamiltonian are combined with the time step ``t`` to
    form ``theta_J = -2 J t`` and ``theta_h = 2 h t``. Refer to "Evidence for the utility of quantum computing before fault
    tolerance" (https://doi.org/10.1038/s41586-023-06096-3) for more details.

    Note that this function is written to optimize for ``backend_coupling_map`` with heavy-hex topology.
    """

    # Validate coupling map has enough qubits
    if backend_coupling_map:
        max_qubit_in_map = max(max(edge) for edge in backend_coupling_map)
        if num_qubits > max_qubit_in_map + 1:
            raise ValueError(
                f"Cannot create {num_qubits}-qubit circuit: "
                f"coupling map only has {max_qubit_in_map + 1} qubits available"
            )

    cmap_full = sorted(backend_coupling_map)
    cmaps = [[], [], []]
    cmap_qubitss = [set(), set(), set()]

    for i, j in cmap_full:
        if i >= num_qubits or j >= num_qubits:
            continue

        for k in range(3):
            if [j, i] in cmaps[k]:
                break
            elif i not in cmap_qubitss[k] and j not in cmap_qubitss[k]:
                cmaps[k].append([i, j])
                cmap_qubitss[k].add(i)
                cmap_qubitss[k].add(j)
                break
        else:
            raise ValueError(
                f"Cannot fit edge ({i}, {j}) into any of the 3 parallel layers. "
                f"The coupling map has too much connectivity for the 3-layer decomposition. "
                f"This typically means the device topology is too densely connected."
            )

    # Check that all qubits have interactions
    all_covered_qubits = set()
    for cmap in cmaps:
        for i, j in cmap:
            all_covered_qubits.add(i)
            all_covered_qubits.add(j)

    if len(all_covered_qubits) < num_qubits:
        raise ValueError(
            f"Cannot create {num_qubits}-qubit circuit: "
            f"coupling map only covers {len(all_covered_qubits)} qubits with edges"
        )

    qc = QuantumCircuit(num_qubits)
    theta_h = Parameter("θ_h")

    for _ in range(num_steps):
        # Apply transverse field: exp(-i * h * t * X_i) for each qubit
        for i in range(num_qubits):
            qc.rx(theta_h, i)

        # Apply ZZ interactions: exp(i * J * t * Z_i Z_j) for each edge
        for cmap in cmaps:
            for i, j in cmap:
                qc.rzz(theta_J, i, j)
    return qc


def load_json_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def plot_magnetization(transverse_field_params, ideal_mag, noisy_mag, ibm_mag, haiqu_mag):
    """Plot magnetization curves for all scenarios vs transverse field strength."""
    import matplotlib.pyplot as plt

    legend_labels = ["Ideal (noise-free simulation)", "Noisy (no error mitigation)",
                     "Baseline (SPL + ZNE + PEA)", "Haiqu (proprietary mitigation)"]
    markers, linestyles = ["x", "o", "s", "d"], [":", "--", "--", "-"]
    colors = ["#333333", "#D9D9D9", "#CDE9F7", "#236CE6"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#FAFAFA"); ax.grid(True, color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)

    x = np.asarray(transverse_field_params).ravel()
    for i, mag in enumerate([ideal_mag, noisy_mag, ibm_mag, haiqu_mag]):
        ax.plot(x, np.asarray(mag).ravel(), label=legend_labels[i],
                marker=markers[i], linestyle=linestyles[i], color=colors[i],
                linewidth=2, markersize=8, markeredgecolor="#333", markeredgewidth=1.2)

    fig.suptitle("127-qubit 2D Transverse-Field Ising Model on Baseline Boston",
                 fontsize=14, color="#333", fontweight="bold", y=0.98)
    ax.set(xlabel="Transverse Field Strength $h$", ylabel="Magnetization $M$")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#333", labelsize=11)
    ax.legend(framealpha=0.95, edgecolor="#CCC", fontsize=10)
    fig.text(0.5, 0.01, "IBM Baseline reference: Kim et al., Nature 618, 500–505 (2023)",
             ha="center", fontsize=8, color="#888")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return plt


def plot_performance(labels, costs, times):
    """Plot cost and time bar charts for scenarios with runtime data."""
    import matplotlib.pyplot as plt

    color_map = {"Haiqu": "#236CE6", "Baseline": "#CDE9F7"}
    bar_colors = [color_map.get(lbl, "#D9D9D9") for lbl in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("Performance Comparison: Cost & Time", fontsize=14, color="#333", fontweight="bold", y=1.02)

    bar_kw = dict(color=bar_colors, edgecolor="#333", linewidth=1.5, width=0.6)
    for ax, vals, ylabel, title, fmt in [
        (ax1, costs, "Quantum Cloud Bill ($)", "Cost", lambda v: f"${v:,.2f}"),
        (ax2, times, "Runtime (minutes)", "Time",
         lambda v: f"{v:.0f} min" if v >= 1 else f"{v * 60:.0f} sec"),
    ]:
        bars = ax.bar(labels, vals, **bar_kw)
        ax.set_yscale("log"); ax.set_ylabel(ylabel, color="#333", fontsize=12)
        ax.set_title(title, fontsize=12, color="#555", pad=25)
        ax.tick_params(colors="#333", labelsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                    fmt(v), ha="center", va="bottom", color="#333", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return plt


def plot_pareto(ideal_mag, scenario_names, scenario_costs, scenario_mags):
    """Plot Pareto front of quality (MAE %) vs cost ($)."""
    import matplotlib.pyplot as plt

    color_map = {"Haiqu": "#236CE6", "Baseline": "#CDE9F7", "Noisy": "#D9D9D9"}
    ideal = np.asarray(ideal_mag)
    pareto = [(name, cost,
               np.mean(np.abs(np.asarray(mag) - ideal)) * 100,
               color_map.get(name, "#D9D9D9"))
              for name, cost, mag in zip(scenario_names, scenario_costs, scenario_mags)]
    p_labels, p_costs, p_mae, p_colors = zip(*pareto)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#FAFAFA"); ax.grid(True, color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)
    ax.scatter(p_costs, p_mae, s=200, c=p_colors, edgecolors="#333", linewidths=1.5, zorder=5)

    for lbl, cost, mae in zip(p_labels, p_costs, p_mae):
        ax.annotate(f"{lbl}\n${cost:,.0f} | {mae:.2f}% error", (cost, mae),
                    textcoords="offset points", xytext=(0, 25), ha="center",
                    fontsize=10, fontweight="bold", color="#333",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Quantum Cloud Bill ($)", fontsize=12, color="#333")
    ax.set_ylabel("Mean Absolute Error (%)  — lower is better", fontsize=12, color="#333")
    fig.suptitle("Pareto Front: Quality vs. Cost", fontsize=14, color="#333", fontweight="bold", y=0.98)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#333", labelsize=11)
    y_lo, y_hi = ax.get_ylim(); ax.set_ylim(max(0, y_lo - 0.3), y_hi + 1.2)
    x_lo, x_hi = ax.get_xlim(); ax.set_xlim(x_lo * 0.5, x_hi * 2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt


# Utility function to visualize device connectivity
def show_device_connectivity(
    coupling_map=None,
    filename="utils/coupling_map.json",
    figsize=(8, 8),
    label_qubits=True,
):
    """
    Visualize a quantum device coupling map.

    Parameters:
    -----------
    coupling_map : list, optional
        List of qubit pairs representing connections. If None, loads from file.
    filename : str, default "coupling_map"
        Filename to load coupling map from if coupling_map is None.
    figsize : tuple, default (20, 20)
        Figure size in inches (width, height).
    label_qubits : bool, default True
        Whether to label qubits in the visualization.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the coupling map visualization.
    """
    from qiskit.visualization import plot_coupling_map
    import matplotlib.pyplot as plt
    from IPython.display import display

    # Load coupling map if not provided
    if coupling_map is None:
        coupling_map = load_json_data(filename)

    # Get number of qubits from the coupling map
    num_qubits = max(max(edge) for edge in coupling_map) + 1

    # Plot the coupling map (qubit_coordinates=None will auto-layout)
    fig = plot_coupling_map(
        num_qubits=num_qubits,
        qubit_coordinates=None,  # Auto-layout
        coupling_map=coupling_map,
        figsize=figsize,
        label_qubits=label_qubits,
    )

    # Explicitly display the figure
    display(fig)
