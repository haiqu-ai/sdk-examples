import json

from qiskit import QuantumCircuit


def make_kicked_ising_circuit(
    num_qubits: int,
    num_steps: int,
    backend_coupling_map: list,
    theta_zz: float = 1.0,
    theta_x: float = 1.6,
    theta_z: float = 0.3,
):
    """
    Create a kicked Ising (Floquet) evolution circuit on ``num_qubits`` qubits with ``num_steps`` steps.

    The kicked Ising model is a periodically driven system whose single Floquet step applies three
    non-commuting terms:

        H_ZZ = theta_zz * sum_{<i,j>} Z_i Z_j   (nearest-neighbor ZZ interaction)
        H_X  = theta_x  * sum_i X_i              (transverse field in X direction)
        H_Z  = theta_z  * sum_i Z_i              (longitudinal field in Z direction)

    Parameters theta_zz=1.0, theta_x=1.6, theta_z=0.3 are chosen far from any integrable point,
    making the model classically intractable. The extra longitudinal field breaks integrability,
    large non-commuting gate angles drive rapid entanglement growth that breaks MPS/TN methods,
    and far-from-Clifford parameters defeat Pauli propagation approaches.

    Refer to "Scalable quantum error mitigation of mid-circuit measurements"
    (https://arxiv.org/abs/2508.10997) for more details.

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

    # Apply 3-color edge coloring for parallel gate execution
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

    for _ in range(num_steps):
        # Each Floquet step applies gates in 3 layers to parallelize RZZ gates
        for cmap in cmaps:
            # Apply RX and RZ gates on all qubits (divided by 3 for each layer)
            for i in range(num_qubits):
                qc.rx(theta_x / 3, i)
                qc.rz(theta_z / 3, i)
            # Apply ZZ interactions for this layer
            for q1, q2 in cmap:
                qc.rzz(theta_zz, q1, q2)

    return qc


def load_json_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def plot_magnetization(num_steps_list, ideal_mag, noisy_mag, baseline_mag, haiqu_mag):
    """Plot magnetization curves for all scenarios vs Floquet steps."""
    import matplotlib.pyplot as plt
    import numpy as np

    legend_labels = ["Ideal (noise-free simulation)", "Noisy (no error mitigation)",
                     "Baseline (PEC)", "Haiqu (proprietary mitigation)"]
    markers, linestyles = ["x", "o", "s", "d"], [":", "--", "--", "-"]
    colors = ["#333333", "#D9D9D9", "#CDE9F7", "#236CE6"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#FAFAFA"); ax.grid(True, color="#E0E0E0", linewidth=0.7); ax.set_axisbelow(True)

    steps = np.asarray(num_steps_list)
    for i, mag in enumerate([ideal_mag, noisy_mag, baseline_mag, haiqu_mag]):
        y = np.asarray(mag).ravel()
        x = steps[:len(y)]
        ax.plot(x, y, label=legend_labels[i],
                marker=markers[i], linestyle=linestyles[i], color=colors[i],
                linewidth=2, markersize=8, markeredgecolor="#333", markeredgewidth=1.2)

    fig.suptitle("103-qubit Kicked Ising (Floquet) Model on IBM Kingston",
                 fontsize=14, color="#333", fontweight="bold", y=0.98)
    ax.set(xlabel="Floquet Steps", ylabel="Magnetization $M$")
    ax.set_xticks(num_steps_list)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#333", labelsize=11)
    ax.legend(framealpha=0.95, edgecolor="#CCC", fontsize=10)
    fig.text(0.5, 0.01, "Baseline reference: arXiv:2508.10997 (2025)",
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


def plot_pareto(num_steps_list, ideal_mag, haiqu_mag, haiqu_bill, baseline_mag, baseline_bill, noisy_mag, noisy_bill):
    """Plot Pareto analysis of cost per circuit vs accuracy."""
    import matplotlib.pyplot as plt
    import numpy as np

    ideal_mag, haiqu_mag, comp_mag, noisy_mag = [
        np.array(m) for m in [ideal_mag, haiqu_mag, baseline_mag, noisy_mag]
    ]
    steps, rzz = num_steps_list, 108
    h_gates = [s * rzz for s in steps]
    c_gates = [s * rzz for s in steps[:len(comp_mag)]]
    n_gates = [s * rzz for s in steps]

    h_acc = [100 * (1 - abs(haiqu_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(steps))]
    c_acc = [100 * (1 - abs(comp_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(comp_mag))]
    n_acc = [100 * (1 - abs(noisy_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(steps))]

    # Haiqu: flat cost + visual offset; Baseline: (1.003)^gamma scaled to sum=$17,856; Noisy: flat cost
    h_base = haiqu_bill / len(steps)
    h_costs = [h_base + i * 3 for i in range(len(steps))]
    raw_w = [1.003 ** g for g in c_gates]
    c_costs = [w / sum(raw_w) * baseline_bill for w in raw_w]
    n_base = noisy_bill / len(steps)
    n_costs = [n_base + i * 2.5 for i in range(len(steps))]

    sz = lambda g: 60 + (g - 500) / 500 * 240

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="both", color="#E0E0E0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    # Pareto frontier & data points
    ax.plot([h_costs[0], h_costs[-1], c_costs[0]], [h_acc[0], h_acc[-1], c_acc[0]],
            "--", color="#333", lw=2, zorder=2, label="Pareto Frontier")
    for costs, acc, gates, color, lbl, tc in [
        (h_costs, h_acc, h_gates, "#236CE6", "Haiqu", "#236CE6"),
        (c_costs, c_acc, c_gates, "#CDE9F7", "Baseline", "#555"),
        (n_costs, n_acc, n_gates, "#D9D9D9", "Noisy", "#777"),
    ]:
        ax.scatter(costs, acc, s=[sz(g) for g in gates], c=color,
                  edgecolors="#333", linewidths=1.5, zorder=3, label=lbl)
        for i, g in enumerate(gates):
            ax.annotate(str(g), (costs[i], acc[i]), xytext=(0, 14),
                        textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color=tc)

    ax.set(xscale="log", xlim=(1, 15000), ylim=(85, 101),
           xlabel="Cost ($ log scale)", ylabel="Accuracy (%)")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["$1", "$10", "$100", "$1K", "$10K"])
    ax.set_title("Pareto Analysis: Cost per Circuit vs Accuracy",
                 fontsize=14, fontweight="bold", color="#333")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#333", labelsize=11)
    ax.legend(framealpha=0.95, edgecolor="#CCC", fontsize=10, loc="lower right")
    plt.tight_layout()
    return plt

def plot_pareto(num_steps_list, ideal_mag, haiqu_mag, haiqu_bill, baseline_mag, baseline_bill, noisy_mag, noisy_bill):
    """Plot Pareto analysis of cost per circuit vs accuracy."""
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgb

    ideal_mag, haiqu_mag, comp_mag, noisy_mag = [
        np.array(m) for m in [ideal_mag, haiqu_mag, baseline_mag, noisy_mag]
    ]
    steps, rzz = num_steps_list, 108
    h_gates = [s * rzz for s in steps]
    c_gates = [s * rzz for s in steps[:len(comp_mag)]]
    n_gates = [s * rzz for s in steps]

    h_acc = [100 * (1 - abs(haiqu_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(steps))]
    c_acc = [100 * (1 - abs(comp_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(comp_mag))]
    n_acc = [100 * (1 - abs(noisy_mag[i] - ideal_mag[i]) / abs(ideal_mag[i])) for i in range(len(steps))]

    # Haiqu: flat cost + visual offset; Baseline: (1.003)^gamma scaled to sum=$17,856; Noisy: flat cost
    h_base = haiqu_bill / len(steps)
    h_costs = [h_base + i * 3 for i in range(len(steps))]
    raw_w = [1.003 ** g for g in c_gates]
    c_costs = [w / sum(raw_w) * baseline_bill for w in raw_w]
    n_base = noisy_bill / len(steps)
    n_costs = [n_base + i * 2.5 for i in range(len(steps))]

    sz = lambda g: 60 + (g - 500) / 500 * 240

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, which="both", color="#E0E0E0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    # Hacky way to add a number for annotations to the lengend 
    plt.plot([], [], marker=f'${972}$', markersize=16, markerfacecolor="#333", markeredgecolor="#333", color="#333", linestyle='none', label="2-qubit gate count")
    # Pareto frontier & data points
    ax.plot([n_costs[0], h_costs[0], h_costs[-1], c_costs[0]], [n_acc[0], h_acc[0], h_acc[-1], c_acc[0]],
            ":", color="#333", lw=2, zorder=2, alpha=0.55, label="Pareto Frontier")
    for costs, acc, gates, color, lbl, tc in [
        (h_costs, h_acc, h_gates, "#236CE6", "Haiqu", "#333"), # "#236CE6"
        (c_costs, c_acc, c_gates, "#CDE9F7", "Baseline", "#777"), # "#555"
        (n_costs, n_acc, n_gates, "#D9D9D9", "Noisy", "#777"),
    ]:
        ax.scatter(costs, acc, s=160, c=color, #s=[sz(g) for g in gates],
                  edgecolors="#333", linewidths=1.5, zorder=3, label=lbl)
        for i, g in enumerate(gates):
            ax.annotate(str(g), (costs[i], acc[i]), xytext=(5, 10),
                        textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color=tc)
        
        x, y = np.array(costs), np.array(acc)
        if len(x) > 1:
            log_x = np.log10(x)            
            # Calculate the Center (Midpoint of extremes for guaranteed coverage)
            cx = (np.max(log_x) + np.min(log_x)) / 2
            cy = (np.max(y) + np.min(y)) / 2         
            # Calculate Rotation (Angle) using Covariance
            cov = np.cov(log_x, y)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta_deg = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            # Calculate Radii with Padding
            padding_w = 1.7 
            padding_h = 2.2 
            width = np.sqrt(vals[0]) * padding_w
            height = np.sqrt(vals[1]) * padding_h
            # Generate the Rotated Shape
            t = np.linspace(0, 2*np.pi, 100)
            # Basic ellipse coordinates
            ex = width * np.cos(t)
            ey = height * np.sin(t)
            # Rotate the coordinates
            R = np.array([[np.cos(np.radians(theta_deg)), -np.sin(np.radians(theta_deg))],
                          [np.sin(np.radians(theta_deg)),  np.cos(np.radians(theta_deg))]])
            rotated_coords = R @ np.vstack([ex, ey])
            # Move to center and convert X back from Log to Linear
            final_x = 10**(rotated_coords[0, :] + cx)
            final_y = rotated_coords[1, :] + cy
            # Plot as a Polygon to handle the log-transformation perfectly
            poly = patches.Polygon(np.column_stack([final_x, final_y]), closed=True,
                           edgecolor=(*to_rgb(color), 0.65),
                           facecolor=(*to_rgb(color), 0.1), 
                           linewidth=2.5, linestyle='--', zorder=1)
            ax.add_patch(poly)

    ax.set(xscale="log", xlim=(1, 15000), ylim=(82.5, 101),
           xlabel="Cost ($ log scale)", ylabel="Accuracy (%)")
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["$1", "$10", "$100", "$1K", "$10K"])
    ax.set_title("Pareto Analysis: Cost per Circuit vs Accuracy",
                 fontsize=14, fontweight="bold", color="#333")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333"); ax.spines["left"].set_color("#333")
    ax.tick_params(colors="#333", labelsize=11)
    ax.legend(framealpha=0.95, edgecolor="#CCC", fontsize=10, loc="lower right", handlelength=3, handletextpad=1.2)
    plt.tight_layout()
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
    filename : str, default "utils/coupling_map.json"
        Filename to load coupling map from if coupling_map is None.
    figsize : tuple, default (8, 8)
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
