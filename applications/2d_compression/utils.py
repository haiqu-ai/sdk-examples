import numpy as np
import json
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, ClassicalRegister


def create_floquet_circuit(theta=0.08*np.pi, num_floquet_layers=[10,], seed=42, defected_qubits=[46, 85, 116, 146]):
    """
    Generating a floquet circuit. Note that the circuit assumes the layout of Heron R2 or R3 devices
    and generates circuit on all 156 device qubits except for the defected qubits included in the 
    function arguments. 

    Args:
        theta (float): parameter controlling non-ergodicity
        num_floquet_layers (list[int]): numbers of trotter steps to return the circuit
        seed (int): random seed for phase gate angles generation
        defected_qubits (list[int]): qubits which are deleted from the circuit due to large errors on the real device

    Returns:
        list[QuantumCircuit]
    """
    # first, we find information about the device geometry
    num_qubits = 156
    rng = np.random.default_rng(seed)
    phase_angles = rng.uniform(-np.pi, np.pi, num_qubits)
    qc = QuantumCircuit(num_qubits)

    # preparing initial antiferromagnetic state
    with open('compression_2d_counts_and_classical_data/nodes_coloring.json', 'r') as f:
        c_nodes = json.load(f)
    # bipartitioning the graph into 2 groups to define antiferromagnetic state
    for n in range(num_qubits):
        if n in defected_qubits:
            continue
        if c_nodes[str(n)] == 0:
            qc.x(n)

    # partitioning (coloring) graph edges into 3 nonoverlapping groups and ordering the edges
    with open('compression_2d_counts_and_classical_data/edge_coloring.json', 'r') as f:
        c_str = json.load(f)
        c_edges = {}
        for k, v in c_str.items():
            c_edges[tuple(map(int, k.split(",")))] = v

    n_colors = max(c_edges.values()) + 1
    colored_edges = []
    for color in range(n_colors):
        edges = []
        for k, v in c_edges.items():
            if v == color:
                edges.append(k)
        colored_edges.append(edges)

    qcs = []
    for r in range(max(num_floquet_layers)):
        for class_edges in colored_edges:
            for na, nb in class_edges:
                if na in defected_qubits or nb in defected_qubits:
                    continue
                else:
                    qc.cz(na, nb)
                    qc.z(na)
                    qc.ry(theta, na)
                    qc.z(nb)
                    qc.ry(theta, nb)
            for i in range(num_qubits):
                if i in defected_qubits:
                    continue
                else:
                    qc.p(phase_angles[i], i)
        if r+1 in num_floquet_layers:
            # adding measurements
            circ = qc.copy()
            cr = ClassicalRegister(num_qubits - len(defected_qubits))
            circ.add_register(cr)
            cl_bit = 0
            for q in range(num_qubits):
                if q not in defected_qubits:
                    circ.measure(q, cl_bit)
                    cl_bit += 1
            qcs.append(circ)
    return qcs


def compute_magnetization(shots):
    """
    Computes magnetization from the shots dictionary.

    Args:
        shots (dict): mapping of bitstrings to counts

    Returns:
        float: average magnetization
    """
    normalization = 0.0
    magnetization = 0.0
    for string, v in shots.items():
        spins = [int(s) for s in list(string)]
        normalization += v
        num_qubits = len(spins)
        string_magnetization = 2 * np.sum(spins) / num_qubits - 1
        magnetization += v * string_magnetization
    magnetization = magnetization / normalization
    return magnetization


def print_compression_info(info, label=None):
    """Pretty-prints a compression info dictionary."""
    header = label or "Compression results"
    print(f"{header}")
    print(f"  Status       : {info.get('compression_status', 'N/A')}")
    print(f"  Quality      : {info.get('compression_quality', 0):.4f}")
    print(f"  Compression  : {info.get('compression_percent', 0):.2f}%")


def _style_axes(ax):
    """Consistent plot styling."""
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E0E0E0", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def compute_magnetizations(counts_original, counts_compressed):
    """
    Computes magnetizations from lists of shot count dictionaries.

    Args:
        counts_original (list[dict]): shot counts for original circuits
        counts_compressed (list[dict]): shot counts for compressed circuits

    Returns:
        tuple[list, list]: magnetizations for original and compressed circuits
    """
    magnetizations_original = []
    magnetizations_compressed = []
    for shots_original, shots_compressed in zip(counts_original, counts_compressed):
        magnetizations_original.append(compute_magnetization(shots_original))
        magnetizations_compressed.append(compute_magnetization(shots_compressed))
    return magnetizations_original, magnetizations_compressed


def plot_quality_vs_compression(layers, qualities, compressions):
    """
    Scatter plot of compression quality vs compression percent, annotated with layer counts.

    Args:
        layers (list[int]): Floquet layer counts
        qualities (list[float]): compression quality values
        compressions (list[float]): compression percent values
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    _style_axes(ax)

    ax.scatter(compressions, qualities,
               color="#CDE9F7", edgecolors="#333", linewidths=0.8, s=90, zorder=3)

    for layer, q, c in zip(layers, qualities, compressions):
        ax.annotate(str(layer), (c, q),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="#333")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Compression (%)", fontsize=11)
    ax.set_ylabel("Quality", fontsize=11)
    ax.set_title("Quality vs. compression by Floquet layer count",
                 fontsize=13, fontweight="bold", color="#333")
    plt.tight_layout()
    plt.show()
    return fig


def plot_magnetizations(layers, magnetizations_classical, magnetizations_original, magnetizations_compressed, device_id="ibm_boston"):
    """
    Plots magnetization vs Floquet layers for classical simulation and QPU results.

    Args:
        layers (list[int]): Floquet layer counts
        magnetizations_classical: classical simulation magnetizations
        magnetizations_original: original circuit QPU magnetizations
        magnetizations_compressed: compressed circuit QPU magnetizations
        device_id (str): device name used in the plot title
    """
    COLOR_CLASSICAL  = "#236CE6";  LINE_CLASSICAL  = "#1A4FA0";  MARKER_CLASSICAL  = "^"
    COLOR_ORIGINAL   = "#D9D9D9";  LINE_ORIGINAL   = "#888888";  MARKER_ORIGINAL   = "s"
    COLOR_COMPRESSED = "#CDE9F7";  LINE_COMPRESSED = "#5B9BC4";  MARKER_COMPRESSED = "o"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    _style_axes(ax)

    ax.plot(layers, np.real(magnetizations_classical),
            marker=MARKER_CLASSICAL, linestyle=":", color=LINE_CLASSICAL,
            linewidth=2, markersize=10,
            markerfacecolor=COLOR_CLASSICAL, markeredgecolor="#333", markeredgewidth=0.8,
            label="Classical simulation", zorder=4)
    ax.plot(layers, magnetizations_original,
            marker=MARKER_ORIGINAL, linestyle=":", color=LINE_ORIGINAL,
            linewidth=2, markersize=10,
            markerfacecolor=COLOR_ORIGINAL, markeredgecolor="#333", markeredgewidth=0.8,
            label="Original circuit (QPU)", zorder=2)
    ax.plot(layers, magnetizations_compressed,
            marker=MARKER_COMPRESSED, linestyle=":", color=LINE_COMPRESSED,
            linewidth=2, markersize=10,
            markerfacecolor=COLOR_COMPRESSED, markeredgecolor="#333", markeredgewidth=0.8,
            label="Compressed circuit (QPU)", zorder=3)

    ax.set_xticks(layers)
    ax.set_xlabel("Floquet layers", fontsize=11)
    ax.set_ylabel("Magnetization", fontsize=11)
    ax.set_title(f"2D State Compression: Magnetization on {device_id}",
                 fontsize=13, fontweight="bold", color="#333")
    ax.legend(framealpha=0.95, edgecolor="#CCC", fontsize=10)
    plt.tight_layout()
    plt.show()
    return fig
