from datetime import datetime
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

COLOR_HAIQU = "#092382"
COLOR_QISKIT = "#C4D5FF"
COLOR_BASELINE = "#333333"

# image preprocessing


def get_image():
    img = Image.open("eiffel_tower.jpg").convert("L")
    img_np = np.asarray(img, dtype=np.float64) / 255.0

    H, W = img_np.shape
    min_dim = min(H, W)

    start_h = (H - min_dim) // 2
    start_w = (W - min_dim) // 2

    cropped = img_np[start_h: start_h + min_dim, start_w: start_w + min_dim]

    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8))
    return cropped_img


def get_resized_image(cropped_img, target_size):
    target_height, target_width = target_size
    resized_img = cropped_img.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

    resized_np = np.asarray(resized_img, dtype=np.float64) / 255
    return resized_np


def pad_to_power_of_two(vec, normalize=True):
    """
    Pads a vector with zeros until its length is a power of two.

    Parameters
    ----------
    vec : array-like
    normalize : bool
        Whether to renormalize after padding.

    Returns
    -------
    padded_vec : np.ndarray
    """
    vec = np.asarray(vec, dtype=np.float64)

    n = len(vec)
    target_len = 1 << (n - 1).bit_length()  # next power of two

    if n == target_len:
        padded = vec.copy()
    else:
        padded = np.zeros(target_len, dtype=vec.dtype)
        padded[:n] = vec

    if normalize:
        norm = np.linalg.norm(padded)
        if norm > 1e-6:
            padded = padded / norm

    return padded


def get_num_qubits(res, tiles):
    num_q = np.log2((res[0] / tiles[0]) *
                    (res[1] / tiles[1])) * tiles[0] * tiles[1]
    return num_q


# postprocessing


def serialize_analytics(analytics_obj):
    return dict(analytics_obj) if hasattr(analytics_obj, "__iter__") else analytics_obj


def serialize_cost(dry_job):
    return {
        label: {
            "amount": float(estimate["amount"]),
            "unit": estimate["unit"],
        }
        for label, estimate in dry_job.estimated_qpu_cost.items()
    }

# image postproceessing


def get_normalization_coefficents(image_resized, NUM_ROW_BLOCKS, NUM_COL_BLOCKS):
    H, W = image_resized.shape
    rows = np.linspace(0, H, NUM_ROW_BLOCKS + 1, dtype=int)
    cols = np.linspace(0, W, NUM_COL_BLOCKS + 1, dtype=int)

    normalization_coeffs = []
    for i in range(NUM_ROW_BLOCKS):
        for j in range(NUM_COL_BLOCKS):
            h1, h2 = rows[i], rows[i + 1]
            w1, w2 = cols[j], cols[j + 1]
            block = image_resized[h1:h2, w1:w2]

            flat_block = block.flatten()
            norm_coeff = np.linalg.norm(flat_block)
            normalization_coeffs.append(norm_coeff)
    return normalization_coeffs


def split_measurements_for_block_encoding(measurement_dict, block_qubit_sizes):
    block_probs = [np.zeros(2**q) for q in block_qubit_sizes]
    ranges = []
    pos = 0
    for q in block_qubit_sizes:
        ranges.append((pos, pos + q))
        pos += q

    for bitstring, prob in measurement_dict.items():
        for i, (start, end) in enumerate(ranges):
            block_bits = bitstring[start:end]
            idx = int(block_bits, 2)
            block_probs[i][idx] += prob

    return block_probs


def reconstruct_image(
    block_probs, normalization_coeffs, target_size, num_row_blocks, num_col_blocks
):
    block_height = target_size // num_row_blocks
    block_width = target_size // num_col_blocks

    reconstructed_blocks = []
    for probs, norm in zip(block_probs, normalization_coeffs):
        s = np.sum(probs)
        if s > 1e-6:
            probs /= s
        probs = np.clip(probs, 0.0, None)  # guard against float-negative QPU probabilities
        amps = np.sqrt(probs)
        block_pixels = amps * norm
        reconstructed_blocks.append(
            block_pixels.reshape((block_height, block_width)))

    image = np.zeros((target_size, target_size))
    idx = 0
    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            h1, h2 = i * block_height, (i + 1) * block_height
            w1, w2 = j * block_width, (j + 1) * block_width
            image[h1:h2, w1:w2] = reconstructed_blocks[idx]
            idx += 1

    return image


def image_from_block_results(res_dict, image_resized, tiles):
    NUM_ROW_BLOCKS, NUM_COL_BLOCKS = tiles
    block_height = image_resized.shape[0] // NUM_ROW_BLOCKS
    block_width = image_resized.shape[1] // NUM_COL_BLOCKS
    pixels_per_block = block_height * block_width
    qubits_per_block = int(np.ceil(np.log2(pixels_per_block)))

    block_qubit_sizes = [qubits_per_block] * (NUM_ROW_BLOCKS * NUM_COL_BLOCKS)
    block_probs = split_measurements_for_block_encoding(
        res_dict, block_qubit_sizes)
    normalization_coeffs = get_normalization_coefficents(
        image_resized, NUM_ROW_BLOCKS, NUM_COL_BLOCKS
    )
    reconstructed = reconstruct_image(
        block_probs,
        normalization_coeffs,
        image_resized.shape[0],
        NUM_ROW_BLOCKS,
        NUM_COL_BLOCKS,
    )
    return reconstructed


def bitstringdict_to_prob(res_dict, resolution):
    """Takes results dictionary and converts it into a probability array"""
    num_qubits = len(
        list(res_dict.keys())[0]
    )  # find number of qubits by bitstring length
    res = np.zeros(2**num_qubits)
    for bs, p in res_dict.items():
        res[int(bs, 2)] = p
    image = res.reshape(resolution)
    return image


def prep_state_and_disp(img):
    img = img.astype(float)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0.0, None)

    state = np.sqrt(img).ravel()
    norm = np.linalg.norm(state)
    if norm > 1e-12:
        state /= norm
    else:
        state = np.zeros_like(state)
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    disp = img - img.min()
    mx = disp.max()
    if mx > 1e-6:
        disp /= mx
    return state, disp


def plot_image_encoding_grid(
    results,
    cropped_img,
    save_path="figures/clean_grid_6x5.png",
    dpi=300,
    show=True,
    column_names=None,
):
    if column_names is None:
        column_names = [
            "Original",
            "Classical Block",
            "QPU Block",
            "Classical Qiskit",
            "QPU Qiskit",
        ]

    cmap = plt.cm.gray.copy()

    n = len(results["experiments"])
    fig, axes = plt.subplots(n, 5, figsize=(
        8, 1.0 * n), constrained_layout=True)

    last_im = None
    for idx, resolution_data in enumerate(results["experiments"]):
        resolution = resolution_data["resolution"]
        res = resolution_data["images"]

        images_to_show = [
            get_resized_image(cropped_img, target_size=resolution),
            res.get("classical", {}).get("block_vector_loading"),
            res.get("qpu", {}).get("block_vector_loading"),
            res.get("classical", {}).get("qiskit_sp"),
            res.get("qpu", {}).get("qiskit_sp"),
        ]

        for col_idx, img_data in enumerate(images_to_show):
            ax = axes[idx, col_idx]

            missing = img_data is None
            if not missing:
                arr = np.asarray(img_data, dtype=object)
                missing = arr.size == 0 or np.all(arr == None)

            if missing:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                _, disp = prep_state_and_disp(np.array(img_data))
                last_im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=1)

            if idx == 0:
                ax.set_title(column_names[col_idx], pad=10)

            if col_idx == 0:
                ax.set_ylabel(
                    f"Res: {resolution}",
                    rotation=45,
                    labelpad=40,
                    va="center",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

    if last_im is not None:
        cb = fig.colorbar(last_im, ax=axes, location="right", shrink=0.4)
        cb.set_label("Intensity")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes


def plot_fidelity_vs_resolution(
    results,
    cropped_img,
    save_path="figures/fidelity_vs_resolution_y_axis.png",
    dpi=300,
    show=True,
):
    plot_data = {
        "Haiqu Block (QPU)": {"labels": [], "areas": [], "fids": []},
        "Qiskit SP (QPU)": {"labels": [], "areas": [], "fids": []},
    }

    for resolution_data in results["experiments"]:
        res_tuple = resolution_data["resolution"]
        res_label = f"({res_tuple[0]}, {res_tuple[1]})"
        res_area = res_tuple[0] * res_tuple[1]

        res_images = resolution_data["images"]
        ideal = get_resized_image(cropped_img, target_size=res_tuple)
        ideal_state, _ = prep_state_and_disp(ideal)
        encodings = [
            (
                "Haiqu Block (QPU)",
                res_images.get("qpu", {}).get("block_vector_loading"),
            ),
            ("Qiskit SP (QPU)", res_images.get("qpu", {}).get("qiskit_sp")),
        ]

        for label, img in encodings:
            missing = img is None
            if not missing:
                arr = np.asarray(img, dtype=object)
                missing = arr.size == 0 or np.all(arr == None)

            if not missing:
                st, _ = prep_state_and_disp(np.array(img))
                fid = float(np.abs(np.dot(ideal_state, st)) ** 2)
                plot_data[label]["labels"].append(res_label)
                plot_data[label]["areas"].append(res_area)
                plot_data[label]["fids"].append(fid)

    fig, ax = plt.subplots(figsize=(10, 8))
    style_map = {
        "Haiqu Block (QPU)": {"color": COLOR_HAIQU, "marker": "x"},
        "Qiskit SP (QPU)": {"color": COLOR_QISKIT, "marker": "o"},
    }

    unique_res = []
    seen_areas = set()
    for category in plot_data.values():
        for label, area in zip(category["labels"], category["areas"]):
            if area not in seen_areas:
                unique_res.append((area, label))
                seen_areas.add(area)
    unique_res.sort()

    res_to_pos = {item[1]: i for i, item in enumerate(unique_res)}
    y_ticks_labels = [item[1] for item in unique_res]
    y_positions = np.arange(len(unique_res))

    for label, data in plot_data.items():
        if not data["fids"]:
            continue

        y_vals = [res_to_pos[l] for l in data["labels"]]
        x_vals = data["fids"]
        sort_idx = np.argsort(y_vals)
        style = style_map.get(label, {"color": COLOR_BASELINE, "marker": "o"})

        ax.plot(
            np.array(y_vals)[sort_idx],
            np.array(x_vals)[sort_idx],
            label=label,
            marker=style["marker"],
            color=style["color"],
            linestyle="-",
            linewidth=2,
            markersize=8,
            alpha=0.95,
        )

    ax.set_xticks(y_positions)
    ax.set_xticklabels(y_ticks_labels)
    ax.set_xlabel("Resolution $(H, W)$", fontdict={"fontsize": 12})
    ax.set_ylabel("Quality", fontdict={"fontsize": 12})
    ax.set_title("Quality across Encodings and Resolutions",
                 fontsize=15, color="#333", fontweight="bold", ha="center")
    ax.set_ylim(0, 1.05)
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, plot_data


def plot_metrics_vs_resolution(
    results,
    metric="depth_2q",  # "depth_2q" or "cost"
    save_path="figures/metric_vs_resolution.png",
    dpi=300,
    show=True,
    logscale=False,
):

    plot_data = {
        "Haiqu Block": {"labels": [], "areas": [], "values": []},
        "Qiskit SP": {"labels": [], "areas": [], "values": []},
    }

    for resolution_data in results["experiments"]:
        res = tuple(resolution_data["resolution"])
        res_label = f"({res[0]}, {res[1]})"
        res_area = res[0] * res[1]

        circuits = resolution_data["circuits"]

        # --- Haiqu Block (always exists) ---
        if "block_vector_loading" in circuits:
            if metric == "depth_2q":
                val = circuits["block_vector_loading"]["analytics"]["depth_2q"]
            elif metric == "cost":
                val = circuits["block_vector_loading"]["dry_run_cost"]["converted"][
                    "amount"
                ]
            else:
                raise ValueError("Unsupported metric")

            plot_data["Haiqu Block"]["labels"].append(res_label)
            plot_data["Haiqu Block"]["areas"].append(res_area)
            plot_data["Haiqu Block"]["values"].append(val)

        if "qiskit_sp" in circuits:
            if metric == "depth_2q":
                val = circuits["qiskit_sp"]["analytics"]["depth_2q"]
            elif metric == "cost":
                val = circuits["qiskit_sp"]["dry_run_cost"]["converted"]["amount"]

            plot_data["Qiskit SP"]["labels"].append(res_label)
            plot_data["Qiskit SP"]["areas"].append(res_area)
            plot_data["Qiskit SP"]["values"].append(val)

    unique_res = []
    seen = set()

    for category in plot_data.values():
        for label, area in zip(category["labels"], category["areas"]):
            if area not in seen:
                unique_res.append((area, label))
                seen.add(area)

    unique_res.sort()

    res_to_pos = {label: i for i, (_, label) in enumerate(unique_res)}
    x_tick_labels = [label for _, label in unique_res]
    x_positions = np.arange(len(unique_res))

    fig, ax = plt.subplots(figsize=(10, 6))
    style_map = {
        "Haiqu Block": {"color": COLOR_HAIQU, "marker": "o"},
        "Qiskit SP": {"color": COLOR_QISKIT, "marker": "o"},
    }

    for label, data in plot_data.items():
        if not data["values"]:
            continue

        x_vals = [res_to_pos[l] for l in data["labels"]]
        y_vals = data["values"]
        style = style_map.get(label, {"color": COLOR_BASELINE, "marker": "o"})

        idx = np.argsort(x_vals)
        x_vals = np.array(x_vals)[idx]
        y_vals = np.array(y_vals)[idx]

        ax.plot(
            x_vals,
            y_vals,
            label=label,
            marker=style["marker"],
            color=style["color"],
            linestyle="-",
            linewidth=2,
            markersize=7,
            alpha=0.95,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45)

    ax.set_xlabel("Resolution (H, W)", fontdict={"fontsize": 12})
    label = "2-qubit Gate Depth" if metric == "depth_2q" else "Estimated Cost ($)" if metric == "cost" else metric
    ax.set_ylabel(label, fontdict={"fontsize": 12})
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Resolution",
                 fontsize=15, color="#333", fontweight="bold", ha="center")

    if logscale:
        ax.set_yscale("log")

    ax.grid(False)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, plot_data


def plot_pareto(results, cropped_img):
    """Pareto front: Quality (y) vs Cost (x), with resolution labels."""

    # ── Extract data from results ──────────────────────────────────────────
    haiqu_res, haiqu_qual, haiqu_cost = [], [], []
    qiskit_res, qiskit_qual, qiskit_cost = [], [], []

    for resolution_data in results["experiments"]:
        res_tuple = resolution_data["resolution"]
        res_label = f"{res_tuple[0]}×{res_tuple[1]}"
        circuits = resolution_data["circuits"]
        images = resolution_data["images"]

        ideal = get_resized_image(cropped_img, target_size=res_tuple)
        ideal_state, _ = prep_state_and_disp(ideal)

        if "block_vector_loading" in circuits:
            qpu_img = images.get("qpu", {}).get("block_vector_loading")
            if qpu_img is not None:
                arr = np.asarray(qpu_img, dtype=object)
                if arr.size > 0 and not np.all(arr == None):
                    st, _ = prep_state_and_disp(np.array(qpu_img))

                    fid = float(np.abs(np.dot(ideal_state, st)) ** 2)
                    cost = circuits["block_vector_loading"]["dry_run_cost"]["converted"]["amount"]
                    haiqu_res.append(res_label)
                    haiqu_qual.append(fid)
                    haiqu_cost.append(cost)

        if "qiskit_sp" in circuits:
            qpu_img = images.get("qpu", {}).get("qiskit_sp")
            if qpu_img is not None:
                arr = np.asarray(qpu_img, dtype=object)
                if arr.size > 0 and not np.all(arr == None):
                    st, _ = prep_state_and_disp(np.array(qpu_img))
                    fid = float(np.abs(np.dot(ideal_state, st)) ** 2)
                    cost = circuits["qiskit_sp"]["dry_run_cost"]["converted"]["amount"]
                    qiskit_res.append(res_label)
                    qiskit_qual.append(fid)
                    qiskit_cost.append(cost)

    haiqu_qual = np.array(haiqu_qual)
    haiqu_cost = np.array(haiqu_cost)
    qiskit_qual = np.array(qiskit_qual)
    qiskit_cost = np.array(qiskit_cost)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # ── Scatter ───────────────────────────────────────────────────────────
    ax.scatter(haiqu_cost, haiqu_qual * 100, s=220, c=COLOR_HAIQU,
               edgecolors="#333", linewidths=1.4, zorder=5, label="Haiqu Block (QPU)")
    ax.scatter(qiskit_cost, qiskit_qual * 100, s=220, c=COLOR_QISKIT,
               edgecolors="#333", linewidths=1.4, zorder=5, marker="s", label="Qiskit SP (QPU)")

    # ── Pareto frontiers (connect points per method) ─────────────────────
    h_order = np.argsort(haiqu_cost)
    ax.plot(haiqu_cost[h_order], haiqu_qual[h_order] * 100,
            color=COLOR_HAIQU, alpha=0.45, linewidth=1.8, linestyle="--", zorder=4)

    q_order = np.argsort(qiskit_cost)
    ax.plot(qiskit_cost[q_order], qiskit_qual[q_order] * 100,
            color=COLOR_QISKIT, alpha=0.45, linewidth=1.8, linestyle="--", zorder=4)

    # ── Haiqu annotations — fanned left / right ──────────────────────────
    _haiqu_offset_pool = [
        (-65, 55), (-65, 33), (-65, 11), (65, 30), (65, 10), (-65, -10),
    ]
    haiqu_offsets = [_haiqu_offset_pool[i % len(_haiqu_offset_pool)] for i in range(len(haiqu_res))]
    for res, cost, qual, (dx, dy) in zip(
            haiqu_res, haiqu_cost, haiqu_qual, haiqu_offsets):
        ax.annotate(
            f"{res}\n${cost:.0f} | {qual*100:.1f}%",
            (cost, qual * 100),
            textcoords="offset points", xytext=(dx, dy), ha="center",
            fontsize=8, fontweight="bold", color="#333",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec="#ccc", alpha=0.85),
        )

    # ── Qiskit annotations ───────────────────────────────────────────────
    _qiskit_offset_pool = [
        (0, -25), (40, -25), (0, 20), (0, 20),
    ]
    qiskit_offsets = [_qiskit_offset_pool[i % len(_qiskit_offset_pool)] for i in range(len(qiskit_res))]
    for res, cost, qual, (dx, dy) in zip(
            qiskit_res, qiskit_cost, qiskit_qual, qiskit_offsets):
        ax.annotate(
            f"{res}\n${cost:.0f} | {qual*100:.1f}%",
            (cost, qual * 100),
            textcoords="offset points", xytext=(dx, dy), ha="center",
            fontsize=8, fontweight="bold", color="#333",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.7),
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec="#ccc", alpha=0.85),
        )

    # ── Axes (linear scale) ───────────────────────────────────────────────
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Cost ($)", fontsize=13, color="#333", labelpad=10)
    ax.set_ylabel("Quality (%)  — higher is better",
                  fontsize=13, color="#333", labelpad=10)
    fig.suptitle("Pareto Front: Quality vs. Cost",
                 fontsize=15, color="#333", fontweight="bold", ha="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(colors="#333", labelsize=11)

    # Limits with breathing room
    ax.set_ylim(0, 105)
    ax.set_xlim(5, 165)

    ax.legend(loc="lower left", fontsize=11, frameon=True, facecolor="white",
              edgecolor="#ccc", framealpha=0.9)

    plt.tight_layout()
    return fig
