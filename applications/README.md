# Application Notebooks

End-to-end experiments using Haiqu SDK on real quantum hardware. These notebooks cover research-grade problems at realistic qubit counts and benchmark Haiqu against published baselines.

Cached results are included so you can inspect outputs without hardware re-runs, and the full code is there if you want to replicate or build on the experiments. To run on QPU, set the variable `RUN_ON_DEVICE` to `True` in the notebook. Please note that this requires credentials for access to the QPU and will a runtime incur cost.


## Quantum Dynamics

### [127-Qubit Ising Model](standard_ising/standard_ising.ipynb)

Simulates the dynamics of a 127-qubit transverse-field Ising model on a real IBM Eagle processor, reproducing results from IBM's 2023 Nature paper. Both Haiqu and IBM's error-mitigation pipeline reach the same magnetisation curve -- Haiqu gets there in ~41 seconds at ~$33, versus IBM's ~4 hours at ~$11,520 (350x faster, 350x cheaper). The notebook walks through the full setup, execution, and comparison.

---

### [103-Qubit Floquet Ising Model](kicked_ising/kicked_ising.ipynb)

Simulates the dynamics of a 103-qubit kicked Ising (Floquet) model on a real IBM Heron processor, with non-integrable parameters that make the circuit harder to classically simulate. Compared against a PEC-based error-mitigation baseline, Haiqu reaches the same magnetisation curve at 531x lower cost (~$34 vs ~$17,856). Includes a detailed breakdown of cost and accuracy across both approaches.


## Variational Algorithms

### [LiH VQE on Real Hardware](variational_optimization/lih_vqe.ipynb)

Runs VQE for the Lithium Hydride molecule on 10 qubits on real IBM hardware, comparing three configurations: Haiqu with classical pretraining, Haiqu with random initialization, and the Qiskit Runtime baseline (NFT optimizer). Haiqu's circuit packing and session management deliver \~5x faster QPU runtime (550s to 115s) at about 5x lower cost (\~$880 to \~$184). Classical pretraining gives an additional 2.7x improvement in energy error at no extra quantum cost.

---

### 100-Qubit LR-QAOA (Coming Soon)

Full benchmark of LR-QAOA for a 100-qubit weighted Max-Cut problem. Runs ideal, noisy, and Haiqu-compressed scenarios across circuit depths 1 to 100, then analyzes cut quality and cost. Haiqu's state compression maintains solution quality across all depths while dramatically reducing the shot overhead, dominating the Pareto frontier of quality vs cost at every depth tested.

## Data Encoding

### [Quantum Image Encoding](image_encoding/main_estimation_block_with_qpu_res.ipynb)

Demonstrates amplitude encoding of grayscale images into quantum states on a real QPU, at resolutions from 16x16 to 256x256. Compares two strategies: Qiskit StatePreparation (full-vector encoding, exact but exponentially expensive) and Haiqu Block Vector Loading (tiled encoding, approximate but practical). Haiqu achieves 2x fidelity improvement at 64x64 and 89% cost reduction at 128x128, while maintaining >90% quality at all tested resolutions (vs Qiskit dropping to 20% at higher resolutions).

## Circuit Compression

### [State Compression on 2D Heavy-Hex Lattice](2d_compression/haiqu_stateCompression_2d.ipynb)

Applies state compression to Floquet circuits on the 156-qubit IBM Heron R3 device with a heavy-hex lattice topology. Targets a non-ergodic Floquet circuit chosen specifically for its high compressibility. `haiqu.state_compression_2d()` achieves up to 85% CNOT reduction on 10-20 layer circuits while maintaining compression quality above 91%. Magnetisation observables from the compressed circuits match classical tensor network references.

## Hybrid Algorithms

### [SKQD for Single-Impurity Anderson Model](skqd_demo/skqd_siam_model.ipynb)

Estimates the ground-state energy of a 40-qubit single-impurity Anderson model (SIAM) using Sample-based Krylov Quantum Diagonalization (SKQD) -- a hybrid algorithm that builds a Krylov subspace via quantum time evolution and diagonalises it with classical sample-based post-processing. Haiqu's circuit compression cuts gate count by 98%, yielding 3.4x lower energy error, 2.9x fewer SQD determinants, and ~5x faster classical post-processing compared to uncompressed Qiskit circuits.
