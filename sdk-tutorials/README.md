# SDK Tutorials

Tutorial notebooks covering the main Haiqu SDK features. Each notebook focuses on a single function or parameter and includes a self-contained example you can run on a simulator or real hardware.

If you're new to Haiqu, start with [haiqu_run.ipynb](run/haiqu_run.ipynb) and work through the notebooks roughly in the order they appear below.


## Circuit Execution

### [`haiqu.run()`](run/haiqu_run.ipynb)

Demonstrates how to use `haiqu.run()` to execute circuits on quantum hardware or simulators. Haiqu SDK currently supports devices via IBM Quantum and AWS Braket, along with noisy simulators and Qiskit Aer. Covers job submission, result retrieval, dry runs for cost estimation, and listing past jobs with `haiqu.list_jobs()`.

---

### [`haiqu.run()` with `use_mitigation`](run/haiqu_run_use_mitigation.ipynb)

Shows how to enable automatic error mitigation with `haiqu.run(use_mitigation=True)`. Haiqu can apply lightweight mitigation to both sampling and expectation value estimation tasks. For an adder circuit starting at around 17.3% success probability, mitigation can boost the success probability up to 85% (4.9x).

---

### [`haiqu.run()` with `error_mitigation_options`](run/haiqu_run_mitigation_options.ipynb)

Shows how to selectively control individual mitigation layers by passing `error_mitigation_options` inside the `options` dictionary. Available layers are `dynamical_decoupling`, `readout_mitigation`, `noise_tailoring`, and `advanced_mitigation`. Useful when you want to experiment with different mitigation combinations for a specific circuit and device.

---

### [`haiqu.run()`: quality estimate](run/haiqu_run_quality_estimate.ipynb)

Shows how to assess the quality of `haiqu.run()` results on noisy devices and simulators. For sampling tasks, quality is reported as Hellinger fidelity against an ideal distribution. For observable estimation, it's reported as 1 - relative error against a statevector reference. Per-circuit metrics are available in `job.info["quality_assessment"]["per_circuit"]`. Quality assessment requires a statevector simulation and is skipped automatically for circuits above 20 qubits.

---

### [`haiqu.run()`: uncertainty estimate](run/haiqu_run_uncertainty_estimate.ipynb)

Shows how shot count and error mitigation affect the statistical uncertainty of expectation value estimates. Haiqu reports per-observable uncertainty in `job.info["uncertainty"]`. Uncertainty decreases as shots increase, following a roughly 1/sqrt(shots) scaling. Enabling `use_mitigation=True` corrects the expectation value toward the ideal but may change the reported uncertainty -- this notebook lets you compare the two directly.

## Data Encoding

### [`haiqu.vector_loading()`](vectorLoading/haiqu_vectorLoading.ipynb)

Shows how to prepare an arbitrary quantum state from a vector of amplitudes. For a 12-qubit sin wave, traditional amplitude encoding requires 4083 CNOT gates and a two-qubit gate depth of 4083. Haiqu's method prepares the same state with 21 CNOT gates (194.4x improvement) and a depth of 11 (371.2x improvement). Also covers hyperparameter tuning with `num_layers` and `truncation_cutoff` to control the fidelity-cost tradeoff.

---

### [`haiqu.block_vector_loading()`](blockVectorLoading/haiqu_blockVectorLoading.ipynb)

Shows how to encode large 1D and 2D vectors using a tiled approach: the input is sliced into blocks that are encoded independently. The method trades extra qubits for higher fidelity on large inputs. For a 64x64 image, standard vector loading achieves ~87% fidelity while block vector loading reaches ~97%.

---

### [`haiqu.distribution_loading()`](distributionLoading/haiqu_distributionLoading.ipynb)

Shows how to efficiently prepare a probability distribution in a quantum state using `haiqu.distribution_loading()`. Works with scipy distributions by specifying the distribution name and parameters directly. Loading a log-normal distribution on 12 qubits traditionally requires 4083 CNOT gates; Haiqu prepares it with 21 (194.4x improvement) and a depth of 11 (371.2x improvement).

---

### [`haiqu.entangled_manifold_embedding()`](EME/haiqu_EME.ipynb)

Shows how to encode a vector of real classical features into a quantum state using Entangled Manifold Embedding (EME). Standard techniques would require as many qubits as features: for 1000 features, that's 1000 qubits, well beyond current hardware. Haiqu's EME compresses the same vector into 100 qubits or fewer, with qubit count controlled by a `density` parameter.

## Circuit Compression

### [`haiqu.state_compression()`](stateCompression/haiqu_stateCompression.ipynb)

Shows how to compress quantum circuits to improve performance on noisy hardware. For a 10-qubit adder circuit, state compression increases the likelihood of measuring the correct result from 7.7% to 88.8% (11.5x). Combining state compression with noise mitigation delivers 99.7% success (13.0x). Also covers using compression together with `use_mitigation=True`.

---

### [`haiqu.state_compression()`: `compression_level`](stateCompression/haiqu_stateCompression_compressionLevel.ipynb)

Shows how to use the `compression_level` parameter (`low`, `balanced`, `high`) to control the tradeoff between gate reduction and state quality. For a 15-step Heisenberg evolution circuit, the highest compression level can reduce two-qubit gate count by up to 71%. The notebook runs all three levels and compares their results side by side.

---

### [`haiqu.state_compression()`: `fine_tuning`](stateCompression/haiqu_stateCompression_fineTuning.ipynb)

Shows how to use the `fine_tuning` parameter to optimize compressed circuit performance. For a 10-step Heisenberg evolution circuit, fine-tuning increases compression quality from 73% to 82% (12.3% improvement). Options are `disabled`, `low`, and `heavy`, with heavier tuning taking more classical compute time.

---

### [`haiqu.observable_backpropagation()`](observableBackpropagation/haiqu_observableBackpropagation.ipynb)

Shows how to reduce circuit depth by classically absorbing circuit layers into the measurement observable via Pauli backpropagation. For a 193-qubit Heisenberg time evolution circuit, this reduces two-qubit gate count from 1344 to 496 for a single-qubit observable. The tradeoff is that the observable gains more (and often heavier) Pauli terms.

## Optimization

### [`haiqu.variational_optimization()`](qml/haiqu_variational_optimization.ipynb)

Shows how to run variational quantum optimization using the NFT (Nakanishi-Fujii-Todo) gradient-free optimizer. You define a parameterized ansatz and an observable, submit a job to the Haiqu API, and retrieve optimized parameters, minimum loss, and loss history. Supports initial parameter warm-starting, error mitigation, and circuit packing to reduce cost per iteration.

---

### [`haiqu.solve_qubo()`](optimization/haiqu_solve_qubo.ipynb)

Shows the `haiqu.solve_qubo()` API for end-to-end QUBO solving. Automatically handles LR-QAOA circuit construction, CVaR analysis, and classical post-processing. For a 20-variable Max-Cut problem, the raw quantum results are suboptimal -- built-in post-processing recovers the optimal solution at no extra quantum cost.

---

### [`haiqu.postprocess()`](optimization/haiqu_postprocess.ipynb)

Shows how to use `haiqu.postprocess()` to improve optimization results through classical bit-flip search. For a 120-qubit QUBO problem, the quantum results alone return a suboptimal solution. Post-processing finds the optimal solution without any additional circuit runs.
