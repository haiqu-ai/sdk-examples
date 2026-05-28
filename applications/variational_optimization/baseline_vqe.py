"""
Variational optimization using standard IBM Qiskit Runtime.

This module provides a concise interface to train parameterized quantum circuits
on IBM backends for comparison against Haiqu's variational_optimization.

Can be used for any QML task: VQE, QAOA, quantum classifiers, etc.
"""

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_algorithms.optimizers import NFT


IBM_USD_PER_SEC = 1.60  # IBM Qiskit Runtime cost rate


def _resolve_ibm_backend(device_id: str, options: dict):
    """Resolve an IBM backend from a device identifier.

    Real devices (e.g. ``"ibm_pittsburgh"``) are loaded via ``QiskitRuntimeService`` using
    credentials from ``options``. Fake devices (e.g. ``"fake_torino"``, ``"fake_marrakesh"``)
    are loaded from ``qiskit_ibm_runtime.fake_provider`` — no credentials needed.
    """
    if device_id.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        class_name = "Fake" + "".join(p.capitalize() for p in device_id.split("_")[1:])
        return getattr(fake_provider, class_name)()

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=options.get("ibm_quantum_token"),
        instance=options.get("ibm_quantum_instance"),
    )
    return service.backend(device_id)


def run_ibm_vqe(
    ansatz,
    hamiltonian,
    device_id,
    options=None,
    shots=1000,
    maxiter=5,
    initial_parameters=None,
    seed=None,
):
    """
    Run variational optimization using IBM Qiskit Runtime.

    This function trains a parameterized quantum circuit to minimize
    the expectation value of an observable. Can be used for VQE, QAOA,
    quantum classifiers, or any QML task.

    Args:
        ansatz: Parameterized quantum circuit (ansatz)
        hamiltonian: SparsePauliOp representing the cost function observable
        device_id: Backend identifier (required). Real IBM devices like
            ``"ibm_pittsburgh"`` require credentials in ``options``. Fake devices like
            ``"fake_torino"``, ``"fake_marrakesh"``, or ``"fake_fez"`` load the
            corresponding ``qiskit_ibm_runtime.fake_provider.*`` simulator without
            credentials.
        options: Dict with ``"ibm_quantum_token"`` and ``"ibm_quantum_instance"`` for real
            devices. Pass ``{}`` (or omit) for fake devices.
        shots: Number of shots per circuit evaluation
        maxiter: Maximum NFT optimizer iterations
        initial_parameters: Initial parameter values. If None, random initialization is used.
        seed: Random seed for initial parameters (ignored if initial_parameters is provided)

    Returns:
        dict: Results in the unified Haiqu-style schema (``loss_history``, ``min_loss``,
              ``qpu_cost``, ``session_cost``) plus IBM-specific ``cost_history``,
              ``job_time_metrics``, ``session_id``, ``iterations``, ``func_evals``,
              ``optimal_params``.
    """
    backend = _resolve_ibm_backend(device_id, options or {})

    # Transpile ansatz for the target backend
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    ansatz_isa = pm.run(ansatz)

    # Apply layout to Hamiltonian
    hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

    # Track optimization history and per-job metrics
    cost_history = []
    job_time_metrics = []
    iteration_energies = []
    prev_eval_count = 0

    def cost_func(params, ansatz, hamiltonian, estimator):
        pub = (ansatz, [hamiltonian], [params])
        job = estimator.run(pubs=[pub])
        result = job.result()
        energy = result[0].data.evs[0]
        cost_history.append(energy)

        # Collect server-side QPU metrics per job
        usage = job.metrics()["usage"]
        job_time_metrics.append({"qpu_time": usage["quantum_seconds"]})

        return energy

    def iteration_callback(x):
        nonlocal prev_eval_count
        iteration = len(iteration_energies) + 1
        # Compute the NFT predicted minimum (c - a) from this iteration's evaluations,
        # matching Haiqu's loss_history which reports the analytically predicted optimum
        # rather than the best of the three actually evaluated energies (min(z0, z1, z3)).
        iter_evals = cost_history[prev_eval_count:]
        if len(iter_evals) == 3:
            z0, z1, z3 = iter_evals
        elif len(iter_evals) == 2:
            z1, z3 = iter_evals
            # z0 was recycled from previous iteration
            z0 = iteration_energies[-1] if iteration_energies else cost_history[0]
        else:
            z0 = iter_evals[-1]
            z1 = z3 = z0
        c = (z1 + z3) / 2
        a = np.sqrt((z0 - (z1 + z3 - z0)) ** 2 + (z1 - z3) ** 2) / 2
        predicted_energy = c - a
        iteration_energies.append(predicted_energy)
        prev_eval_count = len(cost_history)
        print(f"  Iter {iteration}: energy = {predicted_energy:.6f}")

    # Initialize parameters
    if initial_parameters is not None:
        x0 = np.array(initial_parameters)
    elif seed is not None:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-0.1 * np.pi, 0.1 * np.pi, ansatz_isa.num_parameters)
    else:
        x0 = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, ansatz_isa.num_parameters)

    # Run optimization within a Session
    with Session(backend=backend) as session:
        session_id = session.session_id
        print(f"IBM Session ID: {session_id}")

        estimator = Estimator(mode=session, options={"resilience_level": 1})
        estimator.options.default_shots = shots

        optimizer = NFT(
            maxiter=maxiter,
            callback=iteration_callback,
            args=(ansatz_isa, hamiltonian_isa, estimator),
        )
        res = optimizer.minimize(cost_func, x0)

    # QPU time: sum of per-job quantum_seconds
    total_qpu_time = sum(m["qpu_time"] for m in job_time_metrics)

    # Session time: wall-clock duration from first job start to session close.
    # session.usage() returns this directly from IBM's Session API.
    total_session_time = session.usage() or 0.0

    return {
        "optimal_params": res.x,
        "loss_history": iteration_energies,  # per-iteration NFT-predicted minima (matches Haiqu's loss_history shape)
        "min_loss": float(res.fun),
        "qpu_cost": {
            "native": {"amount": total_qpu_time, "unit": "s"},
            "converted": {"amount": total_qpu_time * IBM_USD_PER_SEC, "unit": "USD"},
        },
        "session_cost": {
            "native": {"amount": total_session_time, "unit": "s"},
            "converted": {"amount": total_session_time * IBM_USD_PER_SEC, "unit": "USD"},
        },
        "cost_history": cost_history,  # per-evaluation energies (IBM-only)
        "job_time_metrics": job_time_metrics,  # per-evaluation timing (IBM-only)
        "session_id": session_id,
        "iterations": res.nit,
        "func_evals": res.nfev,
    }
