import os
import numpy as np
import picos as pic
import qutip as qt
from utils import *


"""
Performs a seesaw optimization to find the maximum Bell-inequality violation
(both energy-assisted and non-energy-assisted) for a range of energy parameters w.

Results are stored as tuples (w, non_EA_value, EA_value) and optionally saved to:
'Data/data_viol_det_ineq.txt'
"""

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------
    save = True  # Set to False if you don't want to save results
    output_filename = "data_viol_det_ineq.txt"

    energyrange = np.arange(0.30, 0.50, 0.01)
    dimS, dimM = 2, 3
    max_restarts = 100000       # number of random restarts
    num_trials = 5              # number of independent minimizations per w
    tol = 1e-3                  # convergence threshold
    error = 1e-4
    precisionopt = 1e-4         # solver precision

    results = []

    # ----------------------------------------------------------------------
    # Prepare Data directory
    # ----------------------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")

    if save:
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, output_filename)
        print(f"Saving results to {data_path}")
    else:
        data_path = None
        print("Saving disabled (save=False).")

    # ----------------------------------------------------------------------
    # Main loop over energy values
    # ----------------------------------------------------------------------
    for w in energyrange:
        print(f"\n⟶ Energy parameter: w = {w:.2f}")
        w0Avg, w1Avg = w, w
        nonEA_E1value = 2 * (1 - 2 * w)**2 - 1

        best_EA_value = np.inf  # we minimize over the 5 trials
        success_any = False

        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")

            success = False
            newE1value = np.nan

            # --- Multiple random initializations for this trial ---
            for attempt in range(max_restarts):
                try:
                    # Random initial measurement
                    PiSA0 = qt.ket2dm(
                        qt.rand_ket_haar(dimS * dimM, dims=[[dimS, dimM], [1, 1]])
                    ).full()
                    measurement = [
                        PiSA0,
                        np.kron(np.eye(dimS), np.eye(dimM)) - PiSA0
                    ]
                    ground = [[1, 0], [0, 0]]

                    # --- Seesaw optimization ---
                    newE1value, states = findStateMinViolDet(
                        dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt
                    )
                    oldE1value = 0

                    while newE1value - oldE1value > tol or newE1value - nonEA_E1value > -error:
                        oldE1value = newE1value
                        measurement = findMeasurementMinViolDet(
                            dimS, dimM, states, precisionopt
                        )
                        newE1value, states = findStateMinViolDet(
                            dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt
                        )

                    success = True
                    break  # converged → stop restarts for this trial

                except pic.SolutionFailure:
                    continue  # retry random initialization

            if success:
                success_any = True
                best_EA_value = min(best_EA_value, newE1value)

        if not success_any:
            print(f" Optimization did not converge for w = {w:.2f}")
            best_EA_value = np.nan

        # Store the *minimum* EA value across all trials
        results.append((w0Avg, nonEA_E1value, best_EA_value))
        print(f" → Final EA(min) = {best_EA_value:.6f}, non-EA = {nonEA_E1value:.6f}")

    # ----------------------------------------------------------------------
    # Save results (if enabled)
    # ----------------------------------------------------------------------
    if save:
        np.savetxt(
            data_path,
            results,
            delimiter=",",
            header="omega,non_EA,EA",
            comments='',   # prevents "#" in header
            fmt="%.8f"
        )
        print(f"\n Results successfully saved to {data_path}")
    else:
        print("\n Results not saved (save=False).")

    # ----------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------
    plot_deterministic_inequality_violation(
        results,
        save_as="Fig_viol_det_ineq.png",
        save=True
    )
