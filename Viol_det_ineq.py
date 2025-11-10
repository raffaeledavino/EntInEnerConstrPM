import os
import numpy as np
import picos as pic
import qutip as qt
from utils import *

"""
Performs a seesaw optimization to find the maximum Bell-inequality violation
(both energy-assisted and non-energy-assisted) for a range of energy parameters w.

Results are saved to 'Data/data_viol_det_ineq.txt' as tuples
(omega, non_EA_value, EA_value) , unless save=False is specified.
"""

if __name__ == "__main__":

    # Parameters
    energyrange = np.arange(0.01, 0.50, 0.01)   # Note: for small omega values, the optimization may take 
                                                # significantly longer to find a feasible solution
    dimS, dimM = 2, 3
    num_trials = 10             # Number of independent minimizations per w
    tol = 1e-8                  # Convergence threshold
    error = 1e-4
    precisionopt = 1e-8         # Solver precision


    # Prepare Data directory
    save = False  # Set to False if you don't want to save results
    output_file = "data_viol_det_ineq.txt"
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    data_path = os.path.join(data_dir, output_file)
    os.makedirs(data_dir, exist_ok=True)
 

    # ----------------------------------------------------------------------
    # Main loop over energy values
    # ----------------------------------------------------------------------
    results = []

    for w in energyrange:
        print(f"\n⟶ Energy parameter: w = {w:.2f}")
        w0Avg, w1Avg = w, w
        nonEA_E1value = 2 * (1 - 2 * w)**2 - 1

        best_EA_value = np.inf  # we minimize over the num_trials trials
        success_any = False

        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")

            success = False
            newE1value = np.nan

            # --- Multiple random initializations for this trial ---
            while True:
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
                    
                    max_iters = 100  # avoid getting stuck in the inner loop
                    iters = 0

                    while (
                        newE1value - oldE1value > tol
                        or newE1value - nonEA_E1value > -error
                    ) and iters < max_iters:
                        oldE1value = newE1value
                        measurement = findMeasurementMinViolDet(
                            dimS, dimM, states, precisionopt
                        )
                        newE1value, states = findStateMinViolDet(
                            dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt
                        )
                        iters += 1

                    success = True
                    break  # found a valid result → stop

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

    # ------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------
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
        results,  # Use `results` to plot data from this run, or "Data/filename.txt" to plot previously saved data
        save_as="Fig_viol_det_ineq.png",
        save=False
    )
