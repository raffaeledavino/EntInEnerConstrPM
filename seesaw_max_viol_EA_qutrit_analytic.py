import numpy as np
import picos as pic
import qutip as qt
import os
from utils import * 

"""
Performs a seesaw optimization to find the maximum value of I_corr,
as defined in Eq. (6), in the entanglement-assisted (EA) scenario with a qutrit memory M.
The results are compared with both the non–entanglement-assisted (non-EA) scenario
and the analytical results obtained for a qubit memory M, across a range of energy parameters omega.

Results are saved to 'Data/data_viol_det_ineq.txt' as tuples
(omega, non_EA, EA, EA_analytic), unless save=False is specified.
"""

if __name__ == "__main__":

    # Paramaters
    energyrange = np.arange(0.01, 0.51, 0.01)
    dimS, dimM = 2, 3
    num_trials = 20        # number of independent minimizations per w
    tol = 1e-7               # Convergence threshold

    # ----------------------------------------------------------------------
    # Load analytical data of the entanglement-assisted violation of I_corr
    # (computed in analytical_viol_EA_qubits.py)
    # ----------------------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)

    EA_viol_analytic_file = os.path.join(data_dir, "data_EA_viol_analytic.txt")

    # Load the file, skipping commented lines (starting with '#')
    data_loaded = np.loadtxt(EA_viol_analytic_file, comments="#")

    # Columns: [omega, p_opt, EA_viol_analytic]
    EAvalue_analytic = data_loaded[:, 2]

    # Prepare Data directory
    save = False                # Set to False if you don't want to save results
    output_file = "data_EA_max_viol.txt"
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    data_path = os.path.join(data_dir, output_file)
    os.makedirs(data_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Main loop over energy values
    # ------------------------------------------------------------------
    results = []

    for i, (w, ea_analytic_val) in enumerate(zip(energyrange, EAvalue_analytic)):
        print(f"⟶ Energy parameter: w = {w:.2f}")

        w0Avg, w1Avg = w, w
        success = False

        # Try multiple random initializations until one converges
        for attempt in range(num_trials):
            try:
                # Initialize measurement operator and ground state
                PiSA0 = qt.ket2dm(
                    qt.rand_ket_haar(dimS * dimM, dims=[[dimS, dimM], [1, 1]])
                ).full()
                measurement = [PiSA0, np.kron(np.eye(dimS), np.eye(dimM)) - PiSA0]
                ground = [[1, 0], [0, 0]]

                # Step 1: optimize over states given initial measurement
                newEAvalue, states = findStateMaxViolation(
                    dimS, dimM, w0Avg, w1Avg, measurement, ground
                )

                oldEAvalue = 0

                # Seesaw loop: alternate between optimizing measurement and states
                while newEAvalue - oldEAvalue > tol:
                    oldEAvalue = newEAvalue
                    measurement = findMeasurementMaxViol(
                        dimS, dimM, states
                    )
                    newEAvalue, states = findStateMaxViolation(
                        dimS, dimM, w0Avg, w1Avg, measurement, ground
                    )

                success = True
                break  # Exit retry loop upon success

            except pic.SolutionFailure as e:
                print(f"  Solver failed on attempt {attempt + 1}: {e}")
                continue  # Retry with a new random initialization

        if not success:
            raise RuntimeError(
                f"Seesaw optimization failed for w = {w:.2f} after {num_trials} restarts."
            )

        # Compute non-entanglement-assisted (classical-correlated) value
        nonEAvalue, _ = computeMaxIneqViolation(w0Avg, w1Avg)

        # Store results
        results.append((w0Avg, nonEAvalue, newEAvalue, ea_analytic_val))
        print(
            f"EA = {newEAvalue:.6f}, non-EA = {nonEAvalue:.6f}, EA_analytic = {ea_analytic_val:.6f}\n"
        )

    # ------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------
    if save:
        np.savetxt(
            data_path,
            results,
            delimiter=",",
            header="omega,non_EA,EA,EA_analytic",
            comments="# ",
            fmt="%.8f",
        )
        print(f"\n Results successfully saved to {data_path}")
    else:
        print("\n Results not saved (save=False).")
    

    # ----------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------
    plot_EA_violation_qutrit_analytic(
        results,   # Use `results` to plot data from this run, or "Data/filename.txt" to plot previously saved data
        save_as="Fig_correlations_adv.png",
        save=False
    )
