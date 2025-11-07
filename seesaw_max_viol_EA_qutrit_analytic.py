import numpy as np
import picos as pic
import qutip as qt
import os
from utils import * 

if __name__ == "__main__":
    """
    Performs a seesaw optimization to find the maximum Bell-inequality violation
    (both entanglement-assisted and non-entanglement-assisted) for a range of
    energy parameters omega.

    Results are saved as (omega, non_EA, EA, EA_analytic) to:
        ./Data/data_EA_max_viol.txt
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    energyrange = np.arange(0.01, 0.51, 0.01)
    dimS, dimM = 2, 3
    max_restarts = 20        # Number of random restarts
    tol = 1e-7               # Convergence threshold
    precisionopt = 1e-12     # Solver precision for PICOS

    data = []

    # ----------------------------------------------------------------------
    # Load analytical data of the entanglement-assisted violation of I_corr
    # (Appendix C)
    # ----------------------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)

    EA_viol_analytic_file = os.path.join(data_dir, "data_EA_viol_analytic.txt")

    # Load the file, skipping commented lines (starting with '#')
    data_loaded = np.loadtxt(EA_viol_analytic_file, comments="#")

    # Columns: [omega, p_opt, EA_viol_analytic]
    EAvalue_analytic = data_loaded[:, 2]

    # ------------------------------------------------------------------
    # Main loop over energy values
    # ------------------------------------------------------------------
    for i, (w, ea_analytic_val) in enumerate(zip(energyrange, EAvalue_analytic)):
        print(f"⟶ Energy parameter: w = {w:.2f}")

        w0Avg, w1Avg = w, w
        success = False

        # Try multiple random initializations until one converges
        for attempt in range(max_restarts):
            try:
                # Initialize measurement operator and ground state
                PiSA0 = qt.ket2dm(
                    qt.rand_ket_haar(dimS * dimM, dims=[[dimS, dimM], [1, 1]])
                ).full()
                measurement = [PiSA0, np.kron(np.eye(dimS), np.eye(dimM)) - PiSA0]
                ground = [[1, 0], [0, 0]]

                # --- Step 1: optimize over states given initial measurement ---
                newEAvalue, states = findStateMaxViolation(
                    dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt
                )

                oldEAvalue = 0

                # ------------------------------------------------------------------
                # Seesaw loop: alternate between optimizing measurement and states
                # ------------------------------------------------------------------
                while newEAvalue - oldEAvalue > tol:
                    oldEAvalue = newEAvalue
                    measurement = findMeasurementMaxViol(
                        dimS, dimM, states, precisionopt
                    )
                    newEAvalue, states = findStateMaxViolation(
                        dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt
                    )
                    
                    #newEAvalue = sum(behavior[k] * ineq[k] for k in range(4))

                success = True
                break  # Exit retry loop upon success

            except pic.SolutionFailure as e:
                print(f"  Solver failed on attempt {attempt + 1}: {e}")
                continue  # Retry with a new random initialization

        if not success:
            raise RuntimeError(
                f"Seesaw optimization failed for w = {w:.2f} after {max_restarts} restarts."
            )

        # ------------------------------------------------------------------
        # Compute non-entanglement-assisted (classical-correlated) value
        # ------------------------------------------------------------------
        nonEAvalue, _ = computeMaxIneqViolation(w0Avg, w1Avg, precisionopt)

        # Store results
        data.append((w0Avg, nonEAvalue, newEAvalue, ea_analytic_val))
        print(
            f"EA = {newEAvalue:.6f}, non-EA = {nonEAvalue:.6f}, EA_analytic = {ea_analytic_val:.6f}\n"
        )

    # ------------------------------------------------------------------
    # Save results to ./Data/
    # ------------------------------------------------------------------
    data_path = os.path.join(data_dir, "data_EA_max_viol.txt")

    np.savetxt(
        data_path,
        data,
        delimiter=",",
        header="omega,non_EA,EA,EA_analytic",
        comments="# ",
        fmt="%.8f",
    )

    print("\nAll runs completed successfully.")
    print(f"Results saved to '{data_path}'.")

    # ----------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------
    plot_EA_violation_qutrit_analytic(data,
                                      save_as="Fig_correlations_adv.png",
                                      save=True
    )
