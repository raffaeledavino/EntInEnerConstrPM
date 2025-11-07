import numpy as np
import picos as pic
import qutip as qt

from utils import *

                               
if __name__ == "__main__":
    """
    Performs a seesaw optimization to find the maximum Bell-inequality violation
    (both energy-assisted and non-energy-assisted) for a range of energy parameters w.

    The results are stored as tuples (w, EA_value, nonEA_value)
    in the list `results`, which can be saved or plotted later.

    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    
    energyrange = np.arange(0.01, 0.51, 0.01)
    dimS, dimM = 2, 3
    max_restarts = 20        # number of random restarts
    tol = 1e-7               # convergence threshold
    precisionopt = 1e-12   # precision parameter for picos optimization
    ineq = [1, -1, -1, 1]    # coefficients for the target inequality


    data = []

    # === Load optimal_distance data ===
    optimal_distance_file = "optimal_distance_data.txt"

    # Load the file, skipping commented lines (starting with '#')
    optimal_data = np.loadtxt(optimal_distance_file, comments="#")

    # Columns: [omega, p_opt, optimal_distance]
    omega_optimal = optimal_data[:, 0]
    EAvalue_analytic = optimal_data[:, 2]  # ← use this in your plot

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
                PiSA0 = qt.ket2dm(qt.rand_ket_haar(dimS * dimM, dims=[[dimS, dimM], [1, 1]])).full()
                measurement = [PiSA0, np.kron(np.eye(dimS), np.eye(dimM)) - PiSA0]
                ground = [[1, 0], [0, 0]]

                # --- Step 1: optimize over states given initial measurement ---
                behavior, states = findStateMaxViolation(dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt)

                # Compute initial Bell inequality value
                newEAvalue = sum(behavior[k] * ineq[k] for k in range(4))
                oldEAvalue = 0

                # ------------------------------------------------------------------
                # Seesaw loop: alternate between optimizing measurement and states
                # ------------------------------------------------------------------
                while newEAvalue - oldEAvalue > tol:
                    behavior, measurement = findMeasurementMaxViol(dimS, dimM, states, precisionopt)
                    behavior, states = findStateMaxViolation(dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt)
                    oldEAvalue = newEAvalue
                    newEAvalue = sum(behavior[k] * ineq[k] for k in range(4))

                success = True
                break  # exit retry loop upon success

            except pic.SolutionFailure as e:
                print(f"  Solver failed on attempt {attempt + 1}: {e}")
                continue  # retry with a new random initialization

        if not success:
            raise RuntimeError(f"Seesaw optimization failed for w = {w:.2f} after {max_restarts} restarts.")

        # ------------------------------------------------------------------
        # Compute non-energy-assisted (classical-correlated) value for comparison
        # ------------------------------------------------------------------
        nonEAvalue, _ = computeMaxIneqViolation(w0Avg, w1Avg, ineq, precisionopt)

        # Store results
        data.append((w0Avg, nonEAvalue, newEAvalue, ea_analytic_val))
        print(f"EA = {newEAvalue:.6f}, non-EA = {nonEAvalue:.6f}, EA_analytic = {ea_analytic_val:.6f}\n")


    # ------------------------------------------------------------------
    # Save results to file for later plotting
    # ------------------------------------------------------------------

    # Define output filename
    output_file = "data_EA_max_viol.txt"

    # Save with header
    np.savetxt(
        output_file,
        data,
        delimiter=",",
        header="omega,non_EA,EA,EA_analytic",
        comments='',   # prevents "#" at the start of the header line
        fmt="%.8f"     # 8 decimal digits for precision
    )

    print("All runs completed successfully.")
    print("Results saved to 'data_EA_max_viol.txt'.")

    plot_EA_data_from_txt("data_EA_max_viol.txt", save_as="Fig_correlations_adv.png")


    