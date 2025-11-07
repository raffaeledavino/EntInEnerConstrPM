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
    
    energyrange = np.arange(0.09, 0.51, 0.01)
    dimS, dimM = 2, 3
    max_restarts = 100000        # number of random restarts
    tol = 1e-3               # convergence threshold
    error = 1e-8
    precisionopt = 1e-4   # precision parameter for picos optimization



    data = []

    # ------------------------------------------------------------------
    # Main loop over energy values
    # ------------------------------------------------------------------
    for w in energyrange:
        print(f"⟶ Energy parameter: w = {w:.2f}")

        w0Avg, w1Avg = w, w
        success = False

        nonEA_E1value = 2*(1-2*w)**2-1
        # Try multiple random initializations until one converges
        for attempt in range(max_restarts):
            try:
                # Initialize measurement operator and ground state
                PiSA0 = qt.ket2dm(qt.rand_ket_haar(dimS*dimM, dims=[[dimS,dimM], [1,1]])).full()
                measurement = [PiSA0,np.kron(np.eye(dimS),np.eye(dimM))-PiSA0]
                ground = [[1, 0], [0, 0]]  # reset ground state explicitly

                # First step: optimize over states given initial measurement
                newE1value, states = findStateMinViolDet(dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt)


                oldE1value = 0

                # ------------------------------------------------------------------
                # Seesaw loop: alternate between optimizing measurement and states
                # ------------------------------------------------------------------
                while newE1value - oldE1value > tol or newE1value - nonEA_E1value > -error:
                    oldE1value = newE1value
                    measurement = findMeasurementMinViolDet(dimS, dimM, states, precisionopt)
                    newE1value, states = findStateMinViolDet(dimS, dimM, w0Avg, w1Avg, measurement, ground, precisionopt)
                    

                success = True
                break  # exit retry loop upon success

            except pic.SolutionFailure as e:
                #print(f"  Solver failed on attempt {attempt + 1}: {e}")
                continue  # retry with a new random initialization

        #if not success:
        #    raise RuntimeError(f"Seesaw optimization failed for w = {w:.2f} after {max_restarts} restarts.")

        # ------------------------------------------------------------------
        # Compute non-energy-assisted (classical) value for comparison
        # ------------------------------------------------------------------


        
        # Store results
        data.append((w0Avg, nonEA_E1value, newE1value))
        print(f"EA = {newE1value:.6f}, non-EA = {nonEA_E1value:.6f}\n")




    # ------------------------------------------------------------------
    # (Optional) Save results to file for later plotting
    # ------------------------------------------------------------------
    # Define output filename
    output_file = "data_viol_det_ineq.txt"

    # Save with header
    np.savetxt(
        output_file,
        data,
        delimiter=",",
        header="omega,non_EA,EA",
        comments='',   # prevents "#" at the start of the header line
        fmt="%.8f"     # 8 decimal digits for precision
    )

    print("All runs completed successfully.")
    print("Results saved to 'EA_max_viol_data.txt'.")

    plot_deterministic_inequality_violation("data_viol_det_ineq.txt", save_as="viol_det_ineq.png")


    