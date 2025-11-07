import numpy as np
import qutip as qt
import os
from utils import *

"""
Script to compute quantum quantum guessing probabilities in Eq. (8)
for different energy values.

Results are saved to 'Data/data_avg_pg.txt' as tuples (w, cPguess, qPguess_min),
unless save=False is specified.
"""

if __name__ == "__main__":


    # Configuration
    save = True  # Set to False if you don't want to save result
    output_file = 'data_avg_pg.txt'
    energyrange = np.arange(0.01, 0.51, 0.01)
    num_attempts = 10
    dim, dimM = 2, 3
    precision = 1e-12

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    data_path = os.path.join(data_dir, output_file)
    os.makedirs(data_dir, exist_ok=True)
    #if save:
    #    os.makedirs(data_dir, exist_ok=True)
    #    data_path = os.path.join(data_dir, output_file)
    #    print(f"Saving results to {data_path}")
    #else:
    #    data_path = None
    #    print("Saving disabled (save=False).")
    
    # ------------------------------------------------------------
    # Main computation loop
    # ------------------------------------------------------------
    results = []

    for w in energyrange:
        print(f"\n⟶ Energy parameter: w = {w:.2f}")

        w0Avg, w1Avg = w, w

        # --- Classical-correlated (non-entangled) part ---
        nonEABellValue, _ = computeMaxIneqViolation(w0Avg, w1Avg, precision)
        cPguess = ComputeGuessingProbability(w0Avg, w1Avg, 1, 1, nonEABellValue, precision)

        qPguess_min = 0.0
        for attempt in range(num_attempts):
            qPguess = cPguess
            tol = 1e-9

            while qPguess < cPguess + tol:
                if w > 0.46:  # numerical safeguard for high energy
                    tol = -10

                # Random measurement initialization
                PiSA0 = qt.ket2dm(qt.rand_ket_haar(dim * dimM, dims=[[dim, dimM], [1, 1]])).full()
                measurement = [PiSA0, np.kron(np.eye(dim), np.eye(dimM)) - PiSA0]
                ground = [[1, 0], [0, 0]]

                newPguess = 1.0
                firstPguess = 0.5

                try:
                    error = 1e-4 if w < 0.06 else 1e-7
                    while newPguess - firstPguess > error:
                        firstPguess, states = findStatesGuessProb(
                            dim, dimM, w0Avg, w1Avg, measurement, ground, nonEABellValue, precision
                        )
                        newPguess, measurement = findMeasurementGuessProb(
                            dim, dimM, states, nonEABellValue, precision
                        )
                    qPguess = newPguess

                except Exception:
                    pass  # ignore failed attempts

            qPguess_min = max(qPguess_min, qPguess)

        results.append((w0Avg, cPguess, qPguess_min))
        print(f"Result: w={w0Avg:.2f}, cP={cPguess:.6f}, qP={qPguess_min:.6f}")

    # ------------------------------------------------------------
    # Save results to file
    # ------------------------------------------------------------
    if save:
        with open(data_path, "w") as f:
            f.write("# w0Avg, cPguess, qPguess_min\n")
            for w, c, q in results:
                f.write(f"{w:.6f}, {c:.8f}, {q:.8f}\n")
        print(f"\n Results successfully saved to {data_path}")
    else:
        print("\n Results were not saved (save=False).")

    # ----------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------
    plot_min_entropy(results,
                     save_as="Fig_pguess.png",
                     save=True
    )
