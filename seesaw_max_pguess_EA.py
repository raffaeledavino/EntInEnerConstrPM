import numpy as np
import qutip as qt

from utils import *

"""
Script to compute quantum quantum guessing probabilities in Eq. (8)
for different energy values.

Results are saved to 'data_avg_pg.txt' as tuples (w, cPguess, qPguess_min).
"""

if __name__ == "__main__":
    # Configuration
    output_file = 'data_avg_pg.txt'
    energyrange = np.arange(0.20, 0.47, 0.01)
    num_attempts = 1
    dim, dimM = 2, 3
    precision = 1e-8

    # Main computation
    with open(output_file, 'w') as f:
        f.write("# w0Avg, cPguess, qPguess\n")
        for w in energyrange:
            print(f"Processing energy: {w:.2f}")

            # Use identical energies for both inputs (can be generalized)
            w0Avg, w1Avg = w, w
            ineq = [1, -1, -1, 1]

            # Classical-correlated (non-entangled) part
            nonEABellValue, _ = computeMaxIneqViolation(w0Avg, w1Avg, ineq, precision)
            cPguess = ComputeGuessingProbability(w0Avg, w1Avg, 1, 1, nonEABellValue, precision)

            qPguess_min = 0.0
            for attempt in range(num_attempts):
                qPguess = cPguess
                tol = 1e-9

                while qPguess < cPguess + tol:
                    # Numerical safeguard for high-energy region
                    if w > 0.46:
                        tol = -10  # effectively breaks loop

                    PiSA0=qt.ket2dm(qt.rand_ket_haar(dim*dimM, dims=[[dim,dimM], [1,1]])).full()
                    measurement = [PiSA0,np.kron(np.eye(dim),np.eye(dimM))-PiSA0]
                    ground = [[1, 0], [0, 0]] 
                    # ground=(qt.ket2dm(np.sqrt(w0Avg)*qt.basis(2,0)+np.sqrt(1-w0Avg)*qt.basis(2,1))).full()
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
                        # Ignore failed attempts
                        pass

                qPguess_min = max(qPguess_min, qPguess)

            print((w0Avg, cPguess, qPguess_min), file=f)
            print(f"Result: w={w0Avg:.2f}, cP={cPguess:.6f}, qP={qPguess_min:.6f}")
    
    # Read results from file
    results = []
    with open(output_file, 'r') as the_file:
        for line in the_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("(", "").replace(")", "")
            item = [x.strip() for x in line.split(",")]
            results.append(tuple(float(x) for x in item))

    plot_min_entropy("data_avg_pg.txt", save_as="Pguess.png")
