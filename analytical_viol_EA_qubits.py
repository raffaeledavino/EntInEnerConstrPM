import os
import numpy as np
import qutip as qt
from tqdm import tqdm

"""
Script to compute the maximum value of I_corr using 
two quantum states sigma_SA^0 and sigma_SA^1 constructed analytically
for a range of energy-like parameters w.

This script scans over a grid of (w, p) values, constructs the two
bipartite states using QuTiP, and computes the maximum trace distance
between them for each w.

Results are saved to 'Data/data_EA_viol_analytic.txt' unless save=False.
"""

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
save = True  # Set to False if you don’t want to save results
output_filename = "data_EA_viol_analytic.txt"
omega_num = 50
p_values = np.linspace(0.01, 0.50, 1000)

# ----------------------------------------------------------------------
# Prepare output directory
# ----------------------------------------------------------------------
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "Data")
if save:
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, output_filename)
    print(f"Saving results to {output_path}")
else:
    output_path = None
    print("Saving disabled (save=False).")

# ----------------------------------------------------------------------
# Basis and variable initialization
# ----------------------------------------------------------------------
b00, b01, b10, b11 = [qt.basis(4, i) for i in range(4)]
EA_viol_analytic = np.zeros(omega_num)
p_opt = np.zeros(omega_num)

# ----------------------------------------------------------------------
# Main computation loop
# ----------------------------------------------------------------------
for count in tqdm(range(omega_num), desc="Computing EA violations"):
    w = (count + 1) * 0.01

    for p in p_values:
        # Build analytical states sigma_SA^0 and sigma_SA^1 (Appendix C)
        sigmaSA0 = qt.Qobj(
            ((1 - w) * b00 * b00.dag()
             + w * p * b10 * b10.dag()
             + w * (1 - p) * b11 * b11.dag()
             + np.sqrt((1 - w) * w * p) * (b00 * b10.dag() + b10 * b00.dag())
             + np.sqrt((1 - w) * w * (1 - p)) * (b00 * b11.dag() + b11 * b00.dag())
             + w * np.sqrt(p * (1 - p)) * (b10 * b11.dag() + b11 * b10.dag())),
            dims=[[2, 2], [2, 2]]
        )

        sigmaSA1 = qt.Qobj(
            ((2 * p * w - 2 * w + 1) * b00 * b00.dag()
             + (w - 2 * p * w) * b01 * b01.dag()
             + w * (1 - p) * b10 * b10.dag()
             + w * p * b11 * b11.dag()
             - np.sqrt((2 * p * w - 2 * w + 1) * w * (1 - p)) * (b00 * b10.dag() + b10 * b00.dag())
             - np.sqrt((2 * p * w - 2 * w + 1) * w * p) * (b00 * b11.dag() + b11 * b00.dag())
             + w * np.sqrt(p * (1 - p)) * (b10 * b11.dag() + b11 * b10.dag())),
            dims=[[2, 2], [2, 2]]
        )

        # Compute trace distance
        dist = 2 * qt.metrics.tracedist(sigmaSA0, sigmaSA1)
        if dist >= EA_viol_analytic[count]:
            EA_viol_analytic[count] = dist
            p_opt[count] = p

# ----------------------------------------------------------------------
# Stack and save results
# ----------------------------------------------------------------------
omega = np.array([(i + 1) * 0.01 for i in range(omega_num)])
data_to_save = np.column_stack((omega, p_opt, EA_viol_analytic))

if save:
    np.savetxt(
        output_path,
        data_to_save,
        fmt="%.6f",
        header="omega p_opt EA_viol_analytic",
        comments="# "
    )
    print(f"\n Data successfully saved to {output_path}")
else:
    print("\n Results were not saved (save=False).")

