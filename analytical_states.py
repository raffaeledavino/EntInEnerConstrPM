import numpy as np
import qutip as qt
from tqdm import tqdm

from utils import trace_distance


"""
Script to compute the optimal parameter p_opt and the corresponding
trace distance between two quantum states sigma_SA^0 and sigma_SA^1
for a range of energy-like parameters w.

This script scans over a grid of (w, p) values, constructs the two
bipartite states using QuTiP, and computes the maximum trace distance
between them for each w. The results are saved to a text file.
"""


# Parameters
omega_num = 50
p_values = np.linspace(0.01, 0.50, 1000)
optimal_distance = [0 for _ in range(omega_num)]
p_opt = [0 for _ in range(omega_num)]

# Basis definitions for a two-qubit (4-dimensional) Hilbert space
b00, b01, b10, b11 = [qt.basis(4, i) for i in range(4)]

# ------------------------------------------------------------------------------
# Main computation loop
# ------------------------------------------------------------------------------

for count in tqdm(range(omega_num)):
    w = (count + 1) * 0.01
    print(w)
    for p in p_values:
        sigmaSA0 = qt.Qobj(
            ((1 - w) * b00 * b00.dag() + w * p * b10 * b10.dag() + w * (1 - p) * b11 * b11.dag()
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

        dist = 2 * qt.metrics.tracedist(sigmaSA0, sigmaSA1)
        if dist >= optimal_distance[count]:
            p_opt[count] = p
            optimal_distance[count] = dist

print(p_opt)
print(optimal_distance)

# Reconstruct omega values
omega = np.array([(i + 1) * 0.01 for i in range(omega_num)])
p_opt = np.array(p_opt)
optimal_distance = np.array(optimal_distance)

# Stack the arrays column-wise
data_to_save = np.column_stack((omega, p_opt, optimal_distance))

# Save to TXT
np.savetxt(
    "optimal_distance_data.txt",
    data_to_save,
    fmt="%.6f",
    header="omega p_opt optimal_distance",
    comments="# "   # adds "# " at the start of the header line
)

print("Data saved to optimal_distance_data.txt")