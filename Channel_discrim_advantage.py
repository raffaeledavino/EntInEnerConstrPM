from ncpol2sdpa import generate_variables, SdpRelaxation
import numpy as np
import qutip as qt
import picos as pic
import sympy as sp
from tqdm import tqdm

from utils import *

# =============================================================================
# Utility Functions
# =============================================================================


def random_qubit_state():
    """
    Generate a random qubit pure state as a density matrix.
    """
    psi = np.random.randn(2) + 1j * np.random.randn(2)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())



"""
Script to compute and compare various distinguishability measures
(trace distance, diamond norm, induced norm, etc.)
for a parametrized pair of quantum states sigma_SA^0 and sigma_SA^1.

For each energy parameter omega, the script scans over a small range of noise
parameters p and computes:

  • Optimal trace distance between sigma_SA^0 and sigma_SA^1
  • Diamond norm distance of the corresponding channel
  • Induced trace distance (with and without error correction)
  • Advantage ratio (fraction_ec / fraction_no_ec)

The results can be used to analyze the relationship between
different distinguishability measures under energy constraints.
"""

# =============================================================================
# Main Numerical Simulation
# =============================================================================

# Parameters
omega_num = 2
p_values = np.linspace(0.00, 0.002, 3)
p_opt = np.zeros(omega_num)
optimal_distance = np.zeros(omega_num)
diamond_dist = np.zeros(omega_num)
ind_trace_dist = np.zeros(omega_num)
ind_trace_dist_ec = np.zeros(omega_num)

# Basis definitions for a two-qubit (4-dimensional) Hilbert space
b00, b01, b10, b11 = [qt.basis(4, i) for i in range(4)]

for count in tqdm(range(omega_num)):

    w = (count + 1) * 0.01
    advantage = 0

    for p in p_values:
        # Build joint states sigma_SA^0 and sigma_SA^1 (as in the paper)
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

        # ----------------------------------------------------------------------
        # Trace distance between the two states
        # ----------------------------------------------------------------------
        optimal_distance_EC = 2 * qt.metrics.tracedist(sigmaSA0, sigmaSA1)

        # Compute Kraus operators E0, E1
        eigenvalues0, eigenvectors0 = sigmaSA0.eigenstates()
        eigenvalues1, eigenvectors1 = sigmaSA1.eigenstates()
    
        

        # Build operator A from the eigenbasis of sigma_SA^0
        A = qt.Qobj([
            [np.sqrt(2) * eigenvectors0[3][0][0][0], np.sqrt(2) * eigenvectors0[3][1][0][0]],
            [np.sqrt(2) * eigenvectors0[3][2][0][0], np.sqrt(2) * eigenvectors0[3][3][0][0]]
        ])
    
        # Build operators B0, B1 from the eigenbasis of sigma_SA^1
        B0 = qt.Qobj([
            [np.sqrt(2) * eigenvectors1[2][0][0][0], np.sqrt(2) * eigenvectors1[2][1][0][0]],
            [np.sqrt(2) * eigenvectors1[2][2][0][0], np.sqrt(2) * eigenvectors1[2][3][0][0]]
        ])
        B1 = qt.Qobj([
            [np.sqrt(2) * eigenvectors1[3][0][0][0], np.sqrt(2) * eigenvectors1[3][1][0][0]],
            [np.sqrt(2) * eigenvectors1[3][2][0][0], np.sqrt(2) * eigenvectors1[3][3][0][0]]
        ])
        
        # Kraus operators of the effective channel
        E0 = np.sqrt(eigenvalues1[2])*B0*A.inv()
        E1 = np.sqrt(eigenvalues1[3])*B1*A.inv()
        
        # ----------------------------------------------------------------------
        # Compute diamond norm of the channel defined by {E0, E1}
        # ----------------------------------------------------------------------    
        kraus = [E0, E1]
        choi_sigma_SA = choi_matrix(kraus)
        dnorm, _ = diamond_norm_distance(choi_sigma_SA.real)
        diamond_dist_noEC = dnorm

        # ---------------------------------------------------------------------
        # 2. Compute the induced trace norm via seesaw optimization (no EC)
        # ---------------------------------------------------------------------
        E0_np, E1_np = E0.full().real, E1.full().real
        new_norm_max = 0.0

        for _ in range(10):
            rho = random_qubit_state()
            newnorm, oldnorm = 1.0, 0.0

            # Seesaw iterations until convergence
            while abs(newnorm.real - oldnorm.real) > 1e-7:
                oldnorm = newnorm
                _, Y = induced_norm_distance_seesaw1(E0_np, E1_np, rho)
                newnorm, rho = induced_norm_distance_seesaw2(E0_np, E1_np, Y)

            if newnorm > new_norm_max:
                new_norm_max = newnorm

        ind_trace_dist_noEC = 2 * new_norm_max

        # ---------------------------------------------------------------------
        # 3. Compute the induced trace norm with error correction via SDP relaxation
        # ---------------------------------------------------------------------
        E0_sym, E1_sym = sp.Matrix(E0), sp.Matrix(E1)
        x = generate_variables('x', 7)

        # Variables:
        # x[0:3]  → Measurement operator M (2×2 Hermitian)
        # x[4:6]  → State ψ (2×2 Hermitian, parametrized by Bloch components)
        
        M = sp.Matrix([
            [x[0],        x[1] + 1j * x[2]],
            [x[1] - 1j * x[2],  x[3]]
        ])
        
        psi = sp.Matrix([
            [x[4],        x[5] + 1j * x[6]],
            [x[5] - 1j * x[6], 1 - x[4]]
        ])


        # Channel action on psi
        E0_psi_E0dag = E0_sym * psi * E0_sym.H
        E1_psi_E1dag = E1_sym * psi * E1_sym.H

        # ---------------------------------------------------------------------
        # Inequalities: physical constraints (positivity, normalization, etc.)
        # ---------------------------------------------------------------------

        inequalities = [
            # Positivity of M
            x[0], x[3], x[0] * x[3] - x[1] ** 2 - x[2] ** 2,
            1 - x[0], 1 - x[3],
            (1 - x[0]) * (1 - x[3]) - x[1] ** 2 - x[2] ** 2,

            # Positivity of ψ
            x[4], 1 - x[4],
            x[4] * (1 - x[4]) - x[5] ** 2 - x[6] ** 2,

            # Energy constraint
            x[4] - (1 - w),

            # Average energy after channel action
            E0[0, 0] ** 2 * x[4] + 2 * E0[0, 1] * E0[0, 0] * x[5] + E0[0, 1] ** 2 * (1 - x[4])
            + E1[0, 0] ** 2 * x[4] + 2 * E1[0, 1] * E1[0, 0] * x[5] + E1[0, 1] ** 2 * (1 - x[4]) - (1 - w)
        ]

        # ---------------------------------------------------------------------
        # Objective: maximize Tr[M (Lambda(psi) - psi)]
        # ---------------------------------------------------------------------

        objective = sp.trace(M * (E0_psi_E0dag + E1_psi_E1dag - psi))

        # ---------------------------------------------------------------------
        # 4. SDP Relaxation
        # ---------------------------------------------------------------------

        sdp = SdpRelaxation(x)
        sdp.get_relaxation(3, objective=-objective, inequalities=inequalities)
        sdp.solve()

        ind_trace_dist_EC = -2 * sdp.primal
        
        # Ratio of discrimination probabilities
        fraction_no_ec = (1 + 0.5 * diamond_dist_noEC) / (1 + 0.5 * ind_trace_dist_noEC)
        fraction_ec = (1 + 0.5 * optimal_distance_EC) / (1 + 0.5 * ind_trace_dist_EC)
        
        
        if fraction_ec/fraction_no_ec>=advantage:
            advantage = fraction_ec/fraction_no_ec
            p_opt[count] = p
            diamond_dist[count] = diamond_dist_noEC
            ind_trace_dist[count] = ind_trace_dist_noEC 
            optimal_distance[count] = optimal_distance_EC
            ind_trace_dist_ec[count] = ind_trace_dist_EC



# === Save computed data ===
output_file = "Channel_discr_adv_data.txt"

# Stack the arrays column-wise (each column: omega, non-entangled, entangled)
omega = np.array([(i + 1) * 0.01 for i in range(omega_num)])

non_ea = (1 + 0.5 * np.array(diamond_dist)) / (1 + 0.5 * np.array(ind_trace_dist))
ea = (1 + 0.5 * np.array(optimal_distance)) / (1 + 0.5 * np.array(ind_trace_dist_ec))

data_to_save = np.column_stack((omega, ea, non_ea))

# Save to a .txt file with a header, same as in your Pguess script
with open(output_file, "w") as f:
    f.write("# omega, entangled_adv, non_entangled_adv\n")
    for row in data_to_save:
        f.write(f"({row[0]:.5f}, {row[1]:.10f}, {row[2]:.10f})\n")

print(f"Data saved to {output_file}")

# =============================================================================
# Results and Plot
# =============================================================================
plot_channel_discrimination_advantage("Channel_discr_adv_data.txt", save_as="Fig_advantage.png")

