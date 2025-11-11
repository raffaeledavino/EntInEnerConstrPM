import numpy as np
import qutip as qt
import sympy as sp
from tqdm import tqdm
import os
from ncpol2sdpa import generate_variables, SdpRelaxation
from utils import *

"""
Script to compute the diamond distance and the induced trace norm between the identity channel
and the channel defined in Appendix C. These quantities are used to construct the upper bound
on the guessing probability advantage without an energy constraint (Eq. (22)), and the lower
bound with an energy constraint (Eq. (23)).

Results are saved to 'Data/data_channel_discr_adv.txt' as tuples
(omega, entangled_adv, non_entangled_adv), unless save=False is specified.
"""

if __name__ == "__main__":

    # Parameters
    energyrange = np.arange(0.01, 0.50, 0.01)
    p_values = np.linspace(0.00, 0.001, 2)

    # Prepare output directory
    save = False  # Set to False if you don't want to save result
    output_file = "data_channel_discr_adv.txt"
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "Data")
    data_path = os.path.join(data_dir, output_file)
    os.makedirs(data_dir, exist_ok=True)

    # Variables initialization
    num_omega = len(energyrange)
    p_opt = np.zeros(num_omega)
    diamond_dist_EC_list = np.zeros(num_omega)
    diamond_dist_noEC_list = np.zeros(num_omega)
    ind_trace_dist_noEC_list = np.zeros(num_omega)
    ind_trace_dist_EC_list = np.zeros(num_omega)

    # Basis definitions for a two-qubit (4-dimensional) Hilbert space
    b00, b01, b10, b11 = [qt.basis(4, i) for i in range(4)]

    for count, w in enumerate(tqdm(energyrange)):
        advantage = -np.inf

        for p in p_values:
            # Build joint states sigma_SA^0 and sigma_SA^1 (as in the paper)
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

            # Compute Kraus operators E0, E1
            eigenvalues0, eigenvectors0 = sigmaSA0.eigenstates()
            eigenvalues1, eigenvectors1 = sigmaSA1.eigenstates()
        
            

            # Build operator A from the eigenbasis of sigma_SA^0 (Eq. C15)
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
            
            # Kraus operators of the effective channel (Eq. C13)
            E0 = np.sqrt(eigenvalues1[2])*B0*A.inv()
            E1 = np.sqrt(eigenvalues1[3])*B1*A.inv()
            

            # ---------------------------------------------------------------------
            # 1. Upper bound on the entanglement advantage without energy constraint.
            # ---------------------------------------------------------------------
            
            # Compute diamond distance (no EC)  
            kraus = [E0, E1]
            choi_sigma_SA = choi_matrix(kraus)
            dnorm, _ = diamond_norm_distance(choi_sigma_SA.real)
            diamond_dist_noEC = dnorm


            # Compute the induced trace distance via seesaw optimization (no EC)
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
            # 2. Lower bound on the entanglement advantage
            # in the energy-constrained scenario.
            # ---------------------------------------------------------------------

            # Compute a lower bound on the diamond distance (EC)
            diamond_dist_EC = 2 * qt.metrics.tracedist(sigmaSA0, sigmaSA1)

            # Compute an upper bound on the induced trace norm (EC) via SDP relaxation
            E0_sym, E1_sym = sp.Matrix(E0), sp.Matrix(E1)
            x = generate_variables('x', 7)

            # Variables:
            # x[0:3]  → Measurement operator M (2×2 Hermitian)
            # x[4:6]  → State psi (2×2 Hermitian, parametrized by Bloch components)
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

            # Inequalities: physical constraints (positivity, normalization, etc.)
            inequalities = [
                # Positivity of M
                x[0], x[3], x[0] * x[3] - x[1] ** 2 - x[2] ** 2,
                1 - x[0], 1 - x[3],
                (1 - x[0]) * (1 - x[3]) - x[1] ** 2 - x[2] ** 2,

                # Positivity of psi
                x[4], 1 - x[4],
                x[4] * (1 - x[4]) - x[5] ** 2 - x[6] ** 2,

                # Energy constraint
                x[4] - (1 - w),

                # Average energy after channel action
                E0[0, 0] ** 2 * x[4] + 2 * E0[0, 1] * E0[0, 0] * x[5] + E0[0, 1] ** 2 * (1 - x[4])
                + E1[0, 0] ** 2 * x[4] + 2 * E1[0, 1] * E1[0, 0] * x[5] + E1[0, 1] ** 2 * (1 - x[4]) - (1 - w)
            ]

            # Objective: maximize Tr[M (Lambda(psi) - psi)]
            objective = sp.trace(M * (E0_psi_E0dag + E1_psi_E1dag - psi))

            sdp = SdpRelaxation(x)
            sdp.get_relaxation(3, objective=-objective, inequalities=inequalities)
            sdp.solve()

            ind_trace_dist_EC = -2 * sdp.primal
            
            # Ratio of discrimination probabilities
            fraction_noEC = (1 + 0.5 * diamond_dist_noEC) / (1 + 0.5 * ind_trace_dist_noEC)
            fraction_EC = (1 + 0.5 * diamond_dist_EC) / (1 + 0.5 * ind_trace_dist_EC)
            
            
            if fraction_EC>=advantage:
                advantage = fraction_EC
                p_opt[count] = p
                diamond_dist_noEC_list[count] = diamond_dist_noEC
                ind_trace_dist_noEC_list[count] = ind_trace_dist_noEC 
                diamond_dist_EC_list[count] = diamond_dist_EC
                ind_trace_dist_EC_list[count] = ind_trace_dist_EC



    # Stack the arrays column-wise (each column: omega, non-entangled, entangled)
    omega = np.array([(i + 1) * 0.01 for i in range(num_omega)])

    non_ea = (1 + 0.5 * np.array(diamond_dist_noEC_list)) / (1 + 0.5 * np.array(ind_trace_dist_noEC_list))
    ea = (1 + 0.5 * np.array(diamond_dist_EC_list)) / (1 + 0.5 * np.array(ind_trace_dist_EC_list))

    results = np.column_stack((omega, ea, non_ea))

    # ----------------------------------------------------------------------
    # Stack and save results
    # ----------------------------------------------------------------------
    if save:
        with open(data_path, "w") as f:
            f.write("# omega, entangled_adv, non_entangled_adv\n")
            for row in results:
                f.write(f"({row[0]:.5f}, {row[1]:.10f}, {row[2]:.10f})\n")
        print(f"Data saved to {data_path}")
    else:
        data_path = None
        print("Saving disabled (save=False).")

    # ----------------------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------------------
    plot_channel_discrimination_advantage(
        results,    # Use `results` to plot data from this run, or "Data/filename.txt" to plot previously saved data
        save_as="Fig_channel_discr_adv.png",
        save=False
    )
