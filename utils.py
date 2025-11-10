import numpy as np
import picos as pic
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import ConnectionPatch
from picos.modeling.problem import Problem



# ----------------------------------------------------------------------
# Configure numerical precision for PICOS solvers
# ----------------------------------------------------------------------
def setNumericalPrecisionForSolver(problem, precision):
    """
    Sets high numerical precision parameters for PICOS solvers.
    """
    problem.options["rel_ipm_opt_tol"] = precision
    problem.options["rel_prim_fsb_tol"] = precision
    problem.options["rel_dual_fsb_tol"] = precision
    problem.options["max_footprints"] = None



# ----------------------------------------------------------------------
# Utilities to compute the maximum violation of I_corr (Eq. (6)),
# both for separable strategies and when entanglement is allowed.
# ----------------------------------------------------------------------

# Compute the maximum I_corr violation without entanglement (Eq. (7))
def computeMaxIneqViolation(w1, w2, precision):
    """
    Compute the maximal value of I_corr (Eq. 7) under separable
    (non-entangled) strategies using semidefinite programming.

    Parameters
    ----------
    w1 : float
        Average energy parameter for the first state.
    w2 : float
        Average energy parameter for the second state.
    precision : float
        Numerical precision tolerance for the solver (e.g. 1e-8).

    Returns
    -------
    value : float
        Maximum inequality value achieved.
    probs : list of float
        Probabilities [p00, p10, p01, p11] corresponding to the optimal solution.
    """
    # Initialize the SDP problem
    prob = pic.Problem()

    # Define variables
    gamma = pic.SymmetricVariable("gamma", (4, 4))
    eta1 = pic.RealVariable("eta1")
    eta2 = pic.RealVariable("eta2")

    # Coefficients of the target inequality
    f = [1, -1, -1, 1]

    # Constraints
    prob.add_constraint(gamma >> 0)
    prob.add_constraint(eta1 <= w1)
    prob.add_constraint(eta2 <= w2)

    for i in range(4):
        prob.add_constraint(gamma[i, i] == 1)

    prob.add_list_of_constraints([
        gamma[0, 3] == 2 * eta1 - 1,
        gamma[1, 3] == 2 * eta2 - 1
    ])

    # Define expectation values
    E0, E1 = gamma[0, 2], gamma[1, 2]

    # Define joint probabilities
    p00 = 0.5 * (1 + E0)
    p10 = 1 - p00
    p01 = 0.5 * (1 + E1)
    p11 = 1 - p01

    # Probability constraints
    prob.add_list_of_constraints([
        p00 >= 0, p10 >= 0, p01 >= 0, p11 >= 0
    ])

    # Objective function
    objective = f[0]*p00 + f[1]*p10 + f[2]*p01 + f[3]*p11
    prob.set_objective("max", objective)

    # Set numerical precision and solve
    setNumericalPrecisionForSolver(prob, precision)
    prob.solve(solver="mosek", verbosity=False)

    # Return optimal value and probabilities
    return prob.value, [p00.value, p10.value, p01.value, p11.value]



# Seesaw in which we fix the states and optimize
# over the measurement to maximize I_corr (Eq. 6)
def findMeasurementMaxViol(dimS, dimM, states, precision):
    """
    Given the system and ancilla states sigma_SA^x, find the optimal
    two-outcome measurement {Pi_SA^0, Pi_SA^1} that maximizes I_corr.

    Parameters
    ----------
    dimS : int
        Dimension of the system.
    dimM : int
        Dimension of the ancilla (or measurement register).
    states : list of np.ndarray
        List containing two bipartite density matrices [sigma_SA^0, sigma_SA^1].
    precision : float
        Numerical precision used by the SDP solver.

    Returns
    -------
    measurement : list of np.ndarray
        The optimized measurement operators [Pi_SA^0, Pi_SA^1].
    """
    problem = Problem()

    # Measurement effects
    PiSA0 = pic.HermitianVariable("PiSA0", shape=(dimS * dimM, dimS * dimM))
    PiSA1 = pic.HermitianVariable("PiSA1", shape=(dimS * dimM, dimS * dimM))

    # Positivity and completeness constraints
    problem.add_constraint(PiSA0 >> 0)
    problem.add_constraint(PiSA1 >> 0)
    problem.add_constraint(PiSA0 + PiSA1 == np.eye(dimS * dimM))

    # Objective: maximize I_corr
    sigmaSA0, sigmaSA1 = states
    objective = (
        pic.trace(sigmaSA0 * PiSA0)
        - pic.trace(sigmaSA0 * PiSA1)
        - pic.trace(sigmaSA1 * PiSA0)
        + pic.trace(sigmaSA1 * PiSA1)
    )

    problem.set_objective("max", objective)
    setNumericalPrecisionForSolver(problem, precision)

    problem.solve(verbosity=False)

    # Extract optimal measurement
    PiSA0_opt = np.array(PiSA0.value_as_matrix)
    measurement = [PiSA0_opt, np.eye(dimS * dimM) - PiSA0_opt]

    return measurement


# Seesaw in which we fix the measurement and optimize
# over the states to maximize I_corr (Eq. 6)
def findStateMaxViolation(dimS, dimM, w0, w1, measurement, ground, precision):
    """
    Given a fixed measurement, find the optimal bipartite states
    sigma_SA^0 and sigma_SA^1 that maximize I_corr, under energy constraints.

    Parameters
    ----------
    dimS, dimM : int
        Dimensions of system and ancilla subsystems.
    w0, w1 : float
        Energy constraints for sigma_SA^0 and sigma_SA^1 respectively.
    measurement : list of np.ndarray
        Measurement operators [Pi_SA^0, Pi_SA^1].
    ground : np.ndarray
        Ground state projector on the system.
    precision : float
        Numerical precision used by the SDP solver.

    Returns
    -------
    (float, list of np.ndarray)
        Maximum I_corr value and the corresponding optimal states
        [sigma_SA^0, sigma_SA^1].
    """
    PiSA0, PiSA1 = measurement
    problem = Problem()

    # Variables: bipartite states sigma_SA^0 and sigma_SA^1
    states = [
        pic.HermitianVariable(f"sigma_{x}", shape=(dimS * dimM, dimS * dimM))
        for x in range(2)
    ]
    sigmaSA0, sigmaSA1 = states

    # Basic constraints: positivity and unit trace
    problem.add_list_of_constraints([rho >> 0 for rho in states])
    problem.add_list_of_constraints([pic.trace(rho) == 1 for rho in states])

    # Equal marginal constraint: Tr_S(sigma_SA^0) = Tr_S(sigma_SA^1)
    problem.add_constraint(
        pic.partial_trace(sigmaSA0, 0, [dimS, dimM])
        == pic.partial_trace(sigmaSA1, 0, [dimS, dimM])
    )

    # Energy constraints: ⟨ground|sigma_SA^x|ground⟩ ≥ 1 - w_x
    problem.add_constraint(
        pic.trace(np.kron(ground, np.eye(dimM)) * sigmaSA0) >= 1 - w0
    )
    problem.add_constraint(
        pic.trace(np.kron(ground, np.eye(dimM)) * sigmaSA1) >= 1 - w1
    )

    # Objective: maximize I_corr
    objective = (
        pic.trace(sigmaSA0 * PiSA0)
        - pic.trace(sigmaSA0 * PiSA1)
        - pic.trace(sigmaSA1 * PiSA0)
        + pic.trace(sigmaSA1 * PiSA1)
    )

    problem.set_objective("max", objective)
    setNumericalPrecisionForSolver(problem, precision)

    problem.solve(verbosity=False)

    # Extract optimal states and value
    value = problem.value
    sigmaSA_opt = [np.array(rho.value_as_matrix) for rho in states]

    return value, sigmaSA_opt




# ----------------------------------------------------------------------
# Utilities to compute the maximum guessing probability (P_guess)
# achievable with quantum resources and entanglement assistance (EA),
# as defined in Eq. (8).
# ----------------------------------------------------------------------


# Add normalization constraints (for ComputeGuessingProbability function)
def addNormalizationConstraints(prob, gammas):
    """
    Enforces normalization constraints on the list of moment matrices gamma.

    Each gamma corresponds to a conditional probability distribution p(γ),
    which appears as gamma[0,0] on the diagonal. The constraint ensures that:
        - p(gamma) ≥ 0
        - gamma[i,i] = p(gamma) for i = 1, 2, 3
        - The probabilities sum to 1: Σ p(gamma) = 1

    Parameters
    ----------
    prob : pic.Problem
        The optimization problem to which constraints are added.
    gammas : list of pic.MatrixVariable
        List of moment matrices gamma_AX associated with different settings.
    """
    total_prob = 0

    for gamma in gammas:
        p_gamma = gamma[0, 0]

        # Basic positivity and normalization conditions
        prob.add_constraint(p_gamma >= 0)
        for i in range(1, 4):
            prob.add_constraint(gamma[i, i] == p_gamma)

        total_prob += p_gamma

    # Global normalization
    prob.add_constraint(total_prob == 1)


# Add average energy constraints for the states states^x (for ComputeGuessingProbability function)
def addEnergyConstraints(prob, gammas, w0Avg, w1Avg, w0Peak, w1Peak):
    """
    Adds energy constraints linking the moment matrices gamma to average and
    peak energy parameters (w0Avg, w1Avg, w0Peak, w1Peak).

    For each gamma:
        - Introduce auxiliary variables w0, w1 ≥ 0
        - Bound them by the local probabilities and energy peaks
        - Impose consistency relations connecting gamma entries to w0, w1

    Global constraints ensure:
        sum w0 ≤ w0Avg   and   sum w1 ≤ w1Avg

    Parameters
    ----------
    prob : pic.Problem
        The optimization problem to which constraints are added.
    gammas : list of pic.MatrixVariable
        List of moment matrices gamma.
    w0Avg, w1Avg : float
        Average (expected) energy limits for x = 0,1.
    w0Peak, w1Peak : float
        Peak energy values for x = 0,1.
    """
    w0_vars, w1_vars = [], []

    for idx, gamma in enumerate(gammas):
        p_gamma = gamma[0, 0]

        # Auxiliary real variables representing weighted energies
        w0 = pic.RealVariable(f"w0_{idx}")
        w1 = pic.RealVariable(f"w1_{idx}")
        w0_vars.append(w0)
        w1_vars.append(w1)

        # Local energy bounds
        prob.add_constraint(w0 >= 0)
        prob.add_constraint(w1 >= 0)
        prob.add_constraint(w0 <= p_gamma * w0Peak)
        prob.add_constraint(w1 <= p_gamma * w1Peak)

        # Consistency with γ matrix elements
        prob.add_constraint(gamma[0, 3] + p_gamma <= 2 * w0)
        prob.add_constraint(gamma[1, 3] + p_gamma <= 2 * w1)

    # Global energy budget constraints
    prob.add_constraint(sum(w0_vars) <= w0Avg)
    prob.add_constraint(sum(w1_vars) <= w1Avg)


""" def computeGuessingProbability(w0Avg, w1Avg,w0Peak,w1Peak,expectedBehavior):
    prob=pic.Problem()

    gamma00=pic.SymmetricVariable('gamma00',(4,4))
    gamma01=pic.SymmetricVariable('gamma01',(4,4))
    gammas0=[gamma00,gamma01]

    gamma10=pic.SymmetricVariable('gamma10',(4,4))
    gamma11=pic.SymmetricVariable('gamma11',(4,4))
    gammas1=[gamma10,gamma11]
    
    for gamma in gammas0:
        prob.add_constraint(gamma>>0)
    addNormalizationConstraints(prob,gammas0)
    addEnergyConstraints(prob,gammas0,w0Avg, w1Avg,w0Peak,w1Peak)
    
    for gamma in gammas1:
        prob.add_constraint(gamma>>0)
    addNormalizationConstraints(prob,gammas1)
    addEnergyConstraints(prob,gammas1,w0Avg, w1Avg,w0Peak,w1Peak)
    
    #Eve's guessing probability
    E000,E100=gamma00[0,2],gamma00[1,2]
    E001,E101=gamma01[0,2],gamma01[1,2]
    E010,E110=gamma10[0,2],gamma10[1,2]
    E011,E111=gamma11[0,2],gamma11[1,2]
    
    E0=E000+E001
    E1=E110+E111
    p00=1/2*(1+E0)
    p10=1-p00
    p01=1/2*(1+E1)
    p11=1-p01
    
    prob.add_constraint(E000+E001==E010+E011)
    prob.add_constraint(E100+E101==E110+E111)
 
    tol = 1e-6
    prob.add_constraint(p00<=expectedBehavior[0]+tol)
    prob.add_constraint(p00>=expectedBehavior[0]-tol)
    
    prob.add_constraint(p10<=expectedBehavior[1]+tol)
    prob.add_constraint(p10>=expectedBehavior[1]-tol)
    
    prob.add_constraint(p01<=expectedBehavior[2]+tol)
    prob.add_constraint(p01>=expectedBehavior[2]-tol)
    
    prob.add_constraint(p11<=expectedBehavior[3]+tol)
    prob.add_constraint(p11>=expectedBehavior[3]-tol)
    

    pg0=1/2*(1+E000-E001)
    pg1=1/2*(1+E110-E111)
    setNumericalPrecisionForSolver(prob, precision=1e-8)

    prob.set_objective('max',pg0)#h*(pg0+pg1))
    prob.solve(solver = "mosek", verbosity = False)
    return prob.value """

# Guessing probability for a separable initial state (Ref. [23])
def ComputeGuessingProbability(w0Avg, w1Avg, w0Peak, w1Peak, nonEABellValue, precision):
    """
    Computes Eve's optimal guessing probability (Eq. 8) for a separable
    initial state under an average energy constraint.

    The optimization is formulated as a semidefinite program (SDP)
    using the moment matrices gamma0, gamma1 representing conditional states.
    The constraints enforce:
        - Positivity of gamma_x
        - Normalization and consistency conditions
        - Energy constraints on gamma_x
        - Bell inequality value ≥ nonEABellValue

    Parameters
    ----------
    w0Avg, w1Avg : float
        Average energy constraints for inputs x = 0 and x = 1.
    w0Peak, w1Peak : float
        Peak energy values for inputs x = 0 and x = 1.
    nonEABellValue : float
        Classical (non-entanglement-assisted) Bell inequality value.
    precision : float
        Numerical precision used for solver tolerances.

    Returns
    -------
    float
        Optimal (maximum) guessing probability achievable by Eve.
    """
    # --- Initialize the optimization problem ---
    prob = pic.Problem()

    # Define symmetric moment matrices gamma0 and gamma1 (one per input)
    gamma0 = pic.SymmetricVariable("gamma00", (4, 4))
    gamma1 = pic.SymmetricVariable("gamma01", (4, 4))
    gammas = [gamma0, gamma1]

    # Positivity of the moment matrices
    for gamma in gammas:
        prob.add_constraint(gamma >> 0)

    # Add normalization and energy constraints
    addNormalizationConstraints(prob, gammas)
    addEnergyConstraints(prob, gammas, w0Avg, w1Avg, w0Peak, w1Peak)

    # --- Define Bell correlators and probabilities ---
    # E_xy = ⟨A_x B_y⟩ terms
    E00, E10 = gamma0[0, 2], gamma0[1, 2]
    E01, E11 = gamma1[0, 2], gamma1[1, 2]

    # E0, E1 represent marginal combinations over Bob’s outcomes
    E0 = E00 + E01
    E1 = E10 + E11

    # Classical joint probabilities p(a|x)
    p00 = 0.5 * (1 + E0)
    p10 = 1 - p00
    p01 = 0.5 * (1 + E1)
    p11 = 1 - p01

    # --- Bell inequality constraint ---
    tol = 1e-6
    prob.add_constraint(p00 - p10 - p01 + p11 >= nonEABellValue - tol)

    # --- Objective: Eve's guessing probability ---
    pg0 = 0.5 * (1 + E00 - E01)
    pg1 = 0.5 * (1 + E10 - E11)
    prob.set_objective("max", pg0)

    # --- Set solver precision and solve ---
    setNumericalPrecisionForSolver(prob, precision)
    prob.solve(solver="mosek", verbosity=False)

    return prob.value


def findStatesGuessProb(dimS, dimM, w0, w1, measurement, ground, nonEABellValue, precision):
    """
    Solve for the optimal states maximizing the guessing probability P_guess
    given a fixed measurement.

    Args:
        dimS (int): Dimension of the system S.
        dimM (int): Dimension of the memory M.
        w0, w1 (float): Energy parameters.
        measurement (list): List of measurement operators [PiSA0, PiSA1].
        ground (np.ndarray): Ground state projector.
        nonEABellValue (float): Lower bound from the separable strategy.
        precision (float): Desired solver precision (e.g., 1e-6).

    Returns:
        tuple:
            - pguess (float): Optimal guessing probability.
            - states (list of np.matrix): Optimal states achieving pguess.
    """

    PiSA0, PiSA1 = measurement
    problem = Problem()

    # Four states: rho_{S,A}^{x,g} where x in {0,1}, g in {0,1}
    states = [pic.HermitianVariable(f"phi_{a}", shape=(dimS * dimM, dimS * dimM)) for a in range(4)]
    problem.add_list_of_constraints([rho >> 0 for rho in states])

    # Eve’s probability of guessing outcome "0"
    p0 = pic.RealVariable("p00")
    problem.add_constraint(0 <= p0 <= 1)

    # Normalization: Tr[ρ_{x,g}] = p(g)
    problem.add_constraint(pic.trace(states[0]) == p0)
    problem.add_constraint(pic.trace(states[2]) == p0)
    problem.add_constraint(pic.trace(states[1]) == 1 - p0)
    problem.add_constraint(pic.trace(states[3]) == 1 - p0)

    # Average states prepared for x = 0 and x = 1
    sigmaSA0 = states[0] + states[1]
    sigmaSA1 = states[2] + states[3]

    # Consistency of marginal states on S
    problem.add_constraint(pic.partial_trace(states[0], 0, [dimS, dimM]) == pic.partial_trace(states[2], 0, [dimS, dimM]))
    problem.add_constraint(pic.partial_trace(states[1], 0, [dimS, dimM]) == pic.partial_trace(states[3], 0, [dimS, dimM]))

    # Bell-type constraint (Eq. (6))
    lhs = (
        pic.trace(sigmaSA0 * PiSA0)
        - pic.trace(sigmaSA0 * PiSA1)
        - pic.trace(sigmaSA1 * PiSA0)
        + pic.trace(sigmaSA1 * PiSA1)
    )
    problem.add_constraint(lhs >= nonEABellValue)

    # Energy constraints
    problem.add_constraint(pic.trace(np.kron(ground, np.eye(dimM)) * sigmaSA0) >= 1 - w0)
    problem.add_constraint(pic.trace(np.kron(ground, np.eye(dimM)) * sigmaSA1) >= 1 - w1)

    # Objective: Guessing probability for x = 0
    objective = pic.trace(states[0] * PiSA0) + pic.trace(states[1] * PiSA1)
    problem.set_objective("max", objective)

    # Solver precision
    setNumericalPrecisionForSolver(problem, precision)

    # Solve the SDP
    problem.solve(solver="mosek", verbosity=0)

    return problem.value, [np.matrix(rho.value_as_matrix) for rho in states]


def findMeasurementGuessProb(dimS, dimM, states, nonEABellValue, precision):
    """
    Solve for the optimal measurement maximizing P_guess given fixed states.

    Args:
        dimS (int): Dimension of the system S.
        dimM (int): Dimension of the memory M.
        states (list): List of prepared states from findStatesGuessProb().
        nonEABellValue (float): Bell-type constraint value.
        precision (float): Desired solver precision (e.g., 1e-6).

    Returns:
        tuple:
            - pguess (float): Optimal guessing probability.
            - measurement (list of np.matrix): Optimal POVM elements [PiSA0, PiSA1].
    """

    problem = Problem()

    # Two POVM elements
    PiSA0 = pic.HermitianVariable("PiSA0", shape=(dimS * dimM, dimS * dimM))
    PiSA1 = pic.HermitianVariable("PiSA1", shape=(dimS * dimM, dimS * dimM))

    problem.add_constraint(PiSA0 >> 0)
    problem.add_constraint(PiSA1 >> 0)
    problem.add_constraint(PiSA0 + PiSA1 == np.eye(dimS * dimM))

    sigmaSA0 = states[0] + states[1]
    sigmaSA1 = states[2] + states[3]

    # Bell-type constraint (Eq. (6))
    lhs = (
        pic.trace(sigmaSA0 * PiSA0)
        - pic.trace(sigmaSA0 * PiSA1)
        - pic.trace(sigmaSA1 * PiSA0)
        + pic.trace(sigmaSA1 * PiSA1)
    )
    problem.add_constraint(lhs == nonEABellValue)

    # Objective: Guessing probability for x=0
    objective = pic.trace(states[0] * PiSA0) + pic.trace(states[1] * PiSA1)
    problem.set_objective("max", objective)

    # Solver precision
    setNumericalPrecisionForSolver(problem, precision)

    # Solve the SDP
    problem.solve(solver="mosek", verbosity=0)

    firstEffect = np.matrix(PiSA0.value_as_matrix)
    measurement = [firstEffect, np.eye(dimS * dimM) - firstEffect]

    return problem.value, measurement


# ----------------------------------------------------------------------
# Utilities to compute the maximal violation achievable by deterministic
# behaviours, as defined in Eq. (13).
# ----------------------------------------------------------------------
def findMeasurementMinViolDet(dimS, dimM, states, precision):
    
    problem = Problem()
    PiSA0=pic.HermitianVariable('PiSA0',shape=(dimS*dimM,dimS*dimM))
    PiSA1=pic.HermitianVariable('PiSA1',shape=(dimS*dimM,dimS*dimM))

    sigmaSA0=states[0]
    sigmaSA1=states[1]

    problem.add_constraint(PiSA0>>0)
    problem.add_constraint(PiSA1>>0)
    problem.add_constraint(PiSA0+PiSA1==np.eye(dimS*dimM))
    problem.add_constraint(pic.trace(sigmaSA0*PiSA0)==1)

    
    objective=pic.trace(sigmaSA1*PiSA0)-pic.trace(sigmaSA1*PiSA1)

    
    problem.set_objective("min",objective)
    setNumericalPrecisionForSolver(problem, precision)
    
    problem.solve()
    
    firstEffect = np.matrix(PiSA0.value_as_matrix)
    measurement = [firstEffect,np.eye(dimS*dimM)-firstEffect]
  
    return measurement

def findStateMinViolDet(dimS,dimM,w0,w1,measurement,ground, precision):
    
    PiSA0 = measurement[0]
    PiSA1 = measurement[1]
    
    problem = Problem()

    states=[pic.HermitianVariable('phi_'+str(a),shape=(dimS*dimM,dimS*dimM)) for a in range(2)]
    problem.add_list_of_constraints([aState>>0 for aState in states])
    problem.add_list_of_constraints([pic.trace(aState)==1 for aState in states])
    
    sigmaSA0=states[0]
    sigmaSA1=states[1]
    
    "Marginal's on A of the states sigma_{SA}^x should be equal"
    problem.add_constraint(pic.partial_trace(sigmaSA0,0,[dimS,dimM])==pic.partial_trace(sigmaSA1,0,[dimS,dimM]))
    
    "Overlap with groundstate should be large"
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA0)>=1-w0)
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA1)>=1-w1)
    problem.add_constraint(pic.trace(sigmaSA0*PiSA0)==1)

    objective=pic.trace(sigmaSA1*PiSA0)-pic.trace(sigmaSA1*PiSA1)
    
    problem.set_objective("min",objective)
    setNumericalPrecisionForSolver(problem, precision)
    
    problem.solve(verbosity=False)
    
    return problem.value, [np.matrix(aState.value_as_matrix) for aState in states]
    


# ----------------------------------------------------------------------
# Utilities to compute the upper bound on P_adv without energy constraint (Eq. (22)),
# and the lower bound in the case with energy constraint (Eq. (23)).
# ----------------------------------------------------------------------
def random_qubit_state():
    """
    Generate a random qubit pure state as a density matrix.
    """
    psi = np.random.randn(2) + 1j * np.random.randn(2)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def diamond_norm_distance(choi_E):
    """
    Compute the diamond-norm distance between a given channel and the identity.

    Parameters
    ----------
    choi_E : np.ndarray
        Choi matrix of the quantum channel.

    Returns
    -------
    float
        Diamond norm distance / 2.
    np.ndarray
        Optimal auxiliary state sigma achieving the maximum.
    """
    dim = 2  # Qubit system
    I_choi = np.array([[1, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1, 0, 0, 1]])
    J_delta = I_choi - choi_E

    prob = pic.Problem()
    Y = pic.HermitianVariable("Y", (dim**2, dim**2))
    sigma = pic.HermitianVariable("sigma", (dim, dim))
    prob.add_constraint(pic.trace(sigma) == 1)
    prob.add_constraint(sigma >> 0)

    J_param = pic.Constant("J_delta", J_delta)
    I_d = pic.Constant("I_d", np.eye(dim))

    prob.set_objective("max", pic.trace(Y * J_param))
    prob.add_constraint(-dim * (I_d @ sigma) << Y)
    prob.add_constraint(Y << dim * (I_d @ sigma))

    setNumericalPrecisionForSolver(prob, precision=1e-12)
    prob.solve(solver="mosek")

    return 0.5 * prob.value, sigma.value


def choi_matrix(kraus_ops):
    """
    Compute the Choi matrix of a quantum channel given its Kraus operators.
    """
    d = 2
    phi_plus = np.eye(d).flatten()
    J = np.zeros((d**2, d**2), dtype=complex)
    for K in kraus_ops:
        K_ext = np.kron(K, np.eye(d))
        J += K_ext @ np.outer(phi_plus, phi_plus) @ K_ext.conj().T
    return J


def induced_norm_distance_seesaw1(k1, k2, rho):
    """
    First step of the seesaw optimization for the induced trace norm.
    Maximize over Y given a fixed state rho.
    """
    d = 2
    prob = pic.Problem()
    Lambda_rho = k1 @ rho @ k1.conj().T + k2 @ rho @ k2.conj().T - rho
    Lambda_rho_pic = pic.Constant('Lambda_rho', Lambda_rho)
    I_pic = pic.Constant("I", np.identity(d))
    Y = pic.HermitianVariable("Y", (d, d))
    prob.set_objective("max", pic.trace(Y * Lambda_rho_pic).real)
    prob.add_constraint(Y - I_pic << 0)
    prob.add_constraint(Y >> 0)
    prob.solve()
    return prob.value, Y.value


def induced_norm_distance_seesaw2(k1, k2, Y):
    """
    Second step of the seesaw optimization for the induced trace norm.
    Maximize over rho given a fixed operator Y.
    """
    d = 2
    prob = pic.Problem()
    rho = pic.HermitianVariable("rho", (d, d))
    Lambda_rho = k1 * rho * k1.conj().T + k2 * rho * k2.conj().T - rho
    obj = pic.trace(Y * Lambda_rho).real
    prob.set_objective("max", obj)
    prob.add_constraint(rho >> 0)
    prob.add_constraint(pic.trace(rho) == 1)
    prob.solve()
    return prob.value.real, rho.value



# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_EA_violation_qutrit_analytic(data_source,
                                      save_as="Fig_correlations_adv.png",
                                      save=True):
    """
    Plot entanglement-assisted (EA) and non-entanglement-assisted (non-EA)
    correlations replicating Fig. 3 of the paper.

    Parameters
    ----------
    data_source : str or array-like
        Either:
        - Path to the .txt file containing four comma-separated columns:
              omega, non_EA, EA, EA_analytic
        - Or a NumPy array/list of shape (N, 4) with those values directly.
    save_as : str, optional
        Name of the output figure file (default: "Fig_correlations_adv.png").
    save : bool, optional
        Whether to save the plot. If False, the figure is only displayed.
    """

    # --- Setup directories ---
    base_dir = os.path.dirname(__file__)
    plots_dir = os.path.join(base_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load data ---
    if isinstance(data_source, str):
        # Load from file
        omega, non_ea, ea, ea_analytic = np.loadtxt(
            data_source, delimiter=",", unpack=True, comments="#"
        )
    else:
        # Use data directly (from memory)
        data_array = np.array(data_source)
        if data_array.shape[1] < 4:
            raise ValueError("Data must have four columns: omega, non_EA, EA, EA_analytic.")
        omega, non_ea, ea, ea_analytic = data_array.T

    # --- Main plot ---
    fig, ax = plt.subplots()
    ax.plot(omega, non_ea, label=r'non-EA', linewidth=2)
    ax.plot(omega, ea, label=r'EA (numerical)', linewidth=2)
    ax.plot(omega, ea_analytic, label=r'EA (analytic)', linewidth=2)

    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.set_ylabel(r'max $I_{\mathrm{corr}}$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # --- Inset plot (zoomed region) ---
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=2)
    axins.plot(omega, non_ea, linewidth=2)
    axins.plot(omega, ea, linewidth=2)
    axins.plot(omega, ea_analytic, linewidth=2)

    # Define zoom window
    x1, x2 = 0.01, 0.0105
    axins.set_xlim(x1, x2)

    
    axins.set_ylim(0.3975, 0.409)

    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(axis='both', which='major', labelsize=9)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(1.0)

    axins.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray",
               lw=1.2, linestyle="--", alpha=0.7)

    # --- Save or show ---
    if save:
        save_path = os.path.join(plots_dir, save_as)
        fig.savefig(save_path, dpi=1000, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        print("Plot not saved (save=False).")

    plt.show()



def plot_min_entropy(data_source,
                     save_as="Fig_pguess.png",
                     save=True):
    """
    Plot the min-entropy H_min from a data file containing
    (omega, non_ea, ea) values, replicating Fig. 4 of the paper.

    Parameters
    ----------
    data_source : str | array-like
        Either the filename (e.g., 'Data/data_avg_pg.txt')
        or directly a numpy array/list of tuples (omega, non_ea, ea).
    save_as : str, optional
        Output filename for saving the figure (default: 'Pguess.png').
    save : bool, optional
        If True, saves the figure in the /Plots directory.
    """


    # --- Prepare output directory ---
    plots_dir = os.path.join(os.path.dirname(__file__), "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or assign data
    # ------------------------------------------------------------------
    if isinstance(data_source, str):
        # Load from file
        omega, non_ea, ea = [], [], []
        with open(data_source, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace("(", "").replace(")", "")
                parts = line.split(",")
                omega.append(float(parts[0]))
                non_ea.append(float(parts[1]))
                ea.append(float(parts[2]))
        omega, non_ea, ea = np.array(omega), np.array(non_ea), np.array(ea)
        print(f"Data loaded from file: {data_source}")
    else:
        # Directly use in-memory data
        data_source = np.array(data_source)
        omega, non_ea, ea = data_source[:, 0], data_source[:, 1], data_source[:, 2]
        print("Using in-memory data (not loaded from file).")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()

    ax.plot(omega, -np.log2(non_ea), label=r'$H_{\min}^{*, \mathrm{sep}}$ [22]', linewidth=2)
    ax.plot(omega, -np.log2(ea), label=r'Seesaw upper bound to $H_{\min}^*$', linewidth=2)

    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.set_ylabel(r'$H_{\min}$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # --- Inset plot ---
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)
    axins.plot(omega, -np.log2(non_ea), linewidth=2)
    axins.plot(omega, -np.log2(ea), linewidth=2)

    x1, x2 = 0.32, 0.50
    axins.set_xlim(x1, x2)

    y_data = (
        [-np.log2(y) for x, y in zip(omega, ea) if x1 <= x <= x2] +
        [-np.log2(y) for x, y in zip(omega, non_ea) if x1 <= x <= x2]
    )
    if y_data:
        y_min, y_max = min(y_data), max(y_data)
        padding = 0.05 * (y_max - y_min)
        axins.set_ylim(y_min - padding, y_max + padding)

    axins.grid(True, linestyle="--", alpha=0.5)
    axins.tick_params(axis='both', which='major', labelsize=9)
    axins.set_facecolor("white")
    for spine in axins.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(1.0)

    axins.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axins.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # Connect inset with main plot
    con1 = ConnectionPatch(xyA=(x1, y_min), coordsA=axins.transData,
                           xyB=(x1, y_min), coordsB=ax.transData,
                           color="gray", linestyle="--", lw=1.0, alpha=0.6, clip_on=False)
    con2 = ConnectionPatch(xyA=(x2, y_max), coordsA=axins.transData,
                           xyB=(x2, y_max), coordsB=ax.transData,
                           color="gray", linestyle="--", lw=1.0, alpha=0.6, clip_on=False)
    ax.add_artist(con1)
    ax.add_artist(con2)

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if save:
        save_path = os.path.join(plots_dir, save_as)
        fig.savefig(save_path, dpi=1000, bbox_inches="tight")
        print(f"Figure saved as: {save_path}")
    else:
        print("Plot not saved (save=False).")

    plt.show()



def plot_deterministic_inequality_violation(data_source,
                                            save_as="Fig_viol_det_ineq.png",
                                            save=True):
    """
    Plot the violation of the deterministic inequality for separable and 
    entanglement-assisted scenarios, replicating Fig. 5 of the paper.

    Parameters
    ----------
    data_source : str | array-like
        Either a filename (e.g., 'Data/data_viol_det_ineq.txt')
        or directly a numpy array/list of (omega, ea, non_ea) tuples.
    save_as : str, optional
        Output filename for saving the figure (default: 'Fig_viol_det_ineq.png').
    save : bool, optional
        If True, saves the figure in the /Plots directory.
    """

    # --- Prepare output directory ---
    plots_dir = os.path.join(os.path.dirname(__file__), "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or assign data
    # ------------------------------------------------------------------
    if isinstance(data_source, str):
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"File not found: {data_source}")
        omega, ea, non_ea = [], [], []
        with open(data_source, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace("(", "").replace(")", "")
                parts = [p.strip() for p in line.split(",")]
                try:
                    omega.append(float(parts[0]))
                    non_ea.append(float(parts[1]))
                    ea.append(float(parts[2]))
                except (ValueError, IndexError):
                    continue
        omega, ea, non_ea = np.array(omega), np.array(ea), np.array(non_ea)
        print(f"Data loaded from file: {data_source}")
    else:
        # Directly use in-memory data
        data_source = np.array(data_source)
        if data_source.shape[1] != 3:
            raise ValueError("In-memory data must have 3 columns: (omega, ea, non_ea)")
        omega, non_ea, ea = data_source[:, 0], data_source[:, 1], data_source[:, 2]
        print("Using in-memory data (not loaded from file).")


    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()

    ax.plot(omega, non_ea, label=r'$I_{\mathrm{det}}^{\omega, \mathrm{sep}}$', linewidth=2)
    ax.plot(omega, ea, label=r'Seesaw upper bound to Eq. (13)', linewidth=2)

    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=12)

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if save:
        save_path = os.path.join(plots_dir, save_as)
        fig.savefig(save_path, dpi=1000, bbox_inches="tight")
        print(f"Figure saved as: {save_path}")
    else:
        print("Plot not saved (save=False).")

    plt.show()


def plot_channel_discrimination_advantage(data_source,
                                          save_as="Fig_channel_discr_adv.png",
                                          save=True):
    """
    Plot the channel discrimination advantage data, replicating Fig. 6.

    Parameters
    ----------
    data_source : str | array-like
        Either a filename (e.g., 'Data/data_channel_discr_adv.txt')
        or directly a numpy array/list of (omega, ea, non_ea) tuples.
    save_as : str, optional
        Output filename for saving the figure (default: 'Fig_channel_discr_adv.png').
    save : bool, optional
        If True, saves the figure in the /Plots directory.
    """

    # --- Prepare output directory ---
    plots_dir = os.path.join(os.path.dirname(__file__), "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or assign data
    # ------------------------------------------------------------------
    if isinstance(data_source, str):
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"File not found: {data_source}")
        omega, ea, non_ea = [], [], []
        with open(data_source, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                line = line.replace("(", "").replace(")", "")
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    omega.append(float(parts[0]))
                    ea.append(float(parts[1]))
                    non_ea.append(float(parts[2]))
                except ValueError:
                    continue
        omega, ea, non_ea = np.array(omega), np.array(ea), np.array(non_ea)
        print(f"Data loaded from file: {data_source}")
    else:
        # Directly use in-memory data
        data_source = np.array(data_source)
        if data_source.shape[1] != 3:
            raise ValueError("In-memory data must have 3 columns: (omega, ea, non_ea)")
        omega, ea, non_ea = data_source[:, 0], data_source[:, 1], data_source[:, 2]
        print("Using in-memory data (not loaded from file).")

    # ------------------------------------------------------------------
    # Plot setup
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()

    ax.plot(omega, non_ea, label='Eq. (22)', linewidth=2)
    ax.plot(omega, ea, label='Eq. (23)', linewidth=2)

    # Analytical bound (Eq. 19)
    y_val = 0.5 + 1 / np.sqrt(2)
    ax.hlines(y=y_val, xmin=min(omega), xmax=max(omega),
              colors='green', linestyles='--', linewidth=2, label='Eq. (19)')

    ax.set_xlabel(r'$\omega$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=12, loc='best')

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if save:
        save_path = os.path.join(plots_dir, save_as)
        fig.savefig(save_path, dpi=1000, bbox_inches="tight")
        print(f"Figure saved as: {save_path}")
    else:
        print("Plot not saved (save=False).")

    plt.show()
