import numpy as np
import picos as pic
import qutip as qt
from picos.modeling.problem import Problem
import matplotlib.pyplot as plt

def computeMaxIneqViolation(w1,w2,f):
    prob=pic.Problem()

    gamma=pic.SymmetricVariable('gamma',(4,4))
    eta1=pic.RealVariable('eta1')
    eta2=pic.RealVariable('eta2')
    prob.add_constraint(gamma>>0)
    prob.add_constraint(eta1<=w1)
    prob.add_constraint(eta2<=w2)
    prob.add_constraint(gamma[0,0] == 1)
    prob.add_constraint(gamma[1,1] == 1)
    prob.add_constraint(gamma[2,2] == 1)
    prob.add_constraint(gamma[3,3] == 1)
    prob.add_constraint(gamma[0,3]==2*eta1-1)
    prob.add_constraint(gamma[1,3]==2*eta2-1)
    E0,E1=gamma[0,2],gamma[1,2]
    p00=1/2*(1+E0)
    p10=1-p00
    p01=1/2*(1+E1)
    p11=1-p01
    prob.add_constraint(p00>=0)
    prob.add_constraint(p10>=0)
    prob.add_constraint(p01>=0)
    prob.add_constraint(p11>=0)
    prob.set_objective('max',(f[0]*p00+f[1]*p10+f[2]*p01+f[3]*p11))
    setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "mosek", verbosity = 0)
    #return prob.value
    return prob.value,[p00.value,p10.value,p01.value,p11.value]

def setNumericalPrecisionForSolver(problem):
    problem.options["rel_ipm_opt_tol"]=10**-12
    problem.options["rel_prim_fsb_tol"]=10**-12
    problem.options["rel_dual_fsb_tol"]=10**-12
    problem.options["max_footprints"]=None


def findMeasurementMaxViol(dim, dimM, states):
    
    problem = Problem()
    PiSA0=pic.HermitianVariable('PiSA0',shape=(dim*dimM,dim*dimM))
    PiSA1=pic.HermitianVariable('PiSA1',shape=(dim*dimM,dim*dimM))
    problem.add_constraint(PiSA0>>0)
    problem.add_constraint(PiSA1>>0)
    problem.add_constraint(PiSA0+PiSA1==np.eye(dim*dimM))

    sigmaSA0=states[0]
    sigmaSA1=states[1]
    
    objective=pic.trace(sigmaSA0*PiSA0)-pic.trace(sigmaSA0*PiSA1)-pic.trace(sigmaSA1*PiSA0)+pic.trace(sigmaSA1*PiSA1)

    
    problem.set_objective("max",objective)
    setNumericalPrecisionForSolver(problem)
    
    problem.solve()
    
    firstEffect = np.matrix(PiSA0.value_as_matrix)
    measurement = [firstEffect,np.eye(dim*dimM)-firstEffect]
    behavior = [np.trace(sigmaSA0@measurement[0]),\
                     np.trace(sigmaSA0@measurement[1]),\
                              np.trace(sigmaSA1@measurement[0]),\
                                       np.trace(sigmaSA1@measurement[1])]
  
    
    return behavior,measurement

def initialMeasurementAndGroundState(dim,dimM,w0Avg):
    #PiSA0=qt.rand_dm(dim**2,density=1,dims=[[dim,dim], [dim,dim]],pure=True).full()
    PiSA0=qt.ket2dm(qt.rand_ket_haar(dim*dimM, dims=[[dim,dimM], [1,1]])).full()
    #ground=(qt.ket2dm(np.sqrt(w0Avg)*qt.basis(2,0)+np.sqrt(1-w0Avg)*qt.basis(2,1))).full()
    ground=qt.ket2dm(qt.rand_ket_haar(dim, dims=[[dim], [1]])).full()
    return [PiSA0,np.kron(np.eye(dim),np.eye(dimM))-PiSA0],ground

def findStateMaxViolation(dim,dimM,w0,w1,measurement,ground):
    
    PiSA0 = measurement[0]
    PiSA1 = measurement[1]
    
    problem = Problem()

    states=[pic.HermitianVariable('phi_'+str(a),shape=(dim*dimM,dim*dimM)) for a in range(2)]
    problem.add_list_of_constraints([aState>>0 for aState in states])
    problem.add_list_of_constraints([pic.trace(aState)==1 for aState in states])
    
    sigmaSA0=states[0]
    sigmaSA1=states[1]
    
    "Marginal's on A of the states sigma_{SA}^x should be equal"
    problem.add_constraint(pic.partial_trace(sigmaSA0,0,[dim,dimM])==pic.partial_trace(sigmaSA1,0,[dim,dimM]))
    
    "Overlap with groundstate should be large"
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA0)>=1-w0)
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA1)>=1-w1)

    objective=pic.trace(sigmaSA0*PiSA0)-pic.trace(sigmaSA0*PiSA1)-pic.trace(sigmaSA1*PiSA0)+pic.trace(sigmaSA1*PiSA1)
    
    problem.set_objective("max",objective)
    setNumericalPrecisionForSolver(problem)
    
    problem.solve(verbose=False)
    
    behavior = [np.trace(np.matrix(sigmaSA0.value_as_matrix)@PiSA0),\
                     np.trace(np.matrix(sigmaSA0.value_as_matrix)@PiSA1),\
                              np.trace(np.matrix(sigmaSA1.value_as_matrix)@PiSA0),\
                                       np.trace(np.matrix(sigmaSA1.value_as_matrix)@PiSA1)]
    return [p.real for p in behavior],[np.matrix(aState.value_as_matrix) for aState in states]
                               
if __name__ == "__main__":
    """
    Main execution block.

    Performs a seesaw optimization to find the maximum Bell-inequality violation
    (both energy-assisted and non-energy-assisted) for a range of energy parameters w.

    The results are stored as tuples (w, EA_value, nonEA_value)
    in the list `results`, which can be saved or plotted later.

    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    
    energyrange = np.arange(0.01, 0.51, 0.01)
    dim, dimM = 2, 3
    max_restarts = 20        # number of random restarts
    tol = 1e-7               # convergence threshold
    ineq = [1, -1, -1, 1]    # coefficients for the target inequality


    results = []


    # ------------------------------------------------------------------
    # Main loop over energy values
    # ------------------------------------------------------------------

    for w in energyrange:
        print(f"⟶ Energy parameter: w = {w:.2f}")

        w0Avg, w1Avg = w, w
        success = False

        # Try multiple random initializations until one converges
        for attempt in range(max_restarts):
            try:
                # Initialize measurement operators and ground state
                measurement, ground = initialMeasurementAndGroundState(dim, dimM, w0Avg)
                ground = [[1, 0], [0, 0]]  # reset ground state explicitly

                # First step: optimize over states given initial measurement
                behavior, states = findStateMaxViolation(dim, dimM, w0Avg, w1Avg, measurement, ground)

                # Compute the initial Bell inequality value
                newEAvalue = sum(behavior[i] * ineq[i] for i in range(4))
                oldEAvalue = 0

                # ------------------------------------------------------------------
                # Seesaw loop: alternate between optimizing measurement and states
                # ------------------------------------------------------------------
                while newEAvalue - oldEAvalue > tol:
                    behavior, measurement = findMeasurementMaxViol(dim, dimM, states)
                    behavior, states = findStateMaxViolation(dim, dimM, w0Avg, w1Avg, measurement, ground)
                    oldEAvalue = newEAvalue
                    newEAvalue = sum(behavior[i] * ineq[i] for i in range(4))

                success = True
                break  # exit retry loop upon success

            except pic.SolutionFailure as e:
                print(f"  Solver failed on attempt {attempt + 1}: {e}")
                continue  # retry with a new random initialization

        if not success:
            raise RuntimeError(f"Seesaw optimization failed for w = {w:.2f} after {max_restarts} restarts.")

        # ------------------------------------------------------------------
        # Compute non-energy-assisted (classical) value for comparison
        # ------------------------------------------------------------------
        nonEAvalue, _ = computeMaxIneqViolation(w0Avg, w1Avg, ineq)

        # Store results
        results.append((w0Avg, newEAvalue, nonEAvalue))
        print(f"   ✅ Success: EA = {newEAvalue:.6f}, non-EA = {nonEAvalue:.6f}\n")

    # ------------------------------------------------------------------
    # (Optional) Save results to file for later plotting
    # ------------------------------------------------------------------
    np.savetxt(
        "energy_assisted_results.csv",
        results,
        delimiter=",",
        header="w,EA_value,nonEA_value",
        comments="",
        fmt="%.8f"
    )

    print("All runs completed successfully.")
    print("Results saved to 'energy_assisted_results.csv'.")

    
    #fig, ax = plt.subplots()
    #ea=ax.plot([w for (w,_,_) in results], [EAvalue for (_,EAvalue,nonEAvalue) in results],label='EA')
    #non_ea=ax.plot([w for (w,_,_) in results], [nonEAvalue for (_,EAvalue,nonEAvalue) in results],label='non-EA')
    #ax.legend( ['EA', 'non-EA'], fontsize = 14, loc = 'lower right')
    #ax.grid()
    
    #plt.show()
    
    fig, ax = plt.subplots()
    max_viol_analytic = [0.39840575308375803, 0.5610721862123891, 0.6842006621693072, 0.7865421733836533, 0.8753885443690308, 0.9544929985254028, 1.026092608550252, 1.0916493386099253, 1.1521814807120587, 1.2084322383153452, 1.2609637678647776, 1.3102133902716437, 1.356529305407324, 1.4001939663691745, 1.4414400878904914, 1.480461955421329, 1.5174235569008163, 1.5524647888462924, 1.5857058629292466, 1.6172507941205472, 1.6471902009886958, 1.675603311442658, 1.702559754465343, 1.728121024090647, 1.752341274969706, 1.7752685661304888, 1.796945385787492, 1.8174093185494065, 1.836693644615321, 1.8548275238894891, 1.8718365713351464, 1.8877430448216408, 1.9025659766954834, 1.9163214690857266, 1.9290226805557558, 1.9406799692516763, 1.9513009131916994, 1.9608902329557305, 1.9694496232302088, 1.976977708670437, 1.9834696260643936, 1.9889165870670744, 1.9933053535563499, 1.9966172219554312, 1.9988266741740572, 1.9998992288199449, 1.999999999810664, 1.9999999999769713, 1.9999999998434954, 2.0]
    non_ea=ax.plot([w for (w,_,_) in results], [nonEAvalue for (_,EAvalue,nonEAvalue) in results],label='non-EA')
    ea=ax.plot([w for (w,_,_) in results], [EAvalue for (_,EAvalue,nonEAvalue) in results],label='EA')
    ea_analytic = ax.plot([w for (w,_,_) in results], [trace_norm for trace_norm in max_viol_analytic],label='EA_analytic')
    ax.legend( [r'classically-correlated $ I_\text{corr}$',r'EA qubit(S)-qutrit(M) $ I_\text{corr}$',r'EA qubit(S)-qubit(M) (analytic) $ I_\text{corr}$'], fontsize = 11, loc = 'lower right')
    #ax.plot([w for (w,_,_) in results], [value for value in fidelity],label='Fidelity')
    #ax.plot(x, y_fit, label=f'Fit: y = {m:.2f}x + {c:.2f}', color='red')
    #ax.legend( ['EA', 'non-EA'], fontsize = 14, loc = 'lower right')
    ax.grid()
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'$\rm max$ $I_{\text{corr}} $', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)  # Make both x and y ticks larger
    plt.show()
    fig.savefig('EAcorrelations1.png', dpi=1000, bbox_inches='tight')  # high-res export
    
    # Extract the data from your lists
    omega = np.array([w for (w, _, _) in results])
    non_ea_values = np.array([nonEAvalue for (_, _, nonEAvalue) in results])
    ea_values = np.array([EAvalue for (_, EAvalue, _) in results])
    ea_analytic_values = np.array(max_viol_analytic)  # already corresponds to omega
    
    # Stack them as columns
    data_to_save = np.column_stack((omega, non_ea_values, ea_values, ea_analytic_values))
    
    # Save to CSV
    np.savetxt("EA_plot_data.csv", data_to_save, delimiter=",", 
               header="omega,non_ea,ea,ea_analytic", comments="")
    
    print("Data saved to EA_plot_data.csv")

    