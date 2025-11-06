import numpy as np
import picos as pic
from math import ceil
import qutip as qt
from picos.modeling.problem import Problem
from picos.expressions.algebra import kron, partial_transpose
import matplotlib.pyplot as plt

def addNormalizationConstraints(prob,gammas):

    #p(gammaAX) in the diagonal of gammaAX
    norm=0
    for g in gammas:
        pGamma=g[0,0]
        prob.add_constraint(pGamma>=0)
        prob.add_constraint(g[1,1]==pGamma)
        prob.add_constraint(g[2,2]==pGamma)
        prob.add_constraint(g[3,3]==pGamma)
        norm=norm+pGamma

    #p(Gamma00)+p(gamma01)+p(gamma10)+p(gamma11)==1
    prob.add_constraint(norm==1)

def addEnergyConstraints(prob,gammas,w0Avg,w1Avg,w0Peak,w1Peak):

    w0s=[]
    w1s=[]
    for gamma in gammas:
        pGamma=gamma[0,0]

        w0=pic.RealVariable('w0'+str(gamma))
        w0s.append(w0)

        w1=pic.RealVariable('w1'+str(gamma))
        w1s.append(w1)

        prob.add_constraint(w0>=0)
        prob.add_constraint(w1>=0)
        prob.add_constraint(w0<=pGamma*w0Peak)
        prob.add_constraint(w1<=pGamma*w1Peak)
        prob.add_constraint(gamma[0,3]+pGamma<=2*w0)
        prob.add_constraint(gamma[1,3]+pGamma<=2*w1)

    suma=0
    for w in w0s:
        suma+=w
    prob.add_constraint(suma<=w0Avg)

    suma=0
    for w in w1s:
        suma+=w
    prob.add_constraint(suma<=w1Avg)

def computeGuessingProbability(w0Avg, w1Avg,w0Peak,w1Peak,expectedBehavior):
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
    
    #prob.add_constraint(E000+E010==E101+E111)
    #prob.add_constraint(E000+E010==E001+E011)
    #prob.add_constraint(E001>=-E101)
    #prob.add_constraint(-E010>=E110)
    #prob.add_constraint(-E011>=-E111)
    

    pg0=1/2*(1+E000-E001)
    pg1=1/2*(1+E110-E111)
    setNumericalPrecisionForSolver(prob)

    prob.set_objective('max',pg0)#h*(pg0+pg1))
    prob.solve(solver = "mosek", verbosity = 0)
    return prob.value


def oldComputeGuessingProbability(w0Avg,w1Avg,w0Peak,w1Peak,nonEABellValue):
    prob=pic.Problem()
    
    gamma0=pic.SymmetricVariable('gamma00',(4,4))
    gamma1=pic.SymmetricVariable('gamma01',(4,4))
    gammas=[gamma0,gamma1]

    for gamma in gammas:
        prob.add_constraint(gamma>>0)
    addNormalizationConstraints(prob,gammas)
    addEnergyConstraints(prob,gammas,w0Avg, w1Avg,w0Peak,w1Peak)

    #Eve's guessing probability
    E00,E10=gamma0[0,2],gamma0[1,2]
    E01,E11=gamma1[0,2],gamma1[1,2]

    E0=E00+E01
    E1=E10+E11
    p00=1/2*(1+E0)
    p10=1-p00
    p01=1/2*(1+E1)
    p11=1-p01
    
    tol=1e-6
    prob.add_constraint(p00-p10-p01+p11>=nonEABellValue-tol)
    
    pg0=1/2*(1+E00-E01)
    pg1=1/2*(1+E10-E11)
    #prob.set_objective('max',1/2*(pg0+pg1))
    prob.set_objective('max',pg0)
    
    
    #setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "mosek", verbosity=0)
    
    return prob.value

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
    prob.add_constraint(gamma[2,2] <= 1)
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
    #prob.add_constraint(f[0]*p00+f[1]*p10+f[2]*p01+f[3]*p11<=1.95*(np.sqrt(w0Avg*(1-w1Avg))+np.sqrt(w1Avg*(1-w0Avg))))
    prob.set_objective('max',(f[0]*p00+f[1]*p10+f[2]*p01+f[3]*p11))
    #setNumericalPrecisionForSolver(prob)
    prob.solve(solver = "mosek", verbosity = 0)
    #return prob.value
    return prob.value,[p00.value,p10.value,p01.value,p11.value]

def setNumericalPrecisionForSolver(problem):
    problem.options["rel_ipm_opt_tol"]=10**-8
    problem.options["rel_prim_fsb_tol"]=10**-8
    problem.options["rel_dual_fsb_tol"]=10**-8
    problem.options["max_footprints"]=None


#def initialMeasurementAndGroundState(dim,w0Avg):
#    #PiSA0=qt.rand_dm(dim**2,density=1,dims=[[dim,dim], [dim,dim]],pure=True).full()
#    PiSA0=qt.ket2dm(qt.rand_ket_haar(dim**2, dims=[[dim,dim], [1,1]])).full()
#    ground=(qt.ket2dm(np.sqrt(w0Avg)*qt.basis(2,0)+np.sqrt(1-w0Avg)*qt.basis(2,1))).full()
#    #ground=(qt.ket2dm(qt.rand_ket_haar(2))).full()
#    return [PiSA0,np.kron(np.eye(dim),np.eye(dim))-PiSA0],ground

def initialMeasurementAndGroundState(dim,dimM,w0Avg):
    #PiSA0=qt.rand_dm(dim**2,density=1,dims=[[dim,dim], [dim,dim]],pure=True).full()
    PiSA0=qt.ket2dm(qt.rand_ket_haar(dim*dimM, dims=[[dim,dimM], [1,1]])).full()
    ground=(qt.ket2dm(np.sqrt(w0Avg)*qt.basis(2,0)+np.sqrt(1-w0Avg)*qt.basis(2,1))).full()
    #ground=qt.ket2dm(qt.rand_ket_haar(dim, dims=[[dim], [1]])).full()
    return [PiSA0,np.kron(np.eye(dim),np.eye(dimM))-PiSA0],ground


def findStates(dim, dimM, w0, w1, measurement, ground,nonEABellValue):

    PiSA0 = measurement[0]
    PiSA1 = measurement[1]
    
    problem = Problem()

    states=[pic.HermitianVariable('phi_'+str(a),shape=(dim*dimM,dim*dimM)) for a in range(4)]
    problem.add_list_of_constraints([aState>>0 for aState in states])
    
    "Eve's outcome probability"
    p0=pic.RealVariable('p00')
    problem.add_constraint(p0>=0)
    problem.add_constraint(p0<=1)
    
    "States she prepares on guess 0"
    problem.add_constraint(pic.trace(states[0])==p0)
    problem.add_constraint(pic.trace(states[2])==p0)
    
    "States she prepares on guess 1"
    problem.add_constraint(pic.trace(states[1])==1-p0)
    problem.add_constraint(pic.trace(states[3])==1-p0)
    
    "Average states prepared for inputs X=0 and X=1"
    sigmaSA0=states[0]+states[1]
    sigmaSA1=states[2]+states[3]
    
    "Marginal's on A of the states sigma_{SA}^x should be equal"
    problem.add_constraint(pic.partial_trace(states[0],0,[dim,dimM])==pic.partial_trace(states[2],0,[dim,dimM]))
    problem.add_constraint(pic.partial_trace(states[1],0,[dim,dimM])==pic.partial_trace(states[3],0,[dim,dimM]))
    
    
    problem.add_constraint(pic.trace(sigmaSA0*PiSA0)-pic.trace(sigmaSA0*PiSA1)-pic.trace(sigmaSA1*PiSA0)+pic.trace(sigmaSA1*PiSA1)>=nonEABellValue)
    
    "Overlap with groundstate should be large"
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA0)>=1-w0)
    problem.add_constraint(pic.trace(np.kron(ground,np.eye(dimM))*sigmaSA1)>=1-w1)
    
    'Average pguess'
    #objective=1/2*(pic.trace(states[0]*PiSA0)+pic.trace(states[1]*PiSA1)+pic.trace(states[2]*PiSA0)+pic.trace(states[3]*PiSA1))
    
    'pguess for x=0'
    objective=pic.trace(states[0]*PiSA0)+pic.trace(states[1]*PiSA1)
    
    problem.set_objective("max",objective)
    #setNumericalPrecisionForSolver(problem)
    
    problem.solve(verbose=False)
    
    pguess=problem.value
    return pguess,[np.matrix(aState.value_as_matrix) for aState in states]

def findMeasurement(dim, dimM, w0, w1, states,nonEABellValue):
    
    problem = Problem()
    PiSA0=pic.HermitianVariable('PiSA0',shape=(dim*dimM,dim*dimM))
    PiSA1=pic.HermitianVariable('PiSA1',shape=(dim*dimM,dim*dimM))
    problem.add_constraint(PiSA0>>0)
    problem.add_constraint(PiSA1>>0)
    problem.add_constraint(PiSA0+PiSA1==np.eye(dim*dimM))
    
    sigmaSA0=states[0]+states[1]
    sigmaSA1=states[2]+states[3]
    
    
    problem.add_constraint(pic.trace(sigmaSA0*PiSA0)-pic.trace(sigmaSA0*PiSA1)-pic.trace(sigmaSA1*PiSA0)+pic.trace(sigmaSA1*PiSA1)==nonEABellValue)
   
   
    'Average pguess'
    #objective=1/2*(pic.trace(states[0]*PiSA0)+pic.trace(states[1]*PiSA1)+pic.trace(states[2]*PiSA0)+pic.trace(states[3]*PiSA1))
    
    'pguess for x=0'
    objective=pic.trace(states[0]*PiSA0)+pic.trace(states[1]*PiSA1)
    
    problem.set_objective("max",objective)
    #setNumericalPrecisionForSolver(problem)
    
    problem.solve()
    
    pguess=problem.value
    
    firstEffect = np.matrix(PiSA0.value_as_matrix)
    return pguess,[firstEffect,np.eye(dim*dimM)-firstEffect]

                            
def extractBehavior(states, measurement):
    sigmaSA0=states[0]+states[1]
    sigmaSA1=states[2]+states[3]
    p00=np.trace(measurement[0]@sigmaSA0)
    p10=np.trace(measurement[1]@sigmaSA0)
    p01=np.trace(measurement[0]@sigmaSA1)
    p11=np.trace(measurement[1]@sigmaSA1)
    return [p00.real,p10.real,p01.real,p11.real]

def optimize_with_restarts(dim, dimM, w0Avg, w1Avg, nonEABellValue, n_restarts=5):
    best_pguess = -np.inf
    best_states = None
    best_measurement = None

    for _ in range(n_restarts):
        measurement, ground = initialMeasurementAndGroundState(dim, dimM, w0Avg)
        newPguess = 1
        firstPguess = 0
        iter_count = 0
        max_iter = 10
        error = 1e-7
        
        try:
            while newPguess - firstPguess > error and iter_count < max_iter:
                firstPguess, states = findStates(dim, dimM, w0Avg, w1Avg, measurement, ground, nonEABellValue)
                newPguess, measurement = findMeasurement(dim, dimM, w0Avg, w1Avg, states, nonEABellValue)
                iter_count += 1
            
            if newPguess > best_pguess:
                best_pguess = newPguess
                best_states = states
                best_measurement = measurement
        except:
            pass

    return best_pguess, best_states, best_measurement


if __name__ == '__main__':
    fileName='data.txt'
    with open(fileName, 'w') as the_file:
        energyrange = np.arange(0.25,0.34,0.01)
        for w in energyrange:
            print(w)
            # Here we take the same energy for both inputs,
            # but they can be different
            w0Avg, w1Avg = w, w
    
            ineq=[1,-1,-1,1]
            nonEABellValue,behavior=computeMaxIneqViolation(w0Avg,w1Avg,ineq)
            cPguess=oldComputeGuessingProbability(w0Avg,w1Avg,1,1,nonEABellValue)
            
            dim = 2
            dimM = 3
            
            qPguess = 0
            n_restarts = 10
            for _ in range(n_restarts):
                measurement, ground = initialMeasurementAndGroundState(dim, dimM, w0Avg)
                newPguess = 0
                oldPguess = 1e-6
                while abs(newPguess - oldPguess) > 1e-7:
                    oldPguess = newPguess
                    oldPguess, states = findStates(dim, dimM, w0Avg, w1Avg, measurement, ground, nonEABellValue)
                    newPguess, measurement = findMeasurement(dim, dimM, w0Avg, w1Avg, states, nonEABellValue)
                qPguess = max(qPguess, newPguess)

            print((w0Avg,cPguess,qPguess),file=the_file)
    
            #res.append((w0Avg,cPguess,qPguess))
    
    res = []
    with open(fileName, 'r') as the_file:
        res=[]
        for line in the_file:
            item = line.strip('()\n').split(',')
            res.append((float(item[0]),float(item[1]),float(item[2])))

    fig, ax = plt.subplots()
    
    ax.plot([w for (w,_,_) in res], [-np.log2(cPguess) for (_,cPguess,_) in res])
    ax.plot([w for (w,_,_) in res], [-np.log2(qPguess) for (_,_,qPguess) in res])
    ax.legend( [r'classically-correlated randomness [2]',r'entanglement-assisted randomness'], fontsize = 12, loc = 'upper right')
    
    #ax.set(xlabel=r'$\omega$', ylabel=r'randomness')
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'randomness', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)  # Make both x and y ticks larger
    #ax.legend( ['Eq.(4)'], fontsize = 14, loc = 'lower right')
    ax.grid()
    fig.savefig('pguesses2.png')

    plt.show()
        
        # print('-----------------------------------------------------')
        # print('Classical pguess =',cPguess)
        # print('Ineq viol:',sum([behavior[i]*ineq[i] for i in range(4)])-2*(w0Avg+w1Avg))
        # print('Entanglement-assisted pguess >=',qPguess)
        # print('Energy bound:',w0Avg)
        # print('Behavior giving separation:',behavior)
