# EntInEnerConstrPM
The GitHub repository is organized as follows:

1. **`analytical_viol_EA_qubits.py`** – Computes the value of $I_\text{corr}$ using the analytical states derived in Appendix C, optimizing over the parameter $p$.  
2. **`seesaw_max_viol_EA_qutrit_analytic.py`** – Implements a seesaw SDP to optimize $I_\text{corr}$ for entanglement-assisted (EA) strategies with a qutrit memory. It compares the results with both the non-EA case and the analytical strategy using a qubit memory.  
3. **`seesaw_max_pguess_EA.py`** – Implements a seesaw SDP to compute the quantum guessing probability of an adversary using entangled strategies, and compares it with the case without EA.  
4. **`Viol_det_ineq.py`** – Implements a seesaw SDP to find deterministic behaviors when entanglement is allowed, and compares it with the case without EA.  
5. **`Channel_discr_advantage.py`** – Computes a lower bound on the advantage provided by entanglement without energy constraints in channel discrimination, as well as an upper bound for the advantage under energy constraints.  
6. **`utils.py`** – Contains all helper functions used across the other scripts.


The code is associated with the following paper on arXiv: https://arxiv.org/abs/2510.27559
