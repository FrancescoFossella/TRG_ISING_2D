from code.TRG import TRG
import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad


def U_exact(beta):

    def g(phi):
        x=2*np.sinh(2*beta)/((np.cosh(2*beta) / np.sinh(2*beta))**2)
        return np.log((1+np.sqrt(1-x**2*np.sin(phi)**2))/2.0)

    integral, error = quad(g,0,np.pi)
    U=(np.log(2*np.cosh(2*beta))+integral/(2*np.pi))

    return U 


(N ,truncate)= (3, 6)

beta_values=np.arange(0.00000,2,0.01)
(free_energy_true, free_energy_TRG) = ([],[])

for beta in beta_values : 
    trg = TRG(N, beta, truncate)  
    Z = trg.solve()
    free_energy_TRG.append(-np.log(Z)/beta)
    free_energy_true.append(U_exact(beta))

plt.plot(beta_values,free_energy_true,label="Onsager")
#plt.plot(beta_values,free_energy_TRG,label="TRG")
# Add labels and a legend
plt.xlabel(r'$\beta J$')
plt.ylabel(r'$-\beta U$')
#plt.xlim(0.1,1.8)
#plt.ylim(-10,0.5)
plt.legend()
plt.subplots_adjust(wspace=0.25, hspace=0.15, left=0.1, right=0.98, top=0.95, bottom=0.1)  
plt.savefig("energy.pdf",dpi=4000)