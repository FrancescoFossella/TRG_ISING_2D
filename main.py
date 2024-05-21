from code.TRG import TRG
import matplotlib.pyplot as plt 
import numpy as np


(N ,truncate)= (10, 6)

beta_values=np.arange(0.01,2,0.001)
free_energy=[]
for beta in beta_values : 
    trg = TRG(N, beta, truncate)  
    Z = trg.solve()
    free_energy.append(np.log(Z))

plt.plot(beta_values,free_energy)
# Add labels and a legend
plt.xlabel(r'$\beta J$')
plt.ylabel(r'$-\beta U$')
plt.title(r'Internal energy density $-\beta \, U$')
plt.legend()
plt.subplots_adjust(wspace=0.25, hspace=0.15, left=0.07, right=0.98, top=0.95, bottom=0.1)  
plt.savefig("energy.pdf",dpi=4000)