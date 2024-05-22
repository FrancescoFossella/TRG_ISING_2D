## Importing Block
from code.TRG import TRG
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib import gridspec
from code.physical_analysis import analysis_H, analysis_U, compute_H_exact,compute_Z_num
from code.physical_analysis import compute_U_exact, analysis_U


############################################
N = 10
delta_beta=0.01
beta_values = np.arange(0.1,2,delta_beta)
truncation = 0.5
############################################




#### Plotting
(fontsize,labelsize,legendsize,axes_size,titlesize) = (25,25,25,25,25) 

fig=plt.figure(figsize=(6*3, 4*3))
gs = gridspec.GridSpec(2,4, width_ratios=[1,1,1,1]) #il secondo numero si riferisce al numero di colonne


# first plot with Helmholtz free energy  
ax1 = fig.add_subplot(gs[0,0:2])
plt.plot(beta_values,compute_H_exact(beta_values),label="Onsager",lw=1.1)
plt.plot(beta_values,np.log(compute_Z_num(N, beta_values, truncation)),label="TRG",lw=1.1)
plt.xlabel(r'$\beta$',fontsize=axes_size)
plt.ylabel(r'$-\beta H(\beta)=\partial_\beta(Z^{1/N})$',fontsize=axes_size)
plt.title('Helmholtz Free Energy',fontsize=titlesize)
plt.grid()
plt.legend(loc='best', fontsize=legendsize)


# second plot with internal energy per site
internal_energy=analysis_U(N, beta_values, truncation,delta_beta)
ax1 = fig.add_subplot(gs[0,2:])
plt.plot(beta_values,internal_energy[0],label="Onsager",lw=1.1)
plt.plot(beta_values,internal_energy[1],label="TRG",lw=1.1)
plt.xlabel(r'$\beta$',fontsize=axes_size)
plt.ylabel(r'$U/N=-\partial_\beta log(Z^{1/N})$',fontsize=axes_size)
plt.title('Internal Energy per site',fontsize=titlesize) 
plt.grid()
plt.legend(loc='best', fontsize=legendsize)


# third plot for heat capacity 
internal_energy=analysis_U(N, beta_values, truncation,delta_beta)
ax1 = fig.add_subplot(gs[1,1:3])
plt.plot(beta_values,internal_energy[0],label="Onsager",lw=1.1)
plt.plot(beta_values,internal_energy[1],label="TRG",lw=1.1)
plt.xlabel(r'$\beta$',fontsize=axes_size)
plt.ylabel(r'$U/N=-\partial_\beta log(Z^{1/N})$',fontsize=axes_size)
plt.title('Internal Energy per site',fontsize=titlesize) 
plt.grid()
plt.legend(loc='best', fontsize=legendsize)








##### save the picture
plt.subplots_adjust(wspace=0.40, hspace=0.15, left=0.07, right=0.98, top=0.95, bottom=0.05)  
plt.savefig("picture.pdf",dpi=4000)