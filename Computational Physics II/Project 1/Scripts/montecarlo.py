#!/usr/bin/env python

# packes
import numpy as np
from VMCM.Algorithm.Metroplis import Metropolis 

from matplotlib import pyplot as plt

from VMCM.Hamiltonian.ExactSolution import ExactSolution
from VMCM.utils.PathSaveOutput import SaveOutput


print('--------------------------------------------------------------------------------')
a = int(input('Number of Particles = '))
b = int(input('Dimension = '))
c = int(input('Number of Monte Carlo cycles = '))
print('--------------------------------------------------------------------------------')
print('working ...')


# Here starts the main program with variable declarations
Number_particles = a
Dimension = b
Variations_alfa = 20
Alpha_start = 0.1
StespAlpha = 0.05
Number_MC_cycles = c

# Save all variations 
metropolis = Metropolis() # instantiation class Metropolis
metropolis = metropolis.metropolis_algorithm


# Save infomation
Alpha_values = np.zeros(Variations_alfa)
Energies_analytic = np.zeros(Variations_alfa)
Variances_anayltic = np.zeros(Variations_alfa)
errors_anayltic = np.zeros(Variations_alfa)
Time_consumings_anayltic_alpha = np.zeros(Variations_alfa)

Energies_numeric = np.zeros(Variations_alfa)
Variances_numeric = np.zeros(Variations_alfa)
errors_numeric = np.zeros(Variations_alfa)
Time_consumings_numeric_alpha= np.zeros(Variations_alfa)

# Start variational parameter
alpha = Alpha_start
for ia in range(Variations_alfa):
    alpha += StespAlpha
    Alpha_values[ia] = alpha
    Energies_analytic[ia],Variances_anayltic[ia],errors_anayltic[ia],Time_consumings_anayltic_alpha[ia] = metropolis(alpha, Number_particles,Dimension, Number_MC_cycles, Step_size_jumping = 1.0,Type_calculations='analytic')
    Energies_numeric[ia],Variances_numeric[ia],errors_numeric[ia],Time_consumings_numeric_alpha[ia] = metropolis(alpha, Number_particles,Dimension, Number_MC_cycles, Step_size_jumping = 1.0,Type_calculations='numeric')  
    
#------------------------------------------------------------------------------------------    
#------------------------------------------------------------------------------------------

Ect = ExactSolution() # Instantiation exact solution
save = SaveOutput('Result-comparison-ext-anly-num') # Instantiation SaveOutput solution

# Exact energy
Exact_energies = Ect.exact_energy_ho_no_interact(Number_particles,Dimension,Alpha_values) 
Exact_variance = Ect.exact_variance_ho_no_interact(Number_particles,Dimension,Alpha_values) 

# Simple subplot
plt1 = plt
fig1 = plt1.figure()
plt1.plot(Alpha_values, Energies_numeric, 'kx--', label='MCMN')
plt1.plot(Alpha_values, Energies_analytic, 'bo--', label='MCMA')
plt1.plot(Alpha_values, Exact_energies,'r*-',label='Exact')
#plt1.title('Energy')
plt1.ylabel('Dimensionless energy')
plt1.xlabel(r'$\alpha$', fontsize=15)
plt1.legend(loc='upper right')
plt1.close(fig1)
fig1.savefig('energy.png', dpi=300)


plt2 = plt
fig2 = plt2.figure()
plt2.plot(Alpha_values, Variances_numeric, 'kx--',label='MCMN')
plt2.plot(Alpha_values, Variances_anayltic, 'bo--',label='MCMA')
plt2.plot(Alpha_values, Exact_variance,'r*-',label='Exact')
#plt1.title('Variance')
plt2.ylabel('Variance')
plt2.xlabel(r'$\alpha$', fontsize=15)
plt2.legend(loc='upper right')
#plt.show()
plt2.close(fig2)
fig2.savefig('variance.png', dpi=300)

print('--------------------------------------------------------------------------------')
print('Enjoy your outcomes')
print('--------------------------------------------------------------------------------')
