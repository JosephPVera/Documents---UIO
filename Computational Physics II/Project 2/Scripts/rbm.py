#!/usr/bin/env python3

# Packages
import numpy as np

from RBM.optimizator.energy_optimizator import Optimizator
from RBM.samplings import Samplings

print('================================================================================')
print('                               Information                                      ')
print('- Algorithm:')
print('  1. Metropolis (Force brute)')
print('  2. MetropolisHastings (Importance sampling)')
print('- Interaction:')
print('  1. False')
print('  2. True')
print('================================================================================')
a = int(input('Number of Particles = '))
b = int(input('Dimension = '))
c = int(input('Number of Monte Carlo cycles = '))
d = int(input('Number hidden layer = '))
e = int(input('Algorithm = '))
f = int(input('Interaction = '))
g = int(input('Iterations (Gradient Descent) = '))
h = float(input('Learning rate = '))
print('================================================================================')
print('working ...')



# Here starts the main program with variable declarations

# Number particle 
Number_particles = a

# Dimension
Dimension = b

# Number Monte Carlos cycles
Number_MC_cycles = c

# Number hidden layer
Number_hidden_layer = d

# Type algorithm
if e == 1:
   Algorithm = 'Metropolis'  # Force brute
elif e == 2:  
   Algorithm = 'MetropolisHastings' # Importance sampling
else:
  print('Incorrect')
  
# Interaction
if f == 1:
   Interaction = False
elif f == 2:
   Interaction = True
else:
  print('Incorrect') 

# Instantce class Optimizator
algorithm_1= Optimizator(Number_particles, Dimension, Number_hidden_layer, Interaction, Algorithm = Algorithm, Number_MC_cycles = Number_MC_cycles) 

# Instantce class Samplings
algorithms = Samplings(Number_particles,Dimension,Number_hidden_layer,Interaction,Algorithm= Algorithm,Number_MC_cycles= Number_MC_cycles)


import time
inicio = time.time()
##############################################################################################################

# Packages 
from RBM.utils.path_save_output import SaveOutput

# Name where will be saving outputs  (file or figure)
save = SaveOutput(f'Result_varying_learning_rate_{Algorithm}_interaction_{Interaction}') 

# Decide to save output  
SAVE_OUTPUT = False

# Maximum the iterations (The best option it is a power 2 if you would like to do the stadistic analysis with blocking)
Maximum_iterations = g

# Number samplings (The best option it is a power 2 if you would like to do the stadistic analysis with blocking)
Number_samplings = 2**10

# Number core (take care how many core has your computer)
Number_core = 4

# Different trying with learning rate 
Low = h
High = 0.8
Number_trying_learning_rate = 1
learning_rate_range = np.linspace(Low,High,Number_trying_learning_rate)

# Set up the number decimal 
learning_rate_range = np.around(learning_rate_range,3)

# Saves output 
Energies_trying_learning_rate = []
Optimal_parameter_a_learning_rate =[]
Optimal_parameter_b_learning_rate =[]
Optimal_parameter_w_learning_rate =[]

# Save the output (energy) for each resampling
Energies_samplings_save = []

# Trying the diffferent learning rate 
for  learning_rate in learning_rate_range:

    Energies, Optimal_parameter_a, Optimal_parameter_b, Optimal_parameter_w, Iteration_number, Time_CPU = algorithm_1.gradient_descent(learning_rate, Maximum_iterations)

    # Save all output
    Energies_trying_learning_rate.append(Energies) 
    Optimal_parameter_a_learning_rate.append(Optimal_parameter_a)
    Optimal_parameter_b_learning_rate.append(Optimal_parameter_b)
    Optimal_parameter_w_learning_rate.append(Optimal_parameter_w)
    
    # Calculate the sampling after finding the optimal parameter 
    Optimal_parameter_a_b_w = (Optimal_parameter_a,Optimal_parameter_b,Optimal_parameter_w) 
    samplings = algorithms.samplings(Optimal_parameter_a_b_w,Number_samplings,Number_core)

    # Save all output
    Energies_sampling, Variances_sampling, Errors_sampling, Time_CPU_sam = samplings
    Energies_samplings_save.append(Energies_sampling)


import pandas as pd
# Saves or does not save the output (format .dat one for each learning rate)
if SAVE_OUTPUT == False :
    data_varing_learning_rate = {}
    data_samplings = {} 
    for i in range(len(Energies_trying_learning_rate)):
        data_varing_learning_rate[f'#Energies_wirh_learning_rate_{learning_rate_range[i]}'] = Energies_trying_learning_rate[i]
        data_samplings[f'#Energies wirh learning rate {learning_rate_range[i]}'] = Energies_samplings_save[i]

# Nice panda viw of the data
    Nice_panda_view_learning_rate = pd.DataFrame(data_varing_learning_rate)

    Nice_panda_view_samplings = pd.DataFrame(data_samplings)


# Decide to save output  
    SAVE_OUTPUT = True

# Saves or does not save the output
    if SAVE_OUTPUT == True:
    # Save a external file in .csv format using panda
        Nice_panda_view_learning_rate.to_csv(save.data_path(f'energies_vs_learning_rate_{Number_particles}p_{Dimension}d_{Number_hidden_layer}h.dat'),sep=' ', index=True)
    Nice_panda_view_learning_rate
        #   Nice_panda_view_samplings.to_csv(save_sampling.data_path(f'energies_samplings_{Number_particles}p_{Dimension}d_{Number_hidden_layer}h.dat'), index=True)

# Read the external .csv data format with pandas  
#save_output = pd.read_csv(save.data_path(f'energies_vs_learning_rate_{Number_particles}p_{Dimension}d_{Number_hidden_layer}h.dat'))

# Choose one of the data 
#Energies_learninf_rate_1 = save_output[f'Energies wirh learning rate: {learning_rate_range[i]}']

# Transform to numpy class
#save_output_numpy = np.array(save_output)

#print(save_output_numpy[:,0])

# Print nice view
     
        
#############################################################################################################################################
fin = time.time()
a = fin-inicio
#print("CPU time =",fin-inicio, "seconds")
#print("CPU time =",(fin-inicio)/60, "minutes")
#print("CPU time =",(fin-inicio)/3600, "hours")
print("===============================================")
import pandas as pd
from pandas import DataFrame
data1 ={'seconds':[a], 'minutes':[a/60], 'hours':[a/3600]}
frame1 = pd.DataFrame(data1)
frame1.to_csv("CPU-time.dat", sep=' ', index=False)  ### change ########
print("CPU time:")
print(frame1)
print("===============================================")



# Statistic analysis with blocking
from RBM.statistical_techniques.statistics import StatisticalTechniques

# Using blocking 
blocking_techniques = StatisticalTechniques().blocking

# Using bootstrap
bootstrap_techniques = StatisticalTechniques().bootstrap

#for i in range(len(Energies_samplings_save)):
#    print(f'learning rate {learning_rate_range[i]}')
#    print('===============================================')
#    blocking_techniques(Energies_samplings_save[i])
#    print('===============================================')
#    print()
    
import pandas as pd
from pandas import DataFrame
print('===============================================')
data ={'Mean, variance, error':blocking_techniques(Energies_samplings_save[i])}
frame = pd.DataFrame(data)
frame.to_csv("statistical.dat", sep=' ', index=False)  ### change ########
print('===============================================')
#print(frame)


print('================================================================================')
print('                               Enjoy your outcomes                              ')
print('================================================================================')
