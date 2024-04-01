# Packes
import numpy as np
from math import exp, sqrt 
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from VMCM.utils import Type

# Quantum state
'''
===============================================================================
Define the quantum state or just a trial object wave funtion as 'funtion' 
of the  particles position and varational parameter alpha

consideraction
- Positions always has to be a matrix [Mumber particles x Dimension]
- The alpha parameter can be jus a scalar o vector could be depend of the how
  mant parameter you going to need to do varaitional Monte Carlos method
===============================================================================
'''
class State:

    def __init__(self) -> None:
        pass

    '''
    ===============================================================================
    Define the wave funtion, it is a funtion of position and alpha parameters

    r : Matrix[Number_particles, Dimension]

    alpha  : Can be a escalar or a vector when you have more that one variational 
    parameter
    ===============================================================================
    '''
    # Wave trial wave funtion Harmonic oscilator
    def wave_funtion(self,
        # Variables
        r: Type.Matrix ,            # Positions
        alpha: Type.Vector,         # Parameter
        ) -> Type.Float:
        
        r_sum = np.sum(r**2)        # Squart sume of all particles

        return exp(-alpha*r_sum)
    
    '''
    ===============================================================================
    Define the natural logarithm (ln) of the wave funtion, it is a funtion of 
    position and alpha parameters

    r : Matrix[Number_particles, Dimension]

    alpha  : Can be a escalar or a vector when you have more that one variational 
    parameter
    ===============================================================================
    '''
    # Define the ln of the trial wave funtion 
    def ln_wave_funtion(self,
        r : Type.Matrix,                            # Positions
        alpha : Type.Vector | Type.Float,           # Parameter
        ) -> Type.Float:
    
        r_sum = jnp.sum(r**2)

        return -alpha*r_sum
    
if __name__ == "__main__":
    print('PROGRAM RUNNING IN THE CURRENT FILE')