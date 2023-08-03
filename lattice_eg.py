import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from square_lattice.sorting_model import CellSorting
import time

##Set the temperature time-series. This could be linear or exponentially decreasing if you wish for annealing
T_t = np.repeat(0.5 ,10**5)

#Set the number of cells of each type. The code is set up for three cell types
N_E,N_T,N_X,scaler = 6,6,0,3

##Let division yes or no
division = False

#Set the random configuration type:
# - "circle", generating an approx circle avoiding gaps
# - "random", randomly choose points in the grid
# - "random_circle", generating a circle, not considering gaps
seed_type = "circle"

##Set the interface energy between the cells and the "white space"
medium_energy=30

##Set the grid size
num_x,num_y = 30,30

#Set the affinity matrix. This should be symmetric.
# Wij = the energy associated with the bond between cell type i and j
W0 = np.array([[18, 10, 7],
               [10, 18, 7],
               [7, 7, 2]])

##Initialise the class
srt = CellSorting()

#Make the lattice
srt.generate_lattice(num_x=num_x,num_y=num_y)

#Set the boundary. By default, it prevents cells from crawling out of the plane.
#You can prescribe the 'well' in which cells self-organise, by prescribing a binary mask of shape (num_x,num_y)
## Here, srt.boundary_definition(well=well)
srt.boundary_definition()

##Prescribe the running parameters. This sets the (potentially varying) 'temperature'.
# The shape of the vector defines the number of time points
srt.define_running_params(T_t=T_t)

#Set up the initial state of the system, populating the grid with cells
#N_E, N_T, N_X are the number of cells of each type
#scaler multiplies the above numbers by some pre-factor
#Note that these are approximate and not exact final numbers, depending on whether the N_E etc is an int (I think)
srt.make_C0(N_E,N_T,N_X,scaler,seed_type=seed_type)

#Set up the division clock. If division is true, then a simple division rule applies
#Cells have some normally distributed. I can dig into the code, but presume you don't want this
srt.make_ID0(division=division)

#Set the number of time-points to record the state. Distributed evenly throughout the simulation
srt.define_saving_params(n_save=200)

##Prescribe the interaction energies, specified in the pre-text
srt.define_interaction_energies(W0,medium_energy)

##Gets some internal cogs running to do book-keeping for cell interactions
srt.initialise_simulation()

#Do the simulation
srt.perform_simulation(division=division)

#Return the state matrix/image over the time-points
C_save = srt.get_C_save()

#Animate
srt.animate_C()