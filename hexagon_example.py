import numpy as np
import matplotlib.pyplot as plt
from hexagonal_lattice.sorting_model import CellSorting

#Set the affinity matrix. This should be symmetric.
# Wij = the energy associated with the bond between cell type i and j
W0 = np.array([[18, 10, 7],
                   [10, 18, 7],
                   [7, 7, 2]])

##Define the energy (treat this as a negative number I think) repelling cells from the medium.
#All of the Ws are increased by the below, meaning the effective energy of a
# cell with a cell is (Wij+boundary_scaler). And a cell with the medium is 0.
boundary_scaler = 1e2

#Set the number of cells of each type. The code is set up for three cell types
N_E, N_T, N_X, scaler = 6,6,0,3


division = False

#Initialise sorting class
srt = CellSorting()

#Count total number of cells
N_tot = (N_E+N_T+N_X)*scaler

#Set up a hexagonal lattice. Nx vs Ny cells
srt.generate_lattice(np.sqrt(N_tot)*3,np.sqrt(N_tot)*3)

#Set if periodic boundary conditions
srt.periodic=False

#Set the boundary. By default, it prevents cells from crawling out of the plane.
#You can prescribe the 'well' in which cells self-organise, by prescribing a binary mask of shape (num_x,num_y)
## Here, srt.boundary_definition(well=well)
srt.boundary_definition(well=None)

#Set up the initial state of the system, populating the grid with cells
#N_E, N_T, N_X are the number of cells of each type
#scaler multiplies the above numbers by some pre-factor
#Note that these are approximate and not exact final numbers, depending on whether the N_E etc is an int (I think)
srt.make_C0(N_E, N_T, N_X,scaler,"circle")

#Generate a static plot of the initialisation
srt.plot_save(srt.C0)

#Set up the division clock.
srt.make_ID0()

##Prescribe the running parameters. This sets the (potentially varying) 'temperature'.
# The shape of the vector defines the number of time points
srt.define_running_params(T_t = np.logspace(2,-4,int(5e4)))

##Prescribe the interaction energies, specified in the pre-text
srt.define_interaction_energies(W0,boundary_scaler=boundary_scaler)

##Gets some internal cogs running to do book-keeping for cell interactions
srt.initialise_simulation()

#Set the number of time-points to record the state. Distributed evenly throughout the simulation
srt.define_saving_params(100)

#Do the simulation
srt.perform_simulation()

#Return the state matrix over the specified time-points
C_save = srt.get_C_save()

#Plot a snapshot of the end of the simulation
srt.plot_save(srt.C_save[-1])

#Animate the simulation
srt.animate_C()
