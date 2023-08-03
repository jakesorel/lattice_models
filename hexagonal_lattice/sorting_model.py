import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import animation, cm
try:
    from sim_analysis import Graph
except ModuleNotFoundError:
    from hexagonal_lattice.sim_analysis import Graph

import cv2
import os
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform,cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection


class CellSorting:
    def __init__(self):
        self.x,self.y,self.X,self.Y = [],[],[],[]
        self.W = []
        self.C0 = []
        self.T_t = []
        self.t_span = []
        self.n_save = []
        self.t_save = [None]
        self.C_save = []
        self.n_print = None
        self.t_print = [None]
        self.division_time = []
        self.division_SD = []
        self.Timer = []
        self.xy_clls = []
        self.edge_mask = []
        self.Timer_edge_mask = []
        self.E = []
        self.well = []
        self.boundary = []

        self.ID0 = []
        self.ID = []
        self.ID_save = []
        self.E_mat = []
        self.E_mat_dynamic = []
        self.adjacency_time = []
        self.t_crit, self.f_fin,self.t_n = [],[],[]
        self.dictionary = []

        self.E_t = []
        self.cell_number = []
        self.cell_number_t = []
        self.subpopulation_number = []
        self.subpopulation_number_t = []
        self.average_self_self = []
        self.average_self_self_t = []
        self.eccentricity = []
        self.eccentricity_t = []
        self.angular_distribution = []
        self.angular_polarity = []
        self.angular_polarity_t  = []

        self.t_equilib = []
        self.XEN_external_timescale = []

        self.lat_dirs = np.array([[2,0],
                                  [1, 1],
                                  [-1, 1],
                                   [-2,0],
                                   [1,-1],
                                   [-1,-1]]) #anticlockwise direction

        self.periodic = False

    def generate_lattice(self,num_x=100,num_y=100):
        """Generates a square lattice, with dimensions (num_x,num_y)"""
        self.x,self.y = np.arange(2*int(num_x)),np.arange(2*int(num_y))
        self.X,self.Y = np.meshgrid(self.x,self.y,indexing="ij")
        self.occupancy_matrix = np.zeros_like(self.X).astype(int)
        self.occupancy_matrix[::2,::2] = 1
        self.occupancy_matrix[1::2,1::2] = 1

    def define_interaction_energies(self,W0 = None,boundary_scaler=0,sigma0=None):
        """Defines interaction energies.

        W0 is a 3x3 symmetric matrix defining the (positive) interaction energies of ES,TS,XEN

        boundary_scaler is added to W0 effectively defining the penalty for swapping with a medium element"""
        W = np.zeros([4,4])
        W[1:,1:] = boundary_scaler+W0
        self.medium_energy = boundary_scaler
        self.W = W
        if sigma0 is None:
            sigma0 = np.zeros_like(W)
        self.sigma0 = sigma0
        unique_IDs = np.arange(self.dictionary.shape[0])
        N = unique_IDs.size
        E_mat = np.zeros([N, N])
        E_mat0 = np.zeros([N, N])
        for id_i in [1, 2, 3]:
            for id_j in [1, 2, 3]:
                E_mat[id_i::3, id_j::3] = -np.random.lognormal(np.log(self.W[id_i, id_j]), self.sigma0[id_i, id_j],
                                                               (int((N - 1) / 3), int((N - 1) / 3)))
                E_mat0[id_i::3, id_j::3] = -self.W[id_i, id_j]
        self.E_mat0 = E_mat0
        self.E_mat = E_mat
        self.adjacency_time = np.zeros_like(E_mat)

    def boundary_definition(self,well=None):
        if self.periodic is True:
            self.well = np.zeros_like(self.X)
            self.boundary = np.zeros_like(self.X)
        else:
            if well is None:
                x0, y0 = self.X.shape[0] / 2, self.X.shape[1] / 2
                r1 = int(self.X.shape[0] / 2) - 4
                well = ((self.X + 1 - x0) ** 2/(r1**2) + (self.Y + 1 - y0) ** 2 /(r1**2 * 3/8) > 1)
            x_clls, y_clls = np.where(~well & self.occupancy_matrix)
            boundary = np.zeros_like(self.X)
            for x, i in enumerate(x_clls):
                j = y_clls[x]
                neighbours = self.neighbourhood_possibility(well, i, j)
                if np.sum(neighbours) != neighbours.size:
                    boundary[i, j] = 1
            self.boundary = boundary
            self.well = well



    def C0_generator_random(self,N_E,N_T,N_X):
        if self.periodic is True:
            x_clls, y_clls = np.where(self.occupancy_matrix)
        else:
            x_clls, y_clls = np.where((~self.well)&self.occupancy_matrix)
        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 1
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_T):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 2
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_X):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 3
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        return C0



    def C0_generator_circle(self,N_E,N_T,N_X):
        N_E_, N_T_, N_X_ = N_E,N_T,N_X
        x0,y0 = self.X.shape[0]/2,self.X.shape[1]/2
        N_c0 = N_X + N_E + N_T
        r1 = np.sqrt((2*N_c0/(np.sqrt(3/8)*np.pi)))
        circ = ((self.X+1-x0)**2/(r1**2) + (self.Y+1-y0)**2/(r1**2 * 3/8)> 1)
        x_clls, y_clls = np.where(((~circ)*self.occupancy_matrix*(~self.well)))
        diff = N_c0-x_clls.size
        if diff > 0:
            N_X, N_E, N_T = N_X - int(np.ceil(diff*N_X/N_c0)), N_E - int(np.ceil(diff*N_E/N_c0)),N_T - int(np.ceil(diff*N_T/N_c0))
        N_c = (N_X + N_E + N_T)
        diff = - N_c + x_clls.size
        for i in range(diff):
            rand = np.random.random()
            if rand < N_X_/N_c:
                N_X +=1
            if rand > N_T_/N_c:
                N_T +=1
            else:
                N_E +=1
            N_c -=1

        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 1
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_T):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 2
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_X):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 3
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        return C0


    def C0_generator_random_circle(self,N_E,N_T,N_X):
        x0,y0 = self.X.shape[0]/2,self.X.shape[1]/2
        N_c0 = N_X + N_E + N_T
        r1 = np.ceil(np.sqrt((2*N_c0/(np.sqrt(3/8)*np.pi))))
        circ = ((self.X+1-x0)**2/(r1**2) + (self.Y+1-y0)**2/(r1**2 * 3/8)> 1)

        x_clls, y_clls = np.where(((~circ)*self.occupancy_matrix*(~self.well)))
        print(x_clls.size,N_c0)
        # fig, ax = plt.subplots()
        # ax.imshow(((~circ)*self.occupancy_matrix*(~self.well)).T,extent=[0,2,0,np.sqrt(3)*2])
        # fig.show()
        # diff = N_c0-x_clls.size
        # if diff > 0:
        #     print("True")
        #     N_X, N_E, N_T = N_X - int(np.ceil(diff*N_X/N_c0)), N_E - int(np.ceil(diff*N_E/N_c0)),N_T - int(np.ceil(diff*N_T/N_c0))
        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 1
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_T):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 2
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        for l in range(N_X):
            k = int(np.random.random()*x_clls.size)
            i,j = x_clls[k],y_clls[k]
            C0[i,j] = 3
            x_clls,y_clls = np.delete(x_clls,k),np.delete(y_clls,k)
        return C0

    def C0_generator_fill(self,N_E_,N_T_,N_X_):
        N_c0 = N_E_ + N_X_ + N_T_
        if self.periodic is True:
            x_clls, y_clls = np.where(self.occupancy_matrix)
        else:
            x_clls, y_clls = np.where(self.occupancy_matrix*(~self.well))
        N_E, N_T,N_X = int(x_clls.size*N_E_/N_c0),int(x_clls.size*N_T_/N_c0),int(x_clls.size*N_X_/N_c0)
        N_c = N_E + N_T + N_X
        diff = x_clls.size - N_c
        for i in range(diff):
            rand = np.random.random()
            if rand < N_X_/N_c:
                N_X +=1
            if rand > N_T_/N_c:
                N_T +=1
            else:
                N_E +=1
            N_c -=1
        C0 = np.zeros_like(self.X)
        for l in range(N_E):
            k = int(np.random.random() * x_clls.size)
            i, j = x_clls[k], y_clls[k]
            C0[i, j] = 1
            x_clls, y_clls = np.delete(x_clls, k), np.delete(y_clls, k)
        for l in range(N_T):
            k = int(np.random.random() * x_clls.size)
            i, j = x_clls[k], y_clls[k]
            C0[i, j] = 2
            x_clls, y_clls = np.delete(x_clls, k), np.delete(y_clls, k)
        for l in range(N_X):
            k = int(np.random.random() * x_clls.size)
            i, j = x_clls[k], y_clls[k]
            C0[i, j] = 3
            x_clls, y_clls = np.delete(x_clls, k), np.delete(y_clls, k)
        return C0


    def make_C0(self,N_E,N_T,N_X,scaler,seed_type="random_circle"):
        """Generates the C0 matrix, using the C0_generator function

        N_E, N_T,N_X defines the (approximate) number of cells of ES,TS and XEN at t=0

        scaler will be used to multiply the values assigned for N_E, N_T, N_X"""

        N_X = int(N_X*scaler)
        N_E = int(N_E*scaler)
        N_T = int(N_T*scaler)
        self.N_E, self.N_X, self.N_T, self.N = N_E, N_X, N_T, N_E+N_X+N_T
        if seed_type == "circle":
            C0 = self.C0_generator_circle(N_E,N_T,N_X)
        if seed_type == "random":
            C0 = self.C0_generator_random(N_E,N_T,N_X)
        if seed_type == "random_circle":
            C0 = self.C0_generator_random_circle(N_E,N_T,N_X)
        if seed_type == "fill":
            C0 = self.C0_generator_fill(N_E,N_T,N_X)
        self.C0 = C0
        return C0

    def make_ID0(self,division=False):
        """Each cell is given a unique ID. The spatial configuration of these cells is kept track of in the ID0 matrix.

        A simple convention to make handing data easier is that ES, TS, and XEN cells are alternately assigned.
        I.e. mod(id-1,3) = type {-1 as need to account for the fact that 0 is assigned for medium, both for id and type}
        """
        if division is False:
            ID0 = np.zeros_like(self.C0)
            dict_len = np.max([np.sum(self.C0==id) for id in [1,2,3]])*3+1
            dictionary = np.zeros([dict_len])
            for id in [1,2,3]:
                x_clls,y_clls = np.where(self.C0==id)
                xy_clls = np.array([x_clls,y_clls]).T
                for i, ij in enumerate(xy_clls):
                    ID0[ij[0],ij[1]] = 3*i+id
                    dictionary[3*i+id] = self.C0[ij[0],ij[1]]
        # if division is True:
        #     ID0 = np.zeros_like(self.C0)
        #     for id in [1,2,3]:
        #         x_clls,y_clls = np.where(self.C0==id)
        #         xy_clls = np.array([x_clls,y_clls]).T
        #         for i, ij in enumerate(xy_clls):
        #             ID0[ij[0],ij[1]] = 3*i+id
        #
        #     dict_len = (np.max([np.sum(self.C0==id) for id in [1,2,3]])*3)*int((2**(self.T_t.size/self.division_time+1)))+1 #this sets an upper bound on the number of cells at the end of the simulation
        #     dictionary = np.zeros([dict_len])
        #     dictionary[1::3],dictionary[2::3],dictionary[3::3] = 1,2,3
        self.dictionary = dictionary.astype(int)
        self.ID0 = ID0
        return ID0

    def define_running_params(self,T_t = np.repeat(0.25,1*10**5)):
        """Defines the temperature regime across the simulation.

        T_t: Temperature at each time-step. Length of T_t defines the number of iterations"""
        self.T_t = T_t
        self.t_span = np.arange(T_t.size)

    def define_saving_params(self,n_save=200):
        """n_save is the number of iterations that the simulation output will be saved
        Generates ID_save, an array that logs ID at each of these time-points.
        Generates t_save, a vector defining the time-points of saving"""
        self.t_save = self.t_span[::int(np.ceil(self.t_span.size/n_save))]
        self.n_save = self.t_save.size
        self.ID_save = np.zeros([n_save,self.ID0.shape[0], self.ID0.shape[1]])

    def define_printing_params(self,n_print=20):
        """n_print is the number of iterations the simulation will print the % progress"""
        self.n_print = n_print
        self.t_print = self.t_span[::int(np.ceil(self.t_span.size/n_print))]
    #
    # def define_division_params(self,division_number=4,mean_over_SD=8,Timer=None):
    #     """Sets up division parameters. Generates a matrix Timer of the same shape as C.
    #
    #     Timer keeps track of the time until the next division.
    #
    #     Timer is initially seeded with a uniform distribution by default, preventing divisions in the first mean/8
    #     iterations. But can be customised (requires that dim(Timer) = dim(C)
    #
    #     division_number defines the number of divisions that each cell should take on average across the simulation
    #
    #     mean_over_SD defines the ratio of the mean to the standard deviation of division times. (After the first
    #     division, division time follows a Normal distribution, with means and SDs prescribed here).
    #     """
    #
    #     self.division_time = self.T_t.size / division_number
    #     self.division_SD = self.division_time / mean_over_SD
    #     if Timer is None:
    #         Timer = np.random.uniform(self.division_time / 8, self.division_time,self.X.shape)
    #     Timer[np.where(self.C0 == 0)] = np.inf
    #     self.Timer = Timer



    def E_tot(self,ID):
        """Defines the total energy of a given configuration (C).
        This is the sum of the individual energies
        """
        ID_compiled = np.array([np.roll(ID,+2,axis=0),
                                np.roll(ID,-2,axis=0),
                                np.roll(np.roll(ID,+1,axis=1),+1,axis=0),
                                np.roll(np.roll(ID,-1,axis=1),+1,axis=0),
                                np.roll(np.roll(ID,+1,axis=1),-1,axis=0),
                                np.roll(np.roll(ID,-1,axis=1),-1,axis=0)])
        return np.sum(self.E_mat[ID,ID_compiled])

    def enumerate_neighbours(self,ID,i,j):
        return np.array([ID[i+2,j],ID[i-2,j],ID[i+1,j+1],ID[i-1,j+1],ID[i+1,j-1],ID[i-1,j-1]])

    def periodicity(self,i,j):
        return np.mod(i,self.X.shape[0]),np.mod(j,self.X.shape[1])


    def dEnergy2(self,ID,i,j,k):
        ii,jj = (np.array([i,j]) + self.lat_dirs[k]).T
        ii,jj = self.periodicity(ii,jj)
        ID2 = self.swapper(ID,ID,i,j,ii,jj)
        return self.E_tot(ID2) - self.E_tot(ID)
    #
    # for k in range(6):
    #     print(dEnergy2(etx, ID, 6, 6, k),dEnergy(etx, ID, 6, 6, k))
    #

    def dEnergy(self,ID, i, j, k):
        """Defines the energy change associated with a swap.
        i,j define the matrix position of the element in question
        di,dj define the direction of the putatively swapped cell (di,dj = -1,0,1)
        k defines the
        NB: verbose code, but optimised for efficiency"""

        ii,jj = (np.array([i,j])+self.lat_dirs[k]).T
        ii,jj = self.periodicity(ii,jj)
        Ani,Anj = (np.array([i,j])+self.lat_dirs[[np.mod(k+2,6),np.mod(k+3,6),np.mod(k+4,6)]]).T
        Bni,Bnj = (np.array([ii,jj])+self.lat_dirs[[np.mod(k-1,6),k,np.mod(k+1,6)]]).T
        Ani,Anj = self.periodicity(Ani,Anj)
        Bni,Bnj = self.periodicity(Bni,Bnj)
        A_gain = self.E_mat[ID[i,j],ID[Bni,Bnj]].sum()
        A_loss = self.E_mat[ID[i,j],ID[Ani,Anj]].sum()
        B_gain = self.E_mat[ID[ii,jj],ID[Ani,Anj]].sum()
        B_loss = self.E_mat[ID[ii,jj],ID[Bni,Bnj]].sum()
        dE = A_gain+B_gain-A_loss-B_loss
        return 2*dE #count both directions of an interaction
    #
    # def get_adjacency(self,ID):
    #     A_mat = np.zeros_like(self.E_mat)
    #     adj_array = np.stack([np.roll(ID,1,axis=0),
    #                   np.roll(ID,-1,axis=0),
    #                   np.roll(ID,1,axis=1),
    #                   np.roll(ID,-1,axis=1),
    #                   np.roll(np.roll(ID,1,axis=1),1,axis=0),
    #                   np.roll(np.roll(ID,-1,axis=1),1,axis=0),
    #                   np.roll(np.roll(ID,1,axis=1),-1,axis=0),
    #                   np.roll(np.roll(ID,-1,axis=1),-1,axis=0)])
    #     rw,cl = np.where(ID!=0)
    #     A_mat[ID[rw,cl],adj_array[:, rw, cl]] = 1
    #     A_mat[0] = 0
    #     A_mat[:,0] = 0
    #     return A_mat
    #
    # def adjacency_timer(self,ID):
    #     """Cells build up +1 adjacency time with each timestep of contact. But this reverts to 0 when contact lost"""
    #     self.adjacency_time += 1
    #     self.adjacency_time = self.adjacency_time*self.get_adjacency(ID)
    #
    # def set_dynamic_params(self,t_crit,f_fin,t_n):
    #     self.t_crit, self.f_fin,self.t_n = t_crit, f_fin,t_n
    #
    # def update_E_mat_dynamic(self):
    #     E_mat_additional = np.zeros_like(self.E_mat)
    #     rw, cl = np.where(self.adjacency_time!=0)
    #     t = self.adjacency_time[rw,cl]
    #     E_mat_additional[rw,cl] = self.f_fin/(1+(self.t_crit/t)**self.t_n)
    #     self.E_mat_dynamic = (self.E_mat-self.medium_energy)*(1 + E_mat_additional)+self.medium_energy
    #
    # def update_T_dynamic(self):
    #     T_mat_additional = np.zeros_like(self.E_mat)
    #     rw, cl = np.where(self.adjacency_time!=0)
    #     t = self.adjacency_time[rw,cl]
    #     T_mat_additional[rw,cl] = self.f_fin/(1+(self.t_crit/t)**self.t_n)
    #     self.dynT = np.sum(T_mat_additional,axis=0)/8
    # #
    # # def kth_diag_indices(self,a, k):
    # #     """
    # #     Finds the indices of the offset diagonal of a matrix.
    # #
    # #     From https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    # #
    # #     a is the matrix
    # #
    # #     k is the offset
    # #     (positively offset in the j direction & negatively offset in the i direction by magnitude k)
    # #     """
    # #     rows, cols = np.diag_indices_from(a)
    # #     if k < 0:
    # #         return rows[-k:], cols[:k]
    # #     elif k > 0:
    # #         return rows[:-k], cols[k:]
    # #     else:
    # #         return rows, cols
    # #
    # # def divider(self,X,i,j,dir,N_cells=None):
    # #     """Performs a division at element i,j. Returns a matrix with the element divided.
    # #
    # #     X is the matrix to undergo a division
    # #
    # #     i,j defines the element to be divided (i.e. multiplied)
    # #
    # #     dir defines the direction of the division:
    # #         0: down
    # #         1: up
    # #         2: left
    # #         3: right
    # #         4: down-right
    # #         5: up-left
    # #         6: down-left
    # #         7: up-right"""
    # #     XX = X.copy()
    # #     if dir == 0:
    # #         XX[i + 1:, j] = XX[i:-1, j]
    # #     if dir == 1:
    # #         XX[:i, j] = XX[1:i + 1, j]
    # #     if dir == 2:
    # #         XX[i, j + 1:] = XX[i, j:-1]
    # #     if dir == 3:
    # #         XX[i, :j] = XX[i, 1:j + 1]
    # #     if dir == 4:
    # #         k = j - i
    # #         l = np.min([i,j])
    # #         rw, cl = self.kth_diag_indices(XX,k)
    # #         XX[rw[l+1:],cl[l+1:]] = XX[rw,cl][l:-1]
    # #     if dir == 5:
    # #         k = j - i
    # #         l = np.min([i, j])
    # #         rw, cl = self.kth_diag_indices(XX, k)
    # #         XX[rw[:l],cl[:l]] = XX[rw,cl][1:l+1]
    # #     if dir == 6:  #
    # #         XXX = np.flip(XX,axis=1)
    # #         jj = XX.shape[1]-j - 1
    # #         k = jj - i
    # #         l = np.min([i, jj])
    # #         rw, cl = self.kth_diag_indices(XXX, k)
    # #         XXX[rw[l + 1:], cl[l + 1:]] = XXX[rw, cl][l:-1]
    # #         XX = np.flip(XXX,axis=1)
    # #     if dir == 7:  # up-right
    # #         XXX = np.flip(XX,axis=1)
    # #         jj = XX.shape[1]-j - 1
    # #         k = jj - i
    # #         l = np.min([i, jj])
    # #         rw, cl = self.kth_diag_indices(XXX, k)
    # #         XXX[rw[:l],cl[:l]] = XXX[rw,cl][1:l+1]
    # #         XX = np.flip(XXX,axis=1)
    # #     if N_cells is not None:
    # #         cid = self.dictionary[XX[i,j]]
    # #         XX[i,j] = N_cells[cid-1] * 3 + cid
    # #         N_cells[cid-1] = N_cells[cid-1]+1
    # #         return XX,N_cells
    # #     else:
    # #         return XX
    # #
    #
    # def generate_iijj_from_dir(self,i,j,dir):
    #     """Finds the new daughter cell post-division.
    #
    #     i,j is defines the location of the mother cell
    #
    #     dir defines the division direction (see above)"""
    #     if dir==0:
    #         return i+1,j
    #     if dir==1:
    #         return i-1,j
    #     if dir==2:
    #         return i,j+1
    #     if dir==3:
    #         return i,j-1
    #     if dir==4:
    #         return i+1,j+1
    #     if dir==5:
    #         return i-1,j-1
    #     if dir==6:
    #         return i+1, j-1
    #     if dir==7:
    #         return i-1,j+1

    def swapper(self,X,C,i,j,ii,jj,xy_clls=False):
        """Performs a swapping event.

        X is the matrix to undergo a swap.

        i,j is the cell chosen by the MH algorithm

        ii,jj is the cell that is going to be swapped with cell (i,j)

        xy_clls keeps track of the non-medium elements (i.e. cells). Updated in cases of a medium-cell swap

        C is the configuration matrix. Used to identify changes to xy_clls when medium-cell swaps occur
        """
        XX = X.copy()
        XX[ii, jj] = X[i, j].copy()
        XX[i, j] = X[ii, jj].copy()
        if xy_clls is not False:
            if C[i,j] ==0:
                id = np.where(np.sum(np.absolute(xy_clls-np.array([ii,jj])),axis=1)==0)[0][0]
                xy_clls = np.delete(xy_clls,id,axis=0)
                xy_clls = np.vstack([xy_clls, np.array([i, j])])
            if C[ii,jj]==0:
                id = np.where(np.sum(np.absolute(xy_clls - np.array([i, j])), axis=1) == 0)[0][0]
                xy_clls = np.delete(xy_clls, id, axis=0)
                xy_clls = np.vstack([xy_clls, np.array([ii, jj])])
            return XX,xy_clls
        else:
            return XX

    def number_of_cells(self,C):
        """Finds the number of ES, TS, and XEN cells"""
        return np.array([np.sum(C == 1), np.sum(C == 2), np.sum(C == 3)])

    def initialise_simulation(self):
        """Generates the initial cell list (xy_clls).

        And generates the boundary masks that remove cells that cross the outside of the matrix"""
        x_clls,y_clls = np.where(self.C0!=0)
        self.xy_clls = np.array([x_clls,y_clls]).T

        edge_mask = np.ones_like(self.C0)
        edge_mask[0:2],edge_mask[-2:] = 0,0
        edge_mask[:,0:2],edge_mask[:,-2:] = 0,0
        self.edge_mask = edge_mask

        Timer_edge_mask = np.zeros_like(self.C0,dtype=np.float64)
        Timer_edge_mask[0:2],Timer_edge_mask[-2:] = np.inf,np.inf
        Timer_edge_mask[:,0:2],Timer_edge_mask[:,-2:] = np.inf,np.inf
        self.Timer_edge_mask = Timer_edge_mask


    def neighbourhood_possibility(self,well, i, j):
        """
                0: down
                1: up
                2: left
                3: right
                4: down-right
                5: up-left
                6: down-left
                7: up-right
        """
        ii,jj = (np.array([i,j]) + self.lat_dirs).T
        sample = (~well&self.occupancy_matrix)[ii,jj]
        return sample

    def boundary_valid_didj(self,well, i, j, k):
        ii, jj = (np.array([i, j]) + self.lat_dirs[k]).T
        return (~well & self.occupancy_matrix)[ii, jj]

    #
    # def self_contacts(self,C):
    #     udlr = (C==np.roll(C,1,axis=0))*1.0+(C==np.roll(C,1,axis=1))*1.0+(C==np.roll(C,-1,axis=0))*1.0+(C==np.roll(C,-1,axis=1))*1.0
    #     diags = np.zeros_like(udlr)
    #     diags[:-1,:-1] += 1.0*(C[:-1,:-1] == C[1:,1:])
    #     diags[1:, 1:] += 1.0*(C[:-1,:-1] == C[1:,1:])
    #     diags[1:, :-1] += 1.0 * (C[1:, :-1] == C[:-1, 1:])
    #     diags[:-1, 1:] += 1.0 * (C[1:, :-1] == C[:-1, 1:])
    #     contacts = udlr+diags
    #     return contacts
    #
    # def self_contact_ij(self,C,i,j):
    #     contacts = np.sum(C[i-1:i+2,j-1:j+2]==C[i,j])-1
    #     return contacts
    #
    # def contact_integration(self,C,V,k_gain,k_loss):
    #     """For now, consider only self-self interactions. EXPAND, using W0 in the future
    #
    #     propose that dtV = k_gain*n_contacts - k_loss
    #
    #     for conv., dt = 1"""
    #     contacts = self.self_contacts(C)
    #     V = V+k_gain*contacts - V*k_loss
    #     return V

    def ij_move(self):
        return int(np.random.random()*6)

    def valid_move(self,i,j,k):
        if self.periodic is True:
            return True
        else:
            ii, jj = (np.array([i, j]) + self.lat_dirs[k]).T
            return (ii >= 0) & (ii < self.x.size - 1) & (jj >= 0) & (jj < self.y.size - 1)

    def perform_simulation(self,swap_rate=10,division=False):
        """Performs Metropolis-Hastings. Starting with C0, iterates a random cell selection and putative
       swapping procedure. When cells reach their division time, they undergo divisions

        swap_rate entails the number of cells (as a  proportion of the total number of cells of the embryoid as a whole)
            that are selected in each iteration.
            When swap_rate = 1, N cells are selected with each iteration, where N is the number of cells
            (Note that cell selection is still stochastic i.e. swap_rate =1 does NOT mean that every cell attempts
            to undergo a swap

        if division is False: only swapping is considered. swap_rate term is ignored, with one time-point
        considering one swap"""

        ID = self.ID0
        xy_clls = self.xy_clls
        N_cells = self.number_of_cells(self.C0)
        # if division is True:
        #     Timer = self.Timer
        #     for t, T in enumerate(self.T_t):
        #
        #         #1. Divide
        #         Timer = Timer - 1
        #         if np.sum(Timer < 0)!=0:
        #             i_div, j_div = np.where(Timer<0)
        #             while i_div.size !=0:
        #                 id = np.random.randint(i_div.size)
        #                 i,j = i_div[id],j_div[id]
        #                 if self.boundary[i,j]:
        #                     poss_dirs = np.where(self.neighbourhood_possibility(self.well, i, j) == 1)[0]
        #                     dir = poss_dirs[int(poss_dirs.size*np.random.random())]
        #                 else:
        #                     dir = int(8*np.random.random())
        #                 ID,N_cells = self.divider(ID,i, j, dir,N_cells=N_cells)
        #                 Timer = self.divider(Timer,i, j, dir)
        #                 Timer[i,j] = np.random.normal(self.division_time,self.division_SD)
        #                 Timer[self.generate_iijj_from_dir(i,j,dir)] = np.random.normal(self.division_time,self.division_SD)
        #                 i_div, j_div = np.where(Timer < 0)
        #             x_clls, y_clls = np.where(ID != 0)
        #             xy_clls = np.array([x_clls, y_clls]).T
        #
        #
        #         for n in range(int(xy_clls.shape[0]/swap_rate)): #approximately scale the rate of swapping with embryo size, while allowing for division to occur psuedo-simultaneously.
        #             #2. Select a random cell
        #             cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
        #             i, j = xy_clls[cll_id]
        #
        #             if self.boundary[i,j]:
        #                 # 3. Define a putative move
        #                 di, dj = self.ij_move()
        #                 while not (self.valid_move(i,j,di,dj)& self.boundary_valid_didj(self.well, i, j, di, dj)):  # in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             else:
        #                 #3. Define a putative move
        #                 di,dj = self.ij_move()
        #                 while not self.valid_move(i,j,di,dj):#in some cases, re-sample if point chosen not within graph
        #                     di, dj = self.ij_move()
        #             ii, jj = i+di, j+dj
        #             dE = self.dEnergy(ID,i,j,di,dj)
        #             if (dE < 0):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #             elif np.random.random()< np.exp(-dE/T):
        #                 ID,xy_clls = self.swapper(ID,ID,i,j,ii,jj,xy_clls)
        #                 Timer = self.swapper(Timer,ID,i,j,ii,jj)
        #
        #             self.XXX = ID.copy()
        #
        #
        #         if t in self.t_save:
        #             self.ID_save[np.where(t==self.t_save)[0][0]] = ID
        #         if t in self.t_print:
        #             print("%d %% completed"%(100*t/self.t_print[-1]))
        #     self.Timer = Timer
        if division is False:
            for t, T in enumerate(self.T_t):
                # 2. Select a random cell
                cll_id = int(np.floor(np.random.random() * xy_clls.shape[0]))
                i, j = xy_clls[cll_id]

                if self.boundary[i, j]:
                    # 3. Define a putative move
                    k = self.ij_move()
                    while not (self.valid_move(i,j,k)& self.boundary_valid_didj(self.well, i, j, k)):  # in some cases, re-sample if point chosen not within graph
                        k = self.ij_move()
                else:
                    # 3. Define a putative move
                    k = self.ij_move()
                    while not self.valid_move(i,j,k):  # in some cases, re-sample if point chosen not within graph
                        k = self.ij_move()
                dE = self.dEnergy(ID, i, j, k)
                ii, jj = (np.array([i, j]) + self.lat_dirs[k]).T
                ii,jj = self.periodicity(ii,jj)
                if (dE < 0):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)
                elif np.random.random() < np.exp(-dE / T):
                    ID, xy_clls = self.swapper(ID, ID, i, j, ii, jj, xy_clls)

                if t in self.t_save:
                        self.ID_save[np.where(t == self.t_save)[0][0]] = ID
                if t in self.t_print:
                    print("%d %% completed" % (100 * t / self.t_print[-1]))
        self.ID = ID
        # self.E = self.E_tot(C)


    def get_C(self,ID):
        """Returns C matrix from the ID matrix"""
        C = np.mod(ID, 3)
        C[C == 0] = 3
        C = C * (ID != 0)
        return C

    def get_C_save(self):
        """Returns C matrix from ID matrix for all t"""
        C_save = np.zeros_like(self.ID_save)
        for i, ID in enumerate(self.ID_save):
            C_save[i] = self.get_C(ID)
        self.C_save = C_save
        return C_save

    def find_energies(self):
        """Calculates the global energy for each time-point in t_save, """
        E_t = np.array([self.E_tot(self.ID_save[i].astype(int)) for i in range(self.n_save)])
        self.E_t = E_t
        return E_t

    def find_cell_numbers(self):
        """Calculates cell number for each cell type for each time point in t_save"""
        cell_number = np.zeros(self.C_save.shape[2])
        for i in range(self.C_save.shape[2]):
            cell_number[i] = np.sum(self.C_save[:,:,i]!=0)
        self.cell_number_t = cell_number
        return cell_number

    def find_subpopulations(self,C):
        """For each cell type, finds the number of subpopulations (i.e. cases where cells or clusters of cells are
        seperated by more than one element-width (including diagonals).

        Returns a list: ES,TS,XEN"""
        clique_sizes = []
        for i in np.arange(1,4):
            grid = 1 * (C == i)
            g = Graph()
            clique_sizes.append(g.numIslandsDFS(grid.tolist()))
        self.subpopulation_number = clique_sizes
        return clique_sizes

    def find_subpopulations_t(self):
        """Finds subpopulations for all t in t_save"""
        clique_size_t = np.array([np.array(self.find_subpopulations(self.C_save[i])) for i in range(self.n_save)])
        self.subpopulation_number_t = clique_size_t
        return clique_size_t

    def find_average_self_self_contacts(self,C):
        """Calculates the average number of self-self contacts per cell. Gives an indication of flattness vs roundness
        of a colony or of colonies if there are multiple"""
        av_ss_c = []
        for i in np.arange(1,4):
            x_clls, y_clls = np.where(C==i)
            sum_contacts = 0
            for j, x in enumerate(x_clls):
                y = y_clls[j]
                sum_contacts += np.sum(C[x-1:x+2,y-1:y+2]==i)-1
            av_ss_c.append(sum_contacts/x_clls.size)
        self.average_self_self = av_ss_c
        return av_ss_c

    def find_average_self_self_contacts_t(self):
        """Finds average self-self contacts for all t in t_save"""
        av_ss_c_t = np.array([np.array(self.find_average_self_self_contacts(self.C_save[i])) for i in range(self.n_save)])
        self.average_self_self_t = av_ss_c_t

    def fit_ellipse(self,C):
        """Fits an ellipse to non-zero values in the configuration matrix C

        x0,y0 are the centre of the ellipse

        xl,yl are the short and long axis lengths

        theta is the angle of rotation"""
        img = np.uint8(255*(C!=0))
        cnts, hiers = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        cntss = []
        for cnt in cnts:
            if len(cnt)>5:
                cntss.append(cnt)
        if len(cntss)>1:
            xls = np.zeros(len(cntss))
            yls = np.zeros(len(cntss))
            for cnt in cntss:
                (x0,y0),(xl,yl),theta = cv2.fitEllipse(cnt)
                xls[xl]
                yls[yl]
            cnt = cntss[np.where(xls*yls==np.max(xls*yls))]
        else:
            cnt = cntss[0]
        (x0, y0), (xl, yl), theta = cv2.fitEllipse(cnt)

        return (x0, y0), (xl, yl), theta

    def find_eccentricity(self,C):
        """Finds the eccentricity of a cluster of cells. This performs ellipse fitting."""
        (x0, y0), __, __ = self.fit_ellipse(C)
        if x0>y0:
            a,b = x0,y0
        else:
            a,b = y0,x0
        ecc = np.sqrt(1 - b ** 2 / a ** 2)
        self.eccentricity = ecc
        return ecc

    def find_eccentricity_t(self):
        """Finds eccentricities of clusters for all t_save"""
        ecc_t = np.array([self.find_eccentricity(C) for C in self.C_save])
        self.eccentricity_t = ecc_t
        return ecc_t

    def find_angular_distribution(self,C):
        """Defines the angular distribution of cells across the embryoid.

        Fits an ellipse to find the centroid. Then considers half-lines from this centroid over a 360deg rotation.
        Tracks the ***proportion*** of cells of each type.

        Should give an indication of polarity: if cells are radially symmetric, the angular distribution is flat. e.g. XEN
        If they are segregated to one side, then expect a sinosoidal wave e.g. TS/ES

        BETA: CHECK!!"""
        (x0, y0), __, __ = self.fit_ellipse(C)
        theta_space = np.linspace(0,2*np.pi,50)
        dtheta = theta_space[1] - theta_space[0]
        r_max = np.min([int(C.shape[0]-x0),int(x0),int(C.shape[0]-y0),int(y0)])
        r_space = np.linspace(0,r_max,100)
        p_E,p_T,p_X = np.zeros(theta_space.size),np.zeros(theta_space.size),np.zeros(theta_space.size)
        for i, theta in enumerate(theta_space):
            theta_sample_n = 10
            x = x0 + np.outer(r_space,np.cos(np.linspace(theta,theta+dtheta,theta_sample_n)))
            y = y0 + np.outer(r_space,np.sin(np.linspace(theta,theta+dtheta,theta_sample_n)))
            xy = np.array([np.round(x.flatten()),np.round(y.flatten())]).astype(int)
            xy = np.unique(xy,axis=0)
            sample = C[xy[0],xy[1]]
            p_E[i] = np.sum(sample==1)/np.sum(sample!=0)
            p_T[i] = np.sum(sample == 2) / np.sum(sample != 0)
            p_X[i] = np.sum(sample == 3) / np.sum(sample != 0)
        self.angular_distribution = np.array([p_E,p_T,p_X])
        return p_E,p_T,p_X

    def find_angular_polarity(self,C):
        """
        Quantifies the angular polarity by finding the angular distribution and determining the
        amplitude of a fitted sinosoidal wave (with period 360deg/2pi).

        Complete polarisation should entail a polarity of 0.5
        Unpolarised entails a polarity of 0
        """
        p_E, p_T, p_X = self.find_angular_distribution(C)

        def fit_sine(X, p):
            """
            Cost function to fit a sinosoidal wave with period 2pi
            X = [y-offset,amplitude,x-offset
            """
            sine = X[0] + X[1] * np.sin(np.pi * 2 * np.arange(p.size) / p.size - X[2])
            return np.sum(np.sqrt((p - sine) ** 2))

        amps = []
        for p in [p_E,p_T,p_X]:
            res = minimize(fit_sine,[0.5,0.5,0],method="Powell",args=(p,))
            amps.append(np.absolute(res.x[1]))
        amps = np.array(amps)
        self.angular_polarity = amps
        return amps

    def find_angular_polarity_t(self):
        """Finds angular polarity for t in t_save"""
        a_pol_t = np.array([np.array(self.find_angular_polarity(self.C_save[i])) for i in range(self.n_save)])
        self.angular_polarity_t = a_pol_t
        return a_pol_t

    def find_XEN_externalisation(self,C,membrane_contacts=1,any_contacts=3):
        """Finds proportion of XEN cells on outside"""
        XEN = 1*(C==3)
        number_of_membrane_contacts = 8*XEN - XEN*(np.roll(XEN,-1,axis=0)+np.roll(XEN,1,axis=0)+np.roll(XEN,-1,axis=1)+np.roll(XEN,1,axis=1))
        ANY = 1 * (C !=0)
        number_of_any_contacts = 8*XEN - XEN*(np.roll(ANY,-1,axis=0)+np.roll(ANY,1,axis=0)+np.roll(ANY,-1,axis=1)+np.roll(ANY,1,axis=1))
        return np.sum((number_of_membrane_contacts>=membrane_contacts)+(number_of_any_contacts>=any_contacts))/np.sum(XEN)

    def find_XEN_externalisation_t(self):
        """FInds proprotion of XEN on outside for all t"""
        return np.array([self.find_XEN_externalisation(C) for C in self.C_save])

    def cluster_index(self,C,i_s = (1,2),radial=False,d_eq = False):
        num_x, num_y = C.shape
        X, Y = np.meshgrid(np.arange(num_x), np.arange(num_y), indexing="ij")
        Dd = cdist(np.array([X.ravel(), Y.ravel()]).T, np.array([X.ravel(), Y.ravel()]).T, metric='chebyshev')
        def get_p(i,d):
            CC = 1.0*(C==i)
            CCC = np.outer(CC,CC)
            NN = 1.0*(C!=0)
            NNN = np.outer(NN,NN)
            if d_eq is False:
                p = np.sum(CCC[Dd<=d])/np.sum(NNN[Dd<=d])
                return p
            else:
                p = np.sum(CCC[Dd == d]) / np.sum(NNN[Dd == d])
                return p
        if radial is False:
            self._cluster_index = [get_p(i, Dd.max()) for  i in i_s]
            return self._cluster_index
        else:
            D_r = [np.array([get_p(i,d) for d in np.unique(Dd)]) for i in i_s]
            self._cluster_index_r = D_r
            return D_r


    def cluster_index_t(self,i_s=(1,2)):
        D_t = np.array([self.cluster(C,i_s,radial=False) for C in self.C_save])
        self._cluster_index_t = D_t
        return D_t

    def get_clustered(self,Ci):
        g = Graph()
        count, grid = g.assign_islands(Ci.tolist())
        return (-np.array(grid)).astype(int)

    def cluster_index2(self, C, i_s=(1, 2), radial=False):
        num_x, num_y = C.shape
        X, Y = np.meshgrid(np.arange(num_x), np.arange(num_y), indexing="ij")
        Dd = cdist(np.array([X.ravel(), Y.ravel()]).T, np.array([X.ravel(), Y.ravel()]).T, metric='chebyshev')
        clustered = [self.get_clustered(C==i) for i in i_s]
        def get_p(ii, d):
            c = clustered[ii]
            ni = np.unique(c)
            ni = ni[ni!=0]
            ns = np.zeros_like(ni)
            k = 0
            P = 0
            for j in ni:
                nn = np.sum(c == j).astype(int)
                ns[k] = nn
                CC = 1.0 * (c == j)
                CCC = np.outer(CC, CC)
                NN = 1.0 * (C != 0)
                NNN = np.outer(NN, NN)
                # p = np.sum(CCC[Dd <= d]) / np.sum(NNN[Dd <= d])
                P += np.sum(CCC[Dd <= d]) / np.sum(NNN[Dd <= d])*nn
                k+=1
            return P/np.sum(ns)
        if radial is False:
            self._cluster_index = [get_p(i, Dd.max()) for i in range(len(i_s))]
            return self._cluster_index
        else:
            D_r = [np.array([get_p(i, d) for d in np.unique(Dd)]) for i in range(len(i_s))]
            self._cluster_index_r = D_r
            return D_r

    def cluster_index_t2(self,i_s=(1,2),radial=False):
        D_t = np.array([self.cluster_index2(C,i_s,radial=radial) for C in self.C_save])
        self._cluster_index_t = D_t
        return D_t

    def circles(self,x, y, s, c='b', ax = None,vmin=None, vmax=None, **kwargs):
        """
        Make a scatter of circles plot of x vs y, where x and y are sequence
        like objects of the same lengths. The size of circles are in data scale.

        Parameters
        ----------
        x,y : scalar or array_like, shape (n, )
            Input data
        s : scalar or array_like, shape (n, )
            Radius of circle in data unit.
        c : color or sequence of color, optional, default : 'b'
            `c` can be a single color format string, or a sequence of color
            specifications of length `N`, or a sequence of `N` numbers to be
            mapped to colors using the `cmap` and `norm` specified via kwargs.
            Note that `c` should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values
            to be colormapped. (If you insist, use `color` instead.)
            `c` can be a 2-D array in which the rows are RGB or RGBA, however.
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with `norm` to normalize
            luminance data.  If either are `None`, the min and max of the
            color array is used.
        kwargs : `~matplotlib.collections.Collection` properties
            Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
            norm, cmap, transform, etc.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`

        Examples
        --------
        a = np.arange(11)
        circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
        plt.colorbar()

        License
        --------
        This code is under [The BSD 3-Clause License]
        (http://opensource.org/licenses/BSD-3-Clause)
        """

        if np.isscalar(c):
            kwargs.setdefault('color', c)
            c = None
        if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
        if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
        if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
        if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

        patches = [RegularPolygon((x_, y_), 6,2*s_) for x_, y_, s_ in np.broadcast(x, y, s)]
        collection = PatchCollection(patches, **kwargs)
        if c is not None:
            collection.set_array(np.asarray(c))
            collection.set_clim(vmin, vmax)
        if ax is None:
            ax = plt.gca()
        ax.add_collection(collection)
        ax.autoscale_view()
        if c is not None:
            plt.sci(collection)
        return collection

    def plot_cells(self,ax,C,id,col,**kwargs):
        """Plots cells of a given id with a specific colour

        ax = the axis on which cells are plotted

        C = configuration matrix

        id = cell id to be plotted

        col = colour"""
        x,y = np.where(C==id)
        self.circles(x,y*np.sqrt(3),s=0.5,ax=ax,color=col,**kwargs)

    def plot_all(self,ax,C,cols=("red","blue","green"),**kwargs):
        """Plots the ETX onto ax"""
        E_col,T_col,X_col = cols
        self.plot_cells(ax,C,1,E_col,label="ES",**kwargs)
        self.plot_cells(ax,C,2,T_col,label="TS",**kwargs)
        self.plot_cells(ax,C,3,X_col,label="XEN",**kwargs)


    def plot_save(self,C,file_name=None,dir_name="plots",xlim=None,ylim=None,**kwargs):
        """Plots the ETX embryoid

        dir_name is the directory name

        file_name is a custom filename. If none is given, then the filename is time-stamped

        xlim,ylim define the size of the box. Will auto-scale if not assigned"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, ax = plt.subplots()
        self.plot_all(ax,C,**kwargs)
        ax.axis('off')

        if xlim is None or ylim is None:
            x, y = np.where(C != 0)
            y = y*np.sqrt(3)
            xlim, ylim = (x.min() - 1, x.max() + 1), (y.min() - 1, y.max() + 1)
        ax.set(aspect=1,xlim=xlim,ylim=ylim)

        if file_name is None:
            file_name = "embryoid%d"%time.time()

        fig.savefig("%s/%s.pdf"%(dir_name,file_name))

    def plot_time_series(self,n=6,xlim=None,ylim=None,file_name=None,dir_name="plots",**kwargs):
        """Plots a time-series of size n.

        xlim,ylim is the box size. Will auto-scale (keeping proportions across plots) if None given"""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig, ax = plt.subplots(1,n,figsize=(n*2,2))
        if xlim is None or ylim is None:
            t,x,y = np.where(self.C_save!=0)
            y = y*np.sqrt(3)
            xlim,ylim= (x.min()-1,x.max()+1),(y.min()-1,y.max()+1)
        for nn,i in enumerate(np.linspace(0,self.C_save.shape[2]-1,n).astype(int)):
            CC = self.C_save[i]
            self.plot_all(ax[nn],CC,**kwargs)
            ax[nn].axis('off')
            ax[nn].set(aspect=1,xlim=xlim,ylim=ylim)
        if file_name is None:
            file_name = "time_series %d"%time.time()
        fig.savefig("%s/%s.pdf"%(dir_name,file_name))


    def animate_C(self,file_name=None,dir_name="plots",xlim=None,ylim=None,plot_boundary=False,**kwargs):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if xlim is None or ylim is None:
            t,x,y = np.where(self.C_save!=0)
            y = y*np.sqrt(3)
            xlim,ylim= (x.min()-1,x.max()+1),(y.min()-1,y.max()+1)

        if plot_boundary is True:
            xlim,ylim=(0,self.x.max()),(0,self.y.max()*np.sqrt(3))
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        def animate(i):
            ax1.clear()
            ax1.set(aspect=1, xlim=xlim, ylim=ylim)
            ax1.axis('off')
            self.plot_all(ax1,self.C_save[i],**kwargs)
            if plot_boundary is True:
                ax1.imshow(self.well,cmap=cm.Greys)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)

        if file_name is None:
            file_name = "animation %d"%time.time()

        an = animation.FuncAnimation(fig, animate, frames=self.n_save,interval=200)
        an.save("%s/%s.mp4"%(dir_name,file_name), writer=writer,dpi=264)
