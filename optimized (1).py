from scipy.signal import convolve2d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
N = 100       # Grid will be N x N
SIM_T = 0.4  # Similarity threshold (that is 1-τ)
EMPTY = 0.15  # Fraction of vacant properties
B_to_R = 1   # Ratio of blue to red people
def rand_init(N, B_to_R, EMPTY):
    """ Random system initialisation.
    BLUE  =  0
    RED   =  1
    EMPTY = -1
    """
    #number spaces
    vacant = int(N * N * EMPTY)
    #number agents
    population = N * N - vacant
    blues = int(population * 1 / (1 + 1/B_to_R))
    reds = population - blues
    M = np.zeros(N*N, dtype=np.int8)
    M[:reds] = 1
    M[-vacant:] = -1
    np.random.shuffle(M)
    return M.reshape(N,N)


def evolve(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or symm
    If the similarity ratio of neighbours
    to the entire neighborhood population
    is lower than the SIM_T,
    then the individual moves to an empty house.
    """
    #filter to apply in this cases tell us what the neighbours to consider
    KERNEL = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)

    kws = dict(mode='same', boundary=boundary)
    b_neighs = convolve2d(M == 0, KERNEL, **kws)
    r_neighs = convolve2d(M == 1, KERNEL, **kws)
    neighs   = convolve2d(M != -1,  KERNEL, **kws)

    b_dissatified = (b_neighs / neighs < SIM_T) & (M == 0)
    r_dissatified = (r_neighs / neighs < SIM_T) & (M == 1)
    M[r_dissatified | b_dissatified] = - 1
    vacant = (M == -1).sum() #number of vacant spaces

    n_b_dissatified, n_r_dissatified = b_dissatified.sum(), r_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:n_b_dissatified] = 0
    filling[n_b_dissatified:n_b_dissatified + n_r_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling
    #return happiness 2
    return
board=rand_init(N, B_to_R, EMPTY)

plt.matshow(board,cmap='RdBu')
plt.title("Begining")

#list matrix to use in animation
matrices=[]
happines=[]
for i in range(100):
    evolve(board)
    matrices.append(board.copy())
plt.matshow(board,cmap='RdBu')
plt.title("Finish")


#function who call for the update matrix
def update_frames(frame):
   animation_image.set_array(matrices[frame])
   ax.set_title(str(frame))
   return animation_image,

fig, ax = plt.subplots(figsize=(8,8))
animation_image = ax.matshow(matrices[0],cmap='RdBu')
#makes bar on animation
fig.colorbar(animation_image,orientation='horizontal')
ani1 = FuncAnimation(fig, update_frames, frames=len(matrices), blit=True)

from IPython.display import HTML
HTML(ani1.to_jshtml())
