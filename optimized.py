from scipy.signal import convolve2d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
N = 100       # Grid will be N x N
SIM_T = 0.7  # Similarity threshold (that is 1-τ)
EMPTY = 0.15  # Fraction of vacant properties
B_to_R = 1   # Ratio of blue to red people
colors=[(1,1,1),(0,0,1),(1,0,0)]
cmap_1=LinearSegmentedColormap.from_list("my_list",colors,N=3)
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


def evolve(M, boundary='wrap',radius=2):
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
    if radius==1:
        KERNEL = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.int8)
    if radius==2:#experience feature
         KERNEL = np.array([[1,1, 1, 1,1],
                            [1,1, 1, 1,1],
                            [1,1, 0, 1,1],
                            [1,1, 1, 1,1],
                            [1,1, 1, 1,1]
                            ], dtype=np.int8)       

    kws = dict(mode='same', boundary=boundary)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    b_neighs = convolve2d(M == 0, KERNEL, **kws)
    r_neighs = convolve2d(M == 1, KERNEL, **kws)
    neighs   = convolve2d(M != -1,  KERNEL, **kws)

    b_dissatified = (b_neighs / neighs < SIM_T) & (M == 0)
    r_dissatified = (r_neighs / neighs < SIM_T) & (M == 1)
    # | is iqual to logic operator or
    #dissatisfied agents vacate their position
    M[r_dissatified | b_dissatified] = - 1
    vacant = (M == -1).sum() #int of vacant spaces

    n_b_dissatified, n_r_dissatified = b_dissatified.sum(), r_dissatified.sum()
    filling = -np.ones(vacant, dtype=np.int8) #1 dimension array to then be filled with dissastified pawns
    filling[:n_b_dissatified] = 0
    filling[n_b_dissatified:n_b_dissatified + n_r_dissatified] = 1

    #randomizes moved
    np.random.shuffle(filling)
    #
    #-1 means vacants
    # M[M==-1] makes pointer to all cells that are -1 \vacants
    M[M==-1] = filling
    #return happiness 2
    if n_b_dissatified==0:
        unsatisfaction_b=0
    else:
        unsatisfaction_b=n_b_dissatified/(M==0).sum()
    if n_r_dissatified==0:
        unsatisfaction_r=0
    else:
        unsatisfaction_r=n_r_dissatified/(M==1).sum()
    kill_signal=0
    return unsatisfaction_b,unsatisfaction_r,kill_signal
board=rand_init(N, B_to_R, EMPTY)

plt.matshow(board,cmap=cmap_1)
plt.title("Begining")
plt.show()
plt.close()

#list matrix to use in animation
matrices=[]
dissastified_b=[]
dissastified_r=[]

for i in range(50):
    per_b_d,per_r_d,kill_signal=evolve(board)
    dissastified_b.append(per_b_d)
    dissastified_r.append(per_r_d)
    matrices.append(board.copy())
plt.matshow(board,cmap=cmap_1)
plt.title("Finish")
plt.show()
plt.close()

plt.plot(range(50),dissastified_b,label="dissastified_b",color="blue")
plt.plot(range(50),dissastified_r,label="dissastified_r",color="red")
plt.xlabel("Iteration")
plt.ylabel("Dissastisfaction")
plt.legend()
plt.show()
plt.close()
#function who call for the update matrix
def update_frames(frame):
   animation_image.set_array(matrices[frame])
   ax.set_title(str(frame))
   return animation_image,

fig, ax = plt.subplots(figsize=(8,8))
animation_image = ax.matshow(matrices[0],cmap=cmap_1)
#makes bar on animation
fig.colorbar(animation_image,orientation='horizontal')
ani1 = FuncAnimation(fig, update_frames, frames=len(matrices), blit=True)

#saves animation
from IPython.display import HTML
HTML(ani1.to_jshtml())