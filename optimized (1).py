from scipy.signal import convolve2d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings#delete after
N = 100     # Grid will be N x N
SIM_T = 0.50  # Similarity threshold (that is 1-τ) aka not tolerant
EMPTY = 0.1  # Fraction of vacant properties
B_to_R = 1   # Ratio of blue to red people
N_agents=3
cycles=100
np.random.seed(42) #seed
cmap_set="tab20c"
color_l= ['#fdae6b',"#9e9ac8",'#d9d9d9']
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


def evolve(M, boundary='wrap',radius=1):
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

def evolve_m_agent(M, boundary='wrap',n_agents=N_agents,balance="majority"):
    
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or symm
    If the similarity ratio of neighbours
    to the entire neighborhood population
    is lower than the SIM_T,
    then the individual moves to an empty house.
    """
    #np.seterr(all = "raise") 
    #filter to apply in this cases tell us what the neighbours to consider
    KERNEL = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.int8)

    kws = dict(mode='same', boundary=boundary)
    #neighbours list per agent type
    agents_n_l=[]
    # agent dissatisfied list
    agents_d_l=[]
    #array map
    array_map:list
    array_map_holder:list

    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    neighs   = convolve2d(M != -1,  KERNEL, **kws)
    
    #for graphs
    dissatified_agent=[]
    segregated_agent=[]

    for i in range(n_agents):

        agents_n_l.append(convolve2d(M == i, KERNEL, **kws))
        dissatified_agent.append((agents_n_l[i] / neighs < SIM_T) & (M == i))
        segregated_agent.append(((agents_n_l[i] / neighs == 1) & (M == i)).sum())

        if balance=="majority":
            if i==0:
                #majority is less tolerant
                agents_d_l.append((agents_n_l[i] / neighs < SIM_T+0.2) & (M == i))
            else:
                #minoroties are more tolerant
                agents_d_l.append((agents_n_l[i] / neighs < SIM_T-0.3) & (M == i))

        else:
            agents_d_l.append((agents_n_l[i] / neighs < SIM_T) & (M == i))

        if i==0:
            array_map_holder=agents_d_l[0]
        #makes map
        else:
            array_map=array_map_holder|agents_d_l[i]
            array_map_holder=array_map
    
    # | is iqual to logic operator or
    #dissatisfied agents vacate their position
    M[array_map_holder] = - 1
    vacant = (M == -1).sum() #int of vacant spaces
    filling = -np.ones(vacant, dtype=np.int8) #1 dimension array to then be filled with dissastified pawns

    before_sum, after_sum =agents_d_l[0].sum(),0
    for i in range(n_agents):
        
        if i==0:
            filling[:before_sum]=0
            after_sum+=before_sum
        else:
            after_sum +=agents_d_l[i].sum()

            filling[before_sum:after_sum] = i
            before_sum=after_sum


    #randomizes moved
    np.random.shuffle(filling)
    #
    #-1 means vacants
    # M[M==-1] makes pointer to all cells that are -1 \vacants
    M[M==-1] = filling
    #return happiness 2
    """
    if n_b_dissatified==0:
        unsatisfaction_b=0
    else:
        unsatisfaction_b=n_b_dissatified/(M==0).sum()
    if n_r_dissatified==0:
        unsatisfaction_r=0
    else:
        unsatisfaction_r=n_r_dissatified/(M==1).sum()
    kill_signal=0
    """
    #return
    return agents_n_l,neighs,filling,agents_d_l,array_map_holder,dissatified_agent,segregated_agent

def n_agents(N:int,agents_n:int,Empty,balance="faire"):
    "init for multiple agents"
    vacant = int(N * N * Empty)
    population = N * N - vacant
    
    population_frac=round(population/agents_n)
    M = -np.ones(N*N, dtype=np.int8)
    agents=[]
    half_pop=round(population/2)
    population_frac=round((population/2)/(agents_n-1))
    for i in range(agents_n):
        #balenced distribution
        if balance=="fair":
            M[i*population_frac:(i*population_frac+population_frac)] = i
        else:
             
            if i==0:
                M[0:half_pop] = 0
            else:
                M[half_pop+(i-1)*population_frac:half_pop+(i*population_frac)] = i
                #print(half_pop+(i-1)*population_frac,":",half_pop+(i*population_frac),"i= ",i)
    np.random.shuffle(M)
    return M.reshape(N,N)

def k_evolve(M, boundary='wrap'):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or symm
    If the similarity ratio of neighbours
    to the entire neighborhood population
    is lower than the SIM_T,
    then the individual moves to an empty house that will make them happy.
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
    r_w_s = (r_neighs / neighs < SIM_T) & (M == -1)
    M[r_dissatified | b_dissatified] = - 1
    vacant = (M == -1).sum() #number of vacant spaces
    #vacant good for blue
    #vacant good for red
    #vacant rest
    #number dissatified
    n_b_dissatified, n_r_dissatified = b_dissatified.sum(), r_dissatified.sum()
    #list with all vacant spots
    filling = -np.ones(vacant, dtype=np.int8)

    filling[:n_b_dissatified] = 0
    filling[n_b_dissatified:n_b_dissatified + n_r_dissatified] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling
    #return happiness 2
    return r_w_s

#board=rand_init(N, B_to_R, EMPTY)
board=n_agents(N,N_agents,EMPTY)

plt.matshow(board,cmap=cmap_set)
plt.title("Begining")
plt.show()
plt.close()
#list matrix to use in animation
matrices=[]
unhappines=[]
dissatified=[]
segregated=[]
for i in range(cycles):
    agents_n_l,neight,filling,agents_d_l,array_map_holder,dissatified_agent,segregated_agent=evolve_m_agent(board)
    matrices.append(board.copy())
    unhappines.append(agents_d_l)
    dissatified.append(dissatified_agent),segregated.append(segregated_agent)
plt.matshow(board,cmap=cmap_set)
plt.title("Finish")
plt.show()
plt.close()

#function who call for the update matrix
def update_frames(frame):
   animation_image.set_array(matrices[frame])
   ax.set_title(str(frame))
   return animation_image,

fig, ax = plt.subplots(figsize=(8,8))
animation_image = ax.matshow(matrices[0],cmap=cmap_set)
#makes bar on animation
fig.colorbar(animation_image,orientation='horizontal')
ani1 = FuncAnimation(fig, update_frames, frames=len(matrices), blit=True)
plt.show()
plt.close(fig)
from IPython.display import HTML
#HTML(ani1.to_jshtml())
transpose_list=list(zip(*unhappines))
for j in range(N_agents):
    plt.plot(range(cycles),[transpose_list[j][i].sum() for i in range(cycles)],label="Unhappiness %i"%j ,color=color_l[j] )
plt.title("Unhappiness")
plt.legend()
plt.show()
plt.close()
#Made redunctant plt.plot( [i[j].sum() for i in dissatified])

transpose_list=list(zip(*segregated))
for j in range(N_agents):
        plt.plot(range(cycles),transpose_list[j],label="segregated %i"%j, color=color_l[j] )
plt.title("Segregation")
plt.legend()
plt.show()
plt.close()