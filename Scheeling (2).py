import numpy as np
import copy
from matplotlib import pyplot as plt
def board(n):
    nx=n;ny=n
    board=np.random.rand(nx,ny)
    x=np.linspace(0, nx, nx)
    y=np.linspace(0, ny, ny)
    xv, yv = np.meshgrid(x,y, indexing='ij')
    return xv,yv, board


def check_move(board,i,j,happy,threshold=0.5):
    if np.isnan(board[i,j]):
        #is a empty space
        return board,happy
    value=board[i,j]
    neighbours=0
    #back
    if board[i-1,j]==value:
        neighbours=neighbours+1
    #back up
    if board[i-1,j-1]==value:
        neighbours=neighbours+1
    #back down 
    #-len(board[0])+1 = foward position this was made to makes no boundaries
    if board[i-1,j-len(board[0])+1]==value: 
        neighbours=neighbours+1
    #foward middle
    if board[i-len(board[0])+1,j]==value:
        neighbours=neighbours+1
    #foward up
    if board[i-len(board[0])+1,j-1]==value:
        neighbours=neighbours+1
    #foward down
    if board[i-len(board[0])+1,j-len(board[0])+1]==value:
        neighbours=neighbours+1
    #midde up
    if board[i,j-1]==value:
        neighbours=neighbours+1
    #midde down
    if board[i,j-len(board[0])+1]==value:
        neighbours=neighbours+1
    
    else:
        #no neigbours dont move
        return board,happy+1
    #print(neighbours)
    if (neighbours/8)>threshold:
        
        #back position
        if np.isnan(board[i-1,j]):
            board[i-1,j]=value
            board[i,j]=np.nan
        #back up
        elif np.isnan(board[i-1,j-1]):
            board[i-1,j-1]=value
            board[i,j]=np.nan
        #back down 
        #-len(board[0])+1 = foward position this was made to makes no boundaries
        elif np.isnan(board[i-1,j-len(board[0])+1]): 
            board[i-1,j-len(board[0])+1]=value
            board[i,j]=np.nan
        #foward middle
        elif np.isnan(board[i-len(board[0])+1,j]):
            board[i-len(board[0])+1,j]=value
            board[i,j]=np.nan
        #foward up
        elif np.isnan(board[i-len(board[0])+1,j-1]):
            board[i-len(board[0])+1,j-1]=value
            board[i,j]=np.nan
        #foward down
        elif np.isnan(board[i-len(board[0])+1,j-len(board[0])+1]):
            board[i-len(board[0])+1,j-len(board[0])+1]=value
            board[i,j]=np.nan
        #midde up
        elif np.isnan(board[i,j-1]):
            board[i,j-1]=value
            board[i,j]=np.nan
        #midde bottom
        elif np.isnan(board[i,j-len(board[0])+1]):
            board[i,j-len(board[0])+1]=value
            board[i,j]=np.nan
    else:
        #no space to move
        return board,happy
    #print("did something")
    return board,happy
will=0.5
N=60
xv,yv,board_v=board(N)
board_c=copy.copy(board_v)
board_c[board_v<0.1]=np.nan
board_c[board_v>0.1]=0
board_c[board_v>0.55]=1

plt.matshow(board_c)
plt.title("Beggining")
plt.show()
board_f=board_c.copy()

happy=0
for k in range(100):
    for i in range(len(board_v)):
        for j in range (len(board_v[0])):
            board_f,happy=check_move(board_f,i,j,happy)

plt.matshow(board_f)
plt.title("Finish")
plt.show()
print("Happines: ",happy/(N*N))

