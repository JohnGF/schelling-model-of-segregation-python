import numpy as np
import random as rng
import pandas as pd
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
#board generation
nx=12;ny=12
board=np.zeros((nx,ny))
x=np.linspace(0, nx, nx)
y=np.linspace(0, ny, ny)
will=0.4
rng.random()
xv, yv = np.meshgrid(x,y, indexing='ij')
for i in range(len(board)):
    for j in range(len(board[i])):
        if rng.random() >0.5:
            board[i][j]=1
        #np.random()
print("Initial board: \n")
plt.contourf( xv,yv,board)
print(pd.DataFrame(board))
counter=0
somation=0
for i in range(len(board)):
    for j in range(len(board[i])):
        if board[i][j]==1:  
            #print("{}:{}".format(i,j)) delet
            if i+1==len(board):
                i=0
            if j+1==len(board[i]):
                j=0
            somation=board[i][j-1]+board[i][j+1] + board[i-1][j]+board[i-1][j-1]+board[i-1][j+1] + board[i+1][j]+board[i+1][j-1]+board[i+1][j+1]             
            if (somation/8)>will:
                counter+=1
                #this needs to be put in another array so it can be parellelize
                board[i][j]=0
loop=0
#print(counter)
while counter>0 and loop<100 :
    loop+=1
    #print(loop)
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j]==0 and rng.random() >0.10:
                board[i][j]=1
                counter-=1
    print("Iterarion: {}\n".format(loop))
    print(pd.DataFrame(board))
plt.contourf( xv,yv,board)
def scheeling(board,dic_pos,prob):
    
    return
#scheeling(np.zeros((10,10)))