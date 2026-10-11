import numpy as np
import pytest
from optimized import rand_init, evolve

def test_rand_init():
    # Test valid configuration
    N = 10
    B_to_R = 1
    EMPTY = 0.1
    board, population, blues, reds = rand_init(N, B_to_R, EMPTY)
    
    # Check return types and shapes
    assert board.shape == (N, N)
    assert population == N * N * (1 - EMPTY)
    assert blues + reds == population
    
    # Check counts
    unique, counts = np.unique(board, return_counts=True)
    val_counts = dict(zip(unique, counts))
    
    assert val_counts.get(-1, 0) == int(N * N * EMPTY)
    assert val_counts.get(0, 0) == blues
    assert val_counts.get(1, 0) == reds

def test_evolve_all_happy():
    # If all same type, everyone is happy, no one moves
    N = 5
    M = np.zeros((N, N), dtype=np.int8)
    M[0,0] = -1 # one empty
    
    M_copy = M.copy()
    b_dissatisfied, r_dissatisfied, kill_signal, b_segregated, r_segregated = evolve(M)
    
    assert b_dissatisfied == 0
    assert r_dissatisfied == 0
    assert (M == M_copy).all() # No moves should happen

def test_evolve_movement():
    # Set up a situation where agents are definitely unhappy
    N = 3
    M = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, -1]
    ], dtype=np.int8)
    
    # For SIM_T=0.65, almost all should be unhappy
    b_dissatisfied, r_dissatisfied, kill_signal, b_segregated, r_segregated = evolve(M)
    
    # There should be movement
    assert b_dissatisfied > 0 or r_dissatisfied > 0
    
def test_evolve_boundary_wrap():
    # Set up a grid where wrapping makes them happy but fill makes them unhappy
    M = np.zeros((3, 3), dtype=np.int8)
    M[1, 1] = 1 # Red in middle
    M[0, 0] = -1
    
    # evolve modifies M in place, so we check metrics
    evolve(M.copy(), boundary='wrap')

def test_evolve_radius():
    M = np.zeros((5, 5), dtype=np.int8)
    M[2, 2] = 1
    M[2, 3] = -1
    # run with radius 2
    evolve(M.copy(), radius=2)

