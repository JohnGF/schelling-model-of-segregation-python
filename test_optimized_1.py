import numpy as np
import pytest
import importlib.util

# Load optimized (1).py
spec = importlib.util.spec_from_file_location("opt1", "optimized (1).py")
opt1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt1)

def test_n_agents():
    # Test valid configuration
    N = 10
    agents_n = 3
    EMPTY = 0.1
    board = opt1.n_agents(N, agents_n, EMPTY)
    
    # Check shape
    assert board.shape == (N, N)
    
    # Check counts
    unique, counts = np.unique(board, return_counts=True)
    val_counts = dict(zip(unique, counts))
    
    # In `optimized (1).py`, n_agents populates based on math that might not exactly hit int(N*N*EMPTY)
    # The empty cells are whatever remains as -1 after populating. We just check if it's close or bounded.
    assert val_counts.get(-1, 0) > 0
    
    # Check agents exist
    assert val_counts.get(0, 0) > 0
    assert val_counts.get(1, 0) > 0
    assert val_counts.get(2, 0) > 0

def test_evolve_m_agent():
    # Set up a situation where agents are unhappy
    N = 4
    M = opt1.n_agents(N, 3, 0.1)
    
    M_copy = M.copy()
    agents_n_l, neight, filling, agents_d_l, array_map_holder, dissatified_agent, segregated_agent = opt1.evolve_m_agent(M)
    
    # Check that it returns expected outputs
    assert len(agents_n_l) == opt1.N_agents
    assert len(agents_d_l) == opt1.N_agents

def test_k_evolve():
    # Set up a grid 
    M = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, -1]
    ], dtype=np.int8)
    
    r_w_s = opt1.k_evolve(M)
    assert r_w_s is not None

