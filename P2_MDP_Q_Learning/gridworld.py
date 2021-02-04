# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

'''
Team Members FOR PROJECT - 2:
SUBBA RAO ILLA (C16280847)
SUNDARESH NARAYANAN (C73923755)
'''

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import math
import numpy as np


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    
    previous_v= v.copy()
    for i in range(0,max_iterations):
        previous_v= v.copy()
        for s in range(0,NUM_STATES):
            li=[]
            for a in range(0,NUM_ACTIONS):   
                sum=0
                for val in range (0,len(TRANSITION_MODEL[s][a])):
                   p= TRANSITION_MODEL[s][a][val][0]
                   s_ =TRANSITION_MODEL[s][a][val][1]
                   r= TRANSITION_MODEL[s][a][val][2]
                   terminal= TRANSITION_MODEL[s][a][val][3]
                   sum=sum+(p*(r+(gamma*previous_v[s_])))
                li.append(sum)
            v[s]=max(li)
            pi[s]=li.index(max(li))
        logger.log(i+1, v, pi)
        if( max(abs(previous_v[state]-v[state]) for state in range(0,NUM_STATES)) < 1e-4 ):
            break
                   
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.
    See lines 20-40 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    
    previous_v= v.copy()
    for i in range(0,max_iterations):  
        previous_v= v.copy()
        previous_policies=pi.copy()
        while True:
            previous_v= v.copy()
            for s in range(0,NUM_STATES):
                sum=0
                
                dict={}
                for val in range (0,len(TRANSITION_MODEL[s][pi[s]])):
                    p= TRANSITION_MODEL[s][pi[s]][val][0]
                    s_ =TRANSITION_MODEL[s][pi[s]][val][1]
                    r= TRANSITION_MODEL[s][pi[s]][val][2]
                    terminal= TRANSITION_MODEL[s][pi[s]][val][3]
                    sum=sum+(p*(r+(gamma*previous_v[s_])))
                dict[pi[s]]=sum
                v[s]=sum
            if( max(abs(previous_v[state]-v[state]) for state in range(0,NUM_STATES)) < 1e-4 ):
                break    
        for s in range(0,NUM_STATES):
            for a in range(0,NUM_ACTIONS):
                sum=0
                
                for val in range (0,len(TRANSITION_MODEL[s][a])):
                    p= TRANSITION_MODEL[s][a][val][0]
                    s_ =TRANSITION_MODEL[s][a][val][1]
                    r= TRANSITION_MODEL[s][a][val][2]
                    terminal= TRANSITION_MODEL[s][a][val][3]
                    sum=sum+(p*(r+(gamma*v[s_])))
                dict[a]=sum
            keyMax = max(dict, key=dict.get)
            v[s]=dict[keyMax]
            pi[s]=keyMax
        logger.log(i+1, v, pi)
        
        if(pi==previous_policies ):
            break
            
###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model as above. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 40-50 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    
    # Visualize the initial value and policy
    

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1.0
    eps_decay=1/max_iterations
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

### Please finish the code below ##############################################
###############################################################################
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)
    q_values = np.zeros([NUM_STATES,NUM_ACTIONS])
    print(q_values)
    for i in range(0,max_iterations):
        s = env.reset()
        while True:
            if(random.random()<eps):
                a= random.randint(0, NUM_ACTIONS-1)
            else:
                a = pi[s]
            s_, r, terminal, info = env.step(a)
            if terminal:
                target= r 
            else:    
                target= r + (gamma*np.max(q_values[s_]))
            q_values[s,a] = ((1-alpha) * q_values[s,a]) + (alpha * target)
            s = s_
            v[s] = max(q_values[s])
            pi[s] = np.argmax(q_values[s])
            if (eps>0.001):
                eps -= eps_decay
            if terminal:
                break
        logger.log(i+1,v,pi)
        

        
###############################################################################
    return pi


if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()