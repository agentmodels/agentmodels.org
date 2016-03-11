# Todos


## Visuals
- Would be good to be able to have a button that runs code repeatedly (in cases where outcome is stochastic). For instance, you hit the button and you see repeated samples of the gridworld trajectory for the same agent. 


## Existing chapters

### III.1- MDP 1
- Replace the Gridworld examples with newGridworld
- agent -> act, and we should probably get rid of timeLeft and put that into the state (for consistency with later versions). get rid of totalTime (just use timeLeft). Have 'terminateAfterAction' as part of the state. 


### III.2 - MDP 2
- Use newGridworld for hike example.
- Update agent mode here to be consistent with later on. (Same changes as above). 
- Need to make expectedUtilities work.


### III.3 POMDP
Use math formalism of the paper and from Kaelbling et al paper. Introduce simplified version of beliefAgent.wppl. Bring updateBelief into scope of agent. Simplify *simulate*. Might be worth discussing recursing on state vs. belief but not clear. 

Should we introduce bandit example here? Yes, because we'll want to talk about it for myopic and boundVOI agents and it's good to have multiple examples. If so, showing stochastic bandits also would be ideal -- otherwise we have no stochasticity in the environment for the next few chapters. This also is a good way to introduce the intractability of POMDPs. 


### IV.1 Reasoning about Agents (Inference)
Start with MDP agents in gridworld:
- Condition on single action
- Condition on trajectory (using simulate technique and more tractable 'offPolicy' technique)
- Condition on multiple trajectories (in same world or different world)

 
### IV.2 Reasoning about Agents II
Inference for POMDP agent. Both for IRL bandits and gridworld. Show the Naive and Sophisticated examples. Can be relatively short. Main thing is that putting priors over agent priors and utilities can get confusing and it's good to start with simple examples. 

### V. 
- General introduction to biases. 
- Focused discussion of time-inconsistency / hyperbolic discounting. Show the generative model. Show Naive and Sophisticated examples. Show inference that includes these. 
- Discussion of myopia and boundVOI with links to relevant literature. Examples from IRL bandits and gridworld. 
- Inference examples that combine these things. 

