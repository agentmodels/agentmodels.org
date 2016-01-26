## Spec for Gridworld and Visualization

### Implementation of Gridworld
My aim is to have a Gridworld implementation that fits the standard examples used in RL/MDP (Russell-Norvig 4x3 example, cliff example, and so on). It should also fit the Restaurant Choice examples from our papers.

I've implemented a quick version of the standard Gridworld in ppl_files/models/bookMDP.wppl, after consulting the Berkeley "Pac-Man" python package and the Russell-Norvig AIMA(3e) text. I'll summarize the key features:

#### Utility when leaving states
Pairs of [state,action], i.e. your current state and the action you take in that state, are the argument for the transition function T and the utility function U. In standard Gridworld states have utilities and there is a generic action cost. So U(s,a) = U*(s) - .15 (for example).

On our model, the agent gets utility on *leaving* a state. Assume for illustration that actions don't matter for utility. Suppose the agent has only a single action and starts in state s0. Then they will receive U(s0), no matter what action they take. Note that wherever they would go after s0 doesn't make any difference to their utility. (Also note that we learn nothing about the agent their choice of action). 

This is the model (utility when leaving) that the Berkeley-Pacman group follows. AIMA does it differently. If we have terminal states as in standard Gridworld (where the agent can get a reward from state before the game ends), we need to set up the terminal states such that you get utility. Berkeley does this by having a special "exit" action that you take in terminals. 


#### Cells either open or blocked
There are *open* and *blocked* grid cells. You start in an open cell and if you walk towards a blocked cell you stay put. Likewise, if you walk off the grid (i.e. "against a wall") you stay put.

#### Terminal nodes
Standard MDPs can have unbounded horizon. We have focused on finite horizon. For unbounded horizon, you need *terminal* nodes to end the game. Should we have terminal nodes? In the Restaurant Example, we used terminal nodes. We could simulate having terminal nodes by adding an extra invisible "sink" state that you must go to after visiting what would be the terminal state. (Not having terminals means that we don't need checks in the code for "isTerminal"). On the other hand, having real terminals brings this closer to the standard MDP setting (and might in some cases lead to runtime savings).

#### Stochastic transitions
Standard examples have a noise model where you go LEFT or RIGHT with probability .1 when you choose UP (and so on). This is easy enough to implement. (I don't think it complicates the current MDP code to have stochastic transitions -- but I need to verify this). 

#### Stochastic rewards
Bandit problems have stochastic rewards. It seems fairly easy to change the code to do this for Gridworld. 


### Visualization of Gridworld

Here are the desired features for visualization:

1. Shows blocked/open cells (looking like a path or road map). Highlights terminals/restaurants and shows the utility of cells with non-zero utility.

2. Show the agent's path over time. The simplest way is an animation. However, I think showing the path in a static image is very useful. The path could be a series of arrows (direction of agent from each cell). But best would be a line joining midpoints of successive cells, with occasional arrows showing overall direction. 

3. Label the agent's policy and the EU of each state (a discrete heat map). For MDP and POMDP this is time consistent. For time-inconsistent agents, it will be awesome to show how this heatmap varies over time (as the agent re-computes the EU). 

4. For noisy agents, could show a heatmap of the agent's time spent in each cell. 

5. Interactive: user can manipulate the utilities or other features of the MDP (as in Karpathy's JS DP example) or user can drive the agent and so influence the IRL that is done.


Here are some thoughts on what kind of data structure to provide to the visualization module. The basic idea is that each cell has a set of fixed features {utility:5, blocked:false, terminal:true} and possibly also {agentPolicy:'r', agentQValue:{u:4.6, d:2.2, l:4, r:1}}. Then you have a sequence of agent actions of form [ cell1, cell2, ... ]. You create paths by adding horizontal or vertical lines to cells, based on the diffs of the sequence. These could be extra attributes of cells, or could be represented in a separate layer. (Or whatever the standard solution is here). 

In Webppl, you have a function for producing all gridcells (e.g. by enumerating) and then functions from states to their various features. Features like `isBlocked`, `isTerminal`, and `utility(state)` are part of the `params` for Gridworld MDPs. The agent-dependent features (e.g. the agent's action in a cell, the agent's EU for a cell) have to be computed using the `agent` function. You compute something like zip(allCells, map( getAllFeatures, allCells)) and send this to JS for visualization.
