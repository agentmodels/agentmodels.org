plan

- use karpathy RL algorithms on gridworld. find minimal examples where having a model and doing POMDP type thing beats model-free RL. multi-agent examples might be good for this. also simplified versions of the problematic games for atari.

- faster planning with sampling / MCTS

- speed up POMDP

- speed up inference with HMC / neural net method.


andreas email:
in terms of people using the library, my bigger concern is that the POMDP agents (which are the most interesting agents) are too slow. i imagine a new section with the following chapters:

- use karpathy's RL algorithms on gridworld examples. find the simplest possible examples where having a model and solving the POMDP (maybe using bound-VOI) beats model-free RL. the multi-agent examples could be great for this. we can also make simple versions of the atari games that are hard for RL.

- speed up POMDP agents: try MCTS, standard approximate solvers for POMDPs, some basic engineering to speed up caching and optimize bound-VOI.

- speed up inference with HMC and other gradient based methods


doing all three is a fair amount of work. but they are modular and doing just one would be useful. (my plan is to switch to 'new project' in may -- but i want to at least register these as possibilities for later). 
