## Coding Goals. Deadline Tuesday 23rd Feb

###1. POMDP: Generative model for Gridworld and multi-arm bandits.
It seems valuable to do the standard stochastic bandits. This requires a modification of our code (which assumes deterministic rewards and doesn't do belief updating on the basis of rewards). Stochastic bandits will be slow but we can at least show some small examples. 

###2. Inference on trajectories for POMDP agents.
Bandits and gridworld examples. In Gridworld, you can explain Naive and Sophisticated trajectories in terms of false beliefs about Donut and Noodle respectively. For bandits, you infer the agent's prior belief about the utility of an arm by whether they try the arm. Daniel has been working towards this. One thing new task is to use the improved agent/world factoring from John's hyperbolic code. 

###3. Hyperbolic discounting agent generative model and inference
Have it run fast on gridworld with the naive and sophisticated behavior. John is close to finishing here. (Need to make code part of src, add comments and clearer variable names, and also have documented library functions for constructing gridworld and hyperbolic agent). 

###4. Myopic and bounded-VOI generative model
Extending the model for the hyperbolic-discounting agent. Show how myopic agents do less exploration in bandit problems.

###5. Integrate pomdp and hyperbolic generative model
Are there distinctive behaviors of this model on the restaurant problem? In bandit problems, hyperbolic agent will explore less due to discounting. 

###6. Inference in POMDP + hyperbolic model on Gridworld
This should result in multi-modal explanations of Naive and Sophisticated scenarios for the donut problem. (Show strength of inference of the different explanations. The belief and preference based explanations are more plausible because they need less specific parameter settings).

###7. Reproduce graphs from NIPS paper.
These compare inferences of the flexible model (which includes biases) vs. the optimal model. (We should be able to improve on this by having a broader flexible model than in the NIPS paper).

###8. Visualizations
For the various examples, we need the Gridworld showing the trajectory we condition on and the graph (using viz.print) showing comparing the optimal model to our flexible model. 






