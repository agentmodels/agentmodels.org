---
layout: chapter
title: Myopic planning and sampling
description: Myopic and bounded-VOI examples (NIPS workshop) on bandit problems and Monte Carlo sampling example (or a better example).

---


### PLAN
- motivation. as we saw pomdp planning is exponential in the number of arms. even smallish problems are hard to compute. one approach is to be myopia / greedy. in bandits, coordination of your actions across time isn't so important. you can the same states available to you at every timestep. so intuitively it seems that being short terms is not as bad as in cases where taking very long terms plans is important.

the simple kind of myopia is to treat the time horizon as short. thi is a time inconsistent model -- you model your future self as not caring about anything after time k, but then your future self will do so. we can implement this easily with delays. unlike the HD agent, we now (at each step) only simulate the next k steps, and so even with many arms this is more tractable. this results in less exploration, as we see.

more interesting myopia model is boundVOI. in a study a humans on bandits, this model was a reasonable fit. the model is again time inconsistent. the agent models his future self as not updating after a delay of d (but continuing to choose based on a now fixed set of beliefs). in fact, his future self will always update after observations. one can think of the boundVOI agent as assuming a fixed cutoff for exploration after which he will only exploit. boundVOI is used in generalized bandit problems such as Bayesian Optimization of real-valued functions. even with a VOI bound / look-ahead of 1, performance can be good.

One step look ahead stochastic bandits. Optimal for 2-arm bandits? For speed up, might need to rewrite the fast belief update. Have Daniel look at this. 

Monte Carlo example? Use rejection sampling instead of enumerate to compute the expected utility. ? Was in the MDP setting. Is it worth doing?  

Can we do it for stochastic bandits? You have some arm that has really low probability of being bad. So you normally wouldn't try it but you pull it if you sample (with non-trivial probability). Bandits with dangerous arms.  

