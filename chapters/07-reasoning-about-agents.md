---
layout: chapter
title: Reasoning about agents
description: Overview of inverse planning / IRL. WebPPL examples of inferring utilities and beliefs from choices (online and batch).
status: stub
is_section: true
---

PLAN

1. Simplest example. Restaurant grid. Fix action cost to be small (-0.1). Then learn preference over restaurants. Grid enumeration and rejection sampling. Can't identify ratios. Show online inference (illustrating inverse planning). 

2. Show inference from multiple trajectories (same graph) and from varying the graph. Stronger inference if you walk longer distance for something.

3. Joint inference. Chooses different place on different occasions. (Example: either start near donut or start near noodle). Can infer high noise or indifferent preferences. (Maybe show full tempt case that's only explicable on high noise as a cliffhanger). 

4. Discounting world: joint inference over preferences on summits, cost of cliff, action cost, softmax cost, and possible transition noise probability. Maybe discuss MH and rejection in continuous space. 


Goals in service of this plan:
1. Simple code for doing inference most efficiently given sequences of actions. This should include examples illustrating the efficiency of doing this for each state-action pair vs. enumerating over whole sequences.

2. For the more advanced example, i.e. with more params (discounting world, which might have 6-7 params), grid-search will be worse. Like to show the benefits of using MH with cts parameters in this case. Show runtime and show 









- (Interactive Gridworld where user can control agent and see inferences?)
