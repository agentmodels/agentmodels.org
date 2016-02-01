---
layout: chapter
title: "Sequential decision problems (MDPs)"
description: Mathematical framework, implementation in WebPPL with explicit recursion (could compare to value iteration), Gridworld examples.

---


PLAN:

1. We only consider very simple problems. In the following chapters, we consider more complex problems of two kinds. First, we look at sequential decision making. Cases where the selection of action now depends on how the agent will choose actions in the future. These problems are still simple in that they only involve a single agent. In later chapters, we consider game-theoretic situations involving multiple agents. 

2. Introduce MDP. Give example of moving round city to choose restaurants. To eat at a restaurant, need to first walk to it. Also prefer a shorter to longer route. Show example. 

3. Go through math of this case. Restaurant example can be solved by shortest path type algorithms. But for probabilistic case we can't do that.

4. Discounting example. Two summits. Might be unknown which is more preferred (if you just have satellite image and movement data). Cliff is just a steep hill that would hurt if you fell down (and probably end the hike). Could think of graph more abstractly: cliff as states that you reach if you take a very fast route (or if there's a route with worse heights, you might get vertigo and have to stop). 

good to think about what's stochastic in restaurant street example. attending to tempting thigns might be.

also good to think about andreas example of infinite time horizon but with small probability of death at each age (similar to language models with prob of infinite lenght sentence). 


## Sequential Decision Problems: Introduction
The previous [chapter](/chapters/03-one-shot-planning) chapter introduced agent models for solving very simple decision problems. The rest of the tutorial looks at more complex and interesting problems. Later chapters will look at problems where the outcome depends on the decison of another rational agent (as in *game theory*). The next few chapters look at single-agent problems that are *sequential* rather than *one-shot*. In sequential decision problems, an agent's choice of action *now* depends on the action they'll choose in the future. (Agents must *co-ordinate* with their future selves).

As a simple illustration of a sequential decision problem, suppose that an agent, Alice, is looking for somewhere to eat. Alice gets out of work in a particular location (labeled "start"). She knows the streets and the restaurants nearby. So the decision she faces is to choose a restaurant that (a) she likes, and (b) is not too long a walk.




~~~~
var element = makeResultContainer();

var world = {
  width: 200,
  height: 200,
  fromX: 10,
  fromY: 50,
  incX: 100,
  incY: 0
};

GridWorld.draw(element, world)
~~~~
