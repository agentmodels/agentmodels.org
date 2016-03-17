---
layout: chapter
title: "Cognitive biases and bounded rationality"
description: Discuss soft-max noise, limited memory, heuristics/biases, motivation from intractability of POMDPs.
is_section: true
---

- hawthorne paper

- discuss which biases show up for single decisions already, and which require multiple sequential decisions?


### Biases Plan

- Humans aren't optimal. Already have softmax noise. This predicts random errors only when utility differences are small. And even then, doesn't predict systematic errors.

- Some ways in which humans plausibly deviate from previous models. Humans are cognitively bounded. Given lots of observations we may not take everything in. POMDP agent will incorporate every observation. If it learns about some element of environment, it never forgets. But a human could learn where a restaurant is, but might forget if the knowledge is not used for a long time. (Also bayesian inference hard in general). 

- The intrinsic difficulty of optimal planning. A POMDP involves considering every possible belief state you might end up in, which depends all the states you might end up in (due to stochastic T) and all possible observations. Consider a scientist embarking on a five-year project which is expected to include hundreds of experiments. Imagine planning the first experiment by first considering every possible belief state from doing all 100 hundred experiments. (More plausible, the scientist only thinks a few experiments ahead).

- This suggests an interest in cognitive bounds that would apply to any agent. However, humans may also have some distinctive biases that might not apply to an AI we implement. Time inconsistency. 

