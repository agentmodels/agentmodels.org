## Changes and remaining tasks for library

### Changes
- `updateBelief` and `simulate` have constructor functions in beliefAgent.wppl. These constructors are used by both beliefAgent and beliefDelay. There are also a number of helper functions using within `makeBeliefAgent` and `makeBeliefDelayAgent` which are at the top of `makeBeliefAgent.wppl`. Note that `beliefDelayAgent.wppl` depends on `beliefAgent.wppl` as well as on the POMDPUtils.wppl. 

- The `makeAgent` and `simulate` functions take arguments of the same form (but `beliefDelay` requires boundVOI and myopia params etc.). So most tests that only depend on belief (not delays) can be written to work with both functions. You would use `getSimulateFunction('belief')` for the `beliefAgent` simulate function and `getMakeAgentFunction` for the makeAgent function. Note that the function `getPriorBelief` in POMDPutils.wppl is useful for building priors.


### Todo

- In AgentModels, many of the codeboxes need completing. You can use 'git grep TODO' in the agentmodels repo to find these.

- time inconsitency - add expected utilities visualization

- myopia exploration - add new gridworld restaurant example and stochastic bandit examples

- joint inference: number of examples to add. possibly add procrastination example from NIPS paper.

- Fix citations and internal links (Owain)

- Either: (a) store examples in .wppl scripts and then auto-add them to the webbook or (b) run a script that grabs all codeboxes from webbook and stores them as runnable wppl scripts.

- Document the library functions (esp. those used in webbook). The agent models will be explained in detail in the text. But we need to document the "makeGridWorld" functions and give them informative names.

- fix bug with updateBeliefLatent (should be observation.observation == 'noObservation'). Change name of .observation attribute. Change default of using getFullObservation and getBeliefToAction. these assume POMDPs written in terms of manifestStates. for general library use, beliefAgent should not assume POMDPs in this form. 

