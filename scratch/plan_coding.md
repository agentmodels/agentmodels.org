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
For the various examples, we need the Gridworld showing the trajectory we condition on and the graph (using viz.print) comparing the optimal model to our flexible model. 



## Specific plans: Sunday 21 Feb

### Hyperbolic (John)
Set Digest == 2 and do big donut world.

### Generative model
In addition to examples showing Naive and Sophisticated behavior, show the changing expected utilities for the Naive agent, which leads to the time inconsistency. (Ideally we would show these dynamically update -- maybe just showing utilities for the square next to donutNorth).

### Inference from naive and sophisticated scenarios.

Example 1 shows the posterior on donut and veg for naive agent (can fix all parameters apart from a single utility parameter for donut and veg to make inference fast) for a fine grid (as shown in the AAAI paper).

Example 2 includes inference on all the parameters. This should show that from a single trajectory, the model can infer either the naive or sophisticated explanations (rather than just assuming high noise). This will need MCMC. Can simplify inference a bit by assuming the noodle is bad (as the main conceptual point here is explanations involving high alpha vs. explanations involving nonzero discount rate and naive/sophisticated planning). The prior on the *discount* should assign a high probability to zero discounting (while the other priors should be broad/uniform as possible). 

Example 3 shows inference from multiple trajectories. This should make the high alpha (random noise) explanation less likely -- assuming we see behavior consistent with hyperbolic discounting both times. One case is where you see the same trajectory (e.g. Naive) twice. Another is you first see the Naive trajectory and then you see the agent starting to the right of noodle and going up and then left to veg. A final example is one where the route directly north to veg/donutNorth is blocked (so the world is actually different). Here the Naive agent would take the long route. So if you see this trajectory and the normal Naive trajectory, this is good evidence for Naive over a noise-based explanation. 


(If we have time on Tuesday, we could implement the procrastination problem). 

--------

### POMDP (Daniel).

## Generative model for pomdp

Specialize the beliefDelayAgent to a "beliefAgent" by getting rid of the delays but keeping everything else the same. Write bandit tests for this agent (basically the tests currently in `beliefDelayAgent.wppl`).

On the same model as the bandit tests, write code for the pomdp version of gridworld (which has an `observe` function and `manifestStateToAction`. This should re-use the `newGridworld` code as much as possible. Daniel should chat with John about the design here and then write a first version. (We can chat about that version when it's done). For now, the `latentState` will just be {donutSouthOpen: true, noodleOpen: false}. The transition function checks whether a restaurant is closed and blocks the agent from moving to the restaurant if it is closed. (Apart from this, it's the same transition function as in the MDP). The agent observes that the restaurant is closed when in the adjacent locations. The function `manifestStateToActions` can allow the agent the action that would move it to the closed restaurant, even though this action won't do anything. (This is simpler than having the available actions depend on the agent's beliefs). 

Once the gridworld code is ready, show examples with the 'Naive' trajectory (where the agent has a high prior that `donutSouthOpen:false`) and with the 'Sophisticated' trajectory (where the agent believes Noodle is open but it's actually closed). 

## Inference for POMDP
Show a simple bandit examples. First, with k arms. Agent is uncertain about value of arms. We don't know his expected values for these arms. If he explores then he thinks EV is high. Otherwise, he'll just stay with the arm he knows.

Then show inference in gridworld for the naive and sophisticated trajectories. The hardest case (where we infer preferences, alpha and beliefs) will probably need MCMC. 


-----

### BeliefDelay agent

- The pomdp gridworld library should also work for the `beliefDelayAgent`. (It generates a world object that can be an input to `beliefDelay` agent). Tests for gridworld for `beliefAgent` should carry over, either using `noDelays` or by setting `discount` parameter to zero. Using `noDelays` should have the same runtime as `beliefAgent`, while using zero `discount` should take longer. (`beliefDelay` with a delta prior should have runtime close to the pure `hyperbolic` agent). 

- It's important to find the cost of a trajectory on the big gridworld for the `beliefDelay` agent with both uncertainty and delays. This will allow an upper-bound for inference. If this is too slow, we can profile (and possibly change caching of ERPs). 

- Need to add myopic and bound-VOI. Need to clear tests for correctness. Example from NIPS paper is not ideal and maybe need a better illustration. Maybe it should just be bandits with 'long corridors'. Bound-VOI will go down a long corridor if it knows the result is good but not otherwise. Myopic agent won't go down a long corridor at all. In the web setting, the myopic agent won't do any activity with a long payoff (e.g. read something that is 'slow at first' but good, do a hard MOOC where you only get certificate at the end). Bound-VOI will only do thing will long payoff if it already knows it's a good payoff or if the gets rapid feedback. (It won't start a MOOC if it would only know at the mid-term whether it's worth continuing -- somewhat weird as a model of humans in this context).


### Design for IRL bandits
We want a version of bandits where the agent's preferences over rewards are not given. We get to see the agent choosing between arms, we (doing the inference) know the results or dist on results of pulling the arms, but we don't know how much the agent prefers each possible result. The example case is that agents are heterogeneous in their preferences. We are watching them choose between different content sources (e.g. blog, youtube channel, tv channel, news site, twitter feed etc.). We know about the distribution the sources put on types of content. We don't know what type of content the agent prefers. We jointly infer the agent's preferences on type of content along their prior distribution on which content each source favors. (We don't have control of the sources or the content a user sees. We don't know what background info or presumptions the agent has about which sources are likely to be a good match for their preferences. We don't know how much they've watched different sources in the past). 

A simple model of the stochastic version is that a source is a multinomial on content-type. The agent has Dirichlet priors on each source which are updated analytically from samples. The unknowns are the agent priors and then the agent's mapping from the types to utilities. (What about arms that give numerical rewards. Then we could learn a function on the numbers, e.g. a log function or a sigmoid function or some other threshold function). 


### Myopic and bound VOI

Myopia can be implemented using `perceivedTotalTime`. If actualTotalTIme is 50, k is 5, then simulate calls agent with perceivedTotalTime==5 for every time step (overriding `transition` which would otherwise be counting this down). Do this until the time remaining is 5 and then do the normal thing. This works without having any delays.

With delays, you can implement *sophisticated* myopia. If you are myopic with k=2, then agent simulates fact that after one time step they will care about third timestep (even though they don't care about it now). This is a pretty weird kind of agent. (For example, suppose options are A (u=1) or B (road to C and D), where C=2, D=3, and C and D are two and three steps away from start respectively. Then soph agent would take A to avoid being tempted by D when going to C.)

We can elegantly express myopic with delays, assuming we set agent to Naive. Seems it will be similar in terms of efficiency to modifying simulate. When doing inference, we want get agent action distributions for given states. In the belief case, we also need to pass a current belief and observation to the agent. How does this interact with delays? We would always set the delay to zero, because we are computing how likely the trajectory is for the agent --- not how likely a given move is. 

~~~~
// if we did this via modifying *simulateBeliefAgent*
// (to do this by delays, we just kill state when delay > myopia constant)

var myopiaConstant = 5;

var sampleSequence = function(state, currentBelief, actualTimeLeft) {
	if (cutoffCondition(actualTimeLeft, state.manifestState) ) {
	    return [];
	} else {
        var manifestState = actualTimeLeft < myopiaConstant ? state.manifestState :
        update(state.manifestState,{perceivedTotalTime: myopiaConstant});
        
	    var nextAction = sample(agentAction(manifestState, currentBelief, observe(state)));
	    
	    var nextState = transition(state, nextAction.action);
	    var out = {states:state, actions:nextAction, both:[state, nextAction],
                       stateBelief: [state, currentBelief]}[outputStatesOrActions];
	    
	    return [out].concat( sampleSequence(nextState, nextAction.belief,
						actualTimeLeft - 1));
        }
    };
    return sampleSequence(startState, priorBelief, actualTotalTime);
}
~~~~

For boundVOI, we check the delay and only get a non-trivial observation if the `delay < boundVOI_constant`. Suppose you are sophisticated about boundVOI. Suppose you have `boundVOI_constant==1`. You simulate yourself at timestep 2 getting an observation. Since your simulation of your future self is accurate, you simulate yourself updating on this observation. Hence sophistication makes the VOI bound irrelevant. So best to ensure that boundVOI agent is *naive*. 


~~~~
// modification of beliefDelayAgent

assert.ok( !(agentParams.boundVOI && !agentParams.sophisticatedOrNaive == 'naive'), 'if boundVOI then not sophisticated');

var getObservation = function(state,delay){
    return (delay < boundVOI_constant) ? observe(state) : 'noObservation';
};

 var expectedUtility = dp.cache(
    function (manifestState, currentBelief, action, delay) {
      return expectation(
        Enumerate(function () {
          var latentState = sample(currentBelief);
          var state = buildState(manifestState, latentState);
          var u = 1.0 / (1 + agentParams.discount * delay) * utility(state, action);
          if (state.manifestState.dead) {
            return u;
          } else {
            var nextState = transition(state, action);
            var perceivedDelay = getPerceivedDelay(delay);
            var observation = getObservation(nextState);
            var nextAction = sample(_agent(nextState.manifestState, currentBelief, observation, perceivedDelay));
            var futureU = expectedUtility(nextState.manifestState, nextAction.belief, nextAction.action, incrementDelay(delay));                                                                                                                
            return u + futureU;
          }
        }));
        });
        ~~~~






