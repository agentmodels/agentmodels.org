##
 - gridworld viz for trivial line example
 - ch4 get rid of test element
 - docs for library functions so people can see types of arguments (by chapter)
 - add code to hiking under influence box for single trajectory
 - change exercise so it rules out changing noiseProb
 - map, zip, AS
 - Q values in chp5: clear that these arte computecd on the first action


##
#general design issues

u function for agent that takes world and state. so doesnt assume a particular mdp but can be used for multiple mdps. (need this for more interesting inference).

simulate function should take world params {t:, observe:, startState} and agent argument. agent is given the world params (or some of them) and runs based on that. don't need to have long-term planning exp utility agent as we currently have. 
