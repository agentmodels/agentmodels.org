import matplotlib.pyplot as plt

ax = plt.subplot(111)

optimalUtility = [13.586206897456096,15.454545454545457,16.555555555555557,17.125,17.71428571428571,18.333333333333336,18.636363636363097,18.999999999999996]
possiblyGreedyUtility = [13.586206897456096,14.580645161768787,14.86206896604547,14.93805309789204,14.97297297353191,14.990825688643719,14.9907834107112,14.99074074131634]
timeHorizonVals = [2,3,4,5,6,7,8,9]

lines = ax.plot(timeHorizonVals, optimalUtility, 'b', timeHorizonVals, possiblyGreedyUtility, 'orange')
ax.set_xlabel('Time horizon values')
ax.set_ylabel('Posterior expectation of utility of chocolate')
ax.text(4, 16, 'Blue is optimal agent, orange is possibly greedy agent')
plt.savefig('./procrastination_alpha_inference.png')
plt.show()

