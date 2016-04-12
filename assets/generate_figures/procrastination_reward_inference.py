import matplotlib.pyplot as plt

ax = plt.subplot(111)

optimalReward = [4.4375,3.2241999915252406,2.335948649635492,1.5803777093574267,1.0755027822509946,0.7883794743390911,0.6395028004733034,0.5658815677841956,0.5302344777200582,4.76736477871761]
discountReward = [4.4375,3.4214495972758527,3.156543389439561,3.0114371540991383,2.934497749618596,2.893384944816865,2.8704285710108746,2.856565202659828,2.8486682237625467,6.156940395271026]
actionsSeen = range(10)

lines = ax.plot(actionsSeen, optimalReward, 'b', discountReward, 'orange')
ax.set_xlabel('Actions observed')
ax.set_ylabel('Posterior expectation of reward')
ax.text(2, 5, 'Blue is optimal agent, orange is possibly myopic agent')
plt.savefig('./procrastination_reward_inference.png')
plt.show()
