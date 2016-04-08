import matplotlib.pyplot as plt

ax = plt.subplot(111)

optimalWorkLastMinute = [0.8312761438116328,0.5287806206544768,0.3479811741842922,0.20134554597137466,0.10602976448636145,0.0527475216184923,0.025450158782398206,0.012074845007475686,0.005644696754705079,0.756525629311924]
discountWorkLastMinute = [0.6535839805255447,0.3711885496371404,0.30090029572400545,0.261656426925387,0.2401759374094676,0.22840969592594185,0.22181525623219134,0.21796047590718726,0.21580239100959675,0.9560395193239476] 
actionsSeen = range(10)

lines = ax.plot(actionsSeen, optimalWorkLastMinute, 'b', discountWorkLastMinute, 'orange')
ax.set_xlabel('Actions observed')
ax.set_ylabel('Posterior probability that the agent would choose to work at the last minute')
ax.text(2, 0.5, 'Blue is optimal agent, orange is possibly myopic agent')
plt.savefig('./procrastination_workLastMinute_inference.png')
plt.show()
