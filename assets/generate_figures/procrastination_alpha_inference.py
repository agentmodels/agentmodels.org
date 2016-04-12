import matplotlib.pyplot as plt
plt.style.use('ggplot')

ax = plt.subplot(111)

optimalAlpha = [322.2099999999999,153.20372902296901,250.19731277690136,335.8316861421096,394.0191716895453,427.35722441540383,444.7022308197206,453.3012132768806,457.4828455286439,0.6824259303759362]
discountAlpha = [322.20999999999987,299.6891023261345,375.9555599453344,425.6921958182417,455.9723999413466,474.0086757832103,484.8604088613864,491.58188051278137,495.71998677732006,572.6276167309703]
actionsSeen = range(10)

lines = ax.plot(actionsSeen, optimalAlpha, 'b', discountAlpha, 'orange')
ax.set_xlabel('Actions observed')
ax.set_ylabel('Posterior expectation of alpha')
ax.text(2, 100, 'Blue is optimal agent, orange is possibly myopic agent')
plt.savefig('./procrastination_alpha_inference.png')
plt.show()
