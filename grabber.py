from bs4 import BeautifulSoup
import os

htmlPath  = '/Users/owainevans/agents/agentmodels.org/_site/chapters/'
htmlName = '1-introduction.html'

with open(htmlPath + htmlName, 'r') as htmlFile:
    soup = BeautifulSoup(htmlFile, 'html.parser')

# For every <pre> tag, select string inside the <code> tag
codeboxes = [pre.code.string for pre in soup("pre")]

scriptPath = '/Users/owainevans/agents/agentmodels.org/_codeboxes/'

for i, box in enumerate(codeboxes):
    label = box.split()[1]  # take first word after initial comment tag
    scriptName = htmlName.split('.')[0] + '_' + str(i) + '_' + label + '.wppl'
    with open(scriptPath + scriptName,'w+') as file:
        file.write(box)

