from bs4 import BeautifulSoup
import os


#TODO:
# run jekyll serve to generate the new html (or switch to grabbing from the markdown)
# delete all old codeboxes
# make sure print, viz.print, draw work in both editor and node
# remove "bad" symbols from possible filenames
# start adding names to codeboxes, esp. long ones

# could do a code substitute for viz.print or viz.vegaPrint
# (but could just define them in webppl-gridworld so they do nothing
# and could define print as console.log. 


chapterPath  = '_site/chapters/'

def chapterToScripts(chapterName):
    
    chapterFilename = chapterName + '.html'
    
    with open(chapterPath + chapterFilename, 'r') as chapterFile:
        soup = BeautifulSoup(chapterFile, 'html.parser')
        
    # For every <pre> tag, select string inside the <code> tag
    codeboxes = [pre.code.string for pre in soup("pre")]

    scriptPath = '_codeboxes/' + chapterName
    if not os.path.exists(scriptPath):
        os.makedirs(scriptPath)
    
    for i, box in enumerate(codeboxes):
        label = box.split()[1]  # take first word after initial comment tag
        scriptName = str(i+1) + '_' + label + '.wppl'
        with open(scriptPath + '/' + scriptName,'w+') as file:
            file.write(box)


# chapter names without html extension
chapterNames = [fullName.split('.')[0] for fullName in os.listdir('_site/chapters')]

map(chapterToScripts, chapterNames)
