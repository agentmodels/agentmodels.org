from bs4 import BeautifulSoup
import os


#TODO:
# run jekyll serve to generate the new html (or switch to grabbing from the markdown)

# make sure print, viz.print, draw work in both editor and node



chapterPath  = '_site/chapters/'  # where chapters (html) are stored
scriptDir = '_codeboxes/'  # where the wppl scripts go

def cleanLabel(label):
    return ''.join( [ch for ch in label if (ch.isalnum() or ch=='_' or ch=='-')] )

def chapterToScripts(chapterName):
    
    chapterFilename = chapterName + '.html'
    
    with open(chapterPath + chapterFilename, 'r') as chapterFile:
        soup = BeautifulSoup(chapterFile, 'html.parser')
        
    # For every <pre> tag, select string inside the <code> tag
    codeboxes = [pre.code.string for pre in soup("pre")]

    scriptPath = scriptDir + chapterName
    os.makedirs(scriptPath)
    
    for i, box in enumerate(codeboxes):
        label = box.split()[1]  # take first word after initial comment tag
        label = cleanLabel(label)
        scriptName = str(i+1) + '_' + label + '.wppl'
        with open(scriptPath + '/' + scriptName,'w+') as file:
            file.write(box)


# chapter names without html extension
chapterNames = [fullName.split('.')[0] for fullName in os.listdir('_site/chapters')]

# remove existing codeboxes
if os.path.exists(scriptDir):
    os.system('rm -r ' + scriptDir)
    
map(chapterToScripts, chapterNames)
