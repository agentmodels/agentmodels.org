#!/usr/bin/env python
from bs4 import BeautifulSoup
import os

# Directions:
# 1. edit markdown files, 2. do "jekyll serve" to create html (TODO: automate running jekyll)
# 3. run this script.


# TODOS:
# 1. come up with system for adding codebox scripts to webppl-agents
# such that we can test them all (by checking for errors in execution)

# 2. (maybe should go based on markdown rather than html. since html
# is only updated when we do jekyll serve. but that would require
# rewriting what's below). 




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
