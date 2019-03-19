import os

targetPath = '/home/data/corpora/wikipedia/chinese-wiki/' 
firstNames = os.listdir(targetPath)





for firstName in firstNames:
    fileNames = os.listdir(targetPath + firstName + '/')

    def parseFile(name):
        finalPath = targetPath + firstName + '/' + name
        os.system('/home/jbai/Parsers/stanford-segmenter-2018-10-16/segment.sh pku ' + finalPath + ' UTF-8 0 > /home/data/corpora/wikipedia/parsedChineseWiki/' + firstName + name)

    for name in fileNames:
        parseFile(name)
        
        break
    break
