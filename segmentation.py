import os

targetPath = '/home/data/corpora/wikipedia/chinese-wiki/' 
firstNames = os.listdir(targetPath)

files = []

for firstName in firstNames:
    fileNames = os.listdir(targetPath + firstName + '/')

    for name in fileNames:
        finalPath = targetPath + firstName + '/' + name

        files.append(finalPath)

print(len(files))

for firstName in firstNames:
    fileNames = os.listdir(targetPath + firstName + '/')

    for name in fileNames:
        finalPath = targetPath + firstName + '/' + name

        files.append(finalPath)

        os.system('/home/jbai/Parsers/stanford-segmenter-2018-10-16/segment.sh pku ' + finalPath + ' UTF-8 0 > /home/data/corpora/wikipedia/parsedChineseWiki/' + firstName + name)
        break
    break
