import os

targetPath = '/home/data/corpora/wikipedia/chinese-wiki/' 
firstNames = os.listdir(targetPath)

print(firstNames)


files = []

for firstName in firstNames:
    fileNames = os.listdir(targetPath + firstName + '/')

    for name in fileNames:
        finalPath = targetPath + firstName + '/' + name
        print(finalPath)
        files.append(finalPath)

print(files)


# os.system('/home/jbai/Parsers/stanford-segmenter-2018-10-16/segment.sh pku /home/jbai/Parsers/stanford-segmenter-2018-10-16/sample.in UTF-8 0 > sample.out')