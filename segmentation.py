import os

targetPath = '/home/data/corpora/wikipedia/chinese-wiki/' 
firstNames = os.listdir(targetPath)

print(firstNames)

# os.system('/home/jbai/Parsers/stanford-segmenter-2018-10-16/segment.sh pku /home/jbai/Parsers/stanford-segmenter-2018-10-16/sample.in UTF-8 0 > sample.out')