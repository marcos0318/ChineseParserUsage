# Actually the model that I used does not require the segmentation first. Just do it. throw that to the server! 

import os
from multiprocessing import Pool

if __name__ == '__main__':
    targetPath = '/home/data/corpora/wikipedia/chinese-wiki/' 
    firstNames = os.listdir(targetPath)



    for firstName in firstNames:
        fileNames = os.listdir(targetPath + firstName + '/')

        def parseFile(name):
            finalPath = targetPath + firstName + '/' + name
            os.system('/home/jbai/Parsers/stanford-segmenter-2018-10-16/segment.sh pku ' + finalPath + ' UTF-8 0 > /home/data/corpora/wikipedia/parsedChineseWiki/' + firstName + name)

        with Pool(processes=20) as pool:
            pool.map(parseFile, fileNames)

       