# stats 
# pairs counting
# sample table



fileName = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.txt"


wordList = []
pairList = []

with open(fileName, "r") as f:
    for line in f:
        words = line.strip().split()
        print(words)
        break
