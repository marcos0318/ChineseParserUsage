# stats 
# pairs counting
# sample table
from collections import Counter
import pickle

fileName = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.txt"
fileNameamod = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.amod.txt"
fileNamedobj = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.dobj.txt"
fileNamensubj = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.nsubj.txt"

wordList = []
# pairList = []

with open(fileName, "r") as f:
    for line in f:
        words = line.strip().split()
        # pairList.append(tuple(words))
        wordList.append(words[0])
        wordList.append(words[1])


wordCounter = Counter(wordList).most_common(100000)
# pairCounter = Counter(pairList)

wordList = list(dict(wordCounter).keys())

print(wordList[:10])

# corpus_stats.pkl
corpus_stats = dict()
corpus_stats["id2word"] = {i: word for i, word in enumerate(wordList)}
corpus_stats["word2id"] = {word: i for i, word in enumerate(wordList)}
corpus_stats["vocab_size"] = len(wordList)




# print(corpus_stats)
with open("corpus_stats.pkl", "wb") as fout:
    pickle.dump(corpus_stats, fout)



rels = ["amod", "dobj", "nsubj"]
pairsCountingResult = {}
sampleTableResult = {}
for i, fileName in enumerate([fileNameamod, fileNamedobj, fileNamensubj]):
    pairList = []
    argumentList = []
    with open(fileName, "r") as f:
        for line in f:
            words = line.strip().split()
            try:
                wordids = [corpus_stats["word2id"][word] for word in words]
            except:
                continue

            pairList.append(tuple(wordids))
            argumentList.append(words[1])

    pairCounter = dict(Counter(pairList))
    pairsCountingResult[rels[i]] = pairCounter


    argumentCounter = dict(Counter(argumentList))
    sampleTableResult[rels[i]] = argumentCounter


for rel in rels:
    print(list(pairsCountingResult[rel])[:10])
    print(list(sampleTableResult[rel])[:10])


with open("count.pkl", "wb") as fout:
    pickle.dump(pairsCountingResult, fout)


with open("sample_table.pkl", "wb") as fout:
    pickle.dump(sampleTableResult, fout)

       
       