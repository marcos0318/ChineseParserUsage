import gensim

class MySentences(object):
    def __init__(self, fileName):
        self.fileName = fileName
 
    def __iter__(self):
        for line in open(self.fileName, "r"):
            yield line.split()

target_dir = "/home/data/corpora/MultiRelChineseEmbeddings/"
all_text = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/relationalText.txt"

all_sentences = MySentences(all_text) # a memory-friendly iterator


all_model = gensim.models.Word2Vec(all_sentences, size=300, min_count=3, workers=30)
all_model.save(target_dir + "chinese_multirelation.model")