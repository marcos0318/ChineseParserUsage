import numpy 
import pickle
import json
import numpy as np
from multiprocessing import Pool
from collections import Counter
import os


def count_rel_freq_ppmi(filename):


    enwiki_dir = "/home/data/corpora/wikipedia/ParsedChineseWiki/"
    file_path = enwiki_dir + filename


    target_dir = "/home/data/corpora/wikipedia/ParsedChineseWikiDepText/"
    relation_matrix_dir = target_dir + "relation_matrix/" + filename
    clean_text_dir = target_dir + "clean_text/" + filename + ".txt"
    relation_text_dir = target_dir + "relation_text/" + filename

    clean_text = ""

    amodText = ""
    nsubjText = ""
    dobjText = ""

  
    with open(file_path) as fin:
        data = json.load(fin)
        print("processing:", file_path)

        for sentence in data:
            tokens = sentence["tokens"]
            clean_sentence = " ".join(tokens) + "\n"
            clean_text += clean_sentence


            sentence = sentence["parsed_relations"]

            for rel in sentence:
                try:
                    if rel[1] == "amod":                      
                        amodText += (rel[0][1] + " " + rel[2][1] + "\n")
                        
                    elif rel[1] == "nsubj":
                        nsubjText += (rel[0][1] + " " + rel[2][1] + "\n")
                                          
                    elif rel[1] == "dobj":
                        dobjText += (rel[0][1] + " " + rel[2][1] + "\n")
                except:
                    continue


    with open(clean_text_dir, "w") as f_clean_text:
        f_clean_text.write(clean_text)

    with open(relation_text_dir + ".amod.txt", "w") as f_amod_text:
        f_amod_text.write(amodText)

    with open(relation_text_dir + ".dobj.txt", "w") as f_dobj_text:
        f_dobj_text.write(dobjText)

    with open(relation_text_dir + ".nsubj.txt", "w") as f_nsubj_text:
        f_nsubj_text.write(nsubjText)


   






if __name__ == "__main__":
    wikiPath = "/home/data/corpora/wikipedia/ParsedChineseWiki/"
    file_list = [fileName for fileName in os.listdir(wikiPath) if fileName.endswith(".json")]

    with Pool(processes = 25) as pool:   
        pool.map(count_rel_freq_ppmi, file_list) 


        pool.close()
        pool.join()

