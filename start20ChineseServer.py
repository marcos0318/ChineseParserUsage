import os 
from multiprocessing import Pool

def startOnPort(port):
    os.system(""" java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port """+  str(port)  + """ -timeout 15000  """)


if __name__ == "__main__":
    ports = list(range(10000, 10020))
    with Pool(processes=21) as pool:
        pool.map(startOnPort, ports)
