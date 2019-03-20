import os
import ujson as json

from multiprocessing import Pool
import subprocess


# Use subprocess to catch stdout of program
#  output = subprocess.check_output('ping localhost', stderr=subprocess.STDOUT, shell=True)
valid_chars = set("""qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890`~!@#$%^&*/?., ;:"'""")

def clean_sentence_for_parsing(input_sentence):
    new_sentence = ''
    for char in input_sentence:
        if char in valid_chars:
            new_sentence += char
        else:
            new_sentence += '\n'
    return new_sentence

def parse_sentense_with_stanford(input_sentence, nlp_id=0):
    # Cannot clean for Chinese Charactor
    cleaned_sentence = input_sentence

    # '我喜欢吃美味的寿司，不喜欢吃难吃的炸酱面' 

    # TODO: Replace the tmp output
    request = """wget --post-data '""" + cleaned_sentence + """' 'localhost:10015/?properties={"annotators":"tokenize,depparse,lemma","outputFormat":"json"}' -O - """

    tmp_output = subprocess.check_output(request, stderr=subprocess.STDOUT, shell=True)

    # tmp_output = nlp.annotate(cleaned_sentence,
    #                           properties={'annotators': 'tokenize,depparse,lemma', 'outputFormat': 'json'})


    parsed_examples = list()
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        stored_dependency_list = list()
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_position = relation['governor']
            dependent_position = relation['dependent']
            stored_dependency_list.append(((governor_position, s['tokens'][governor_position - 1]['lemma'],
                                            s['tokens'][governor_position - 1]['pos']), relation['dep'], (
                                               dependent_position, s['tokens'][dependent_position - 1]['lemma'],
                                               s['tokens'][dependent_position - 1]['pos'])))
        tokens = list()
        for token in s['tokens']:
            tokens.append(token['word'])
        parsed_examples.append(
            {'parsed_relations': stored_dependency_list, 'sentence': input_sentence, 'tokens': tokens})
    return parsed_examples




def parse_sentense(folder_name, nlp_id):
    """
    make sure that the last element in counters satisfies x%10000=0
    """
    file_names = os.listdir('/data/xliucr/sActivityNet/Wikipedia/enwiki-20180920-pages-articles/'+folder_name)
    print('We are working on folder:', folder_name)
    print('Number of files:', len(file_names))
    for file_name in file_names:
        all_parsed_result = []
        full_file_name = '/data/xliucr/sActivityNet/Wikipedia/enwiki-20180920-pages-articles/'+folder_name + '/'+file_name
        sentences = list()
        with open(full_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                sentences.append(line)
        for sentence in sentences:
            if len(sentence) < 3 or '<doc' in sentence[:10] or '</doc>' in sentence[:10]:
                continue
            try:
                parsed_result = parse_sentense_with_stanford(sentence, nlp_id)
            except TypeError:
                continue
            for sub_sentence_result in parsed_result:
                all_parsed_result.append(sub_sentence_result)

        # print('We are storing parsing result for', counter, 'sentences')
        file_name = '/data/xliucr/sActivityNet/WIKI/Parsed_data/' + folder_name+'_' + file_name + '.json'
        file = open(file_name, 'w')
        json.dump(all_parsed_result, file)
        file.close()
        print(folder_name, 'finished')


# all_parsed_result = list()
# counter = 0

# folder_names = os.listdir('/data/xliucr/sActivityNet/Wikipedia/enwiki-20180920-pages-articles')

# pool = Pool(20)
# for i, fo_name in enumerate(folder_names):
#     pool.apply_async(parse_sentense, args=(fo_name, i % 15))
# pool.close()
# pool.join()

# # parse_sentense(folder_names[0], 0)

# print('end')

if __name__ == "__main__":
    print(parse_sentense_with_stanford('我喜欢吃美味的寿司，不喜欢吃难吃的炸酱面'))