from tensorflow.keras.preprocessing.text import Tokenizer
import string
from pathlib import Path
from tqdm import tqdm
import pickle
import os
import logging
import optparse

parser = optparse.OptionParser()

parser.add_option('-d', '--text-dataset-dir',
    action="store", dest="dataset_text_dir",
    help="Directory of the text dataset created with apk-parser, type opcodes", default="../dataset_text/")
parser.add_option('-t', '--type',
    action="store", dest="dataset_type",
    help="Type of dataset to build (words_count, tfidf)", default="tfidf")

options, args = parser.parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


data_folder_name = options.dataset_text_dir

notewhorty_words_set = pickle.load( open( "notewhorty_words.p", "rb" ) )
notewhorty_words = sorted(list(notewhorty_words_set))
notewhorty_words_indexes = {}
for i, word in enumerate(notewhorty_words):
    notewhorty_words_indexes[word] = i


def increment_word(dict, key):
    if key not in dict:
        dict[key] = 1
    else:
        dict[key] += 1

def create_words_count(dataset_name):

    total = 0
    for path in Path(data_folder_name).glob('**/*.txt'):
        total += 1

    file = open(dataset_name + ".pv", "w")

    with tqdm(total=total) as pbar:
        for path in Path(data_folder_name).glob('**/*.txt'):
            document = [0 for _ in notewhorty_words]
            f = open(str(path), "r")
            text = f.read()
            f.close()
            file.write(str(path))
            file.write(" ")
            text = text.lower()
            text = ''.join(c for c in text if c not in string.punctuation)
            words = text.split()
            for word in words:
                if word in notewhorty_words_set:
                    document[notewhorty_words_indexes[word]] += 1
            for word in document:
                file.write(str(word))
                file.write(" ")
            for target in get_target(str(path)):
                file.write(str(target))
                file.write(" ")
            file.write('\n')
            pbar.update(1)


def create_tfidf(dataset_name):

    total = 0
    for path in Path(data_folder_name).glob('**/*.txt'):
        total += 1

    file = open(dataset_name + ".pv", "w")

    tokenizer_obj = Tokenizer()
    #tokenizer_obj.fit_on_texts(notewhorty_words_set)

    logging.info("FITTING WORDS")
    with tqdm(total=total) as pbar:
        for path in Path(data_folder_name).glob('**/*.txt'):
            document = [0 for _ in notewhorty_words]
            f = open(str(path), "r")
            text = f.read()
            f.close()
            text = text.lower()
            text = ''.join(c for c in text if c not in string.punctuation)
            words = text.split()
            new_text = []
            for word in words:
                if word in notewhorty_words_set:
                    new_text.append(word)
            tokenizer_obj.fit_on_texts(new_text)
            pbar.update(1)

    logging.info("COMPUTE TF-IDF WORDS")
    with tqdm(total=total) as pbar:
        for path in Path(data_folder_name).glob('**/*.txt'):
            document = [0 for _ in notewhorty_words]
            f = open(str(path), "r")
            text = f.read()
            f.close()
            file.write(str(path))
            file.write(" ")
            text = text.lower()
            text = ''.join(c for c in text if c not in string.punctuation)
            words = text.split()
            new_text = []
            for word in words:
                if word in notewhorty_words_set:
                    new_text.append(word)
            sequence = tokenizer_obj.texts_to_sequences([new_text])
            document = tokenizer_obj.sequences_to_matrix(sequence, mode='tfidf')
            for word in document[0]:
                file.write(str(word))
                file.write(" ")
            for target in get_target(str(path)):
                file.write(str(target))
                file.write(" ")
            file.write('\n')
            pbar.update(1)


def get_target(p):
    result = [0,0,0,0]
    if 'TRIVIAL' in p:
        result[0] = 1
    if 'STRING_ENCRY' in p:
        result[1] = 1
    if 'REFLECTION' in p:
        result[2] = 1
    if 'CLASS_ENCRYPTION' in p:
        result[3] = 1
    return result


if __name__ == '__main__':
    if options.dataset_type == 'tfidf':
        create_tfidf("dataset_tfidf")
    elif options.dataset_type == 'words_count':
        create_words_count("dataset_words_count")
