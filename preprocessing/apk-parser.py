from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.androconf import show_logging
import logging
import glob
import re
import os
import pickle
import string
import math
from tqdm import tqdm
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from scipy import stats
import optparse

import sys
tmp = sys.path
sys.path.append("../")
from common import get_target
sys.path.append(tmp)

parser = optparse.OptionParser()

parser.add_option('-s', '--source-dataset',
    action="store", dest="dataset_dir",
    help="Directory of the dataset to process", default="../dataset/")
parser.add_option('-d', '--text-dataset-dest',
    action="store", dest="dataset_dest",
    help="Destination of the text dataset that will be created (csv file for androdet, directory for opcodes)", default="../dataset_text/")
parser.add_option('-t', '--dataset-type',
    action="store", dest="dataset_type",
    help="Type of dataset to build (androdet_IR, androdet_SE, opcodes)", default="opcodes")

options, args = parser.parse_args()


show_logging(level=logging.ERROR)
logging.basicConfig(level=logging.ERROR)


features_IR = ['Avg_Wordsize_Flds',
            'Avg_Distances_Flds',
            'Num_Flds_L1',
            'Num_Flds_L2',
            'Num_Flds_L3',
            'Avg_Wordsize_Mtds',
            'Avg_Distances_Mtds',
            'Num_Mtds_L1',
            'Num_Mtds_L2',
            'Num_Mtds_L3',
            'Avg_Wordsize_Cls',
            'Avg_Distances_Cls',
            'Num_Cls_L1',
            'Num_Cls_L2',
            'Num_Cls_L3']


features_SE = ['Avg_Entropy',
            'Avg_Wordsize',
            'Avg_Length',
            'Avg_Num_Equals',
            'Avg_Num_Dashes',
            'Avg_Num_Slashes',
            'Avg_Num_Pluses',
            'Avg_Sum_RepChars']


def get_properties_IR(path):
    """
    Get a list of custom properties
    :param path: the path to the
    :rtype: string
    """
    a = APK(path)
    d = DalvikVMFormat(a)
    properties = np.array([])
    classes = {'total': 0, 'total_characters': 0, 'total_distance': 0, 'l1': 0, 'l2': 0, 'l3': 0, 'last_id': None}
    methods = {'total': 0, 'total_characters': 0, 'total_distance': 0, 'l1': 0, 'l2': 0, 'l3': 0, 'last_id': None}
    fields = {'total': 0, 'total_characters': 0, 'total_distance': 0, 'l1': 0, 'l2': 0, 'l3': 0, 'last_id': None}
    for c in d.get_classes():
        #print(c.get_name())
        count_indentifier(classes, c.get_name())
        for m in c.get_methods():
            #print(m.get_name())
            count_indentifier(methods, m.get_name())
        for f in c.get_fields():
            #print(f.get_name())
            count_indentifier(fields, f.get_name())
    properties = np.append(properties, fields['total_characters'] / fields['total'] if fields['total'] != 0 else 0)
    properties = np.append(properties, fields['total_distance'] / fields['total'] if fields['total'] != 0 else 0)
    properties = np.append(properties, fields['l1'])
    properties = np.append(properties, fields['l2'])
    properties = np.append(properties, fields['l3'])
    properties = np.append(properties, methods['total_characters'] / methods['total'] if methods['total'] != 0 else 0)
    properties = np.append(properties, methods['total_distance'] / methods['total'] if methods['total'] != 0 else 0)
    properties = np.append(properties, methods['l1'])
    properties = np.append(properties, methods['l2'])
    properties = np.append(properties, methods['l3'])
    properties = np.append(properties, classes['total_characters'] / classes['total'] if classes['total'] != 0 else 0)
    properties = np.append(properties, classes['total_distance'] / classes['total'] if classes['total'] != 0 else 0)
    properties = np.append(properties, classes['l1'])
    properties = np.append(properties, classes['l2'])
    properties = np.append(properties, classes['l3'])
    return properties


def count_indentifier(resource, identifier):
    resource['total'] += 1
    resource['total_characters'] += len(identifier)
    if resource['last_id'] is not None:
        resource['total_distance'] += SequenceMatcher(None, identifier, resource['last_id']).ratio()
    resource['last_id'] = identifier
    length = len(identifier)
    if length == 1:
        resource['l1'] += 1
    elif length == 2:
        resource['l2'] += 1
    elif length == 3:
        resource['l3'] += 1


def get_properties_SE(path):
    """
    Get a list of custom properties for String Encryption
    :param path: the path to the
    :rtype: string
    """
    a = APK(path)
    d = DalvikVMFormat(a)
    properties = np.array([])
    total = 0
    entropy = 0
    words = {'total': 0, 'size': 0}
    length = 0
    symbols = {'equals': 0, 'dashes': 0, 'slashes': 0, 'pluses': 0}
    rep_chars = 0
    for s in d.get_strings():
        total += 1
        ent = stats.entropy(list(map(ord, list(s))))
        if math.isnan(ent):
            print("-"*100)
            print(s)
            print("-"*100)
            ent = 0
        entropy += ent
        for word in s.split(" "):
            words['total'] += 1
            words['size'] += len(word)
        length += len(s)
        symbols['equals'] += s.count('=')
        symbols['dashes'] += s.count('/')
        symbols['slashes'] += s.count('-')
        symbols['pluses'] += s.count('+')
        rep_chars += count_repetitive_characters(s)
    properties = np.append(properties, entropy / total)
    properties = np.append(properties, words['size'] / words['total'])
    properties = np.append(properties, length / total)
    properties = np.append(properties, symbols['equals'] / total)
    properties = np.append(properties, symbols['dashes'] / total)
    properties = np.append(properties, symbols['slashes'] / total)
    properties = np.append(properties, symbols['pluses'] / total)
    properties = np.append(properties, rep_chars / total)
    return properties


def count_repetitive_characters(str):
    l = list(str)
    counter = 0
    for w1, w2 in zip(l[:-1], l[1:]):
        if w1 == w2:
            counter += 1
    return counter


def parse_apk(path):
    """
    Parse an apk file to my custom bytecode output
    :param path: the path to the
    :rtype: string
    """
    # Load our example APK
    a = APK(path)
    # Create DalvikVMFormat Object
    d = DalvikVMFormat(a)

    return parse_dalvik(d)


def parse_dalvik(d):
    """
    Parse dalvik object to my custom bytecode output
    :param d: the DalvikVMFormat Object
    :rtype: string
    """
    notewhorty_words = pickle.load( open( "notewhorty_words.p", "rb" ) )
    body = ""
    for c in d.get_classes():
        #body += "Cls " + c.get_name() + "\n"
        body += "Class\n"
        for m in c.get_methods():
            m.get_name()
            #body += "Met " + m.get_name() + "\n"
            body += "Method\n"
            for i in m.get_instructions():
                #myPraGuard:----  body += i.get_name() + "\t" + i.get_output() + "\n"

                #simplePraGuard:----
                inst = i.get_name()
                inst = ''.join(c for c in inst if c not in string.punctuation) #remove punctuation
                body += inst + " "

                #dryPraGuard:------
                #inst = i.get_name()
                #inst = ''.join(c for c in inst if c not in string.punctuation) #remove punctuation
                #if inst in notewhorty_words:
                #    body += inst + " "

            body += '\n'
    return body


def create_dataset(root_dir, new_root_dir):

    total = 0
    for _ in glob.iglob(root_dir + '**/*.apk', recursive=True):
        total += 1

    already_done = []
    for apk_file in glob.iglob(new_root_dir + '**/*.txt', recursive=True):
        m = re.match(new_root_dir + "(.*)\/(.+).txt", apk_file)
        path = m.group(1)
        filename = m.group(2)
        already_done.append(path + filename)


    with tqdm(total=total) as pbar:
        for apk_file in glob.iglob(root_dir + '**/*.apk', recursive=True):
            try:
                m = re.match(root_dir + "(.*)\/(.+).apk", apk_file)
                path = m.group(1)
                filename = m.group(2)
                if path + filename not in already_done:
                    if not os.path.exists(new_root_dir + path):
                        os.makedirs(new_root_dir + path)

                    file = open(new_root_dir + path + "/" + filename + ".txt", "w")
                    body = parse_apk(apk_file)
                    file.write(body)
                    file.close()
            except Exception as e:
                logging.error('Failed: ' + apk_file + "\t" + str(e))
            pbar.update(1)


def create_dataset_androguard_IR(root_dir, new_dataset):

    total = 0
    for _ in glob.iglob(root_dir + '**/*.apk', recursive=True):
        total += 1

    data = np.empty((0,len(features_IR) + 4 + 1))

    with tqdm(total=total) as pbar:
        for apk_file in glob.iglob(root_dir + '**/*.apk', recursive=True):
            try:
                properties = get_properties_IR(apk_file)
                #print(np.append(properties, get_target(apk_file)))
                data = np.append(data, [np.append([apk_file], np.append(properties, get_target(apk_file)))], axis=0)
                #data = np.append(data, [np.append([apk_file], np.append(properties, [1,0,0,0]))], axis=0)  #fixed target
                pbar.update(1)
            except Exception as e:
                logging.error('Failed: ' + apk_file + "\t" + str(e))

    targets = ['trivial', 'string', 'reflection', 'class']
    df = pd.DataFrame(data=data, columns=['filename']+features_IR+targets)
    df.to_csv(new_dataset)


def create_dataset_androguard_SE(root_dir, new_dataset):

    total = 0
    for _ in glob.iglob(root_dir + '**/*.apk', recursive=True):
        total += 1

    data = np.empty((0,len(features_SE) + 4 + 1))

    with tqdm(total=total) as pbar:
        for apk_file in glob.iglob(root_dir + '**/*.apk', recursive=True):
            try:
                properties = get_properties_SE(apk_file)
                data = np.append(data, [np.append([apk_file], np.append(properties, get_target(apk_file)))], axis=0)
                #data = np.append(data, [np.append([apk_file], np.append(properties, [0,1,0,0]))], axis=0)  #fixed target
                pbar.update(1)
            except Exception as e:
                logging.error('Failed: ' + apk_file + "\t" + str(e))

    targets = ['trivial', 'string', 'reflection', 'class']
    df = pd.DataFrame(data=data, columns=['filename']+features_SE+targets)
    df.to_csv(new_dataset)





if __name__ == '__main__':
    if options.dataset_type == 'opcodes':
        create_dataset(
            root_dir=options.dataset_dir,
            new_root_dir=options.dataset_dest
        )
    elif options.dataset_type == 'androdet_IR':
        create_dataset_androguard_IR(
            root_dir=options.dataset_dir,
            new_dataset=options.dataset_dest
        )
    elif options.dataset_type == 'androdet_SE':
        create_dataset_androguard_IR(
            root_dir=options.dataset_dir,
            new_dataset=options.dataset_dest
        )
