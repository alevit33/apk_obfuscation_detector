from PIL import Image
from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.androconf import show_logging
from scipy.stats import entropy
import collections
import logging
import glob
import re
import os
import pickle
import string
import math
from tqdm import tqdm
import numpy as np
import optparse

parser = optparse.OptionParser()

parser.add_option('-s', '--source-dataset',
    action="store", dest="dataset_dir",
    help="Directory of the dataset to process", default="../dataset/")

options, args = parser.parse_args()


show_logging(level=logging.ERROR)
logging.basicConfig(level=logging.ERROR)


root_dir = options.dataset_dir
dataset_name = "dataset_entropy"


def parse_apk(path):
    """
    Parse an apk file to my custom bytecode output
    :param path: the path to the
    :rtype: string
    """
    # Load our example APK
    a = APK(path)
    # Create DalvikVMFormat Object
    #d = DalvikVMFormat(a)
    return a.get_dex()


def parse_img(path):
    image = Image.open(path, 'r')
    return np.asarray(image).reshape((-1, 3))


def compute_entropy(path):
    data = parse_apk(path)
    tot = len(data)

    probabilities = np.zeros((256, ), dtype=np.float)

    for (byte, times) in collections.Counter(data).items():
        probabilities[byte] = times / tot

    return [entropy(probabilities)]


def prepare_dataset():
    """
    compute the entropy for every app
    """

    total = 0
    for _ in glob.iglob(root_dir + '**/*.apk', recursive=True):
        total += 1

    file = open(dataset_name + ".csv", "w")

    with tqdm(total=total) as pbar:
        for apk_file in glob.iglob(root_dir + '**/*.apk', recursive=True):
            try:
                m = re.match(root_dir + "(.*)\/(.+).apk", apk_file)
                path = m.group(1)
                filename = m.group(2)
                file.write(str(os.path.join(path, filename)))
                file.write(" ")
                entropy = compute_entropy(apk_file)
                for e in entropy:
                    file.write(str(e))
                    file.write(" ")
                file.write('\n')
            except Exception as e:
                logging.error('Failed: ' + apk_file + "\t" + str(e))
            pbar.update(1)



if __name__ == '__main__':
    prepare_dataset()

