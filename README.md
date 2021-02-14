# Obfuscation detection for Android applications

This repository holds the source code made for the research "Obfuscation Detection in Android Applications using Deep Learning".

All the scripts can be executed with python3. You will find the parameters' description with the help command `python <script_name.py> -h`

## Pre-processing

`apk-parser.py` parse the apk dataset, extracting the features for androdet of the opcodes for NLP

`count_words.py` parse the opcodes dataset (created in the previous step) to create the Bag of words or TF-IDF vectors

`create_entropy_dataset.py` parse the apk dataset and compute the entropy value for each application

`create_images.py` parse the apk dataset and compute the dataset of images

## New_andordet

This is the model Androdet*. It detects Identifier renaming from dataset created with `apk-parser`

`androdet.py` Run the model. You can train a new model with the dataset, or just predict the result with the saved model.

`main.py` Run the hyper-parameters tuning with genetic search

## BOW

This is the "bag of word" model. It detects  [Trivial obfuscation, String encryption, Reflection, Class encryption] from dataset created with `count_words`

`bow.py` Run the model. You can train a new model with the dataset, or just predict the result with the saved model.

`main.py` Run the hyper-parameters tuning with genetic search

## CNN

This is the "image processing with Convolutional neural network" model. It detects  [Trivial obfuscation, String encryption, Reflection, Class encryption] from dataset created with `create_images`

`cnn.py` Run the model. You can train a new model with the dataset, or just predict the result with the saved model.

`evolution_search.py` Run the hyper-parameters tuning with genetic search

## Hybrid

This is the "hybrid" model that combines all the previous one. It detects  [Trivial obfuscation, String encryption, Reflection, Class encryption]

`hybrid.py` Run the model. You can train a new model with the dataset (only part of the network after the merge_layer is trainable), or just predict the result with the saved model.

`evolution_search.py` Run the hyper-parameters tuning with genetic search