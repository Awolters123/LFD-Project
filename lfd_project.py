#!/usr/bin/env python

''''
This python program lets you run three type of models:
SVM baseline: Running the support vector machine (SVM) model with a predefined vectorizer and parameters.
SVM finetuned: Running the SVM model with a Gridsearcv, which finds the optimal vectorizer and parameters
LSTM: Running a Long short-term memory (LSTM) model, where you can specify the parameters: learn rate, batch size, and
maximum sequence length
Transformer: Running three different pretrained language model: BERT (bert-base-uncased), RoBERTa (roberta-base),
and DeBERTa (microsoft/deberta-v3-base), and specifying the learn rate, batch size, epochs, maximum sequence length.

Usage:
python3 lfd_assignment3.py -i -d -t -e -svm_base -svm_opt -lstm -tf -lr -bs -sl -epoch
Note: Please see the create_arg_parse function for a detailed description of each argument!

Example:
Running bert model with default parameters on standard dataset
python3 lfd_assignment3.py -i "train.tsv" -d "dev.tsv" -t "test.tsv" -tf bert
'''

# Importing libraries
import argparse
import os
import pandas as pd
import random as python_random
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import json
import pprint
import emoji
import re
import warnings
warnings.filterwarnings("ignore")
import tensorflow
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Embedding, LSTM, Bidirectional
from keras.initializers import Constant
from sklearn.metrics import accuracy_score,  classification_report
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Ftrl
from tensorflow.keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
# Make reproducible as much as possible
np.random.seed(1234)
tensorflow.random.set_seed(1234)
python_random.seed(1234)


#############
# Helper functions, (e.g. create arguments, reading data/embeddings, plotting confusion matrix)
def create_arg_parser():
    '''This function creates all command arguments, for data input, model selection, and custom parameters,
    please see the help section for a detailed description'''
    parser = argparse.ArgumentParser()
    # Data input arguments
    parser.add_argument("-i", "--train_file", default="train.tsv", type=str,
                        help="Training the model with this data file (default train.tsv)")
    parser.add_argument("-d", "--dev_file", default="dev.tsv", type=str,
                        help="Validating the model with this data file (default dev.tsv)")
    parser.add_argument("-t", "--test_file", default="test.tsv", type=str,
                        help="Evaluating the model with this data file (default test.tsv)")
    parser.add_argument("-e", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file used as embedding layer for LSTM (default glove_reviews.json)")

    # Model arguments
    parser.add_argument("-svm_base", "--svm_baseline", action='store_true',
                        help="This argument runs the SVM baseline with default vectorizer (TFIDF) and "
                             "default parameters (svm.SVC(), cannot be used with parameter arguments")
    parser.add_argument("-svm_opt", "--svm_optimised", action='store_true',
                        help="This argument runs the SVM with gridsearchCV to automatically find the best features"
                             "and parameters, cannot be used with parameter arguments")
    parser.add_argument("-lstm", "--lstm", action='store_true',
                        help="This argument runs the lstm model, you can specify the parameters: learning rate, "
                             "batch size, and sequence length")
    parser.add_argument("-tf", "--transformer", default="bert", type=str,
                        help="This argument runs the pretrained language models, you can choose between: "
                             "bert (bert-base-uncased), roberta (robert-base), and deberta (microsoft/deberta-v3-base),"
                             "you can specify the parameters: learn rate, batch size, sequence length, and epochs"
                             "Note: the default model the script will run is bert (bert-base-uncased)")

    # Parameter arguments
    parser.add_argument("-lr", "--learn_rate", default=5e-5, type=float,
                        help="Set a custom learn rate for the LSTM or transformer model, default is 5e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for the LSTM or transformer model default is 8")
    parser.add_argument("-sl", "--sequence_length", default=100, type=int,
                        help="Set a custom maximum sequence length for the LSTM or transformer model default is 100")
    parser.add_argument("-epoch", "--epochs", default=1, type=int,
                        help="This argument selects the amount of epochs to run the model with, default is 1 epoch")

    args = parser.parse_args()
    return args


def heatmap(cf, color):
    """Create heatmap confusion matrix"""
    labels = ['Not Offensive', 'Offensive']
    sns.heatmap(cf, annot=True, fmt='g', cmap=color, xticklabels=labels, yticklabels=labels)
    plot.xlabel("Predicted Label")
    plot.ylabel("True Label")
    plot.show()


def read_corpus(corpus_file):
    '''Reads in the tsv file and returns separate text and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            # Selecting all texts
            documents.append(" ".join(tokens.split()[:-1]).strip())
            # binary problem: offensive or not offensive
            labels.append(tokens.split()[-1])
    return documents, labels


def read_csv(corpus_file):
    '''Reads in the csv file and returns separate text and labels'''
    data = pd.read_csv(corpus_file)
    documents = data["text"].values.tolist()
    labels = data['label'].values.tolist()
    return documents, labels


def preprocessing(text):
    '''Preprocessing the text, by converting emojis to their textual value'''
    documents = []
    for line in text:
        line = emoji.demojize(line)
        documents.append(line)
    return documents


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


#############
# SVM baseline part
def svm_baseline(X_train, Y_train, X_test, Y_test):
    '''This function takes as input the train, dev and test set with their label sets.
    It trains a linear SVM model with a TF-IDF vectorizer used as baseline on the training data and returns predictions on the dev and test set'''
    print("Running SVM baseline...")
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training, dev and test data
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Get the unique labels from the training data
    unique_labels = list(set(Y_train))

    # Create the model
    cls = SVC()

    # Train the model on texts and labels
    cls.fit(X_train, Y_train)

    # Predict the labels of the test set
    pred_test = cls.predict(X_test)

    # Present classification report for dev set with precison, recall and F1 per label
    print(classification_report(Y_test, pred_test, target_names=unique_labels, digits=3))
    print(confusion_matrix(Y_test, pred_test))
    heatmap(confusion_matrix(Y_test, pred_test), "Blues")


#############
# SVM optimised part
def get_best_parameter_settings(X_train, Y_train, X_dev, Y_dev):
    '''This function takes as input the train, dev and test set with their label sets.
    It trains a SVM model with a TF-IDF vectorizer with different parameter settings.
    The model will be trained and evaluated on the dev set. Best score and best parameter settings will be returned'''
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training and dev data
    X_train = vectorizer.fit_transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    best_score = 0

    for gamma in [1]:  # All the feature values experimented with: for gamma in [0.001, 0.01, 1, 10]
        for C in [0.01]:  # All the feature values experimented with: for C in [0.001, 0.1, 1, 10]
            for kernel in ['linear']:  # All the feature values experimented with: for kernel in ['linear', 'rbf']
                # Train the SVC
                svm = SVC(gamma=gamma, C=C, kernel=kernel)
                svm.fit(X_train, Y_train)

                # Evaluate the set
                score = svm.score(X_dev, Y_dev)
                print("Gamma value: {}, C value: {}, Kernel: {}, Score: {}".format(gamma, C, kernel, score))

                # If we got a better score, store the score and parameters
                if score > best_score:
                    best_score = score
                    best_parameters = {'C': C, 'gamma': gamma, 'kernel': kernel}

    print("Best score: {:.3f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))
    return best_score, best_parameters


def get_best_feature_settings(X_train, Y_train, X_dev, Y_dev, X_test):
    '''This function takes as input the train, dev and test set with their label sets.
    It trains a linear SVM model, with gamma=0.01 and a C=1 TF-IDF vectorizer with different combinations features and
    evaluates on a dev set. It will return the model, vectorized dev data for evaluation, best score and
    the best parameter settings'''
    best_score = 0

    for ngram_range in [(3,5)]: # All the features used: for ngram_range in
        # [(1,1), (2,2), (3,3), (4,4), (5,5), (1,2), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
        for analyzer in ['char']: # All the features used: for analyzer in ['char', 'word']
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)

            # Vectorize the training, dev and test data
            X_train_vec = vectorizer.fit_transform(X_train)
            X_dev_vec = vectorizer.transform(X_dev)
            X_test_vec = vectorizer.transform(X_test)

            # Train the SVC
            svm = SVC(gamma=0.01, C=1, kernel='linear')
            svm.fit(X_train_vec, Y_train)

            # Evaluate the set
            score = svm.score(X_dev_vec, Y_dev)
            print("Ngram range value: {}, Analyzer value: {}, Score: {}".format(ngram_range, analyzer, score))

            # If we got a better score, store the score and parameters
            if score > best_score:
                best_score = score
                best_parameters = {'ngram_range': ngram_range, 'analyzer': analyzer}

    print("Best score: {:.3f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))
    return svm, best_score, best_parameters, X_test_vec


def svm_optimized(X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
    '''This function takes as input the train, dev and test set with their label sets.
    It trains a SVM model with a TF-IDF vectorizer with optimized settings on the training data and returns predictions
    on the dev and test set'''
    print("Running optimised SVM\nSearching for best settings...")
    # Get the best parameter settings
    best_parameter_score, best_parameters = get_best_parameter_settings(X_train, Y_train, X_dev, Y_dev)

    # Get the best feature settings
    svm, best_feature_score, best_features, X_test_vec = get_best_feature_settings(X_train, Y_train, X_dev, Y_dev,
                                                                                   X_test)
    # Make predictions
    predictions = svm.predict(X_test_vec)

    # Print classification report
    print(classification_report(Y_test, predictions, digits=3))
    print(confusion_matrix(Y_test, predictions))
    heatmap(confusion_matrix(Y_test, predictions), "Blues")


#############
# LSTM part
def create_lstm(lr, emb_matrix):
    '''Create the Keras model to use'''
    print("Creating LSTM\nWith learn rate {}".format(lr))

    # Define settings
    loss_function = 'binary_crossentropy'
    optim = Adamax(learning_rate=lr)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = 2

    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=True))

    # Multiple LSTM layers
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(LSTM(embedding_dim, dropout=0.1))

    # Ultimately, end with dense layer with softmax
    model.add(Dense(input_dim=embedding_dim, units=num_labels, activation="sigmoid"))

    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model


def train_lstm(model, bs, X_train, Y_train):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    print("and batch size: {}".format(bs))
    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # Fit the model to our data
    model.fit(X_train, Y_train, verbose=1, epochs=10, callbacks=[callback], batch_size=int(bs))
    return model


def test_lstm(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    print("Testing LSTM on test set")
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)

    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)

    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print("Classification Report:\n", classification_report(Y_test, Y_pred, digits=3))
    print(confusion_matrix(Y_test, Y_pred))
    heatmap(confusion_matrix(Y_test, Y_pred), "Blues")


def lstm(lr, bs, sl, X_train, Y_train, X_test, Y_test):
    '''Calling the create, train, and test functions to run the whole LSTM model'''
    # Read embeddings
    embeddings = read_embeddings("glove_reviews.json")

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=sl)

    # Use train to create vocab
    text_ds = tensorflow.data.Dataset.from_tensor_slices(X_train)
    vectorizer.adapt(text_ds)

    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_test = encoder.fit_transform(Y_test)
    Y_train_bin = np.hstack((1 - Y_train, Y_train))
    Y_test_bin = np.hstack((1 - Y_test, Y_test))

    # Create model
    model = create_lstm(lr, emb_matrix)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()

    # Train the model
    model = train_lstm(model, bs, X_train_vect, Y_train_bin)

    # Test model
    vectorizer.adapt(text_ds)

    # Predicting on the test set
    test_lstm(model, X_test_vect, Y_test_bin, "test")


#############
# Transformer part
def train_transformer(lm, epoch, bs, lr, sl, X_train, Y_train, X_dev, Y_dev):
    '''This function takes as input the train file, dev file, transformer model name, and parameters.
    It trains the model with the specified parameters and returns the trained model'''
    print("Training model: {}\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, Sequence length: {}"
          .format(lm, lr, bs, epoch, sl))
    # Selecting the correct tokenizer for the model, and selecting the model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2)

    # Tokenzing the train and dev texts
    tokens_train = tokenizer(X_train, padding=True, max_length=sl,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=sl,
                           truncation=True, return_tensors="np").data

    # Setting the loss function for binary task and optimization function
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=lr)

    # Encoding the labels with sklearns LabelBinazrizer
    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)
    Y_dev = encoder.fit_transform(Y_dev)
    Y_train_bin = np.hstack((1 - Y_train, Y_train))
    Y_dev_bin = np.hstack((1 - Y_dev, Y_dev))

    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(tokens_dev, Y_dev_bin))

    return model


def test_transformer(lm, epoch, bs, lr, sl, model, X_test, Y_test, ident):
    '''This function takes as input the trained transformer model, name of the model, parameters, and the test files,
    and predicts the labels for the test set and returns the accuracy score with a summarization of the model'''
    print("Testing model: {} on {} set\nWith parameters:\nLearn rate: {}, Batch size: {}\nEpochs: {}, "
          "Sequence length: {}".format(lm, ident, lr, bs, epoch, sl))
    # Selecting the correct tokenizer for the model, and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predicitions on the test set and converting the logits to sigmoid probabilities (binary)
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # converting gold labels with LabelBinarizer
    encoder = LabelBinarizer()
    Y_test = encoder.fit_transform(Y_test)
    Y_test_bin = np.hstack((1 - Y_test, Y_test))

    # Converting the predicitions and gold set to their original numerical label value
    pred = np.argmax(prob, axis=1)
    gold = np.argmax(Y_test_bin, axis=1)

    # Printing classification report (rounding on 3 decimals)
    print("Classification Report on {} set:\n{}".format(ident, classification_report(gold, pred, digits=3)))
    print(confusion_matrix(gold, pred))
    heatmap(confusion_matrix(gold, pred), "Blues")


##############################################################################
# initialising all functions, (e.g. helper functions, SVMs, LSTM, Transformers)
def main():
    '''Main function to train and test neural network and transformer models given command line arguments'''
    # Create the command arguments for the script
    args = create_arg_parser()

    # Creating parameter variables
    lr = args.learn_rate
    bs = args.batch_size
    sl = args.sequence_length
    ep = args.epochs

    # Reading in train/dev data:
    if args.train_file.endswith(".csv"):
        X_train, Y_train = read_csv(args.train_file)
        X_dev, Y_dev = read_csv(args.dev_file)
    else:
        X_train, Y_train = read_corpus(args.train_file)
        X_dev, Y_dev = read_corpus(args.dev_file)

    # Reading in test data
    if args.test_file.endswith(".csv"):
        X_test, Y_test = read_csv(args.test_file)
    else:
        X_test, Y_test = read_corpus(args.test_file)

    # Preprocess the text
    X_train = preprocessing(X_train)
    X_dev = preprocessing(X_dev)
    X_test = preprocessing(X_test)

    # Running the different models by checking if their matching argument is true
    if args.svm_baseline:
        svm_baseline(X_train, Y_train, X_test, Y_test)

    elif args.svm_optimised:
        svm_optimized(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

    elif args.lstm:
        lstm(lr, bs, sl, X_train, Y_train, X_test, Y_test)

    elif args.transformer:
        if args.transformer == "bert": # checking pretrained lm
            tf = train_transformer("bert-base-uncased", ep, bs, lr, sl, X_train, Y_train, X_dev, Y_dev)
            test_transformer("bert-base-uncased", ep, bs, lr, sl, tf, X_test, Y_test, "test")
        elif args.transformer == "roberta": # checking pretrained lm
            tf = train_transformer("roberta-base", ep, bs, lr, sl, X_train, Y_train, X_dev, Y_dev)
            test_transformer("roberta-base", ep, bs, lr, sl, tf, X_test, Y_test, "test")
        elif args.transformer == "deberta": # checking pretrained lm
            tf = train_transformer("microsoft/deberta-v3-base", ep, bs, lr, sl, X_train, Y_train, X_dev, Y_dev)
            test_transformer("microsoft/deberta-v3-base", ep, bs, lr, sl, tf, X_test, Y_test, "test")


if __name__ == '__main__':
    main()
