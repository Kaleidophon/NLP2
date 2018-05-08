import aer
from collections import defaultdict, Counter
import json
from math import log2
import numpy as np
from random import randint
import random
import progressbar
import pickle
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
import os


def read_corpus(file_name, source_language):
    """
    Reads the corpus and saves each sentence in a list.
    """

    corpus = []

    with open(file_name, "r", encoding='utf8') as f:
        for line in f:
            line = line.replace("\n", "")
            sentence = line.split()

            if source_language:
                sentence.insert(0, "null")
            corpus.append(sentence)
    return corpus


def reduce_corpus(corpus):
    """
    Reduces the vocabulary of the corpus by replacing each word that only
    occurs once in the vocabulary by -LOW- in the corpus.
    """

    flat_corpus = [word for sentence in corpus for word in sentence]
    word_counts = Counter(flat_corpus)
    small_corpus = []

    for sentence in corpus:
        small_sentence = []

        for word in sentence:
            if word_counts[word] != 1:
                small_sentence.append(word)
            else:
                small_sentence.append("-LOW-")
        small_corpus.append(small_sentence)
    return small_corpus


def load_parameters(file_path):
    """
    Loads the parameters that obtained the highest validation AER
    from a Pickle file.
    """

    f = open(file_path, "rb")
    parameters = pickle.load(f)
    f.close()
    return parameters


def get_best_aer():
    """
    Finds the file that had the highest AER score and returns the file path.
    """

    dir_path = os.path.dirname(os.path.realpath("__file__"))
    files = [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
    return files[0]


def initialise_parameters(source_corpus, target_corpus, method):
    """
    Initialises the conditional probability of generating a source
    word from a target word for all possible pairs of words in the source
    and target sentences to 5 and then normalises the parameters such that
    the initialisation is uniform.
    """

    if method == "uniform":
        vocabulary = set([word for sentence in source_corpus for word in sentence])
        theta0 = 1/len(vocabulary)
        return defaultdict(lambda: defaultdict(lambda: theta0))
    elif method == "random":
        theta0 = np.random.uniform(0.001, 1)
        return defaultdict(lambda: defaultdict(lambda: theta0))
    elif method == "ibm1":
        file_path = get_best_aer()
        parameters = load_parameters(file_path)
        return parameters


def get_best_alignment(source_corpus, target_corpus, parameters, q):
    """
    Gets the best alignment for each sentence and saves the alignment
    in a list of lists that holds tuples for each position in the sentence
    and looks as follows:
    (sentence_index, target_word_index, source_word_index).
    """
    alignments = []

    print("Getting alignments...")

    with progressbar.ProgressBar(max_value=len(target_corpus)) as bar:
        for n in range(len(source_corpus)):
            source_sentence = source_corpus[n]
            target_sentence = target_corpus[n]
            alignment = []
            l = len(source_sentence)
            m = len(target_sentence)

            for i, target_word in enumerate(target_sentence):
                best_prob = 0
                best_j = 0

                for j, source_word in enumerate(source_sentence):
                    # If a word does not occur in the training data, assign probability 0
                    prob = parameters[source_word].get(target_word, 0)
                    try:
                        prob = prob*q[j].get(i, 0).get(l, 0).get(m, 0)
                    except AttributeError:
                        prob = 0


                    if prob > best_prob:
                        best_prob = prob
                        best_j = j

                if best_j != 0:
                    alignment.append((n, best_j, i+1))
            alignments.append(alignment)
            bar.update(n)
    return alignments


def compute_aer(predictions, file_path):
    """
    Computes the Alignment Error Rate.
    """

    gold_sets = aer.read_naacl_alignments(file_path)
    metric = aer.AERSufficientStatistics()

    for gold, prediction in zip(gold_sets, predictions):
        prediction = set([(alignment[1], alignment[2]) for alignment in prediction])
        metric.update(sure=gold[0], probable=gold[1], predicted=prediction)
    print(metric.aer())
    return metric.aer()


def expectation_maximisation2(source_corpus, target_corpus, val_source,
                              val_target, parameters, num_iterations,
                              min_perplexity_change, model, file_path):
    """
    Runs the EM algorithm for IBM Model 2.
    """

    q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.1))))
    old_perplexity = -100000
    perplexities = []
    aers = []

    with open(file_path, "a") as f:
        for k in range(0, num_iterations):
            print("Iteration #" + str(k), "out of", num_iterations - 1)

            counts_pairs = defaultdict(lambda: defaultdict(lambda: 0))
            counts_alignments = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
            counts_single = defaultdict(lambda: 0)
            counts_pairs, counts_single, counts_alignments = e_step2(source_corpus, target_corpus,
                                                                     counts_pairs, counts_single,
                                                                     counts_alignments, q)
            parameters, q = m_step2(parameters, q, counts_alignments, counts_pairs, counts_single)
            perplexity = compute_perplexity2(parameters, q, source_corpus, target_corpus)
            alignments = get_best_alignment(val_source, val_target, parameters, q)
            val_aer = compute_aer(alignments, "validation/dev.wa.nonullalign")
            f.write("pp " + str(perplexity) + "\n")
            f.write("aer " + str(val_aer) + "\n")
            perplexities.append(perplexity)
            aers.append(val_aer)
        f.close()
    return perplexities, aers, parameters, q


def e_step2(source_corpus, target_corpus, counts_pairs, counts_single, counts_alignments, q):
    """
    Does the E-step for IBM Model 2.
    """

    print("Doing E-step...")

    with progressbar.ProgressBar(max_value=len(source_corpus)) as bar:
        for n in range(len(source_corpus)):
            source_sentence = source_corpus[n]
            target_sentence = target_corpus[n]
            l = len(source_sentence)
            m = len(target_sentence)

            for i, target_word in enumerate(target_sentence):
                delta_denominator = sum([q[j_k][i][l][m]*parameters[source_sentence[j_k]][target_word]
                                         for j_k in range(l)])

                for j, source_word in enumerate(source_sentence):
                    delta = (q[j][i][l][m]*parameters[source_word][target_word])/delta_denominator

                    counts_pairs[source_word][target_word] += delta
                    counts_single[source_word] += delta
                    counts_alignments[l][m][i][j] += delta
            bar.update(n)
    return counts_pairs, counts_single, counts_alignments


def m_step2(parameters, q, counts_alignments, counts_pairs, counts_single):
    """
    Does the M-step for IBM Model 2.
    """

    print("Doing M-step...")

    for j in q.keys():
        for i in q[j].keys():
            for l in q[j][i].keys():
                for m in q[j][i][l].keys():
                    q[j][i][l][m] = counts_alignments[l][m][i][j]/sum(counts_alignments[l][m][i].values())

    for source_word, target_words in parameters.items():
        for target_word in target_words:
            parameters[source_word][target_word] = \
                counts_pairs[source_word][target_word]/counts_single[source_word]
    return parameters, q


def compute_perplexity2(parameters, q, source_corpus, target_corpus):
    """
    Computes the perplexity of the corpus for IBM Model 2.
    """

    perplexity = 0

    print("Calculating perplexity...")

    with progressbar.ProgressBar(max_value=len(source_corpus)) as bar:
        for n in range(len(source_corpus)):
            source_sentence = source_corpus[n]
            target_sentence = target_corpus[n]
            log_sentence = 0
            l = len(source_sentence)
            m = len(target_sentence)

            for i, target_word in enumerate(target_sentence):
                log_sum = []

                for j, source_word in enumerate(source_sentence):
                    log_sum.append(parameters[source_word][target_word]*q[j][i][l][m])
                log_sentence += np.log(np.sum(log_sum))
            perplexity += log_sentence
            bar.update(n)
    print(perplexity)
    return perplexity

if __name__ == '__main__':
    train_source = read_corpus("training/hansards.36.2.e", True)
    train_source = reduce_corpus(train_source)
    train_target = read_corpus("training/hansards.36.2.f", False)
    train_target = reduce_corpus(train_target)
    val_source = read_corpus("validation/dev.e", True)
    val_target = read_corpus("validation/dev.f", False)
    test_source = read_corpus("testing/test/test.e", True)
    test_target = read_corpus("testing/test/test.f", False)

    model = "evolution"
    initial = "ibm1"
    parameters = initialise_parameters(train_source, train_target, initial)
    file_path = "ibm2_" + model + "_" + initial + ".txt"
    perplexities, aers, parameters, q = expectation_maximisation2(train_source, train_target,
                                                   val_source, val_target,
                                                   parameters, 10, 1000, model,
                                                   file_path)
    alignments = get_best_alignment(test_source, test_target, parameters, q)
    test_aer = compute_aer(alignments, "testing/answers/test.wa.nonullalign")

    with open("ibm2_results.txt", "a") as f:
        str = initial + " " + model + " " + str(test_aer) + "\n"
        f.write(str)
    f.close()
