import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling
        self.__vocab = dict()
        self.__V = 0
        self.__corpus_size = 0
        self.__unigram_probs = dict()
        self.__corrected_unigram = dict()
        self.__neg_samples = set()
        self.__pos_samples = set()


    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w


    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        array = line.split()
        ret = []
        for token in array:
            token = ''.join([ch for ch in list(token) if ch.isalpha()])
            print(token)
            if token != "":
                ret.append(token)
        return ret



    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        ret = []
        for left_idx in range(self.__lws, 0, -1):
            if i - left_idx > 0:
                ret.append(self.__w2i[sent[left_idx]])

        for right_idx in range(1, self.__rws + 1):
            if i + right_idx < len(sent):
                ret.append(self.__w2i[sent[right_idx]])

        return ret


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        # build voacabulary
        for line in self.text_gen():
            for word in line:
                if word not in self.__vocab:
                    self.__vocab[word] = 1
                    self.__V += 1
                else:
                    self.__vocab[word] += 1
                self.__corpus_size += 1
        print(self.__vocab)


        # build maps
        words = list(self.__vocab)
        self.__w2i = {words[idx] : idx for idx in range(self.__V)}
        self.__i2w = {idx : words[idx] for idx in range(self.__V)}

        # calculate unigram distributions
        corrected_sum = 0
        for word in self.__vocab.keys():
            self.__unigram_probs[word] = float(self.__vocab[word] / self.__corpus_size)
            corrected_sum += pow(self.__unigram_probs[word], 0.75)
        for word in self.__vocab.keys():
            self.__corrected_unigram[word] = float(pow(self.__unigram_probs[word], 0.75) / corrected_sum)

        focus = []
        context = []
        for line in self.text_gen():
            for idx in range(len(line)):
                focus.append(self.__w2i[line[idx]])
                context.append(self.get_context(line, idx))

        return focus, context


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        neg_samples = set()
        for count in range(number):
            # random_neg = np.random.choice(a=list(self.__corrected_unigram.keys()),
            #                             p=list(self.__corrected_unigram.values()))
            random_neg = random.choices(population=list(self.__corrected_unigram.keys()),
                                        weights=list(self.__corrected_unigram.values()),
                                        k=1)[0]

            while random_neg in self.__pos_samples:
                # random_neg = np.random.choice(a=list(self.__corrected_unigram.keys()),
                #                               p=list(self.__corrected_unigram.values()))
                random_neg = random.choices(population=list(self.__corrected_unigram.keys()),
                                            weights=list(self.__corrected_unigram.values()),
                                            k=1)[0]
            neg_samples.add(random_neg)
        return neg_samples


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x)
        print("Dataset contains {} datapoints".format(N))

        # REPLACE WITH YOUR RANDOM INITIALIZATION
        # self.__W = np.zeros((self.__V, self.__H))
        # self.__U = np.zeros((self.__H, self.__V))
        self.__W = np.random.randn(self.__V, self.__H)
        self.__U = np.random.randn(self.__H, self.__V)

        for ep in range(self.__epochs):
            for i in tqdm(range(N)):

                focus_word = x[i]
                context_vector = np.array(t[i])
                # print(self.__i2w[focus_word])
                # print([self.__i2w[word] for word in context_vector])

                # add all to positive samples
                for vec in context_vector:
                    self.__pos_samples.add(self.__i2w[vec])
                self.__pos_samples.add(self.__i2w[x[i]])
                focus_sum = np.zeros((self.__H, ))

                for idx in context_vector:
                    #calculate gradient for context
                    gradient_context = self.__W[focus_word] * \
                                       (self.sigmoid(np.dot(self.__U[:, idx], self.__W[focus_word]))-1)

                    #calculate gradient for focus
                    gradient_focus = self.__U[:, idx] * \
                                       (self.sigmoid(np.dot(self.__U[:, idx], self.__W[focus_word]))-1)

                    # update focus gradient sum
                    focus_sum += gradient_focus

                    # update vector
                    self.__U[:, idx] -= self.__lr * gradient_context

                    negs = self.negative_sampling(self.__nsample, i, idx)
                    for neg in negs:

                        neg_index = self.__w2i[neg]
                        gradient_neg = self.__W[focus_word] * \
                                       (1 - self.sigmoid(np.dot((-1 * self.__U[:, neg_index]), self.__W[focus_word])))

                        self.__U[:, idx] -= self.__lr * gradient_neg

                        gradient_focus_neg = self.__U[:, neg_index] * \
                                         (self.sigmoid(np.dot(self.__U[:, neg_index], self.__W[focus_word])))

                        focus_sum += gradient_focus_neg

                self.__W[focus_word] -= self.__lr * focus_sum
                self.__pos_samples.clear()
                pass


    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        if not words:
            return [None]
        print(words)
        print(self.__w2i[words[0]])
        print(len(self.__W))
        input = [self.__W[self.__w2i[word]] for word in words]
        if len(input) == 0:
            return [None]

        index_mapping = {}
        samples = []
        idx = 0
        for word in self.__vocab.keys():
            samples.append(self.__W[self.__w2i[word]])
            index_mapping[idx] = word
            idx += 1

        # create scikit-learn classifier
        net = NearestNeighbors(metric=metric, n_neighbors=k)
        net.fit(samples)

        ret = []
        distances, indices = net.kneighbors(X=input, return_distance=True)
        for i in range(0, len(indices)):
            ret.append([(index_mapping[indices[i][j]], distances[i][j]) for j in range(0, len(indices[i]))])
        return ret


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w):
                    f.write(str(w) + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")


    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except:
            print("Error: failing to load the model to the file")
        return w2v


    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
