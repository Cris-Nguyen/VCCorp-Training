### Module imports ###
import os
from collections import defaultdict


### Global Variables ###


### Class declarations ###
class ExtractData:

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        # list to store tuples of sentences and its tag
        self.all_tuples = []

        # dictionary to store the word and all tags of that word
        self.word_tags = defaultdict(set)

        # set to store the unique tags
        self.unique_tags = set()

    def word_tag_tuples(self):
        with open(self.train_path, 'r') as f:
            for line in f:
                # list of tuples for a sentence
                tuples_sentence_l = []
                tuple_words = line.strip('\n').split()

                # add '*' at the start of a sentence
                tuples_sentence_l.append(('*', '*'))
                tuples_sentence_l.append(('*', '*'))

                # word tag
                for tup in tuple_words:
                    split = tup.split('/')
                    if len(split) < 2:
                        continue

                    split_word = '/'.join(split[:-1])
                    split_tag = split[-1]

                    # saving the unique tags
                    if split_tag not in self.unique_tags:
                        self.unique_tags.add(split_tag)

                    # add word-tag pair to list
                    tuples_sentence_l.append((split_word, split_tag))
                    # add all tags of a word to dictionary
                    self.word_tags[split_word].add(split_tag)

                # add sentence to all tuples list
                self.all_tuples.append(tuples_sentence_l)
        # add tag * to word tag
        self.word_tags['*'] = '*'
        return self.all_tuples, self.word_tags, self.unique_tags

    def word_tag_test(self):
        sentence_l, tags_l = [], []
        with open(self.test_path, 'r') as f:
            for line in f:
                # list of tuples for a sentence
                words_l, tag_sent = ['*', '*'], ['*', '*']
                tuple_words = line.strip('\n').split()

                # word tag
                for tup in tuple_words:
                    split = tup.split('/')
                    split_word = '/'.join(split[:-1])
                    split_tag = split[-1]
                    words_l.append(split_word)
                    tag_sent.append(split_tag)
                sentence_l.append(words_l)
                tags_l.append(tag_sent)
        return sentence_l, tags_l


### Function declarations ###
