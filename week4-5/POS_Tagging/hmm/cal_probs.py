### Module imports ###
import math
from collections import defaultdict
import pickle

from read_from_file import ExtractData


### Global Variables ###


### Class declarations ###
class CalProbs:

    def __init__(self, train_path, test_path):
        """ Probabilities attributes """
        self.train_path = train_path
        self.test_path = test_path
        # dictionary to store the number of a given word-tag tuple occurrences
        # ('phụ_nữ', 'N'): 100
        self.word_tag_count = defaultdict(int)

        # dictionary to store the number of a tag occurrences
        self.tag_count = defaultdict(int)

        # dictionary to store the number of trigram of tags occurrences
        self.trigram_tags_count = defaultdict(int)

        # dictionary to store the number of bigram of tags occurrences
        self.bigram_tags_count = defaultdict(int)

        # dictionary to store the emission probability for a given word-tag
        self.emission_probs = {}

        # dictionary to store the transition probability for a given trigram
        self.transition_probs = {}

        # dictionary to store tags for a given word
        self.word_tags_s = {}

        # set to store all unique tags
        self.unique_tags = set()

        # run to save attributes
        self.save()

    def count_dicts(self):
        """
        Calculate dictionary attributes

        Args:
            orig_folder: original folder contain all data text files
        """
        extractor = ExtractData(self.train_path, self.test_path)
        self.all_tuples, self.word_tags_s, self.unique_tags = extractor.word_tag_tuples()

        for sentence in self.all_tuples:
            for i in range(2, len(sentence)):
                self.word_tag_count[sentence[i]] += 1

                # count tag occurrences
                tag = sentence[i][1]
                self.tag_count[tag] += 1

                # make trigram and bigram by tags
                tags_trigram = (sentence[i - 2][1], sentence[i - 1][1], sentence[i][1])
                self.trigram_tags_count[tags_trigram] += 1
                tags_bigram = (sentence[i - 2][1], sentence[i - 1][1])
                self.bigram_tags_count[tags_bigram] += 1

    def cal_emission_probs(self):
        """ Calculate emission probabilities """
        for word_tag, word_tag_count in self.word_tag_count.items():
            self.emission_probs[word_tag] = math.log(float(word_tag_count) / float(self.tag_count[word_tag[1]]))

    def cal_transition_probs(self):
        """ Calculate transition probabilities """
        unique_tags_count = len(self.unique_tags)
        for trigram, trigram_count in self.trigram_tags_count.items():
            bigram_count = self.bigram_tags_count[(trigram[0], trigram[1])]
            # calculate transition probabilities with smoothing
            self.transition_probs[trigram] = math.log(float(trigram_count + 1) / float(bigram_count + unique_tags_count))

    def save(self):
        self.count_dicts()
        self.cal_emission_probs()
        self.cal_transition_probs()

        data = {
            'unique_tags': self.unique_tags,
            'word_tags_s': self.word_tags_s,
            'bigram': self.bigram_tags_count,
            'trigram': self.trigram_tags_count,
            'emission': self.emission_probs,
            'transition': self.transition_probs
        }

        with open('probs/hnnmodel.pkl', 'wb') as f:
            pickle.dump(data, f)


### Function declarations ###
if __name__ == '__main__':
    probs = CalProbs('../processed_data/train.txt', '../processed_data/test.txt')
