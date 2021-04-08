### Module imports ###
import math
import pickle

from read_from_file import ExtractData


### Glocal Variables ###


### Class declarations ###
class Decode:

    def __init__(self, probs_path):
        # load the saved model
        with open(probs_path, 'rb') as f:
            data = pickle.load(f)

            # dictionary to store the emission probabilities
            self.emission_probs = data['emission']

            # dictionary to store the transition probabilities for trigrams
            self.transition_probs = data['transition']

            # unique tags
            self.unique_tags = data['unique_tags']

            # bigram_counts
            self.bigram_counts = data['bigram']

            # dictionary to store a word and its tags
            self.word_tags_s = data['word_tags_s']

        # dictionary to store the probabilities in the sequence
        self.word_tag_viterbi_probs = {}

        # save the index-tag(word-tag) for the last word of the sentence
        self.idx_tag_key = None

    def recursive_probs_cal(self, word_sequence, idx, word_tag_i):
        # base case: if the index is the start tag return the probability as 0
        if idx == 1:
            return 0.0

        # if the probability already exists then return
        if (idx, word_tag_i) in self.word_tag_viterbi_probs:
            return self.word_tag_viterbi_probs[idx, word_tag_i][0]

        # word-tags from idx - 1
        word_tags_i_1 = self.word_tags_s[word_sequence[idx - 1]] if word_sequence[idx - 1] in self.word_tags_s else self.unique_tags
        # word tags from idx - 2
        word_tags_i_2 = self.word_tags_s[word_sequence[idx - 2]] if word_sequence[idx - 2] in self.word_tags_s else self.unique_tags

        max_viterbi_prob = -float("Inf")
        back_pointer_tag = '*'

        # iterate through all word-tag from previous indexes
        for word_tag_i_1 in word_tags_i_1:
            for word_tag_i_2 in word_tags_i_2:
                viterbi_prob = 0.0

                # calculate transition prob
                if (word_tag_i_2, word_tag_i_1, word_tag_i) in self.transition_probs:
                    transition_prob = self.transition_probs[(word_tag_i_2, word_tag_i_1, word_tag_i)]
                else:
                    if (word_tag_i_2, word_tag_i_1) in self.bigram_counts:
                        transition_prob = math.log(1.0 / float(self.bigram_counts[(word_tag_i_1, word_tag_i_2)] + len(self.unique_tags)))
                    else:
                        transition_prob = math.log(1.0 / float(len(self.unique_tags)))

                # calculate viterbi prob
                if (word_sequence[idx], word_tag_i) not in self.emission_probs:
                    transition_prob = 0.0
                    viterbi_prob = self.recursive_probs_cal(word_sequence, idx - 1, word_tag_i_1) + transition_prob
                else:
                    viterbi_prob = self.recursive_probs_cal(word_sequence, idx - 1, word_tag_i_1) + transition_prob + self.emission_probs[(word_sequence[idx], word_tag_i)]

                if max_viterbi_prob < viterbi_prob:
                    max_viterbi_prob = viterbi_prob
                    back_pointer_tag = word_tag_i_1

        self.word_tag_viterbi_probs[idx, word_tag_i] = (max_viterbi_prob, (idx - 1, back_pointer_tag))
        return max_viterbi_prob

    def viterbi_sentence(self, sentence):
        max_viterbi_prob = -float("Inf")
        self.word_tag_viterbi_probs = {}
        l = len(sentence)

        cur = sentence[l - 1]
        if cur in self.word_tags_s:
            word_tags_i = self.word_tags_s[sentence[l - 1]]
        else:
            word_tags_i = self.unique_tags

        for word_tag_i in word_tags_i:
            viterbi_prob = self.recursive_probs_cal(sentence, l - 1, word_tag_i)
            # save the last pointer
            if max_viterbi_prob < viterbi_prob:
                max_viterbi_prob = viterbi_prob
                self.idx_tag_key = (l - 1, word_tag_i)

        tagged_sentence = []
        while self.idx_tag_key[0] >= 2:
            tagged_sentence.insert(0, self.idx_tag_key[1])
            self.idx_tag_key = self.word_tag_viterbi_probs[self.idx_tag_key][1]

        return tagged_sentence


### Function declarations ###
