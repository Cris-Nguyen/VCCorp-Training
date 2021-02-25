### Module imports ###
import os
from datetime import datetime
import argparse

from pyvi import ViTokenizer
from gensim import corpora, matutils
from sklearn.feature_extraction.text import TfidfVectorizer

from load_from_file import DataLoader, FileReader, FileStore
import config as cf


### Global Variables ###


### Class declarations ###
class TextPreprocess(object):

    def __init__(self, text = None):
        self.text = text
        self.get_stopwords()

    def get_stopwords(self):
        self.stopwords = FileReader(cf.STOP_WORDS).read_stopwords()

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(cf.SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []

    def get_words_split(self):
        split_words = self.split_words()
        return [word for word in split_words if word.encode('utf-8') not in self.stopwords]


class FeatureExtraction(object):
    def __init__(self, data):
        self.data = data

    def build_dictionary(self):
        print('Building dictionary')
        dict_words = []
        i = 0
        for text in self.data:
            i += 1
            print("Dictionary Step {} / {}".format(i, len(self.data)))
            words = TextPreprocess(text = text['content']).get_words_split()
            dict_words.append(words)
        FileStore(file_path=cf.DICTIONARY_PATH).store_dictionary(dict_words)

    def load_dictionary(self):
        if not os.path.exists(cf.DICTIONARY_PATH):
            self.build_dictionary()
        self.dictionary = FileReader(cf.DICTIONARY_PATH).load_dictionary()

    def build_dataset(self):
        print('Building dataset')
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            self.features.append(self.get_dense(d['content']))
            self.labels.append(d['category'])

    def get_dense(self, text):
        #remove stopword
        words = TextPreprocess(text).get_words_split()
        # Bag of words
        self.load_dictionary()
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense

    def get_data_and_label_tfidf(self):
        print('Building dataset')
        self.features = []
        self.labels = []
        i = 0
        for d in self.data:
            i += 1
            print("Step {} / {}".format(i, len(self.data)))
            self.features.append(' '.join(TextPreprocess(d['content']).get_words_split()))
            self.labels.append(d['category'])
        return self.features, self.labels

    def get_data_and_label_bow(self):
        self.build_dataset()
        return self.features, self.labels

    def read_feature(self):
        return self.data['features'] , self.data['labels']


### Function declarations ###
def get_feature_dict(value_features, value_labels):
    return {
            'features': value_features,
            'labels': value_labels
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='tfidf', help='feature type for extracting feature in tfidf for bow')
    args = parser.parse_args()

    print('Reading data raw... ',  str(datetime.now()))
    json_train = DataLoader(data_path=cf.DATA_TRAIN_PATH).get_json()
    json_test = DataLoader(data_path=cf.DATA_TEST_PATH).get_json()
    print('Load Data to JSON Done! ', str(datetime.now()))

    # Feature Extraction
    print('Featuring Extraction... ',  str(datetime.now()))
    if args.feature == 'bow':
        features_train, labels_train = FeatureExtraction(data=json_train).get_data_and_label_bow()
        features_test, labels_test = FeatureExtraction(data=json_test).get_data_and_label_bow()
        features_train_path = f'{cf.FEATURES_TRAIN}_bow.pkl'
        features_test_path = f'{cf.FEATURES_TEST}_bow.pkl'
    elif args.feature == 'tfidf':
        features_train, labels_train = FeatureExtraction(data=json_train).get_data_and_label_tfidf()
        features_test, labels_test = FeatureExtraction(data=json_test).get_data_and_label_tfidf()
        vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2))
        features_train = vectorizer.fit_transform(features_train)
        features_test = vectorizer.transform(features_test)
        FileStore(file_path=cf.VECTOR_EMBEDDING).save_pickle(obj=vectorizer)
        features_train_path = f'{cf.FEATURES_TRAIN}_tfidf.pkl'
        features_test_path = f'{cf.FEATURES_TEST}_tfidf.pkl'
    else:
        raise ValueError("Feature type is incorrect")

    # Save feature
    print('Saving Feature... ',  str(datetime.now()))
    features_train_dict = get_feature_dict(value_features=features_train, value_labels=labels_train)
    features_test_dict = get_feature_dict(value_features=features_test, value_labels=labels_test)
    FileStore(file_path=features_train_path).save_pickle(obj=features_train_dict)
    FileStore(file_path=features_test_path).save_pickle(obj=features_test_dict)
    print("Store data DONE!")
