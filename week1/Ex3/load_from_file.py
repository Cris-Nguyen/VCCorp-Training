### Module imports ###
import os
import json
import pickle as cPickle
from random import randint

from gensim import corpora, matutils


### Global Variables ###


### Class declarations ###
class DataLoader(object):

    def __init__(self, data_path):
        self.data_path = data_path

    def get_files(self):
        class_l = os.listdir(self.data_path)
        folders = [os.path.join(self.data_path, folder) for folder in class_l]
        files = {}
        for folder, title in zip(folders, class_l):
            files[title] = [os.path.join(folder, f) for f in os.listdir(folder)]
        self.files = files

    def get_json(self):
        self.get_files()
        data = []
        for topic in self.files:
            for file in self.files[topic]:
                content = FileReader(file_path=file).content()
                data.append({
                    'category': topic,
                    'content': content
                })
        return data


class FileReader(object):

    def __init__(self, file_path, encoder = None):
        self.file_path = file_path
        self.encoder = encoder if encoder != None else 'utf-16le'

    def read(self):
        with open(self.file_path, 'rb') as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s.decode(self.encoder)

    def read_json(self):
        with open(self.file_path, 'rb') as f:
            s = json.load(f)
        return s

    def read_stopwords(self):
        with open(self.file_path, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.file_path)


class FileStore(object):

    def __init__(self, file_path, data = None):
        self.file_path = file_path
        self.data = data

    def store_json(self):
        with open(self.file_path, 'w') as outfile:
            json.dump(self.data, outfile)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=100, no_above=0.5)
        dictionary.save_as_text(self.file_path)

    def save_pickle(self,  obj):
        outfile = open(self.file_path, 'wb')
        fastPickler = cPickle.Pickler(outfile, cPickle.HIGHEST_PROTOCOL)
        fastPickler.fast = 1
        fastPickler.dump(obj)
        outfile.close()


### Function declarations ###
if __name__ == '__main__':
    data_loader = DataLoader('DL_dataset/train/')
    print(data_loader.get_json())
