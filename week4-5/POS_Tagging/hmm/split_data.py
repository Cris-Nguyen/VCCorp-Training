### Module imports ###
import os


### Global Variables ###


### Class declarations ###
class SplitData:

    def __init__(self, orig_folder, train_path, val_path, test_path):
        self.orig_folder = orig_folder
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.text_l = []
        self.read_data()

    def read_data(self):
        self.files = [os.path.join(self.orig_folder, fn) for fn in os.listdir(self.orig_folder)]
        for fn in self.files:
            with open(fn, 'r') as f:
                self.text_l += f.readlines()

    def write_data(self, text_l, out_path):
        with open(out_path, 'w') as f:
            for text in text_l:
                f.write(text)

    def split_data(self):
        l = len(self.text_l)
        self.train_text_l = self.text_l[:int(0.8 * l)]
        self.val_text_l = self.text_l[int(0.8 * l):int(0.9 * l)]
        self.test_text_l = self.text_l[int(0.9 * l):]
        self.write_data(self.train_text_l, self.train_path)
        self.write_data(self.val_text_l, self.val_path)
        self.write_data(self.test_text_l, self.test_path)


### Function declarations ###
if __name__ == '__main__':
    split = SplitData('../data', '../processed_data/train.txt', '../processed_data/val.txt', '../processed_data/test.txt')
    split.split_data()

