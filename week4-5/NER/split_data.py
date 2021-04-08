### Module imports ###
import os


### Global Variables ###


### Class declarations ###
class SplitData:

    def __init__(self, orig_folder, output_folder):
        self.orig_folder = orig_folder
        self.output_folder = output_folder

    def read_data(self):
        text_l = []
        for fn in os.listdir(self.orig_folder):
            file_path = os.path.join(self.orig_folder, fn)
            with open(file_path, 'r') as f:
                data = f.read().split('<s>')[1:]
                for line in data:
                    text = []
                    l = line.split('\n')[1:-2]
                    for pattern in l:
                        split = pattern.split('\t')
                        text.append(split[0] + '/' + split[3])
                    text_l.append(' '.join(text))

        return text_l

    def write_data(self, text_l, out_path):
        with open(out_path, 'w') as f:
            for text in text_l:
                f.write(text + '\n')

    def split_data(self):
        text_l = self.read_data()
        l = len(text_l)
        train_text_l = text_l[:int(0.8 * l)]
        val_text_l = text_l[int(0.8 * l):int(0.9 * l)]
        test_text_l = text_l[int(0.9 * l):]
        self.write_data(train_text_l, os.path.join(self.output_folder, 'train.txt'))
        self.write_data(val_text_l, os.path.join(self.output_folder, 'val.txt'))
        self.write_data(test_text_l, os.path.join(self.output_folder, 'test.txt'))


### Function declarations ###
if __name__ == '__main__':
    split = SplitData('data/', 'processed_data/')
    split.split_data()
