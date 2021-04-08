### Module imports ###
from read_from_file import ExtractData
from decode import Decode

from sklearn_crfsuite.metrics import flat_classification_report


### Global Variables ###


### Class declarations ###


### Function declarations ###
def benchmark(model, train_path, test_path):
    extractor = ExtractData(train_path, test_path)
    sentence_l, tags_l = extractor.word_tag_test()
    y_true, y_pred = [], []
    correct, total = 0, 0
    for i in range(len(sentence_l)):
        tags_infer = model.viterbi_sentence(sentence_l[i])
        tags_true = tags_l[i][2:]
        y_true.append(tags_true)
        y_pred.append(tags_infer)
        for j in range(len(tags_infer)):
            correct += 1 if tags_infer[j] == tags_true[j] else 0
            total += 1
    print(f'Accuracy: {correct / total}')
    print(flat_classification_report(y_true, y_pred))


if __name__ == '__main__':
    model = Decode('probs/hmmmodel.pkl')
    train_path = '../processed_data/train.txt'
    test_path = '../processed_data/val.txt'
    benchmark(model, train_path, test_path)
