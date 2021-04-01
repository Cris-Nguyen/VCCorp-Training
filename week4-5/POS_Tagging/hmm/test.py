### Module imports ###
from read_from_file import ExtractData
from decode import Decode


### Global Variables ###


### Class declarations ###


### Function declarations ###
def benchmark(model, train_path, test_path):
    extractor = ExtractData(train_path, test_path)
    sentence_l, tags_l = extractor.word_tag_test()
    correct, total = 0, 0
    for i in range(len(sentence_l)):
        tags_infer = model.viterbi_sentence(sentence_l[i])
        tags_true = tags_l[i][2:]
        for j in range(len(tags_infer)):
            correct += 1 if tags_infer[j] == tags_true[j] else 0
            total += 1
    print(f'Accuracy: {correct / total}')


if __name__ == '__main__':
    model = Decode('probs/hmmmodel.pkl')
    train_path = '../processed_data/train.txt'
    test_path = '../processed_data/test.txt'
    benchmark(model, train_path, test_path)
