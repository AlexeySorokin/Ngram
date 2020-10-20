from ngram import LanguageModel
from read import read_infile
from utils import get_perplexity


if __name__ == "__main__":
    data = read_infile('data/russian-train-high')
    test_data = read_infile('data/russian-dev')
    words = [elem[0].lower() for elem in data]
    test_words = [elem[0].lower() for elem in test_data]
    for order in range(2, 7):
        model = LanguageModel(order=order, method="wb", reverse=False)
        model.train(words)
        data = [model.score(word) for word in test_words]
        print(order, "{:.2f}".format(get_perplexity(data)))
        # alphabet = list(model.trie[0]["children"].keys())
        # for index in range(len(model.trie)):
        #     total_prob = 0.0
        #     history = model._calculate_repr(index)
        #     for a in model.alphabet:
        #         total_prob += pow(10, model._score(history, a))
        #     if abs(total_prob - 1.0) > 1e-6:
        #         print(order, node["order"], history, total_prob)