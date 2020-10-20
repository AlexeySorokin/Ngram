from collections import defaultdict

import numpy as np


class LanguageModel:

    def __init__(self, order=3, reverse=False, append_first=True, append_last=True,
                 alpha=1.0, add_unknown=False, method=None):
        self.order = order
        self.reverse = reverse
        self._root_index = 0
        # вершина: предок, уровень, суффикс, сыновья, счётчик, вероятность, backoff
        self.append_first = append_first
        self.append_last = append_last
        self.alpha = alpha
        self.add_unknown = add_unknown
        self.method = method
        self.trie = []
        self._add_node()

    @property
    def alphabet(self):
        return list(self.root["children"].keys())

    @property
    def root(self):
        return self.trie[self._root_index]

    def _add_node(self, backoff_node=None, order=0, index=None, label=None):
        if index is None:
            index = len(self.trie) if hasattr(self, "trie") else 0
        node = {"backoff_node": backoff_node, "order": order, "children": dict(), "index": index, "count": 0, "label": label}
        self.trie.append(node)
        return node

    def _add_child(self, prev, word, backoff_node):
        label = word if prev["label"] is None else prev["label"]
        node = self._add_node(backoff_node=backoff_node, order=prev["order"]+1, label=label)
        prev["children"][word] = node
        return node

    def train(self, sents):
        for i, sent in enumerate(sents):
            # if i % 100000 == 1:
            #     print(i)
            sent = list(sent)
            if self.reverse:
                sent = sent[::-1]
            if self.append_first:
                sent = ["<s>"] + sent
            if self.append_last:
                sent += ["</s>"]
            # вершины, соответствующие текущим суффиксам
            nodes = [self.root] + [None] * (self.order - 1)
            for word in sent:
                new_nodes = [self.root] + [None] * (self.order)
                for i, prev in enumerate(nodes, 1):
                    if prev is None:
                        break
                    child = prev["children"].get(word)
                    if child is None:
                        child = self._add_child(prev, word, new_nodes[i-1])
                    child["count"] += 1
                    new_nodes[i] = child
                nodes = new_nodes[:-1]
            self.root['count'] = sum(child["count"] for child in self.root["children"].values())
        self._smooth()
        return self

    def _calculate_repr(self, node, sep=None):
        if node is None:
            return None
        if isinstance(node, int):
            node = self.trie[node]
        s = []
        while node["index"] != self._root_index:
            s.append(node["label"])
            node = node["backoff_node"]
        if sep is not None:
            s = sep.join(s)
        return s


    def _smooth(self):
        """
        Выполняет сглаживание по методу self.method
        """
        if self.method in ['Witten-Bell', 'wb']:
            self.make_WittenBell_smoothing()
        elif self.method in ['Kneser-Ney', 'kn']:
            self.make_KneserNey_smoothing()
        else:
            raise NotImplementedError("Smoothing method {} is not implemented".format(self.method))
        return self

    def make_WittenBell_smoothing(self):
        '''
        Выполняет сглаживание по методу Уиттена-Белла
        '''
        for node in self.trie:
            continuations_count = len(node["children"])
            continuations_sum = sum(child["count"] for child in node["children"].values())
            denominator = continuations_count + continuations_sum
            if node["order"] == 0 and self.add_unknown:
                denominator += 1.0
            if node["order"] == self.order or continuations_count == 0:
                node["backoff"] = 0.0
                continue
            else:
                backoff = continuations_count # / denominator
            for word, child in node["children"].items():
                numerator = child["count"]
                if node["order"] > 0:
                    child_backoff_node = node["backoff_node"]["children"][word]
                    numerator += backoff * child_backoff_node["prob"]
                else:
                    numerator += 1
                child["prob"] = numerator / denominator
            node["backoff"] = np.log10(backoff) - np.log10(denominator)
        for node in self.trie:
            if "prob" in node:
                node["prob"] = np.log10(node["prob"])
        return self

    def make_KneserNey_smoothing(self):
        for node in self.trie:
            continuations_sum = sum(child["count"] for child in node["children"].values())
            if node["order"] == self.order or continuations_sum == 0:
                node["backoff"] = 0.0
                continue
            if node["order"] > 0:
                backoff = sum(min(child["count"], self.alpha) for child in node["children"].values())
            elif self.add_unknown:
                backoff = self.alpha * (1.0 - 1.0 / (len(node["children"]) + 1))
            else:
                backoff = self.alpha
            for word, child in node["children"].items():
                numerator = max(child["count"] - self.alpha, 0.0)
                if node["order"] > 0:
                    child_backoff_node = node["backoff_node"]["children"][word]
                    numerator += backoff * child_backoff_node["prob"]
                else:
                    numerator += backoff
                child["prob"] = numerator / continuations_sum
            node["backoff"] = np.log10(backoff) - np.log10(continuations_sum)
        for node in self.trie:
            if "prob" in node:
                node["prob"] = np.log10(node["prob"])
        return self

    def _find_node(self, history):
        if self.order == 0:
            return self.root
        node = self.root
        history = history[-(self.order-1):]
        for word in history:
            while word not in node["children"]:
                parent = node["backoff_node"]
                if parent is None:
                    node = None
                    break
                node = parent
            if node is None or word not in node["children"]:
                node = self.root
            else:
                node = node["children"][word]
        return node

    def _score(self, history, word):
        parent = self._find_node(history)
        score = 0.0
        while True:
            child = parent["children"].get(word)
            if child is not None:
                score += child["prob"]
                return score
            if parent["index"] == self._root_index:
                raise ValueError(f"Unknown word {word}")
            score += parent["backoff"]
            parent = parent["backoff_node"]

    def score(self, sent, skip_first_letter=False):
        # print(sent)
        sent = list(sent)
        start_index = 0
        if self.reverse:
            sent = sent[::-1]
        if self.append_last:
            sent += ["</s>"]
        if self.append_first:
            sent = ["<s>"] + sent
            start_index = 1
        if skip_first_letter:
            start_index += 1
        probs = []
        for i, word in enumerate(sent[start_index:], start_index):
            history_start_index = max(i-self.order+1, 0)
            history = sent[history_start_index:i]
            probs.append(self._score(history, word))
        return {"probs": probs, "total_prob": sum(probs)}


if __name__ == "__main__":
    model = LanguageModel(order=3, method="wb")
    words = ["abacb", "aabc", "bcbac"]
    model.train(words)
    for node in model.trie:
        print(node["index"], model._calculate_repr(node, sep=""))
        print("backoff:", model._calculate_repr(node["backoff_node"], sep=""), end="\t")
        print("children:", end=" ")
        for child in node["children"].values():
            print(model._calculate_repr(child, sep=""), end=" ")
        print("")