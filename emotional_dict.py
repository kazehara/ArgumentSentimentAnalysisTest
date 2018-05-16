# -*- coding: utf-8 -*-
import pandas as pd
from typing import Dict, List, Set


class EmotionalDict:

    def __init__(self, nouns_dict_path: str, verbs_dict_path: str) -> None:
        self.nouns_dict_path = nouns_dict_path
        self.verbs_dict_path = verbs_dict_path

    def load(self) -> Dict[str, Set[str]]:
        table_nouns = pd.read_table(self.nouns_dict_path, sep='\t')
        table_verbs = pd.read_table(self.verbs_dict_path, sep='\t')
        emotional_dict = {'positive': set(), 'negative': set()}

        for key, row in table_nouns.iterrows():
            if row[1] == 'p':
                emotional_dict['positive'].add(row[0])
            elif row[1] == 'n':
                emotional_dict['negative'].add(row[0])
            else:
                pass

        for key, row in table_verbs.iterrows():
            try:
                verb = row[1].split(' ')[0]
            except AttributeError:
                continue
            if 'ポジ' in row[0]:
                emotional_dict['positive'].add(verb)
            elif 'ネガ' in row[0]:
                emotional_dict['negative'].add(verb)
            else:
                print(row[0], verb)

        return emotional_dict


def test():
    loader = EmotionalDict('dataset/nouns', 'dataset/verbs')
    emotional_dict = loader.load()

    print(len(emotional_dict['positive']))
    print(len(emotional_dict['negative']))


if __name__ == '__main__':
    test()
