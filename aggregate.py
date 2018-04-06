# -*- coding: utf-8 -*-
import glob
import json
import os
import re
from typing import Set, Tuple

import pandas as pd


# id_set : Set[(トピック名, post_id, sentence_id)]
def get_sentences(id_set: Set[Tuple[str, str, str]]):
    paths = glob.glob('dataset/nagoya2016/sentences/*.csv')
    topics = [os.path.splitext(os.path.basename(r))[0] for r in paths]

    sentences = []

    for topic, path in zip(topics, paths):
        # 文章などが入ったCSVを読み込む
        df = pd.read_csv(path)
        df['post_id'] = df['post_id'].apply(str)
        df['sentence_id'] = df['sentence_id'].apply(str)

        for id_ in id_set:
            # id_[0] にトピック名が入っている
            if id_[0] == topic:
                sentence = list(df[(df['post_id'] == id_[1]) & (df['sentence_id'] == id_[2])].sentence)[0]
                sentences.append(sentence)

    assert len(id_set) == len(sentences)

    return sentences


def aggregate():
    rational_count = 0
    emotional_count = 0

    id_set = set()

    emotional_id_set = set()
    rational_id_set = set()

    for path in glob.glob('dataset/nagoya2016_raw/phase1_raw/*.json'):
        with open(path) as f:
            json_dict = json.load(f)

        for topic in json_dict['annotation_data']['topics']:
            if not topic['sentences']:
                continue
            for sentence in topic['sentences']:
                sentence['post_id'] = str(sentence['post_id'])
                sentence['sentence_id'] = str(sentence['sentence_id'])

                if (topic['topic'], sentence['post_id'], sentence['sentence_id']) in id_set:
                    continue
                id_set.add((topic['topic'], sentence['post_id'], sentence['sentence_id']))
                tag = re.sub(r'(PREMISE_|CLAIM_)', '', sentence['tag'])
                if tag == 'EMOTIONAL':
                    emotional_id_set.add((topic['topic'], sentence['post_id'], sentence['sentence_id']))
                    emotional_count += 1
                elif tag == 'RATIONAL':
                    rational_id_set.add((topic['topic'], sentence['post_id'], sentence['sentence_id']))
                    rational_count += 1
                else:
                    pass

    print('EMOTIONAL : {}'.format(emotional_count))
    print('RATIONAL : {}'.format(rational_count))

    return emotional_id_set, rational_id_set


def main():
    emotional_id_set, rational_id_set = aggregate()
    print(emotional_id_set)
    print(rational_id_set)

    emotional_sentences = get_sentences(emotional_id_set)
    rational_sentences = get_sentences(rational_id_set)


if __name__ == '__main__':
    main()
