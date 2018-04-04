# -*- coding: utf-8 -*-
import json
import re
import glob


def main():
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
                if (sentence['post_id'], sentence['sentence_id']) in id_set:
                    continue
                id_set.add((sentence['post_id'], sentence['sentence_id']))
                tag = re.sub(r'(PREMISE_|CLAIM_)', '', sentence['tag'])
                if tag == 'EMOTIONAL':
                    emotional_id_set.add((sentence['post_id'], sentence['sentence_id']))
                    emotional_count += 1
                elif tag == 'RATIONAL':
                    rational_id_set.add((sentence['post_id'], sentence['sentence_id']))
                    rational_count += 1
                else:
                    pass

    print('EMOTIONAL : {}'.format(emotional_count))
    print('RATIONAL : {}'.format(rational_count))

    print('EMOTIONAL : {}'.format(emotional_id_set))
    print('RATIONAL : {}'.format(rational_id_set))


if __name__ == '__main__':
    main()