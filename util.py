import csv
import random

from textattack.augmentation import EmbeddingAugmenter

from analyzer import Analyzer
from preprocessor import PreProcessor


def gen_fine_tune_dataset(fine_tune_suffix: str = '', base_suffix: str = ''):
    fine_tune_set_name = f'dataset/fine_tune_set_{fine_tune_suffix}.jsonl'\
        if fine_tune_suffix != '' else 'dataset/fine_tune_set.jsonl'
    base_set_name = f'dataset/train3098itemPOLARITY_{base_suffix}.csv'\
        if base_suffix != '' else 'dataset/train3098itemPOLARITY.csv'
    with open(fine_tune_set_name, encoding='utf-8', mode='w') as f1:
        with open(base_set_name, encoding='utf-8', mode='r') as f2:
            for line in f2.readlines():
                line = line.strip('\n').strip(' ')
                temp = line.split(';')
                label, text = temp[1], temp[2]
                if len(temp) > 3:
                    for i in range(3, len(temp)):
                        text += ';' + temp[i]
                text = text.replace('\\', '\\\\')
                text = text.replace('"', '\\"')
                line = f'{{"prompt": "{text} ->", "completion": " {label}"}}\n'
                f1.write(line)
            # dataset = []
            # file_reader = csv.reader(f2, delimiter=';')
            # for line in file_reader:
            #     label = line[1]
            #     text = line[2]
            #     if len(line) > 3:
            #         for i in range(3, len(line)):
            #             text += ';' + line[i]
            #     dataset.append((text, label))
            # for text, label in dataset:
            #     text = text.replace('\\', '\\\\')
            #     text = text.replace('"', '\\"')
            #     line = f'{{"prompt": "{text} ->", "completion": " {label}"}}\n'
            #     f1.write(line)
        f2.close()
    f1.close()


def gen_preprocessed_dataset(origin_file_name: str):
    pp = PreProcessor()
    texts, labels = Analyzer.read_data(origin_file_name)
    texts = [pp.preprocess_text_v2(text) for text in texts]
    with open('./dataset/train3098itemPOLARITY_augmented_preprocessed.csv', mode='w', encoding='utf-8') as file:
        index = 1
        # contents = []
        for text, label in zip(texts, labels):
            line = f'{index};{label};{text}\n'
            file.write(line)
            # contents.append(line)
            # print(line, end='')
            index += 1
        # file.writelines(contents)
    file.close()


def split_dataset():
    with open('dataset/fine_tune_set.jsonl', encoding='utf-8', mode='r') as f:
        random.seed(7)
        train_set = []
        validate_set = []
        for line in f.readlines():
            if random.random() < 0.2:
                validate_set.append(line)
            else:
                train_set.append(line)
        with open('./dataset/fine_tune_train_set.jsonl', encoding='utf-8', mode='w') as f1, \
                open('./dataset/fine_tune_validate_set.jsonl', encoding='utf-8', mode='w') as f2:
            f1.writelines(train_set)
            f2.writelines(validate_set)
        f1.close()
        f2.close()
    f.close()


def augment_dataset():
    texts_train, labels_train = Analyzer.read_data('dataset/train3098itemPOLARITY.csv')
    texts_train_augmented = []
    labels_train_augmented = []
    augmenter = EmbeddingAugmenter(transformations_per_example=2)
    index = 0
    print('start augment')
    for text, label in zip(texts_train, labels_train):
        text_aug_res = augmenter.augment(text)
        label_aug_res = [label] * len(text_aug_res)
        texts_train_augmented.extend(text_aug_res)
        labels_train_augmented.extend(label_aug_res)
        print(f'finish augment text {index}')
        index += 1
    print('finish augment')
    with open('dataset/train3098itemPOLARITY_augmented.csv', mode='w', encoding='utf-8') as f:
        for text, label in zip(texts_train_augmented, labels_train_augmented):
            line = f'xx;{label};{text}\n'
            f.write(line)
    f.close()


if __name__ == '__main__':
    # gen_preprocessed_dataset('dataset/train3098itemPOLARITY_augmented.csv')
    gen_fine_tune_dataset(fine_tune_suffix='augmented_preprocessed', base_suffix='augmented_preprocessed')
    # gen_fine_tune_dataset()
