import csv
import random


def gen_fine_tune_dataset():
    with open('dataset/fine_tune_set.jsonl', encoding='utf-8', mode='w') as f1:
        with open('dataset/train3098itemPOLARITY.csv', encoding='utf-8', mode='r') as f2:
            file_reader = csv.reader(f2, delimiter=';')
            dataset = [(line[2], line[1]) for line in file_reader]
            for text, label in dataset:
                text = text.replace('\\', '\\\\')
                text = text.replace('"', '\\"')
                line = f'{{"prompt": "{text} ->", "completion": " {label}"}}\n'
                f1.write(line)
        f2.close()
    f1.close()


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


if __name__ == '__main__':
    gen_fine_tune_dataset()
