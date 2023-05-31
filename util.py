import csv
import re

if __name__ == '__main__':
    # with open('dataset/fine_tune_set.jsonl', encoding='utf-8', mode='w') as f1:
    #     with open('dataset/train3098itemPOLARITY.csv', encoding='utf-8', mode='r') as f2:
    #         file_reader = csv.reader(f2, delimiter=';')
    #         dataset = [(line[2], line[1]) for line in file_reader]
    #         for text, label in dataset:
    #             text = text.replace('\\', '\\\\')
    #             text = text.replace('"', '\\"')
    #             line = '{"prompt": "' + text + ' ->", "completion": "' + label + '"}\n'
    #             f1.write(line)
    #     f2.close()
    # f1.close()

    string = "<a>xxx</a>"
    match = re.match(r"<(\w+)>.*</\1>", string)
    if match:
        print("匹配成功！")
    else:
        print("匹配失败。")