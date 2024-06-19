import os
import string

from pyspark import SparkContext


def get_base_path():
    return os.path.dirname(os.path.realpath(__file__)) + '/'


def main():
    base_path = get_base_path()
    input_text_path = base_path + 'inputs/combined.txt'
    output_path = base_path + 'results'

    if os.path.exists(output_path):
        for filename in os.listdir(output_path):
            file_path = os.path.join(output_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(output_path)

    sc = SparkContext('local', 'Simple WordCount on one machine')

    words = sc.textFile(input_text_path).flatMap(
        lambda text:
        text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    )

    result = words.map(lambda word: (word, 1))\
        .reduceByKey(lambda a, b: a + b)\
        .sortBy(lambda a: a[1], ascending=False)
    result.saveAsTextFile(output_path)


def show_results():
    path = get_base_path() + 'results/part-00000'
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if i > 10:
                break
            print(line.rstrip())


if __name__ == "__main__":
    main()
    show_results()
