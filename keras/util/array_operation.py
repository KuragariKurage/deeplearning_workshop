import re
import numpy as np

def numericalSortKey(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def numericalSort(List):
    return sorted(List, key=numericalSortKey)

def shuffle_two_list(X, y):
    zipped = list(zip(X, y))
    np.random.shuffle(zipped)
    X_result, y_result = zip(*zipped)
    return list(X_result), list(y_result)