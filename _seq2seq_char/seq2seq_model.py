import numpy as np
import time
import tensorflow as tf

def get_data(file):
    with open(file,"r",encoding='utf-8') as f:
        return f.read()


source_data = get_data('./data/letters_source.txt').split('\n')
# target_data = get_data('./data/letters_target.txt')
print(source_data)