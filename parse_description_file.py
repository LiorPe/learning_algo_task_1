import tensorflow as tf
import re
import numpy as np

def extract_feature(cur_line, lines, feature_dictionary):
    line = lines[cur_line]
    feature_name = line.split(':')[0]
    values_list = []
    cur_line+=2
    feature =None
    if (cur_line<=len(lines)):
        while (cur_line<len(lines) and re.search('[a-zA-Z0-9]', lines[cur_line]) and lines[cur_line].startswith(' ')):
            line= lines[cur_line].replace(' ', '')
            value= line.split('\t')[0]
            values_list.append(value)
            cur_line += 1
    if (len(values_list)>0):
        cur_line +=1
    feature_dictionary[feature_name]= values_list
    return cur_line


def create_features_lists(feature_dictionary):
    all_features = []
    feature = None
    dtype =  None
    for feature_name, feature_values in feature_dictionary.items():
        if(len(feature_values)>0):
            feature = tf.contrib.layers.sparse_column_with_hash_bucket(column_name=feature_name,
                                                                       hash_bucket_size=len(feature_values)+1)
            feature = tf.contrib.layers.embedding_column (sparse_id_column= feature, dimension=1)
        else:
            feature = tf.contrib.layers.real_valued_column(column_name=feature_name)
        all_features.append(feature)
    return all_features


def extract_feature_values(path):
    lines = [line.rstrip('\n') for line in open(path)]
    total_lines = len(lines)
    cur_line = 0
    features_dict ={}
    while (cur_line<total_lines):
        cur_line = extract_feature(cur_line, lines, features_dict)
    return features_dict