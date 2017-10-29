# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import parse_description_file

import itertools

features_dict = parse_description_file.extract_feature_values('data/data_description.txt')
feature_cols = parse_description_file.create_features_lists(features_dict)
FEATURES = list(features_dict.keys())

LABEL = 'SalePrice'
test_set = pd.read_csv('data/test.csv')
tf.logging.set_verbosity(tf.logging.INFO)


def convert_categoriacl_values_to_string(df_data, features_dict):
    for feature_name, feature_values in features_dict.items():
        if(len(feature_values)>0):
            df_data[feature_name] = df_data[feature_name].apply(str)
            df_data[feature_name].fillna(df_data[feature_name].mode(), inplace=True)
        else:
            df_data[feature_name].fillna(df_data[feature_name].mean(), inplace=True)
    return df_data

def get_input_fn(data_set, num_epochs=None, shuffle=True):
      return tf.estimator.inputs.pandas_input_fn(
          x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
           y = pd.Series(data_set[LABEL].values),
          num_epochs=num_epochs,
          shuffle=shuffle)


def get_input_fn_predict(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        num_epochs=num_epochs,
        shuffle=shuffle)


def train_model(regressor):
    global training_set
    global test_set
    global features_dict
    global feature_cols
    global FEATURES
    global LABEL
    training_set = pd.read_csv('data/train.csv')
    training_set = convert_categoriacl_values_to_string(training_set,features_dict)
    regressor.train(input_fn=get_input_fn(training_set), steps=10000)



def load_model(regressor):
    global test_set
    test_set = convert_categoriacl_values_to_string(test_set,features_dict)
    y = regressor.predict(input_fn=get_input_fn_predict(test_set,num_epochs=1,shuffle=False))
    id = 1461
    with open ("predictions.txt", mode='w') as file:
        file.write("Id,SalePrice\n")
        for prediction in y:
            print(prediction['predictions'][0])
            file.write(f"{id},{prediction['predictions'][0]}\n")
            id+=1

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[1024, 512, 256],
                                          model_dir='model_dir',
                                          optimizer=tf.train.ProximalAdagradOptimizer(
                                              learning_rate=0.2,
                                              l1_regularization_strength=0.001
                                          ))
train_model(regressor)
load_model(regressor)