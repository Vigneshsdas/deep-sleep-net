# -*- coding: utf-8 -*-
"""RDD_preprocessing.ipynb

Automatically generated by Colaboratory.

"""

# Preprocessing using PySpark DataFrames

from pyspark.sql import SparkSession
from pyspark import SparkContext
import numpy as np

import sys

if __name__ == "__main__":

  spark = SparkSession.builder.master("local[*]").getOrCreate()
  sc = SparkContext.getOrCreate()

  
  input_file_1 = sys.argv[1]
  input_file_2 = sys.argv[2]
  output= sys.argv[3]

  rdd_data = sc.textFile(input_file_1).map(lambda x: x.split(","))
  rdd_label = sc.textFile(input_file_2).map(lambda x: x.split(","))

#Header of raw data and label
  header_raw = rdd_data.first()
  header_label = rdd_label.first()

#Remove the header from the RDD
  rdd_raw = rdd_data.filter(lambda x : x != header_raw)
  label = rdd_label.filter(lambda x : x != header_label)

#Function to create unique identifier
  def combine_id_time(line):

    unique_id = str(str(line[0]) + "_" + str(line[1]))

    return unique_id, line

#Create the unique identifier
  rdd_raw_1 = rdd_raw.map(lambda x: combine_id_time(x))
  label_1   = label.map(lambda x: combine_id_time(x)).map(lambda x: (x[0], [x[1][3]]))


 #Join three channels and join the label
  data =  rdd_raw_1.join(label_1)

  #x[1][0] - list of channels, x[1][1] - label
  # x[1][0] + x[1][1] - [channels, label]  
  
  data_1 = data.map(lambda x: (x[0], x[1][0] + x[1][1]))
  data_2 = data_1.map(lambda x: [x[0], x[1][1], x[1][3], x[1][4], x[1][5], x[1][7]])

#label numeric conversion
  hashmap = { 'Sleep stage W': 0, 
              'Sleep stage 1': 1, 
              'Sleep stage 2': 2, 
              'Sleep stage 3': 3, 
              'Sleep stage 4': 3, 
              'Sleep stage R': 4}

#
  def hash_map(line):

    if line[5] in hashmap.keys():
      label = hashmap.get(line[5],-1)

    return line + [label]

  data_3 = data_2.map(lambda x: hash_map(x))
  data_4 = data_3.map(lambda x: (x[0], x[1:]))

#Sort to add epoch
  data_5 = data_4.sortByKey()
  data_6 = data_5.map(lambda x: [x[0]] + x[1])

#Add epoch with 3000 time step (30s is an epoch)
  def epoch(line):

    epoch = np.floor(int(line[1])/3000)

    return line + [epoch]

  data_7 = data_6.map(lambda x: epoch(x))

  data_7.take(2)

#Save it in CSV format

  def toCSVLine(data):
    return ','.join(str(d) for d in data)

  lines = data_6.map(toCSVLine)
  lines.saveAsTextFile(output + '/RDD_preprocessed.csv')


