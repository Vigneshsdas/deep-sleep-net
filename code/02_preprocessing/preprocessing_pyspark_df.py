# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:19:20 2022

"""

# Preprocessing using PySpark DataFrames

import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pyspark.sql.types as t

if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("Sleep").getOrCreate()
    
    print("\nSYS ARGUMENTS:")
    print(*sys.argv, sep="\n")
    
    path_data = sys.argv[1]
    path_label = sys.argv[2]
    
    # Read the data and labels
    df_data  = spark.read.csv(path_data, header=True)
    df_label = spark.read.csv(path_label, header=True)
    
    # Rename the channels
    df_data = df_data.withColumnRenamed("EEG Fpz-Cz", "fpzcz")\
        .withColumnRenamed("EEG Pz-Oz", "pzoz")\
        .withColumnRenamed("EOG horizontal", "horizontal")
    
    # Drop unnecessary column
    df_data = df_data.drop("EMG submental")
    
    # Cast the columns to the proper data type
    df_data = df_data.withColumn("fpzcz", f.col("fpzcz").cast("float"))\
        .withColumn("pzoz", f.col("pzoz").cast("float"))\
        .withColumn("horizontal", f.col("horizontal").cast("float"))
    
    # Create a primary key to join the data and the labels
    df_data = df_data.withColumn("primary_key", f.concat_ws("_", f.col("patient"), f.col("timestep")))
    
    hashmap = {'Sleep stage W': 0, 
               'Sleep stage 1': 1, 
               'Sleep stage 2': 2, 
               'Sleep stage 3': 3, 
               'Sleep stage 4': 3, 
               'Sleep stage R': 4}
    
    # Convert string labels to numeric
    udf_label = f.udf(lambda x: hashmap.get(x, -1), t.IntegerType())
    
    df_label = df_label.withColumn("label", udf_label(f.col("condition")))
    
    # Create a primary key to join the data and the labels and then drop redundant columns
    df_label = df_label.withColumn("primary_key", f.concat_ws("_", f.col("patient"), f.col("timestep")))\
        .drop("patient", "timestep", "time", "condition")
        
    # Merge the data and the labels
    df_full = df_data.join(df_label, "primary_key", "inner")
    
    # Remove unrecognized sleep stages
    df_full = df_full.filter(df_full.label != -1)
    
    # Sort by patient and timestep
    df_full = df_full.withColumn("timestep", f.col("timestep").cast("integer"))\
        .orderBy(f.col("patient"), f.col("timestep"))
      
    # Create epochs (where 30s is an epoch)
    df_full = df_full.withColumn("epoch", f.floor(f.col("timestep")/3000))\
        .drop("timestep", "primary_key")
        
    print("\nData after proprocessing:")
    
    df_full.show(5)
    
    df_full.coalesce(1)\
        .write.options(header='True', delimiter=',')\
        .csv(sys.argv[3]+"preprocessed")