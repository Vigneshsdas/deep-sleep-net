import numpy as np
import random
import sys

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers

from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

from elephas.ml_model import ElephasEstimator


# Model
from model_elephas import Sleep_Conv_Net

if __name__ == "__main__":
    print("\nSYS ARGUMENTS:")
    print(*sys.argv, sep="\n")
    clf = Sleep_Conv_Net(num_classes=5, 
                         head_size=256, 
                         heads=3, 
                         dim_feed_forw=128, 
                         attention_dropout=0.2)
    
    clf.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])
    
    # Data Processing
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    
    spark = SparkSession.builder.appName("SparkSQL").getOrCreate()
    
    df_train_spark = spark.read.csv(sys.argv[1], header=True, sep=",")
    
    df_train_spark = df_train_spark.withColumn('time', f.col("time").cast("float"))\
      .withColumn('fpzcz', f.col("fpzcz").cast("float"))\
      .withColumn('pzoz', f.col("pzoz").cast("float"))\
      .withColumn('horizontal', f.col("horizontal").cast("float"))\
      .withColumn('label', f.col("label").cast("int"))\
      .withColumn('epoch', f.col("epoch").cast("int"))\
      .persist()
    
    df_train_spark = df_train_spark.withColumn('primary_key', f.concat_ws('_',f.col('patient'), f.col('epoch')))
    
    print("\nData before scaling:\n")
    df_train_spark.show(5)
    
    assembler = VectorAssembler(inputCols=["fpzcz", "pzoz", "horizontal"], outputCol="features")
    df_train_spark = assembler.transform(df_train_spark)
    
    # Scaling
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    pipeline = Pipeline(stages=[scaler])
    scalerModel = pipeline.fit(df_train_spark)
    df_train_spark = scalerModel.transform(df_train_spark)
    
    print("\nData after scaling:\n")
    df_train_spark.show(5)
    
    def extract_elem(vals, idx):
      return float(list(vals)[idx])
    
    udf_extract0 = f.udf(lambda x: extract_elem(x, 0), FloatType())
    udf_extract1 = f.udf(lambda x: extract_elem(x, 1), FloatType())
    udf_extract2 = f.udf(lambda x: extract_elem(x, 2), FloatType())
    
    df_train_spark = df_train_spark.drop("features")\
      .withColumn("scaled_fpzcz", udf_extract0(f.col("scaled_features")))\
      .withColumn("scaled_pzoz", udf_extract1(f.col("scaled_features")))\
      .withColumn("scaled_horiz", udf_extract2(f.col("scaled_features")))\
      .drop("scaled_features", "fpzcz", "pzoz", "horizontal")
    
    subject_list = ['ST7011J0', 'ST7012J0', 'ST7021J0', 'ST7022J0', 'ST7041J0',
                    'ST7042J0', 'ST7051J0', 'ST7052J0', 'ST7061J0', 'ST7062J0',
                    'ST7071J0', 'ST7072J0', 'ST7081J0', 'ST7082J0', 'ST7091J0',
                    'ST7092J0', 'ST7101J0', 'ST7102J0', 'ST7111J0', 'ST7112J0',
                    'ST7121J0', 'ST7122J0', 'ST7131J0', 'ST7132J0', 'ST7141J0',
                    'ST7142J0', 'ST7151J0', 'ST7152J0', 'ST7161J0', 'ST7162J0',
                    'ST7171J0', 'ST7172J0', 'ST7181J0', 'ST7182J0', 'ST7191J0',
                    'ST7192J0', 'ST7201J0', 'ST7202J0', 'ST7211J0', 'ST7212J0',
                    'ST7221J0', 'ST7222J0', 'ST7241J0', 'ST7242J0']
    
    test_subjects = random.sample(subject_list, 8)
    
    df_test_spark = df_train_spark.filter(df_train_spark.patient.isin(test_subjects))\
      .orderBy(f.col("patient"), f.col("epoch"), f.col("time"))
    df_train_spark = df_train_spark.filter(~df_train_spark.patient.isin(test_subjects))\
      .orderBy(f.col("patient"), f.col("epoch"), f.col("time"))
    
    def combine(a,b,c):
      return list(a) + list(b) + list(c)
    
    udf_combine = f.udf(combine, ArrayType(FloatType()))
    
    df_train_spark = df_train_spark.groupBy('primary_key')\
      .agg(f.collect_list(f.col('scaled_fpzcz')).alias("fpzcz"),
           f.collect_list(f.col('scaled_pzoz')).alias("pzoz"),
           f.collect_list(f.col('scaled_horiz')).alias("horizontal"),
           f.avg(f.col("label")).cast("int").alias("label"))
      
    df_train_spark = df_train_spark.withColumn("features", udf_combine(f.col("fpzcz"), f.col("pzoz"), f.col("horizontal")))\
      .drop("fpzcz", "pzoz", "horizontal")
    
    df_train_spark = df_train_spark.withColumn("vectors", array_to_vector(f.col("features")))
    
    df_test_spark = df_test_spark.groupBy('primary_key')\
      .agg(f.collect_list(f.col('scaled_fpzcz')).alias("fpzcz"),
           f.collect_list(f.col('scaled_pzoz')).alias("pzoz"),
           f.collect_list(f.col('scaled_horiz')).alias("horizontal"),
           f.avg(f.col("label")).cast("int").alias("label"))
    
    df_test_spark = df_test_spark.withColumn("features", udf_combine(f.col("fpzcz"), f.col("pzoz"), f.col("horizontal")))\
      .drop("fpzcz", "pzoz", "horizontal")
    
    df_test_spark = df_test_spark.withColumn("vectors", array_to_vector(f.col("features")))
    
    print("\nTrain data:\n")
    df_train_spark.show(5)
    
    adam = optimizers.Adam(learning_rate=1e-5)
    opt_conf = optimizers.serialize(adam)
    
    # Initialize SparkML Estimator and set all relevant properties
    estimator = ElephasEstimator()
    estimator.setFeaturesCol("vectors")             
    estimator.setLabelCol("label")                
    estimator.set_keras_model_config(clf.to_json())  
    estimator.set_categorical_labels(True)
    estimator.set_nb_classes(5)
    estimator.set_num_workers(4)  
    estimator.set_epochs(100) 
    estimator.set_batch_size(32)
    estimator.set_verbosity(1)
    estimator.set_validation_split(0.1)
    estimator.set_optimizer_config(opt_conf)
    estimator.set_mode("synchronous")
    estimator.set_loss("categorical_crossentropy")
    estimator.set_metrics(['acc'])
    
    pipeline = Pipeline(stages=[estimator])
    
    fitted_pipeline = pipeline.fit(df_train_spark)
      
    prediction_train = fitted_pipeline.transform(df_train_spark) 
    prediction_train = prediction_train.select("label", "prediction")
    
    def get_pred_label(probs):
      return int(np.argmax(probs))
    
    udf_pred_labels = f.udf(get_pred_label, IntegerType())
    
    prediction_train = prediction_train.withColumn("y_pred", udf_pred_labels(f.col("prediction")))\
      .drop("prediction")\
      .persist()
    
    print("\nPrediction Samples:\n")
    prediction_train.show(10)
    
    def get_accuracy(y_true, y_pred):
      return int(y_true == y_pred)
    
    udf_accuracy = f.udf(get_accuracy, IntegerType())
    
    accuracy_train = prediction_train.withColumn("equal", udf_accuracy(f.col("label"), f.col("y_pred")))\
      .select(f.avg(f.col("equal")))\
      .collect()[0][0]
    
    prediction_test = fitted_pipeline.transform(df_test_spark)
    
    prediction_test = prediction_test.withColumn("y_pred", udf_pred_labels(f.col("prediction")))\
      .drop("prediction")
    
    accuracy_test = prediction_test.withColumn("equal", udf_accuracy(f.col("label"), f.col("y_pred")))\
      .select(f.avg(f.col("equal")))\
      .collect()[0][0]
    
    print(f"\nTrain accuracy: {100*accuracy_train:.2f} %")
    print(f"\nTest accuracy: {100*accuracy_test:.2f} %")
