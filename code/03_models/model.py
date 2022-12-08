# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 17:20:48 2022

"""

from tensorflow.keras import layers 
from tensorflow.keras import models
import keras_nlp

## Feature Extraction - Small CNN

def small_cnn(input_layer):
    x = layers.Conv1D(filters=64, kernel_size=50, strides=6, padding="same", name="small_conv1")(input_layer)
    x = layers.BatchNormalization(name="small_BN1")(x)
    x = layers.ReLU(name="small_relu1")(x)
    
    x = layers.MaxPool1D(pool_size=8, strides=8, padding="same")(x)
    
    x = layers.Dropout(0.5, name="small_dropout1")(x)
    
    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding="same", name="small_conv2")(x)
    x = layers.BatchNormalization(name="small_BN2")(x)
    x = layers.ReLU(name="small_relu2")(x)
    
    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding="same", name="small_conv3")(x)
    x = layers.BatchNormalization(name="small_BN3")(x)
    x = layers.ReLU(name="small_relu3")(x)
    
    x = layers.Conv1D(filters=128, kernel_size=8, strides=1, padding="same", name="small_conv4")(x)
    x = layers.BatchNormalization(name="small_BN4")(x)
    x = layers.ReLU(name="small_relu4")(x)
    
    x = layers.MaxPool1D(pool_size=8, strides=8, padding="same")(x)
    # x = layers.Dropout(0.4, name="small_dropout2")(x)
  
    return x

## Feature Extraction - Large CNN

def large_cnn(input_layer):

    x = layers.Conv1D(filters=64, kernel_size=400, strides=50, padding="same", name="large_conv1")(input_layer)
    x = layers.BatchNormalization(name="large_BN1")(x)
    x = layers.ReLU(name="large_relu1")(x)
    
    x = layers.MaxPool1D(pool_size=4, strides=4, padding="same", name="large_maxpool1")(x)
      
    x = layers.Dropout(0.5, name="large_dropout1")(x)
      
    x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", name="large_conv2")(x)
    x = layers.BatchNormalization(name="large_BN2")(x)
    x = layers.ReLU(name="large_relu2")(x)
      
    x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", name="large_conv3")(x)
    x = layers.BatchNormalization(name="large_BN3")(x)
    x = layers.ReLU(name="large_relu3")(x)
      
    x = layers.Conv1D(filters=128, kernel_size=6, strides=1, padding="same", name="large_conv4")(x)
    x = layers.BatchNormalization(name="large_BN4")(x)
    x = layers.ReLU(name="large_relu4")(x)
      
    x = layers.MaxPool1D(pool_size=2, strides=2, padding="same", name="large_maxpool2")(x)
    # x = layers.Dropout(0.4, name="large_dropout2")(x)
      
    return x

def transformer(inputs, head_size=256, heads=4, dim_feed_forw=128, attention_dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=heads, dropout=attention_dropout)(x, x)
    x = layers.Dropout(0.4)(x)
    attention_out = x + inputs
    
    x = layers.LayerNormalization(epsilon=1e-6)(attention_out)
    x = layers.Conv1D(filters=dim_feed_forw, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    
    return x + attention_out

def Sleep_Conv_Net(num_classes=5, 
                   head_size=256, 
                   heads=4, 
                   dim_feed_forw=128, 
                   attention_dropout=0):
    
    input_layer = layers.Input(shape=(3000,3))
    x1 = small_cnn(input_layer)
    x2 = large_cnn(input_layer) 
    
    features = layers.Concatenate()([x1,x2])
    features = layers.Dropout(0.4, name="concat_dropout")(features)
    
    pos_encodings = keras_nlp.layers.SinePositionEncoding()(features)
    
    transformer_inputs = features + pos_encodings
    
    x = transformer(transformer_inputs, head_size, heads, dim_feed_forw, attention_dropout)
    x = transformer(x, head_size, heads, dim_feed_forw, attention_dropout)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(num_classes, activation="relu")(x)
    
    residual = layers.Flatten()(features)
    residual = layers.Dense(128, activation="relu")(residual)
    residual = layers.Dense(num_classes, activation="relu")(residual)
    
    x = x + residual
    
    output = layers.Softmax()(x)
    
    model = models.Model(inputs=input_layer, outputs=output)
    return model
    