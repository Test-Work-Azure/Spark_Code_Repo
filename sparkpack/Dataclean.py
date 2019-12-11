# Databricks notebook source
# Importing the Modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SparkSession
import datetime
import os

#Importing pyspark sql functions
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext

#Spark ml
from pyspark.ml.feature import Imputer

class clean:
  
  def __init__(self, Input_data):
    self.input = Input_data
  
  def missing_val_imput(self):    
    check = self.input.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in self.input.columns))
    check.show()
    print("||| Above table shows missing values accross columns |||")
    check_pd = self.input.toPandas()
    val = check_pd.isnull().any().any()
    
    if val == True:
      imputer = Imputer(
          inputCols=self.input.columns, 
          outputCols=["{}".format(c) for c in self.input.columns])
      cleaned_input = imputer.fit(self.input).transform(self.input)
      print("Missing values replaced with mean accross columns")
      print("Returning cleaned data")
      return cleaned_input

    else:
      print("No missing value found")
      return self.input
