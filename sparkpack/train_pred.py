# Databricks notebook source
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression,RandomForestRegressor,GBTRegressor,DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import *
from pyspark import SQLContext,SparkContext,SparkConf


class StackRegression:
  
  def __init__(self,X,y = None,target_col = None,learners = None,meta_learners = None):
    self.X = X
    if y !=None and target_col == None:
      self.X = self.X.withColumn('id',monotonically_increasing_id())
      self.y = self.y.withColumn('id',monotonically_increasing_id())
      self.X = self.X.join(self.y,'id','outer').drop('id')
      self.target_col = self.X.columns[-1]
    elif y==None and target_col != None:
      assert isinstance(target_col,str),print('Need to have target column name as string.')
      self.target_col = target_col
      pass
    elif y== None and target_col == None:
      print('Either provide target dataframe or target column name!!')
      return
    else:
      pass
    
    self.learners = learners
    self.meta_learner = meta_learners
    #self.create_empty_df
    self.X = self.create_data(X)
    
    if (len(self.learners)>1) and (self.meta_learner == None):
      print('When multiple learners are provided, need meta-learner.')
      return
    self.return_df
    
    
  def return_df(self):
    return self.X
    
  def create_empty_df(self):
    learner_names= [ str(x).split('_')[0] for x in self.learners ]
    lear_name = [str(x).split('_')[0] for x in learners ] + [self.target_col]
    sch_all = []
    
    for x in lear_name:
      sch_all.append(StructField(x,DoubleType(),True))
      
    schema = StructType(fields=sch_all)
    self.emp_df = spark.createDataFrame([],schema=schema)
  
  
  def create_data(self,data,data_type = 'train'):
    if data_type == 'train':
      num_cols = [x for x in data.columns if x not in [self.target_col,'label'] ]
      feat_assem = VectorAssembler(inputCols = num_cols,outputCol = 'features')
      data = feat_assem.transform(data)
      if (self.target_col != 'label') and ('label' not in data.columns):
        data = data.withColumnRenamed(self.target_col,'label')
    else:
      num_cols = [x for x in data.columns if x not in [self.target_col,'label'] ]
      feat_assem = VectorAssembler(inputCols = num_cols,outputCol = 'features')
      data = feat_assem.transform(data)
    return data
    
    
  def fit(self):
    self.learner_fit = {}
    for i,clf in enumerate(self.learners):
      self.learner_name = str(clf).split('_')[0]
      self.learner_fit[self.learner_name] = clf.fit(self.X)
      model_preds = self.learner_fit[self.learner_name].transform(self.X)
      print(self.learner_name)
      
      if i == 0:
        label_df = self.X.select('label').withColumn('id',monotonically_increasing_id())
        preds_df = model_preds.select('prediction').withColumn('id',monotonically_increasing_id()).withColumnRenamed('prediction',self.learner_name)
        #print(label_df.columns)
        #print(preds_df.columns)
      else:
        temp_preds = model_preds.withColumn('id',monotonically_increasing_id()).select('prediction','id').withColumnRenamed('prediction',self.learner_name)
        #print(temp_preds.columns)
        #print(preds_df.columns)      
        preds_df = preds_df.join(temp_preds,'id','outer')
    
    preds_df = preds_df.join(label_df,'id','outer')
    preds_df = preds_df.drop('id')
    print("learner model fitting done !")

    if self.meta_learner != None:
      meta_data = self.create_data(preds_df)
      self.meta_fit = self.meta_learner.fit(meta_data)
      print('meta learner fitted !')
    return self  
  
  
  def predict(self,test):
    test = self.create_data(test,data_type = 'test')
    if self.meta_learner == None:
      return self.learner_fit[self.learner_name].transform(test)
    else:
      for i,clf in enumerate(list(self.learner_fit.keys())):
          preds_df1 = self.learner_fit[clf].transform(test).select('prediction').withColumn('id',monotonically_increasing_id()).withColumnRenamed('prediction',clf)
          if i==0:
            preds_df = preds_df1.select('id',clf)
          else:
            preds_df = preds_df.join(preds_df1,'id','outer')
      
      preds_df = preds_df.drop('id')
      meta_data = self.create_data(preds_df,'test')
      test_meta_preds = self.meta_fit.transform(meta_data)
      
    return test_meta_preds
