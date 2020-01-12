

'''''''''''Pablo Lázaro Herrasti y Manuel Jesús Galán Moreu'''''''''''

###########------------------------------------------------###########
###########------------------------------------------------###########
###########------------------------------------------------###########
###########-------------------DATATHON MS------------------###########
###########------------------------------------------------###########
###########------------------------------------------------###########
###########------------------------------------------------###########

###########---------------------IMPORT---------------------###########

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC ### Support Vector Machine
from sklearn.ensemble import RandomForestClassifier ### Random forest
import xgboost as xgb
from xgboost import XGBClassifier ### XGBoosting
warnings.filterwarnings("ignore")

###########---------------------CLASS---------------------###########

class DatathonML:
    
    def __init__(self, dir_datasets_train, dir_datasets_test):
        
        self.elephant_name = 'elephant.csv'
        self.ring_name = 'ring.csv'
        self.yeast1_name = 'yeast1.csv'
        
        self.dir_datasets_train = dir_datasets_train
        self.dir_datasets_test = dir_datasets_test
        
        
    def read_dataframe(self, dir_datasets, name):
        
        return pd.read_csv(dir_datasets + name, sep=';', skipinitialspace=True)
    
    
    def obtain_labels(self, df, pos):
    
        if pos == 1 or pos == 2:
            return list(df.iloc[:,-1])
        else:
            return [0 if label == 'negative' else 1 for label in list(df.iloc[:,-1])]
        
        
    def preprocess_dataframe(self, df, pos, mode):
    
        ### Erase the duplicate samples
        df = df.drop_duplicates() 
        
        ### Obtain labels if we are in training
        if mode == 'Train':
            labels = self.obtain_labels(df, pos)
        else: 
            labels = []
        
        ### Clean columns that are uniques and does not give us any information
        eliminate_columns = []
        df = df.iloc[:,:-1]
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = pd.to_numeric(df[column].str.replace(',','.'), errors='coerce')
            if len(df[column].unique()) == 1:
                eliminate_columns.append(column)
        df_drop = df.drop(eliminate_columns, axis=1)
    
        return df_drop, labels, eliminate_columns
    
    
    def calculate_corr(self, df):
        
        return df.corr() 
    
    
    def eliminate_corr_features(self, df, corr, threshold):
        
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= threshold:
                    if columns[j]:
                        columns[j] = False
        selected_columns = df.columns[columns]
        df = df[selected_columns]
        
        return df


    def auto_ml(self, df_test):
        
        ### Reading dataframes
        df_elephant = self.read_dataframe(self.dir_datasets_train, self.elephant_name)
        df_ring = self.read_dataframe(self.dir_datasets_train, self.ring_name)
        df_yeast1 = self.read_dataframe(self.dir_datasets_train, self.yeast1_name)
        
        ### Preprocessing dataframes
        data_elephant, labels_elephant, eliminate_elephant = self.preprocess_dataframe(df_elephant, 1, 'Train')
        data_ring, labels_ring, eliminate_ring = self.preprocess_dataframe(df_ring, 2, 'Train')
        data_yeast1, labels_yeast1, eliminate_yeast1 = self.preprocess_dataframe(df_yeast1, 3, 'Train')
        
        ### Eliminating correlated features from dataset
        data_elephant_nocorr = self.eliminate_corr_features(data_elephant, self.calculate_corr(data_elephant), 0.8)
        data_ring_nocorr = self.eliminate_corr_features(data_ring, self.calculate_corr(data_ring), 0.8)
        data_yeast1_nocorr = self.eliminate_corr_features(data_yeast1, self.calculate_corr(data_yeast1), 0.8)
        
        ### Data for training
        X1 = data_elephant_nocorr
        X2 = data_ring_nocorr
        X3 = data_yeast1_nocorr
        Y1 = labels_elephant
        Y2 = labels_ring
        Y3 = labels_yeast1
        
        ### Splitting for training
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,Y1,test_size=0.2, random_state=30, stratify=Y1)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,Y2,test_size=0.2, random_state=30, stratify=Y2)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3,Y3,test_size=0.2, random_state=30, stratify=Y3)

        return data_elephant, data_ring, data_yeast1


###########--------------------MAIN--------------------###########

dir_train = 'C:/Users/Pablo.lazaro.herras1/Documents/Datathon/Datasets/'
dir_test = 'C:/Users/Pablo.lazaro.herras1/Documents/Datathon/Datasets/'
datathon = DatathonML(dir_train, dir_test)
a,b,c = datathon.auto_ml(1)
