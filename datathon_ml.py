

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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC ### Support Vector Machine
from sklearn.ensemble import RandomForestClassifier ### Random forest
import xgboost as xgb
from xgboost import XGBClassifier ### XGBoosting
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

###########---------------------CLASS---------------------###########

class DatathonML:
    
    def __init__(self):
        
        self.SVM_parameters = {'SVM':{'SVM__C':[0.001,0.1,10,100,10e5],
                               'SVM__gamma':[0.1,0.01]}}
        self.RF_parameters = {'RF':{'RF__n_estimators':[100, 300, 500, 800, 1200],
                              'RF__max_depth':[5, 8, 15, 25, 30]}}
        
        self.XGB_parameters = {'XGB':{'XGB__n_estimators': range(60, 220, 40),
                           'XGB__max_depth':[3, 4, 5]}}
        
        self.LR_parameters = {'LR':{"LR__C":np.logspace(-3,3,7), 
                                    "LR__penalty":["l1","l2"]}}
        
        
        
        self.all_possible_models = {'RF':RandomForestClassifier(random_state=15325),
                                    'SVM':SVC(),
                                    'XGB':XGBClassifier(),
                                    'LR': LogisticRegression()}
        
        self.threshold_distance = 0.1
        self.results = {}
        
        print('Clase inicializada')
        
        
    def read_dataframe(self, dir_datasets, name):
        
        return pd.read_csv(dir_datasets + name, sep=';', skipinitialspace=True)
    
    
    def obtain_labels(self, df):
        
            le = LabelEncoder()
            return le.fit_transform(list(df.iloc[:,-1]))

        
        
    def preprocess_dataframe(self, df):
    
        ### Erase the duplicate samples
        df = df.drop_duplicates()
        
        ### Obtain labels
        labels = self.obtain_labels(df)
        
        ### Clean columns that are uniques and does not give us any information
        eliminate_columns = []
        df = df.iloc[:,:-1]
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = pd.to_numeric(df[column].str.replace(',','.'), 
                  errors='coerce')
            if len(df[column].unique()) == 1:
                eliminate_columns.append(column)
        df_drop = df.drop(eliminate_columns, axis=1)
    
        return df_drop, labels
    
    
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


    def compute_AUC(self, model, X_test, y_test):
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
        return metrics.auc(fpr, tpr)
    
    
    def first_approximation_training(self, X_train, X_test, y_train, y_test):
        
        best_models = {}
        for name in self.all_possible_models:
            model = self.all_possible_models[name]
            model.fit(X_train, y_train)
            best_models[name] = self.compute_AUC(model, X_test, y_test)
            
        return best_models
    
    
    def training_function(self, X_train, X_test, y_train, y_test, parameters,
                          model, type_clf):
        
        steps = [('scaler', StandardScaler()), (type_clf, model)]
        pipeline = Pipeline(steps)
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
        
        grid.fit(X_train, y_train)
        self.results[type_clf] = self.compute_AUC(grid, X_test, y_test)
        

    def auto_ml(self, df):
        
        
        ### IDEA: HACER TODOS LOS MODELOS CON UN PARÁMETRO POR DEFECTO Y ELIMINAR
        ### AQUELLOS QUE DEN BAJOS. QUEDARSE CON LOS 3 MÁS ALTOS Y HACER GRID_SEARCH
        
        ### Preprocessing dataframes
        data, labels = self.preprocess_dataframe(df)
       
        ### Eliminating correlated features from dataset
        data_nocorr = self.eliminate_corr_features(data, 
                                                   self.calculate_corr(data),
                                                   0.8)
        
        ### Data for training
        X = data_nocorr
        Y = labels
        
        ### Splitting for training and test
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, 
                                                            random_state=30, 
                                                            stratify=Y)

        ### First approximation to eliminate the worst models
        best_models = self.first_approximation_training(X_train, X_test, 
                                                        y_train, y_test)
        
        ### Training all possible models
        self.training_function(X_train, X_test, y_train, y_test,
                          self.RF_parameters, 
                          RandomForestClassifier(random_state=15325), 
                          'RF')
        
        self.training_function(X_train, X_test, y_train, y_test,
                          self.SVM_parameters, 
                          SVC(), 
                          'SVM')
        self.training_function(X_train, X_test, y_train, y_test,
                          self.XGB_parameters, 
                          XGBClassifier(), 
                          'XGB')
        
        
#        self.training_function(self, X_train, X_test, y_train, y_test,
#                          self.RF_parameters, 
#                          SVC(), 
#                          type_clf='RF')
#        
        return self.results


###########--------------------MAIN--------------------###########

### Variables
dir_dataset = 'C:/Users/Pablo.lazaro.herras1/Documents/Datathon/Datasets/'
name = 'ring.csv'

### Creating class
datathon = DatathonML()

### Reading dataframes
df = datathon.read_dataframe(dir_dataset, name)

### Main
Y = datathon.auto_ml(df)
