

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
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
warnings.filterwarnings("ignore")

###########---------------------CLASS---------------------###########

class DatathonML:
    
    
    def __init__(self):
        
        SVM_parameters = {'SVM__C':[0.001,0.1,10,100,10e5],
                               'SVM__gamma':[0.1,0.01]}
        RF_parameters = {'RF__n_estimators':[100, 300, 500, 800, 1200],
                              'RF__max_depth':[5, 8, 15, 25, 30]}
        
        XGB_parameters = {'XGB__n_estimators': range(60, 220, 40),
                           'XGB__max_depth':[3, 4, 5]}
        
        LR_parameters = {"LR__C":np.logspace(-3,3,7), 
                                    "LR__penalty":["l1","l2"]}
        
        LGB_parameters = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                           np.log(1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        }
        
        self.parameters = {'SVM': SVM_parameters,
                           'RF': RF_parameters,
                           'XGB': XGB_parameters,
                           'LR': LR_parameters,
                           'LGB': LGB_parameters}
        
        
        self.all_possible_models = {'SVM':SVC()}
                                    'RF':RandomForestClassifier(random_state=15325),
                                    'XGB':XGBClassifier(),
                                    'LR': LogisticRegression(),
                                    'LGB': lgb.LGBMClassifier()}
        
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
        

    def training_function_bayes(self, param_space, 
                                X_train, y_train, X_test, y_test, num_eval):
    
        def objective_function(params):
        
            clf = lgb.LGBMClassifier(**params)
            score = cross_val_score(clf, X_train, y_train, cv=5).mean()
            return {'loss': -score, 'status': STATUS_OK}
        
        trials = Trials()
        best_param = fmin(objective_function, 
                          param_space, 
                          algo=tpe.suggest, 
                          max_evals=num_eval, 
                          trials=trials,
                          rstate= np.random.RandomState(1))
        
        best_param_values = [x for x in best_param.values()]
        
        if best_param_values[0] == 0:
            boosting_type = 'gbdt'
        else:
            boosting_type= 'dart'
                           
        clf_best = lgb.LGBMClassifier(learning_rate=best_param_values[2],
                                  num_leaves=int(best_param_values[5]),
                                  max_depth=int(best_param_values[3]),
                                  n_estimators=int(best_param_values[4]),
                                  boosting_type=boosting_type,
                                  colsample_bytree=best_param_values[1],
                                  reg_lambda=best_param_values[6],
                                 )
                                      
        clf_best.fit(X_train, y_train)
        
        self.results['LGB'] = self.compute_AUC(clf_best, X_test, y_test)
        
        return 
      
        
    def select_best_models(self, all_models, range_p, flag):
        
        best_models_sort = {k: v for k, v in sorted(all_models.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True)}
                    
        if flag == 0:
            
            final_models = []
            max_value = list(best_models_sort.values())[0]
            for model in best_models_sort:
                if best_models_sort[model] >= (max_value-range_p):
                    final_models.append(model)
                    
        elif flag == 1:
            
            final_models = (list(best_models_sort.keys())[0], list(best_models_sort.values())[0])
            self.results = {}
            
        return final_models
    
            
    def first_approximation_training(self, X_train, X_test, y_train, y_test):
        
        best_models = {}
        for name in self.all_possible_models:
            model = self.all_possible_models[name]
            model.fit(X_train, y_train)
            best_models[name] = self.compute_AUC(model, X_test, y_test)
            
        return best_models
    
    
    def training_function_grid(self, X_train, X_test, y_train, y_test, parameters,
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
        first_approach = self.first_approximation_training(X_train, X_test, 
                                                        y_train, y_test)
        
        print('First approach: ', first_approach)
        
        best_models = self.select_best_models(first_approach, 0.15, 0)
        
        print('Models from first approach: ', self.all_possible_models)
        for name_model in self.all_possible_models:
            
            print(name_model)
            ### Training all possible models
            if name_model != 'LGB':
                self.training_function_grid(X_train, X_test, 
                                            y_train, y_test,
                                  self.parameters[name_model], 
                                  self.all_possible_models[name_model], 
                                  name_model)
            else:
                self.training_function_bayes(self.parameters[name_model], X_train, 
                                             y_train, X_test, 
                                             y_test, 75)
                
                 
        return self.select_best_models(self.results, 0, 1)


###########--------------------MAIN--------------------###########

### Variables
dir_dataset = 'C:/Users/Pablo.lazaro.herras1/Documents/Datathon/Datasets/'
name = 'yeast1.csv'

### Creating class
datathon = DatathonML()

### Reading dataframes
df = datathon.read_dataframe(dir_dataset, name)

### Main
final_results = datathon.auto_ml(df)
