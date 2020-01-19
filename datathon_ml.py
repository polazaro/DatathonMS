

'''''''''''Pablo Lázaro Herrasti y Manuel Jesús Galán Moreu'''''''''''

# #########----------------------------------------------------------######## #
# #########----------------------------------------------------------######## #
# #########----------------------------------------------------------######## #
# #########-----------------------DATATHON MS------------------------######## #
# #########----------------------------------------------------------######## #
# #########----------------------------------------------------------######## #
# #########----------------------------------------------------------######## #

# ##########--------------------------IMPORT------------------------######### #

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn import metrics
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
warnings.filterwarnings("ignore")

# #########--------------------------CLASS--------------------------######### #


class DatathonML:

    def __init__(self):

        '''This is the constructor of the DatathonML class'''

        # All hyperparameters for the grid search
        SVM_parameters = {'SVM__C': [0.001, 0.01, 0.1, 10, 100, 10e5],
                          'SVM__gamma': [0.1, 0.01, 0.5, 0.001]}
        RF_parameters = {'RF__n_estimators': [100, 300, 500, 800, 1000,
                                              1200, 1400]}

        XGB_parameters = {'XGB__n_estimators': [10, 30, 60, 80, 100, 120, 140,
                                                160, 180, 200, 220, 300,
                                                500, 800]}
        LR_parameters = {"LR__C": np.logspace(-3, 3, 7),
                         "LR__penalty": ["l1", "l2"]}
        KNN_parameters = {"KNN__n_neighbors": [3, 5, 7, 9, 11, 13, 15]}
        self.LGB_parameters = {
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                           np.log(1)),
            'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)}
        self.parameters = {'SVM': SVM_parameters,
                           'NB': {},
                           'RF': RF_parameters,
                           'XGB': XGB_parameters,
                           'LR': LR_parameters,
                           'LGB': self.LGB_parameters,
                           'KNN': KNN_parameters}

        # All models for training
        self.all_models = {'SVM': SVC(),
                           'NB': GaussianNB(),
                           'RF': RandomForestClassifier(random_state=15325),
                           'XGB': XGBClassifier(),
                           'LR': LogisticRegression(),
                           'LGB': lgb.LGBMClassifier(),
                           'KNN': KNeighborsClassifier()}

        # These are the dictionary needed to store the results
        self._results = {}
        self._best_parameters = {}

        print('Clase inicializada...')

    def read_dataframe(self, dir_datasets, name):

        '''Function: This function read a .csv data into pandas dataframe

        Input:
                dir_datasets: directory where the data is stored as .csv
                name: name of the .csv that is going to be read
        Output:
                dataframe: a pandas dataframe'''

        try:
            dataframe = pd.read_csv(dir_datasets + name, sep=';',
                                    skipinitialspace=True)
        except:
            print('No se ha podido leer el dataframe correctamente')
            return []

        return dataframe

    def __obtain_labels(self, df):

        '''Function: This function uses a label encoder to transform a
        categorical label into numerical one.

        Input:
                df: a pandas dataframe
        Output:
                label: all labels as list'''

        le = LabelEncoder()  # Label Encoder initialization
        label = le.fit_transform(list(df.iloc[:, -1]))

        return label

    def __preprocess_dataframe(self, df):

        '''Function: This function takes a dataframe as input and is able to
        preprocess and clean all data, first replacing the wrong characters and
        converting the type of the column, avoiding special errors. As well,
        this function erase the unique columns and separate data and labels.

            Input:
                    df: a pandas dataframe
            Output:
                    df_preprocessed: a pandasdataframe preprocessed as it
                    was explained before
                    labels: all labels as list'''

        # Erase the duplicate samples
        df = df.drop_duplicates()

        # Obtain labels
        labels = self.__obtain_labels(df)

        # Clean columns that are uniques and does not give us any information
        eliminate_columns = []
        df = df.iloc[:, :-1]  # Select dataframe except the label (last column)
        for column in df.columns:
            if df[column].dtype == 'object':  # Type replace
                df[column] = pd.to_numeric(df[column].str.replace(',', '.'),
                                           errors='coerce')
            if len(df[column].unique()) == 1:  # If the feature is unique
                eliminate_columns.append(column)
        df_preprocessed = df.drop(eliminate_columns, axis=1)

        return df_preprocessed, labels

    def __calculate_corr(self, df):

        '''Function: This function uses the corr() function to obtain a matrix
        correlation of a pandas dataframe.

            Input:
                    df: a pandas dataframe
            Output:
                    correlation: matrix correlation'''

        correlation = df.corr()

        return correlation

    def __eliminate_corr_features(self, df, threshold):

        '''Function: This function takes a dataframe as input and computes the
        correlation of the data. Than, using a threshold it drops the columns
        that are above this threshold, elminating randomly one of the two
        columns.

            Input:
                    df: a pandas dataframe
                    threshold: a threshold of correlation between features. It
                    is 0.8 by default
            Output:
                    df_no_correlation: dataframe without correlated feature
                    between them'''

        correlation = self.__calculate_corr(df)  # Computing matrix correlation
        columns = np.full((correlation.shape[0],), True, dtype=bool)
        for i in range(correlation.shape[0]):
            for j in range(i+1, correlation.shape[0]):
                if correlation.iloc[i, j] >= threshold:
                    if columns[j]:
                        columns[j] = False
        selected_columns = df.columns[columns]
        df_no_correlation = df[selected_columns]

        return df_no_correlation

    def __compute_auc(self, model, X_test, y_test):

        '''Function: This function computes the AUC for a set of data and a
        model given.

            Input:
                    model: scikit-learn model
                    X_test, y_test: data for testing the AUC
            Output:
                    AUC_metric: the AUC obtained with model'''

        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict(X_test))
        AUC_metric = metrics.auc(fpr, tpr)

        return AUC_metric

    def __select_best_models(self, all_models, range_p, flag):

        '''Function: This function select the best models in two different
        ways. This first one is when the flag = 0, where after sorting all the
        models AUC, it chooses the best model and the other ones that are at
        most a range_p far away from the best one. When flag=1, it just takes
        the best model and restart the main attributes of the class.

            Input:
                    all_models: dictionary with the name of the model and the
                    AUC values
                    range_p: percentage/100 of the maximum distance between the
                    best AUC and the other ones that are going to be considered
                    flag: flag that takes two different ways
            Output:
                    final_models: a list with the best models or the best
                    model'''

        # Sorting the dictionary by value (AUC)
        best_models_sort = {k: v for k, v in sorted(all_models.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True)}

        if flag == 0:  # For the first approximation

            final_models = ['SVM']
            max_value = list(best_models_sort.values())[0]
            for model in best_models_sort:
                if best_models_sort[model] >= (max_value-range_p):
                    final_models.append(model)

        elif flag == 1:  # For the final selection

            final_models = [list(best_models_sort)[0],
                            list(best_models_sort.values())[0],
                            self._best_parameters[list(best_models_sort)[0]]]
            self._best_parameters = {}
            self._results = {}

        else:  # Take the first three elements (best 3 models)
            try:
                auxiliar = [(model, self._best_parameters[model])
                            for model, aux in
                            list(best_models_sort.items())[:3]]
                final_models = []
                for model, params in auxiliar:
                    parameter_changed = {}
                    for param in params:
                        parameter_changed[param.split('__')[1]] = params[param]
                    final_models.append((model, parameter_changed))
            except:
                final_models = []

        return final_models

    def __first_approximation_training(self, X_train, X_test, y_train, y_test):

        '''Function: This function computes with the default hyperparameters
        the AUC of each model..

            Input:
                    X_train, y_train, X_test, y_test: training and testing data
            Output:
                    best models: dictionary with the name of the model and the
                    AUC value'''

        best_models = {}
        for name in self.all_models:
            model = self.all_models[name]
            model.fit(X_train, y_train)
            best_models[name] = self.__compute_auc(model, X_test, y_test)

        return best_models

    def __training_function_bayes(self, param_space,
                                  X_train, y_train, X_test, y_test, num_eval):

        '''Function: This function computes for the LGB classifier a bayesian
        optimization for the hyperparameters search. The output is kept in
        the class attributes to be used in other classes.

            Input:
                    X_train, y_train, X_test, y_test: training and testing data
                    num_eval: number of evaluations for the bayesian work
            Output:
                    None'''

        def objective_function(parameters):

            '''Internal function for the training_function_bayes'''

            clf = lgb.LGBMClassifier(**parameters)
            score = cross_val_score(clf, X_train, y_train, cv=5).mean()
            return {'loss': -score, 'status': STATUS_OK}

        # Bayesian optimization, the objective_function is the loss function
        trials = Trials()
        best_param = fmin(objective_function,
                          param_space,
                          algo=tpe.suggest,
                          max_evals=num_eval,
                          trials=trials,
                          rstate=np.random.RandomState(1))

        # Obtaining the best parameters
        best_param_values = [x for x in best_param.values()]
        if best_param_values[0] == 0:
            boosting_type = 'gbdt'
        else:
            boosting_type = 'dart'

        # Selecting the best model of the bayesian optimization
        best_model = lgb.LGBMClassifier(learning_rate=best_param_values[2],
                                        num_leaves=int(best_param_values[5]),
                                        max_depth=int(best_param_values[3]),
                                        n_estimators=int(best_param_values[4]),
                                        boosting_type=boosting_type,
                                        colsample_bytree=best_param_values[1],
                                        reg_lambda=best_param_values[6])

        # Fitting the best model
        best_model.fit(X_train, y_train)

        # Saving the best result
        self._results['LGB'] = self.__compute_auc(best_model, X_test, y_test)
        parameters = list(self.LGB_parameters.keys())
        self._best_parameters['LGB'] = [(parameters[i], best_param_values[i])
                                        for i in range(len(parameters))]

    def __training_function_grid(self, X_train, X_test, y_train, y_test,
                                 parameters, model, type_clf):

        '''Function: This function computes a GridSearch for the model passed
        by parameter. It saves the results in the main attributes of the class.

            Input:
                    X_train, y_train, X_test, y_test: training and testing data
                    parameters: possible hyperparameters for the grid search
                    model: scikit-learn model
                    type_clf: name of the model
            Output:
                    None'''

        steps = [('scaler', StandardScaler()),
                 ('percentile', SelectPercentile(f_classif, percentile=75)),
                 (type_clf, model)]
        pipeline = Pipeline(steps)
        grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

        grid.fit(X_train, y_train)
        self._results[type_clf] = self.__compute_auc(grid, X_test, y_test)
        self._best_parameters[type_clf] = grid.best_params_

    def __ensemble_voting(self, X_train, X_test, y_train, y_test, models):

        '''Function: This function takes the best three models and create an
        ensemble model with majority voting.

            Input:
                    X_train, y_train, X_test, y_test: training and testing data
                    parameters: possible hyperparameters for the grid search
                    model: best scikit-learn model
            Output:
                    None'''

        # Creating the models with the best parameters
        clf1 = self.all_models[models[0][0]].set_params(**models[0][1])
        clf2 = self.all_models[models[1][0]].set_params(**models[1][1])
        clf3 = self.all_models[models[2][0]].set_params(**models[2][1])
        v_ensembl = VotingClassifier(estimators=[(models[0][0], clf1),
                                                 (models[1][0], clf2),
                                                 (models[2][0], clf3)],
                                     voting='hard')

        # Fitting the model and creating the final result
        v_ensembl.fit(X_train, y_train)
        type_clf = ('Ensemble voting with: ' +
                    models[0][0] + '__' + models[1][0] + '__' + models[2][0])
        self._results[type_clf] = self.__compute_auc(v_ensembl, X_test, y_test)

    def __print_final_solution(self, final_solution):

        '''Function: This function prints the results of the best model and the
        hyperparameters selected by the grid search.

            Input:
                    final_solution: a list with the name, the AUC and the
                    hyperparamets of the best model
            Output:
                    None'''

        print('\n######### SOLUCIÓN #########\n')
        print('Best model: ' + final_solution[0])
        print('AUC = ' + str(final_solution[1]))
        print('Grid Search parameters: ')
        print(final_solution[2])
        print('\n############################\n')

    def auto_ml(self, df):

        '''Function: This function is the main function of the class. Its
        input is a pandas dataframe and it can preprocessed the dataframe,
        eliminate correlated features, obtain the labels of the dataframe and
        the features, divide the process in train and test, computes a first
        approach to choose the potential models and train the final models
        with a grid search or bayesian optimization to calculate the best
        hyperparameters. Finally, it return the best model, with the
        hyperparameters selected and the AUC.

            Input:
                    df: pandas dataframe
            Output:
                    final_solution: a list with the name, the AUC and the
                    hyperparamets of the best model '''

        # Preprocessing dataframes
        data, labels = self.__preprocess_dataframe(df)

        # Eliminating correlated features from dataset
        data_nocorr = self.__eliminate_corr_features(data, 0.8)

        # Data for training
        X = data_nocorr
        Y = labels

        # Splitting for training and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.3,
                                                            random_state=30,
                                                            stratify=Y)

        # First approximation to eliminate the worst models
        first_approach = self.__first_approximation_training(X_train, X_test,
                                                             y_train, y_test)

        # Selecting the best model (flag = 0) and the range = 0.15
        best_models = self.__select_best_models(first_approach, 0.15, 0)

        # For loop to compute the hyperparameter search for each model
        for name_model in best_models:

            print('Entrenando el clasificador ' + name_model + '...')

            if name_model != 'LGB':

                self.__training_function_grid(X_train, X_test,
                                              y_train, y_test,
                                              self.parameters[name_model],
                                              self.all_models[name_model],
                                              name_model)
            else:

                self.__training_function_bayes(self.parameters[name_model],
                                               X_train,
                                               y_train,
                                               X_test,
                                               y_test, 75)

        # Computing the final solution and the best three models
        best_three_models = self.__select_best_models(self._results, 0, 2)
        best_solution = self.__select_best_models(self._results, 0, 1)

        # Computing ensemble voting with the best 3 models and comparing
        if best_three_models:
            voting_mod = self. __ensemble_voting(X_train, X_test,
                                                 y_train, y_test,
                                                 best_three_models)
            if voting_mod[list(voting_mod)[0]] > best_solution[1]:
                final_solution = [list(voting_mod)[0],
                                  voting_mod[list(voting_mod)[0]], {}]
            else:  # Better the other options
                final_solution = best_solution
        else:  # Only the other options
            final_solution = best_solution

        # Print the final solution
        self.__print_final_solution(final_solution)

        return final_solution[1]  # Returning only the AUC
