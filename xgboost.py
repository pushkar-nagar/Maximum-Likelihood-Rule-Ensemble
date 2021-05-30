'''
Implementation of a SDS Classifier so that it can be defined separately from the training file
and the changes in the training file can be minimal
'''

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import xgboost
from utils import segregate_y_in_seq
from warnings import warn
import copy

class SDSClassifier(BaseEstimator, ClassifierMixin):
    '''A classifier class for SDS, implementing the least number of required methods,
    being used in training file for any type of classifier that is to be used'''

    def __init__(self, model_params_dict, **train_params_dict):
        '''Initialize with an external classifier using a dictionary that contains all the parameters and their values
        that will be used by the external classifier'''
        self.model_params_dict = model_params_dict
        self.train_params_dict = train_params_dict

    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, y_val_seq_len=None,
            perf_metric_func=None, **kwargs):
        '''
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        '''

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if X_val is not None:
            X_val, y_val = check_X_y(X_val, y_val)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        num_boost_round = self.train_params_dict['num_boost_round']
        warm_start_classifier = warm_start_model.classifier if kwargs.get('warm_start_model') is not None else None
        if kwargs.get('warm_start_model') is not None:
            self.model_params_dict.update({'process_type': 'update', 'updater': 'refresh',
                                           'refresh_leaf': True})

            num_boost_round = warm_start_classifier.best_ntree_limit


        if self.train_params_dict.get('early_stopping_rounds'):
            val_start_idx = int(round((1-self.train_params_dict['early_stop_eval_frac'])*len(X)))
            X_train, y_train, w_train = X[:val_start_idx], y[:val_start_idx], None

            if sample_weight is not None:
                w_train = sample_weight[:val_start_idx]
                D_eval = xgboost.DMatrix(X[val_start_idx:], y[val_start_idx:], weight=sample_weight[val_start_idx:])
            else:
                D_eval = xgboost.DMatrix(X[val_start_idx:], y[val_start_idx:])

            self.classifier = self.batched_train(self.model_params_dict, X_train, y_train, w_train=w_train,
                                            batch_size=self.train_params_dict.get('batch_size'),
                                            num_boost_round=num_boost_round,
                                            evals=[(D_eval,'eval')],
                                            early_stopping_rounds=self.train_params_dict['early_stopping_rounds'],
                                            verbose_eval=False, xgb_model=warm_start_classifier)
        else:
            X_train, y_train, w_train = X, y, sample_weight
            self.classifier = self.batched_train(self.model_params_dict, X_train, y_train, w_train=w_train,
                                            batch_size=self.train_params_dict.get('batch_size'),
                                            num_boost_round=num_boost_round,
                                            xgb_model=warm_start_classifier)

        if self.train_params_dict.get('staged_best') and X_val is not None and y_val is not None:
            performances = self.staged_performance(X_val, y_val, y_val_seq_len, perf_metric_func)
            opt_boost_round = np.argmin(performances) + 1

            self.classifier = self.batched_train(self.model_params_dict, X_train, y_train, w_train=w_train,
                                            batch_size=self.train_params_dict.get('batch_size'),
                                            num_boost_round=opt_boost_round,
                                            xgb_model=warm_start_classifier)

        self.feature_importances_ = self.classifier.get_fscore()
        self.feature_importances_ = np.array([self.feature_importances_.get('f'+str(i),0)
                                              for i in range(X.shape[1])])

        self.feature_importances_ = self.feature_importances_ / sum(self.feature_importances_)

        return self

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['classifier'])

        # Input validation
        X = check_array(X)

        D_test = xgboost.DMatrix(X)
        proba = self.classifier.predict(D_test)
        proba = np.transpose(np.array([1-proba, proba]))

        return proba

    @staticmethod
    def validate_parameters(tunable_model_params, nontunable_model_params, model_train_params,
                            **kwargs):
        '''Check whether the parameters provided are non-conflicting and valid.'''
        if model_train_params.get('staged_best') and model_train_params.get('early_stopping_rounds'):
            if model_train_params.get('verbose'):
                print('Both Early Stopping and Staged Best have been asked to be performed. Staged Best '
                      'takes precedence.')
            model_train_params['early_stopping_rounds'] = None

        # Validate that if num_boost_round is in tunable_model_params then staged_best is set as False.
        # If not then set it to False
        if model_train_params.get('staged_best') and 'num_boost_round' in {i['param_name'] for i in tunable_model_params}:
            model_train_params['staged_best'] = False
            warn("'staged_best' can't be set as True while 'num_boost_round' is tunable. Setting "
                 "'staged_best' value as False.", RuntimeWarning)

        if kwargs.get('update_train_and_model_params') and 'num_boost_round' in kwargs['model_params_dict']:
            model_train_params['num_boost_round'] = kwargs['model_params_dict']['num_boost_round']
            kwargs['model_params_dict'].pop('num_boost_round')

    def batched_train(self, params_dict, X_train, y_train, w_train=None, batch_size=-1, **kwargs):
        D_train_batches = self.get_batches(X_train, y_train, weight=w_train, batch_size=batch_size)
        classifier = xgboost.train(params_dict, next(D_train_batches), **kwargs)

        self.model_params_dict.update({'process_type': 'update', 'updater': 'refresh',
                                       'refresh_leaf': True})

        for train_batch in D_train_batches:
            kwargs['xgb_model'] = classifier
            classifier = xgboost.train(params_dict, train_batch, **kwargs)

        for key in ('process_type', 'updater', 'refresh_leaf'):
            self.model_params_dict.pop(key)

        return classifier

    def get_batches(self, X_train, y_train, weight=None, batch_size=-1):
        batch_size = X_train.shape[0] if (batch_size == -1 or batch_size is None) else batch_size
        if weight is not None:
            for i in range(0, X_train.shape[0], batch_size):
                yield xgboost.DMatrix(X_train[i:i+batch_size], y_train[i:i+batch_size],
                                               weight=weight[i:i+batch_size])
        else:
            for i in range(0, X_train.shape[0], batch_size):
                yield xgboost.DMatrix(X_train[i:i+batch_size], y_train[i:i+batch_size])

    def wrap_up(self):
        if self.model_params_dict.get('n_gpus') is not None:
            classifier = copy.deepcopy(self.classifier)
            self.classifier.__del__() # needs to be deleted to free the reserved GPU memory
            self.classifier = classifier

    def loss_(self, y, y_pred):
        '''This function not necessarily required as the performance evaluation can be done using other methods too.
        Return the loss value'''
        epsilon = np.finfo(float).eps
        loss = np.sum(-(y*np.log(y_pred+epsilon)+(1-y)*np.log(1-y_pred+epsilon))) / y.shape[0]
        return loss

    def staged_performance(self, X, y, y_seq_len=None, perf_metric_func=None):
        D_val = xgboost.DMatrix(X)
        performances = np.zeros((self.classifier.best_ntree_limit,), dtype=np.float64)

        if y_seq_len is None or perf_metric_func is None:
            for i, y_pred in enumerate(self.staged_decision_function(X)):
                performances[i] = self.loss_(y, y_pred)
        else:
            y_by_seq = segregate_y_in_seq(y, y_seq_len)
            for i, y_pred in enumerate(self.staged_decision_function(X)):
                y_pred_by_seq = segregate_y_in_seq(y_pred, y_seq_len)
                performances[i] = perf_metric_func(y_pred_by_seq, y_by_seq).score

        return performances

    def staged_decision_function(self, X):
        D = xgboost.DMatrix(X)
        for i in range(self.classifier.best_ntree_limit):
            yield self.classifier.predict(D, ntree_limit = i+1)

if __name__ == '__main__':
    from sklearn import datasets
    cancer_data = datasets.load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target

    xgb_clf = SDSClassifier({'learning_rate' : 0.01},
                            **{'num_boost_round': 100})
    xgb_clf.fit(X,y)
    proba = xgb_clf.predict_proba(X)
    print(proba[:5])
    i = 0
