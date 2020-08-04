from django.http import HttpResponse
from clickhouse_driver import Client
import h2o
h2o.init(nthreads=100, max_mem_size='12g',strict_version_check = False)
import json
import ast
##Lin reg
import pandas as pd; import numpy as np
from multiprocessing.pool import ThreadPool
from django.views.decorators.csrf import csrf_exempt
#import scipy
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from Linear_prep.R2P_dataprep import columns_data_type, remove_col, missing_val_impute, GBM_impute, transformation, transformation_inv, ch_sq_test, correlations,vif, variable_importance_h2o, correlations_ds, dt_transformation_inv
import warnings
import sklearn.utils._cython_blas
#import sklearn.neighbors.typedefs
#import sklearn.neighbors.quad_tree
import sklearn.tree._utils
warnings.filterwarnings("ignore")
from sklearn import preprocessing
print('Done')
## Logistic
import Linear_prep.R2P_linregr_dataprep as lp_log
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score#, f1_score, roc_curve, auc
from sklearn.metrics import classification_report

## Clustering
from h2o.estimators.gbm import H2OGradientBoostingEstimator
#from h2o.estimators.kmeans import H2OKMeansEstimator
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#import scipy
import Linear_prep.R2P_cluster_dataprep as lp_c# columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv,correlations_cluster
from Linear_prep.R2P_cluster import cluster_profiling, kMeans_model

## DBScan

#from Linear_prep.R2P_cluster_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv,correlations_cluster
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
print('Done2')
## Decision Tree

from sklearn.tree import DecisionTreeClassifier
#, DecisionTreeRegressor
#from sklearn.tree.export import export_text
#from sklearn.metrics import r2_score
from sklearn.tree import _tree
from imblearn.over_sampling import SMOTE

#from math import sqrt

## Envotting

#from django.shortcuts import render
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

## Forecasting

import Linear_prep.Forecasting_prep as lp_f #is_date, date2stamp, columns_data_type, remove_col, missing_val_impute, GBM_impute, transformation, transformation_inv, ch_sq_test, correlations,vif, variable_importance_h2o
#from sklearn.metrics import mean_squared_error
from scipy import stats
import math
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
#from numpy import array
#from keras.models import Sequential
#from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
print('done3')
## MLP

from sklearn.neural_network import MLPClassifier
'''
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from plot_metric.functions import BinaryClassification
from sklearn.ensemble import ExtraTreesClassifier
'''
## Random forest

from h2o.estimators.random_forest import H2ORandomForestEstimator
from word2number import w2n
import n2w as nw
from h2o.grid.grid_search import H2OGridSearch

## Sentiment Analysis

import gensim
#import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel
import spacy
#import en_core_web_sm
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from nltk.corpus import stopwords
import re
#from textblob import TextBlob
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from pprint import pprint
# Plotting tools
#import pyLDAvis
#import pyLDAvis.sklearn
from collections import Counter
#import Linear_prep.textanalysis as qt
from Linear_prep.textanalysis import sent_to_words, remove_stopwords, remove_stopwords, lemmatization, tuple_to_vector, clean_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from textanalysis import sent_to_words,sentimentscore,remove_stopwords,make_bigrams,make_trigrams,lemmatization,show_topics,sentiment_analysis

## Smart Insights

import Linear_prep.smartinsights as smi
import scipy
'''
## Xgboost
import re
from xgboost import XGBClassifier
import xgboost as xgb
'''
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/datasnapshot", methods = ['POST'])
def datasnap():
    print("Starting")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    print(columnsArray)
    pool = ThreadPool()
    async_result = pool.apply_async(datasnapshotThread, (query,dbName, password,userName,columnsArray,dbHost,columnList))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def datasnapshotThread(query,dbName,password,userName,columnsArray,dbHost,columnList):
    client = Client(dbHost,user=userName,password=password,database=dbName)
    print("in  datasnapshot")
    print(query)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    print("in  datasnapshottdryrtdtrdddrtrd")
    print(df)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    col_namechange = pd.DataFrame(json.loads(columnsArray))
    print(type(col_namechange))
    print(col_namechange)
    #data = data.replace(to_replace = [".",",","@","#","$","%","^","&","*","(",")","-","+","=", "?"],value = np.nan)
    ent_cor,chisq_dependency,miss_cols, data  = correlations_ds(data, columnsArray=columnsArray_e, no_rem_col='none', Val = data.columns)
    cat_data = data.select_dtypes(include=['category', 'object']).copy()
    num_data = data.select_dtypes(include=['number', 'int', 'float']).copy()
    print(chisq_dependency)
    if len(ent_cor)>0:
        for i in ent_cor.columns:
            for j in range(len(ent_cor[i])):
                ent_cor[i].iloc[j] = round(ent_cor[i].iloc[j],3)
    if len(chisq_dependency)>0:           
        for j in range(len(chisq_dependency['Chi.Square'])):
                chisq_dependency['Chi.Square'].iloc[j] = round(chisq_dependency['Chi.Square'].iloc[j],3)
    
    if len(ent_cor)>0:
        l1 = ent_cor.to_dict(orient='records')
        names=ent_cor.columns
        print("Done till checkpoint 1")
        for i in range(0,len(ent_cor.columns)):
            l1[i].update([('_row',names[i])])
        for k in [1,2]:
            for i in range(len(l1)):
                for key, value in l1[i].items():
                    for j in range(len(col_namechange)):
                        if key == col_namechange['columnDisplayName'][j]:
                            temp_val = l1[i][key]
                            del l1[i][key]
                            l1[i][col_namechange['changedColumnName'][j]] = temp_val

        for i in range(len(l1)):
            tem_keys = list(l1[i].keys())
            tem_keys.remove(tem_keys[0])
            for key, value in l1[i].items():
                if key == '_row':
                    #tem_row = l1[i]['_row']
                    del l1[i]['_row']
                    l1[i].update([('_row',tem_keys[i])])
                    break
        list1 = [l1]

    elif (len(ent_cor)==0 and num_data.shape[1] <1):
        l1 = 'No Numerical data'
        list1 = [l1]
    elif num_data.shape[1] == 1:
        l1 = 'Only one Numeric column'
        list1 = [l1]

    print('length of Chi square dep is : ' + str(len(chisq_dependency)))
    print('length of Cat data is : ' + str(len(cat_data.columns)))
    if len(chisq_dependency)>0:
        l2 = [chisq_dependency]
        list2 = l2
        try:
            for i in range(len(chisq_dependency)):
                for j in range(len(col_namechange)):
                    if chisq_dependency['Row'].iloc[i] == col_namechange['columnDisplayName'][j]:
                        chisq_dependency['Row'].iloc[i] = col_namechange['changedColumnName'][j]
                for j in range(len(col_namechange)):
                    if chisq_dependency['Column'].iloc[i] == col_namechange['columnDisplayName'][j]:
                        chisq_dependency['Column'].iloc[i] = col_namechange['changedColumnName'][j]
            l2 = chisq_dependency.to_dict(orient='records')
            if (len(l2) ==0 and cat_data.shape[1] <1):
                print('try 1st elif')
                l2 = 'No Categorical data'
                list2 = [l2]
            elif len(l2) ==0 and cat_data.shape[1] == 1:
                print('try 2nd elif')
                l2 = 'Only one Categorical column'
                list2 = [l2]
            list2 = l2
        except Exception:
            if (len(l2) ==0 and cat_data.shape[1] <1):
                print('except 1st elif')
                l2 = 'No Categorical data'
                list2 = [l2]
            elif cat_data.shape[1] == 1:
                print('except 2nd elif')
                l2 = 'Only one Categorical column'
                list2 = [l2]
    elif (len(chisq_dependency) ==0 and cat_data.shape[1] <1):
                print('Outer 1st elif')
                l2 = 'No Categorical data'
                list2 = [l2]
    elif cat_data.shape[1] == 1:
                print('Outer 2nd elif')
                l2 = 'Only one Categorical column'
                list2 = [l2]
    else:
                print('Outer 3rd elif')
                l2 = 'No Correlation'
                list2 = [l2]

    #list3 = [rm_cols]
    #list4 = data.to_dict(orient='records')
    missing_columns = []
    for i in miss_cols:
        for j in range(len(col_namechange)):
            if i == col_namechange['columnDisplayName'][j]:
                missing_columns.append(col_namechange['changedColumnName'][j])

    listed = [list1, list2, missing_columns]
    print(listed)
    print("Done till checkpoint - listed")
    return listed

##### Linear regression

@app.route("/linear_regression", methods = ['POST'])
def linearRegression():
    print("in linearRegression")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    yValue =request.form.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    print(yValue)
    print("in linearRegression y ")
    parametersObj= request.form.get('parametersObj')
    print(parametersObj)
    pool = ThreadPool()
    async_result = pool.apply_async(linearRegressionThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,yValue,parametersObj))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def linearRegressionThread(query,dbName,password,userName,columnsArray,dbHost,columnList,yValue,parametersObj):
    print("in linearRegressionThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    parametersObj = json.loads(parametersObj)
    #print(yValue)
    #data, columnsArray_e = columns_data_type(df, columnsArray)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    nan_list = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"]
    for i in nan_list:
        data = data.replace(to_replace = i,value = np.nan)
    for i in range(len(columnsArray_e)):
        if yValue == columnsArray_e['changedColumnName'].iloc[i]:
            yValue = columnsArray_e['columnName'].iloc[i]
        else:
            continue

    dat_lis = []
    for i in data.columns:
        for j in range(len(columnsArray_e['changedColumnName'])):
            if i == columnsArray_e['changedColumnName'].iloc[j]:
                dat_lis.append(columnsArray_e['columnName'].iloc[j])
    try:
        data.columns = dat_lis
    except Exception:
        data.columns = list(data.columns)
    print(data.shape[0])
    if data.shape[0]>150:
            print(data)
            if len(data[yValue].unique())>1:
                ent_cor,data,rm_cols, miss_cols, obj_t, sub, drop_columns,trans_col = correlations(data, columnsArray=columnsArray_e, no_rem_col='none', Val = data.columns)
                data = data.loc[:,~data.columns.duplicated()]
                if data.shape[1]>1:
                    xValues = list(filter(lambda x: x not in rm_cols and x in data.columns, data.columns))
                    data_x_unique = data[xValues]
                    for i in data_x_unique.columns:
                        if len(data_x_unique[i].unique())>1:
                            continue
                        else:
                            if data[i].dtypes != 'float' and  data[i].dtypes != 'int':
                                print('dropping ' + str(i))
                                data = data.drop(columns = [i])
                                drop_columns.append(i)
                            else:
                                if data_x_unique.shape[1]>1:
                                    continue
                                else:
                                   data = data.drop(columns = [i])
                                   drop_columns.append(i)
                else:
                    listed = ['There is only one column after preprocessing. Please try giving different inputs or different dataset as the dataset might not be applicable for this model.']
                    return listed
                if data.shape[1]>1:
                    print(data.shape[1])
                    print('inside imp shape')
                    if yValue not in list(rm_cols) and yValue not in drop_columns:
                        xValues = list(filter(lambda x: x not in rm_cols and x in data.columns, data.columns))
                        try:
                            variable_imp = variable_importance_h2o(data, xValues, yValue)
                        except Exception as e:
                            msg = str(e)
                            if 'Response cannot be constant' in msg:
                                listed = ['Target Variable cannot be a constant value.']
                            else:
                                listed = [msg]
                            return listed
                        #labels = xValues
                        var_imp = []; var_name=[]
                        for i in range(len(variable_imp)):
                            for j in range(len(variable_imp[i])):
                                var_imp.append(variable_imp[i][j][2])
                                var_name.append(variable_imp[i][j][0])
                        Var_imp  = pd.DataFrame(columns = ['y'], index = range(len(var_imp)))
                        Var_imp['y'] = var_imp
                        Var_name = pd.DataFrame(columns = ['x'], index = range(len(var_name)))
                        Var_name['x'] = var_name
                        frame_var = [Var_name, Var_imp]
                        Var_imp = pd.concat(frame_var, axis = 1)

                        vif_frame = [data[xValues], data[yValue]]
                        data_vif = pd.concat(vif_frame, axis = 1)
                        data_vif = data_vif.loc[:,~data_vif.columns.duplicated()]
                        print(data_vif.dtypes)
                        num_data = data_vif.select_dtypes(include=['int', 'float', 'number']).copy()

                        #vif_var = vif(num_data, yValue, 10)
                        Nfolds = parametersObj['nfolds']
                        Nfolds = int(Nfolds)
                        ###NLG
                        '''
                        print(num_data)
                        if (num_data.shape[1]>0):
                            num_data_inv = transformation_inv(num_data,obj_t)
                            print(num_data_inv)
                        else:
                            num_data_inv = []
                        num_x = list(filter(lambda x: x in num_data, xValues))
                        len_num = len(num_x)
                        if (len_num>1):
                            x_mean = num_data_inv[num_x].mean()
                            x_min = num_data_inv[num_x].min()
                            x_max = num_data_inv[num_x].max()
                            x_quant25 = num_data_inv[num_x].quantile(0.25)
                            x_quant50 = num_data_inv[num_x].quantile(0.5)
                            x_quant75 = num_data_inv[num_x].quantile(0.75)
                            x_skew = scipy.stats.skew(num_data_inv[num_x])
                        elif (len_num>0):
                            x_mean = num_data_inv[num_x].mean()
                            print(x_mean)
                            x_min = num_data_inv[num_x].min()
                            x_max = num_data_inv[num_x].max()
                            x_quant25 = []
                            x_quant50 = []
                            x_quant75 = []
                            x_skew = []
                        else:
                            x_mean = []
                            x_min = []
                            x_max = []
                            x_quant25 = []
                            x_quant50 = []
                            x_quant75 = []
                            x_skew = []
                        '''
                        ### Data preperation

                        #ind = range(data.shape[0])
                        data_x = data[xValues]
                        cat_data = data_x.select_dtypes(include=['object', 'category']).copy()
                        '''
                        if (cat_data.shape[1]>0):
                            print("Categorical Encoding started")
                            n_data = data_x.select_dtypes(include=['int', 'float', 'number']).copy()
                            # label_encoder object knows how to understand word labels.
                            lab = []
                            for i in range(cat_data.shape[1]):
                                lab.append(preprocessing.LabelEncoder())
                            for j in range(len(lab)):
                                for i in cat_data.columns:
                                    try:
                                        data_x[i] = data_x[i].str.encode('utf-8')
                                        data_x[i] = lab[j].fit_transform(data_x[i])
                                        break
                                    except Exception:
                                        continue
                            print(data_x)
                            data_x_category = []
                            for i in range(cat_data.shape[1]):
                                data_x_cat = pd.get_dummies(data_x[cat_data.columns[i]])
                                print('here')
                                data_x_cat= data_x_cat.astype('Int64')
                                col_val= []
                                for j in range(data_x_cat.shape[1]):
                                    col_val.append(cat_data.columns[i]+str(j))
                                dataframe = pd.DataFrame(columns = col_val, index = range(len(data_x_cat)))
                                for j in range(data_x_cat.shape[1]):
                                    dataframe[col_val[j]] = data_x_cat[j]
                                data_x_category.append(dataframe)

                            data_x_category.append(data_x[n_data.columns])
                            final_datax = pd.concat(data_x_category, axis = 1)
                            print("Categorical Encoding completed")
                        else:
                            final_datax = data_x

                        data_y = pd.DataFrame(data = data[yValue], columns = [yValue], index = ind)
                        frames= [final_datax, data_y]
                        data2 = pd.concat(frames, axis=1)
                        '''
                        data2 = data_vif
                        family = parametersObj['family']
                        family = str(family)
                        data2 = data2.loc[:,~data2.columns.duplicated()]
                        #print(data2[yValue].unique())

                        if data2.shape[1]>1 and len(data2[yValue].unique())>1:
                            if(data[yValue].dtypes == 'float') or (data[yValue].dtypes == 'int64'):
                                try:
                                    hf = h2o.H2OFrame(data2)
                                    if  hf.shape[1]<1:
                                        try:
                                                for i in data2.columns:
                                                    if data2[i].dtypes == 'float':
                                                        continue
                                                    else:
                                                         data2[i] = data2[i].str.encode('utf-8')
                                        except Exception:
                                                data2 = data2
                                        hf = h2o.H2OFrame(data2)
                                    else:
                                        hf = hf
                                except Exception as e:
                                    msg = str(e)
                                    listed = [msg]
                                    return listed
                                print("Finding variable importance by taking given numeric variable as a dependent variable")
                                train, valid, test = hf.split_frame(ratios=[.7, .1])
                                x_Values= list(data2.columns)
                                print("xvalues done")
                                x_Values.remove(yValue)
                                glm_model = H2OGeneralizedLinearEstimator(family = family, nfolds= Nfolds)
                                glm_model.train(x_Values, yValue, training_frame= train, validation_frame=valid)
                                ## Selecting best model from the cross validation score
                                mod = glm_model.cross_validation_models()
                                rmse_list=[]
                                for i in range(len(mod)):
                                    rmse_list.append(mod[i].rmse())
                                rmse_list.sort()
                                for i in range(len(mod)):
                                    if (mod[i].rmse() == rmse_list[0]):
                                        best_mod = mod[i]
                                        break
                                    else:
                                        continue

                                print(glm_model)
                                predicted = best_mod.predict(test_data=test)
                                print('Performing Inverse transformations')
                                test_trueY = h2o.as_list(test)
                                x = list(filter(lambda x: x in num_data.columns and x in xValues and x not in drop_columns, test_trueY.columns))
                                if yValue not in x:
                                    x.append(yValue)
                                try:
                                    temp = list(filter(lambda x: x in obj_t, obj_t))
                                    temp1 = list(filter(lambda x: x in obj_t, obj_t))
                                    test_true_inv = transformation_inv(pd.DataFrame(test_trueY[x]),obj_t)
                                    test_true_inv.index = range(len(test_true_inv))
                                    true_y = pd.DataFrame(columns = ['Actual'], index = range(test_true_inv.shape[0]))
                                except Exception:
                                    return ['Gone']
                                true_y['Actual']= test_true_inv[yValue]
                                true_y.index = range(len(true_y))
                                ### For plotting transformed true Y data
                                #true_y_trnf = test_trueY[yValue]
                                
                                #Inverse Transforming yValue
                                test[yValue] = predicted
                                test_predY = h2o.as_list(test)
                                print('2nd inverse')
                                test_pred_inv = smi.transformation_inv(pd.DataFrame(test_predY[x]),temp)
                                pred_y = pd.DataFrame(columns = ['Predicted_values'], index = range(test_pred_inv.shape[0]))
                                pred_y['Predicted_values']=test_pred_inv[yValue]
                                pred_y.index = range(len(pred_y))
                                print('Inverse transformation completed')
                                ### For plotting transformed predicted Y data
                                #pred_y_trnf = test_predY[yValue]

                                ### Converting encoded categoricals to original values
                                ## Extracting individual categorical column
                                '''
                                if (cat_data.shape[1]>0):
                                    print("Decoding Categorical features")

                                    for i in n_data.columns:
                                        try:
                                            x_Values.remove(i)
                                        except Exception:
                                            continue
                                    test_cat = test_trueY[x_Values]
                                    test_cat_frame=[]
                                    data_x_category.pop(len(data_x_category)-1)
                                    for i in range(cat_data.shape[1]):
                                        l_col = []; l_colname = []
                                        for j in range(data_x_category[i].shape[1]):
                                            l_col.append(test_cat[test_cat.columns[j]])
                                            l_colname.append(test_cat.columns[j])
                                        test_cat = test_cat.drop(columns = l_colname)
                                        categ_col = pd.concat(l_col, axis = 1)
                                        test_cat_frame.append(categ_col)

                                    ## decoding from keras encoder to label encoded data
                                    label_col = []
                                    for i in range(len(lab)):
                                        label= []
                                        for j in range(test_cat_frame[i].shape[0]):
                                            label.append(argmax(np.array(test_cat_frame[i].iloc[j])))
                                        temp_df = pd.DataFrame(label, columns = [cat_data.columns[i]],index = range(len(label)))
                                        label_col.append(temp_df)

                                    ## decoding label encoded data to original message
                                    for i in range(len(lab)):
                                        label_col[i] = lab[i].inverse_transform(label_col[i])
                                        label_col[i] = pd.Series(label_col[i])
                                        label_col[i] = label_col[i].str.decode(encoding ='utf-8', errors= 'ignore')
                                        label_col[i]=pd.DataFrame(label_col[i], index = range(len(label_col[0])))

                                    test_cat_final = pd.concat(label_col, axis = 1)
                                    test_cat_final.columns = cat_data.columns
                                    print(test_cat_final)
                                    print("Categorical features Decoded")
                                else:
                                    test_cat_final = []
                                '''
                                ## Final Dat Frame
                                print('Preparing final Outputs')
                                act_pred=true_y['Actual'] - pred_y['Predicted_values']
                                Diff = pd.DataFrame(columns = ['difference_Actual_predicted'],index=range(len(act_pred)))
                                Diff['difference_Actual_predicted'] = act_pred
                                x.remove(yValue)
                                print(cat_data.shape[1])
                                print(num_data.shape[1])
                                cat_cols = list(cat_data.columns)
                                test_cat_final = test_predY[cat_cols]
                                test_cat_final.index = range(len(test_cat_final))
                                if (cat_data.shape[1]>0) and (num_data.shape[1]==1):
                                    frame = [true_y, pred_y, Diff, test_cat_final]
                                elif (cat_data.shape[1]==0) and (num_data.shape[1]>0):
                                    frame = [true_y, pred_y, Diff, test_true_inv[x]]
                                else:
                                    frame = [true_y, pred_y, Diff, test_cat_final, test_true_inv[x]]
                                Final_pred_data = pd.concat(frame, axis= 1)
                                Actual_predicted = pd.concat([true_y, pred_y], axis = 1)
                                act_pred_cols =  ['dataframe.Actual', 'dataframe.Predicted_values']
                                Actual_predicted.columns = act_pred_cols
                                print(Final_pred_data)
                                '''
                                Actual_predicted = pd.DataFrame(columns = ['Actual', 'Predicted_values'], index = range(len(true_y)))
                                Actual_predicted['Actual'] = true_y
                                Actual_predicted['Predicted_values'] = pred_y
                                #linear_regr = [glm_model.r2(),dataframeNewSet,Actual_predicted, ent_cor,glm_model.coef(),variable_imp,vif_var]
                                '''
                                #linear_regr = [ent_cor, [x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]]
                                list1 = [round(glm_model.r2()*10)]
                                '''
                                final_df_columns= []
                                for i in range(len(Final_pred_data.columns)):
                                    for j in range(len(columnsArray_e)):
                                        if Final_pred_data.columns[i] == columnsArray_e['columnDisplayName'][j]:
                                            final_df_columns.append(columnsArray_e['changedColumnName'][j])

                                final_list = ['Actual', 'Predicted_values', 'difference_Actual_predicted']
                                for i in final_df_columns:
                                    final_list.append(i)
                                Final_pred_data.columns = final_list
                                '''
                                for i in Final_pred_data.columns:
                                    if Final_pred_data[i].dtypes=='float' or Final_pred_data[i].dtypes=='int':
                                        for j in range(len(Final_pred_data[i])):
                                            Final_pred_data[i].iloc[j] = round(Final_pred_data[i].iloc[j], 2)
                                            
                                list2 = Final_pred_data.to_dict(orient='records')
                                #print("Actual_predicted")
                                #print(Actual_predicted)
                                for i in Actual_predicted.columns:
                                    if Actual_predicted[i].dtypes=='float' or Actual_predicted[i].dtypes=='int':
                                        for j in range(len(Actual_predicted[i])):
                                            Actual_predicted[i].iloc[j] = round(Actual_predicted[i].iloc[j], 2)
                                            
                                list3 = Actual_predicted.to_dict(orient='records')
                                #list3_1 = list(Actual_predicted[Actual_predicted.columns[0]])
                                #list3_2 = list(Actual_predicted[Actual_predicted.columns[1]])
                                list4 = [best_mod.rmse()]
                                temp_dict = best_mod.coef()
                                temp_dataf = pd.DataFrame.from_dict(temp_dict, orient = 'index')
                                for i in list(temp_dataf.index):
                                    if temp_dataf[0][i]> 0:
                                        continue
                                    elif temp_dataf[0][i]< 0:
                                        continue
                                    else:
                                        temp_dataf.drop(list(temp_dataf.index[temp_dataf.index==i]), inplace = True)
                                temp_dataf['left'] = 0
                                temp_dataf['right'] = 0
                                temp_dataf['right h2o.coef.model.gaussian.'] = temp_dataf[0]
                                temp_dataf = temp_dataf.drop(columns = [0])
                                for i in range(len(temp_dataf)):
                                    for j in range(len(columnsArray_e)):
                                        ab_coeffname = temp_dataf.index[i].split('.')
                                        if ab_coeffname[0] == columnsArray_e['columnDisplayName'][j]:
                                            temp_dataf['left'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                            try:
                                                temp_dataf['right'].iloc[i] = ab_coeffname[1]
                                            except Exception:
                                                temp_dataf['right'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                            break
                                temp_dataf['left'].iloc[0] = temp_dataf.index[0]
                                temp_dataf['right'].iloc[0] = temp_dataf.index[0]
                                temp_dataf.index = range(len(temp_dataf))

                                list5 = temp_dataf.to_dict(orient='records')
                                #list6 = [variable_imp]
                                #listed7 = NLG(linear_regr, yValue, labels, len_num)
                                print("listed7")
                                for i in range(len(Var_imp)):
                                    for j in range(len(columnsArray_e)):
                                        if Var_imp['x'].iloc[i] == columnsArray_e['columnName'].iloc[j]:
                                             Var_imp['x'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                             break
                                
                                # Top 6 variable importance
                                try:
                                    Var_imp = Var_imp[0:6]
                                except Exception:
                                    Var_imp = Var_imp
                                
                                listed8 = Var_imp.to_dict(orient='records')
                                listed = [list1, list2, list3, listed8, list4, [glm_model.r2()], [], list5, []]
                                print("listed")
                                #print(listed)

                            else:
                                listed = [{"message": "Target Variable (Y-axis) is Categorical data and only works if it is Numerical", "error": "Error"}]

                        else:
                            listed = [{"message": "There are less than two columns for building model or the Response variable is constant. Please make sure that there are atleast 1 Input dependant variables and it is not a constant Value", "error": "Error"}]

                    else:
                        listed = [{"message": "Selected Target Variable(yValue) is removed during preprocessing", "error": "Error"}]

                else:
                    listed = [{"message": "There are less than two columns for building model. Few of the selected Variables have been removed during preprocessing", "error": "Error"}]

            else:
                listed = [{"message": "Target Variable (yValue) cannot be constant", "error": "Error"}]

    else:
            listed = [{"message": "Not enough data to perform modelling, need atleast 150 data points", "error": "Error"}]

    return listed

'''
def NLG(linear_regr, yValue,labels, length_num):
    # correlated variables
    corr = linear_regr[0]
    if len(corr)>0:
        corr = corr.drop(yValue)
        mask1 = corr[yValue] >= 0.3
        mask2 = corr[yValue] <= -0.2
        mask3 = (corr[yValue] >= 0.2) & (corr[yValue] <= 0.3)
        corr1 = []
        corr_val1 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between <strong>' + (corr[yValue][mask1].index) + '</strong><strong>' + yValue + '</strong>, With every unit of increase in '+ (corr[yValue][mask1].index) + ' there is an increase in ' + yValue + '</span></li>'
        for i in range(len(corr_val1)):
            corr1.append(corr_val1[i])
        corr2 = []
        corr_val2 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between<strong> ' + (corr[yValue][mask2].index) + '</strong><strong>' + yValue + '</strong></span>, With every unit of decrease in '+ (corr[yValue][mask2].index) + ' there is an increase in ' + yValue + '</span></li>'
        for i in range(len(corr_val2)):
            corr2.append(corr_val2[i])
        corr3 = []
        corr_val3 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a little or no relationship between <strong> ' + (corr[yValue][mask3].index) + '</strong> and <strong>' + yValue + '</strong></span></li>'
        for i in range(len(corr_val3)):
            corr3.append(corr_val3[i])
    else:
        corr1 = []
        corr2 = []
        corr3 = []
    # quantile and min, max values
    #linear_regr[8] : [x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]
    for i in range(len(linear_regr[1])):
        for j in range(len(linear_regr[1][i])):
            a = round(linear_regr[1][i][j],2)
            linear_regr[1][i][j] = a

    if length_num>0:
        var_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The average value of <strong>'+', '.join(labels)+'</strong> is <strong>'+', '.join(map(str, linear_regr[1][0]))+'</strong></span>''</li>'
        if length_num>1:
            quantile_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>'+', '.join(labels)+'</strong> is <strong>' +', '.join(map(str, linear_regr[1][1]))+'</strong> whereas <strong>25%</strong> of data lies below the value <strong>'+ str(', '.join(map(str,linear_regr[1][3])))+ '</strong> the median of <strong>'+ str(', '.join(labels))+ '</strong> is <strong>'+', '.join(map(str, linear_regr[1][4]))+'</strong> and the <strong>75%</strong> of data lies below the value <strong>'+', '.join(map(str, linear_regr[1][5]))+'</strong> and the max value is <strong>'+', '.join(map(str, linear_regr[1][2]))+'</strong></span></li>'
        else:
            quantile_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>'+str(', '.join(labels))+'</strong> is <strong>' + str(', '.join(map(str, linear_regr[1][1])))+'</strong> and the max value is <strong>'+ str(', '.join(map(str, linear_regr[1][2])))+'</strong></span></li>'
    else:
        quantile_val=[]
        var_val = []
    listed = [corr1,corr2,corr3,var_val,quantile_val]
    return listed
'''

###### Logistic Regression

@app.route("/logistic_regression", methods = ['POST'])

def Logistic_Regression(request):
    print("in Logistic_Regression")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.args.get('columnsArray')
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    xValues = request.form.getlist('xValues')
    xValues = xValues[0]
    parametersObj =request.form.get('parametersObj')
    print(xValues)
    print(type(xValues))
    yValue =request.args.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    print("in Logistic_Regression y ")
    pool = ThreadPool()
    async_result = pool.apply_async(Logistic_RegressionThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue,parametersObj))
    return_val = async_result.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr

def Logistic_RegressionThread(query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue,parametersObj):
    print("in Logistic_RegressionThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    try:
        query_result = client.execute(query)
    except Exception:
        info = ['Please Check whether the values of any of the "query", "dbName", "password", "userName", "columnsArray", "dbHost", "columnList" are properly given or not and then try again. Thank you!']
        return info
    df = pd.DataFrame(query_result, columns = eval(columnList))
    print(df)
    parametersObj = json.loads(parametersObj)
    #print(yValue)
    data, columnsArray_e = lp_log.columns_data_type(df, json.loads(columnsArray))
    #data = data.replace(to_replace = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"],value = np.nan)
    print ("Different Levels in Y Variable are:", data[yValue].unique())
    print("Total Number of Unique Levels in Target variable",len(df[yValue].unique()))
    Nfolds = parametersObj['Nfolds']
    if data[yValue].dtypes != 'float' and data[yValue].dtypes != 'int':
        if len(df[yValue].unique()) == 2:
            ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub, drp_clmns = lp_log.correlations(data, columnsArray=columnsArray_e,no_rem_col='none',Val=data.columns, xVal=xValues,yVal=yValue)
#        if data[yValue].dtypes != 'float' and data[yValue].dtypes != 'int':
            if yValue not in rm_cols and yValue not in drp_clmns:
                #data.fillna(method ='ffill', inplace = True)
                #Replacing special characters to NULL values:
                print('######### Data after correlations #########################')
                print(data.head())
                num_data = data.select_dtypes(include=['number']).copy()
                print('######### Num data #########################')
                print(num_data.head())
                

                cols = data.columns
                num_cols = data._get_numeric_data().columns
                cat_cols = list(set(cols) - set(num_cols))
                try:
                    cat_cols.remove(yValue)
                except Exception:
                    cat_cols = cat_cols
            #   cat_cols = cat_cols.remove('phone number')

                data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                '''
                cols_y = yValue
                le = preprocessing.LabelEncoder()
                le.fit(data_n[yValue])
                yVal = le.transform(data_n[yValue])
            #   yVal = yVal.astype('object')
                yVal = pd.DataFrame(yVal,columns = list(cols_y))
                '''
            #   data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                                     #    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)

                x = data_n.drop(yValue,axis=1)
                y = pd.DataFrame(data = data[yValue], columns = [yValue])
                x.dtypes
                y.dtypes
                print(y.columns)
                y.dtypes
                x.isnull().sum().sum()
                y.isnull().sum().sum()

                print("IS DEBBUGGING HAPPENS TILL HERE PART 1:")
            #   xVal = list(filter(lambda a: a in x.columns and a in xVal, xVal))
                try:
                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)
                except Exception as ex:
                    message = str(ex)
                    if 'The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2' in message:
                        info = ['Please select a Binary Categorical variable as the yValue (Target Variable)']
                    else:
                        info = [message]
                    return info

                logistic = LogisticRegression()
               # Create regularization penalty space
                penalty = ['l1', 'l2']

               # Create regularization hyperparameter space
                C = np.logspace(0, 4, 10)

               # Create hyperparameter options
                hyperparameters_grid = dict(C=C, penalty=penalty)

                kfold = StratifiedKFold(n_splits = Nfolds, shuffle = True, random_state = 7)

                # Create grid search using K-fold cross validation (Here K = 5)
                clf = GridSearchCV(logistic, hyperparameters_grid, n_jobs = -1, cv = kfold, verbose = 1)
                print('done till here 1')
                # Fit grid search
                try:
                    best_model = clf.fit(X_train, y_train)
                    print('done till here 1')
                    # View best hyperparameters
                    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
                    print('Best C:', best_model.best_estimator_.get_params()['C'])

                    y_pred = best_model.predict(X_test)

                    y_pred_Prob1 = list(best_model.predict_proba(X_test)[:,0])
                    y_pred_Prob2 = list(best_model.predict_proba(X_test)[:,1])
                    X_test1 = X_test
                    X_test1['Predictions'] = y_pred
                    X_test1['Probabilities_0'] = y_pred_Prob1
                    X_test1['Probabilities_1'] = y_pred_Prob2
                    '''
                    plt.hist(X_test1['Probabilities_1'], bins=3)
                    plt.ylabel('No of times')
                    plt.show()
                    '''
                    X_test_Low = X_test1.loc[X_test1['Probabilities_1'] <= 0.45]
                    X_test_Medium = X_test1.loc[(X_test1['Probabilities_1'] > 0.45) & (X_test1['Probabilities_1'] <= 0.70)]
                    X_test_High = X_test1.loc[X_test1['Probabilities_1'] > 0.70]

                    #X_test1['Probabilities'] = y_pred_Prob
                    y_pred = list(y_pred)
                    Logistic_Accuracy = accuracy_score(y_test,y_pred)
                    #Logistic_Precision = precision_score(y_test,y_pred)
                    #Logistic_Recall = recall_score(y_test,y_pred)
                    #Logistic_F1_Score = f1_score(y_test,y_pred)
                    Logistic_ROC_AUC_Score = roc_auc_score(y_test,y_pred_Prob2)

                    cm = confusion_matrix(y_test, y_pred)
                    print("Confusion Matrix\n",cm)

                    print("The Accuracy of the Logistic Regression Model is", Logistic_Accuracy)
                    #print("The Precision of the Logistic Regression Model is", Logistic_Precision)
                    #print("The Recall of the Logistic Regression Model is", Logistic_Recall)
                    #print("The F1_Score of the Logistic Regression Model is", Logistic_F1_Score)
                    print("The ROC_AUC_Score of the Logistic Regression Model is", Logistic_ROC_AUC_Score)
                    '''
                    #Receiver Operating Characteristic Curve (ROC AUC):
                    fpr, tpr, threshold = roc_curve(y_test, y_pred_Prob2,pos_label='1')
                    roc_auc = metrics.auc(fpr, tpr)
                    plt.title('Receiver Operating Characteristic')
                    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
                    plt.legend(loc = 'lower right')
                    plt.plot([0, 1], [0, 1],'r--')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.ylabel('True Positive Rate')
                    plt.xlabel('False Positive Rate')
                    plt.show()
                    '''
                    l1 = [Logistic_Accuracy]
                    #l2 = [Logistic_Precision]
                    #l3 = [Logistic_Recall]
                    #l4 = [Logistic_F1_Score]
                    l5 = [Logistic_ROC_AUC_Score]
                    df1 = X_test_Low.to_dict(orient='records')
                    df2 = X_test_Medium.to_dict(orient='records')
                    df3 = X_test_High.to_dict(orient='records')
                    l6 = [l1, l5, df1, df2, df3]
                    return l6
                except Exception as e:
                    msg = str(e)
                    if 'This solver needs samples of at least 2 classes in the data, but the data contains only one class' in msg:
                        listed = ['Training data contains only one class as the number of instances for 2nd class might be far less compared to the 1st.']
                    elif 'Found array with 0 feature(s)' in msg:
                        listed = ['X Variables which were selected got deleted in Data Preprocessing! Kindly select other Variable to build the Logistic Regression Model. Thank you.']
                    else:
                       listed = [msg]
                    return listed
            else:
                listed = ['yValue removed while preprocessing! Please select appropriate yValue as the Target variable.']
            return listed

        elif len(df[yValue].unique()) > 2:
            ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub, drp_clmns = lp_log.correlations(data, columnsArray=columnsArray_e,no_rem_col='none',Val=data.columns,xVal=xValues,yVal=yValue)
            if yValue not in rm_cols and yValue not in drp_clmns:
                #data.fillna(method ='ffill', inplace = True)
                #Replacing special characters to NULL values:
                print('######### Data after correlations #########################')
                print(data.head())
                num_data = data.select_dtypes(include=['number']).copy()
                print('######### Num data #########################')
                print(num_data.head())

                cols = data.columns
                num_cols = data._get_numeric_data().columns
                cat_cols = list(set(cols) - set(num_cols))
                try:
                    cat_cols.remove(yValue)
                except Exception:
                    cat_cols = cat_cols
                    #    cat_cols = cat_cols.remove('phone number')

                data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                '''
                cols_y = yValue
                le = preprocessing.LabelEncoder()
                le.fit(data_n[yValue])
                yVal = le.transform(data_n[yValue])
                #yVal = yVal.astype('object')
                yVal = pd.DataFrame(yVal,columns = list(cols_y))
                '''
                #    data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                                      #    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)

                x = data_n.drop(yValue,axis=1)
                y = pd.DataFrame(data = data[yValue], columns = [yValue])
                x.dtypes
                y.dtypes
                print(y.columns)
                y.dtypes
                x.isnull().sum().sum()
                y.isnull().sum().sum()
                print("IS DEBBUGGING HAPPENS TILL HERE PART 2")
                #    xVal = list(filter(lambda a: a in x.columns and a in xVal, xVal))
                try:
                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)
                except Exception as ex:
                    message = str(ex)
                    if 'The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2' in message:
                        info = ['Please select a Binary Categorical variable as the yValue i.e., Target Variable']
                    else:
                        info = [message]
                    return info

#            logregpipe = Pipeline([('scale', StandardScaler()),
#                   ('logreg',LogisticRegression(multi_class="multinomial",solver="lbfgs"))])

            # Gridsearch to determine the value of C
            #param_grid = {'logreg__C':np.arange(0.01,100,10)}
                '''
                logreg_cv = GridSearchCV(logregpipe,param_grid,cv=5,return_train_score=True)
                logreg_cv.fit(X_train,y_train)
                print(logreg_cv.best_params_)

                bestlogreg = logreg_cv.best_estimator_
                bestlogreg.fit(X_train,y_train)
                bestlogreg.coef_ = bestlogreg.named_steps['logreg'].coef_
                bestlogreg.score(X_train,y_train)
                '''
                # Create one-vs-rest logistic regression object
               # clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')

                # Create regularization penalty space
                penalty = ['l2']

                # Create regularization hyperparameter space
                C = np.logspace(0, 4, base=10.0)

                # Create hyperparameter options
                hyperparameters_grid = dict(C=C, penalty=penalty)

                kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)

                # Create one-vs-rest logistic regression object
                mult_logistic = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')

                # Create grid search using K-fold cross validation (Here k = 5)
                clf = GridSearchCV(mult_logistic, hyperparameters_grid, n_jobs = -1, cv = kfold, verbose = 0, return_train_score=True)

                print('done till here 1')
                # Fit grid search
                try:
                    best_model = clf.fit(X_train, y_train)
                    print('done till here 2')

                    y_pred = best_model.predict(X_test)
                    prob = np.ndarray.tolist(best_model.predict_proba(X_test))
                    X_test1 = X_test
                    X_test1['Predictions'] = y_pred
                    print(type(prob))
                    X_test1['Probabilities'] = list(prob)
                    #print(X_test1)
                    #a = list(best_model.predict_proba(X_test))

                    y_pred = list(y_pred)
                    Logistic_Accuracy = accuracy_score(y_test,y_pred)
                    #Logistic_Precision = precision_score(y_test,y_pred)
                    #Logistic_Recall = recall_score(y_test,y_pred)
                    #Logistic_F1_Score = f1_score(y_test,y_pred)
                    #Logistic_ROC_AUC_Score = roc_auc_score(y_test,y_pred_Prob2)

                    # View best hyperparameters
                    print('K Fold:', kfold)
                    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
                    print('Best C:', best_model.best_estimator_.get_params()['C'])

                    cm = confusion_matrix(y_test, y_pred)
                    print("Confusion Matrix\n",cm)
                    print(classification_report(y_test, y_pred))

                    print("The Accuracy of the Logistic Regression Model is", Logistic_Accuracy)
                    #print("The Precision of the Logistic Regression Model is", Logistic_Precision)
                    #print("The Recall of the Logistic Regression Model is", Logistic_Recall)
                    #print("The F1_Score of the Logistic Regression Model is", Logistic_F1_Score)
                    #print("The ROC_AUC_Score of the Logistic Regression Model is", Logistic_ROC_AUC_Score)
                    l1 = [Logistic_Accuracy, classification_report(y_test, y_pred)]
                    #l2 = [Logistic_Precision]
                    #l3 = [Logistic_Recall]
                    #l4 = [Logistic_F1_Score]
                    #l5 = [Logistic_ROC_AUC_Score]
                    df1 = X_test1.to_dict(orient='records')
                    #    df2 = X_test_Medium.to_dict(orient='records')
                    #    df3 = X_test_High.to_dict(orient='records')
                    l = [l1, df1]
                    return l


                except Exception as e:
                    msg = str(e)
                    if 'This solver needs samples of at least 2 classes in the data, but the data contains only one class' in msg:
                        listed = ['Training data contains only one class as the number of instances for 2nd class might be far less compared to the 1st.']
                    elif 'Found array with 0 feature(s)' in msg:
                        listed = ['X Variables which were selected got deleted in Data Preprocessing! Kindly select other Variable to build the Logistic Regression Model. Thank you.']
                    else:
                        listed = [msg]

                    return listed
            else:
                listed = ['yValue removed while preprocessing! Please select appropriate yValue as the Target variable.']
                return listed

        else:
            msg_yValue = ['Please select the yValue which is Categorical and must have only 2 Levels for Logistic Regression to work or more than 2 Levels for Multinomial Logistic Regression to work. It must not have 1 Level.']
            return msg_yValue
    else:
        listed = ['yValue is Numerical. Logistic Regression works on Categorical Binary yValue and Multinomial Logistic work with Categorical Target variable, yValue.']
        return listed

#### Clustering

@app.route("/clustering", methods = ['POST'])

def Clusterapp():
    print("in clustering")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    parametersObj =request.form.get('parametersObj')
    #yValue = ast.literal_eval(yValue)
    #yValue = yValue[0]
    pool = ThreadPool()
    print('pool')
    async_result = pool.apply_async(clustering, (query,dbName,password,userName,columnsArray,dbHost,columnList, parametersObj))
    print('async_result')
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def clustering(query,dbName,password,userName,columnsArray,dbHost,columnList, parametersObj):
    ##############
    # connecting to BD
    ##############
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    parametersObj = json.loads(parametersObj)
    print('query')
    df = pd.DataFrame(query_result, columns = eval(columnList))
    print(df.head())
    if (df.shape[0]>100):
    #print(columnsArray)
        print(columnsArray)
        data, columnsArray_e = lp_c.columns_data_type(df, json.loads(columnsArray))
        nan_list = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"]
        for i in nan_list:
            data = data.replace(to_replace = i,value = np.nan)
        print('after columns data type')
        print(data)
        data,rm_cols, miss_cols, obj_t,sub = lp_c.correlations_cluster(data, columnsArray=columnsArray_e, no_rem_col='none',Val=data.columns)
        print('Data info before clustering')
        #print(data.info())
        Vals=data.columns
        print(obj_t)
        if(len(data.columns)>2) and data.shape[0]>100:
            output = kMeans_model(data,Vals,parametersObj, obj_t)
            if len(output)>1:

                new_df=pd.DataFrame(output[0])
                print(new_df)
                new_df['cluster'] = new_df['cluster'].astype('int')
                if len(new_df['cluster'].unique())<2:
                        output = [{"message": "All the data falls under a Single cluster as there is no variance between them. So, Clustering cannot be performed on data that is identical.", "error": "Error"}]
                        return output
                try:
                    cluster_prof = cluster_profiling(new_df, Vals = list(new_df.columns))
                except Exception as e:
                    if 'constant' in str(e):
                        output = [{"message": "All the data falls under a Single cluster as there is no variance between them. So, Clustering cannot be performed on data that is identical.", "error": "Error"}]
                        return output
                new_df['cluster'] = new_df['cluster'].astype('category')
                print(cluster_prof)
                #prof_nlg_cluster=profiling_nlg(variable_imp=cluster_prof[0],mean_clust=cluster_prof[1],median_clust=cluster_prof[2],mode_clust=cluster_prof[3])
                #print('check type')
                #print(type(cluster_prof))
                print('profile done')

                listed2 = output[1].to_dict(orient='records')
                listed3 = output[3].to_dict(orient='records')
                listed4 = output[6].to_dict(orient='records')

                top6 = cluster_prof[0][0:6]
                num_top6 = list(filter(lambda x: x in top6['x'].values, list(cluster_prof[1].columns)))
                cluster_prof[1] = cluster_prof[1][num_top6]
                cluster_prof[2]  = cluster_prof[2][num_top6]

                cat_top6 = list(filter(lambda x: x in top6['x'].values, list(cluster_prof[3].columns)))
                cluster_prof[3] = cluster_prof[3][cat_top6]

                df_numlist = []
                for i in list(cluster_prof[2].columns):

                    df_mm = pd.DataFrame(columns = ['var_name', 'means', 'cluster', 'medians'], index = range(cluster_prof[2].shape[0]))
                    df_mm['var_name'] = i
                    df_mm['means'] = cluster_prof[1][i].values
                    df_mm['medians'] =  cluster_prof[2][i].values
                    df_mm['cluster'] = list(cluster_prof[1].index)
                    df_mm = df_mm.to_dict(orient='records')
                    df_numlist.append(df_mm)

                ## Finding mode list

                df_catlist = []

                for i in cluster_prof[3].columns:

                    clust_lis = []
                    for j in new_df['cluster'].unique():
                        for k in new_df[i].unique():
                            clust_lis.append(j)
                    clust_lis.sort()
                    df_mm = pd.DataFrame(columns = ['var_name', 'name', 'value', 'cluster', 'parent'], index = range(len(clust_lis)))
                    df_mm['var_name'] = i
                    df_mm['cluster'] = clust_lis
                    df_mm_name = []
                    for j in range(len(new_df['cluster'].unique())):
                        temp_df_list = list(new_df[i].unique())
                        temp_df_list.sort()
                        df_list = pd.DataFrame(temp_df_list)
                        df_mm_name.append(df_list)

                    df_mm_name = pd.concat(df_mm_name, axis = 0)
                    df_mm['name'] = df_mm_name[0].values
                    for j in range(len(df_mm)):
                        df_mm['parent'].iloc[j] = 'cluster '+str(df_mm['cluster'].iloc[j])

                    for j in range(len(df_mm)):
                        temp_val_df = new_df[new_df[i]==df_mm['name'].iloc[j]]

                        try:
                            temp_val_df = temp_val_df[temp_val_df['cluster']== temp_val_df['cluster'].iloc[j]]
                            df_mm['value'].iloc[j] = len(temp_val_df)
                        except Exception:
                            df_mm['value'].iloc[j] = 0

                    df_mm = df_mm.to_dict(orient='records')
                    df_catlist.append(df_mm)


                listed5=cluster_prof[0].to_dict(orient='records')

                df_cluster_id = pd.DataFrame(columns = ['id', 'name'], index = range(len(list(new_df['cluster'].unique()))))
                for j in range(len(list(new_df['cluster'].unique()))):
                    df_cluster_id['id'].iloc[j] = 'cluster '+str(j)
                    df_cluster_id['name'].iloc[j] = 'Cluster '+str(j)
                df_cluster_id = df_cluster_id.to_dict(orient='records')
                new_df = new_df.to_dict(orient='records')
                listed=[new_df, listed2, [], listed3, [output[4]], [output[5]], listed4, df_numlist, df_catlist,  df_cluster_id, listed5, []]
                print(listed[1:])
            else:
                listed= output
        else:
            listed=[{"message": "In the preprocessing stage all the unnecessary columns and rows are removed. Try again with other dataset", "error": "Error"}]
    else:
        listed=[{"message": "Data length is very less. so model is not applicable for such data", "error": "Error"}]

    return listed

#### DBscan

@app.route("/dbscan", methods = ['POST'])

def Clusterdb(request):
    print("in DBSCAN clustering")
    query = request.args.get('query')
    dbName = request.args.get('dbName')
    password = request.args.get('password')
    userName = request.args.get('userName')
    columnsArray = request.args.get('columnsArray')
    dbHost = request.args.get('dbHost')
    columnList =request.args.get('columnList')
    pool = ThreadPool()
    async_result = pool.apply_async(DBSCAN_Cluster, (query,dbName,password,userName,columnsArray,dbHost,columnList))
    return_val = async_result.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr

def DBSCAN_Cluster(query,dbName,password,userName,columnsArray,dbHost,columnList):
    print("in linearRegressionThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(yValue)
    data, columnsArray_e = lp_c.columns_data_type(df, json.loads(columnsArray))
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub = lp_c.correlations_cluster(data, columnsArray=columnsArray_e, no_rem_col='none',Val=data.columns)
    data.fillna(method ='ffill', inplace = True)
    print('######### Data after correlations #########################')
    print(data.head())
    num_data = data.select_dtypes(include=['number']).copy()
    print('######### Num data #########################')
    print(num_data.head())

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))

    data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)

    x = data_n
    x.dtypes
    x.isnull().sum().sum()


    # Scaling the data to bring all the attributes to a comparable level
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Normalizing the data so that
    # the data approximately follows a Gaussian distribution
    X_normalized = normalize(X_scaled)

    # Converting the numpy array into a pandas DataFrame
    X_normalized = pd.DataFrame(X_normalized)
    # Numpy array of all the cluster labels assigned to each data point
    clustering = DBSCAN(eps = 0.3, min_samples = 21).fit(X_normalized)
    clustering
    clustering.fit(X_normalized)
    labels = list(clustering.labels_)
    print(labels)
    data['Clusters'] = labels
    listed = data.to_dict(orient ='records')

    return listed


#### Decision Tree

@app.route("/decision_tree", methods = ['POST'])

def Decisiontree():
    print("in Decisiontree")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    yValue =request.form.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    fitParameter =request.form.get('fitParameter');
    controlParameter =request.form.get('controlParameter');
    pool = ThreadPool()
    async_result = pool.apply_async(DecisiontreeThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,yValue,fitParameter, controlParameter))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def DecisiontreeThread(query,dbName,password,userName,columnsArray,dbHost,columnList,yValue,fitParameter, controlParameter):
    print("in DecisiontreeThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    fitParameter = json.loads(fitParameter)
    controlParameter = json.loads(controlParameter)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(yValue)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    data.columns = eval(columnList)
    nan_list = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"]
    for i in nan_list:
        data = data.replace(to_replace = i,value = np.nan)
    print(data.shape[0])
    if data.shape[0]>150:
        print(data[yValue].value_counts())
        if len(data[yValue].unique())>1:
            ent_cor,data,rm_cols, miss_cols, obj_t, sub, drop_columns,trans_col = correlations(data, columnsArray=columnsArray_e, no_rem_col='none', Val = data.columns)
            data = data.loc[:,~data.columns.duplicated()]
            if data.shape[1]>1:
                xValues = list(filter(lambda x: x not in rm_cols and x not in yValue, data.columns))
                data_x_unique = data[xValues]
                for i in data_x_unique.columns:
                    if len(data_x_unique[i].unique())>1:
                        continue
                    else:
                        if data[i].dtypes != 'float' and  data[i].dtypes != 'int':
                            print('dropping ' + str(i))
                            data = data.drop(columns = [i])
                            drop_columns.append(i)
                        else:
                            if data_x_unique.shape[1]>1:
                                continue
                            else:
                               data = data.drop(columns = [i])
                               drop_columns.append(i)
            else:
                listed = ['There is only one column after preprocessing. Please try giving different inputs or different dataset as the dataset might not be applicable for this model.']
                return listed
            if data.shape[1]>1:
                if yValue not in list(rm_cols) and yValue not in drop_columns:
                    xValues = list(filter(lambda x: x not in rm_cols and x in data.columns, xValues))
                    try:
                            variable_imp = variable_importance_h2o(data, xValues, yValue)
                    except Exception as e:
                            msg = str(e)
                            if 'Response cannot be constant' in msg:
                                listed = ['Target Variable cannot be a constant value.']
                            else:
                                listed = [msg]
                            return listed
                        #labels = xValues
                    var_imp = []; var_name=[]
                    for i in range(len(variable_imp)):
                            for j in range(len(variable_imp[i])):
                                var_imp.append(variable_imp[i][j][2])
                                var_name.append(variable_imp[i][j][0])
                    Var_imp  = pd.DataFrame(columns = ['Overall'], index = range(len(var_imp)))
                    Var_imp['Overall'] = var_imp
                    Var_name = pd.DataFrame(columns = ['colnames'], index = range(len(var_name)))
                    Var_name['colnames'] = var_name
                    frame_var = [Var_name, Var_imp]
                    Var_imp = pd.concat(frame_var, axis = 1)
                    
                    vif_frame = [data[xValues], data[yValue]]
                    data_vif = pd.concat(vif_frame, axis = 1)
                    print(data_vif.dtypes)
                    num_data = data_vif.select_dtypes(include=['float', 'int', 'number']).copy()
                    cat_data = data_vif.select_dtypes(include=['category', 'object']).copy()
                    #vif_var = vif(num_data, yValue, 10)

                    ### Data preperation
                    data_x = data[xValues]
                    print(data_x.dtypes)
                    cat_data = data_x.select_dtypes(include=['category', 'object']).copy()
                    if (cat_data.shape[1]>0):
                        print("Starting Categorical Encoding")
                        # Label Encoding categorical data
                        lab = []
                        for i in range(cat_data.shape[1]):
                            lab.append(preprocessing.LabelEncoder())
                        j = 0
                        for i in cat_data.columns:
                                data_x[i] = lab[j].fit_transform(data_x[i])
                                j = j+1

                    data_y = data[yValue]
                    frames= [data_x, data_y]
                    data2 = pd.concat(frames, axis=1)
                    for i in range(len(data2)):
                        data2[yValue].iloc[i] = str(data2[yValue].iloc[i])
                    data2[yValue] = data2[yValue].astype('category')
                    print('counts before removing')
                    print(data2[yValue].value_counts())
                    value_counts = data2[yValue].value_counts()
                    value_count = pd.DataFrame(value_counts)
                    indexes = list(value_count.index)
                    for i in range(len(value_counts)):
                        if value_counts[-1] <=15:
                            data2 = data2[data2[yValue]!=indexes[-1]]
                            indexes.remove(indexes[-1])
                            value_counts = value_counts[indexes[0]:indexes[len(indexes)-1]]
                    x_values = list(data2.columns)
                    x_values.remove(yValue)
                    print('Counts after removing')
                    print(data2[yValue].value_counts())
                    ## Model Building ##

                    X_train, X_test, y_train, y_test = train_test_split(data2[x_values], data2[yValue], test_size=0.2, random_state=0)
                    Nfolds = int(controlParameter['decisiontreerepeats'])
                    Tunelength = int(fitParameter['decisiontreetunelength'])
                    Split = 'entropy'
                    print("done")
                    
                    ### Dealing with imbalaced data of train if any
                    try:
                        Y_Train = pd.DataFrame(y_train)
                        sampl_list = list(Y_Train[yValue].value_counts())
                        if len(sampl_list)<8:
                            smote = SMOTE(random_state = 0)
                            X_train, y_train = smote.fit_sample(X_train, y_train)
                            print(y_train.value_counts())
                        else:
                            smote = SMOTE(random_state = 0, sampling_strategy = 'not majority')
                            X_train, y_train = smote.fit_sample(X_train, y_train)
                            print(y_train.value_counts())
                    except Exception as e:
                        print(e)

                    if data2.shape[1]>1 and len(data2[yValue].unique())>1:
                        if (data2[yValue].dtypes == 'float') or (data2[yValue].dtypes == 'int'):
                            '''
                            if Split == 'mse':
                                print("Building the model based on the Numerical Y considering all values")
                                print("Starting Auto Tuning")
                                clf = DecisionTreeRegressor(random_state=564, criterion=Split, max_depth = Tunelength)
                                #Hyper Parameters Set
                                params = {'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
                                #Making models with hyper parameters sets
                                model = GridSearchCV(clf, param_grid=params, n_jobs=-1, cv = 10)
                                print("Started tuning")
                                model.fit(X_train,y_train)
                                print("Tuning completed")
                                bp_list = model.best_params_
                                best_mod = DecisionTreeRegressor(max_features = bp_list['max_features'],min_samples_split = bp_list['min_samples_split'], min_samples_leaf= bp_list['min_samples_leaf'], criterion=Split, max_depth = Tunelength)
                                best_mod.fit(X_train,y_train)
                                r = export_text(best_mod, feature_names=xValues)
                                print(r)
                                ### Prediction
                                predicted = best_mod.predict(X_test)
                                print("Done Predictions")

                                X_test.index = range(X_test.shape[0])
                                y_test.index = range(y_test.shape[0])
                                test_yValue=pd.DataFrame(y_test, index= range(len(y_test)))
                                test_frame = [X_test,test_yValue]
                                test_trueY = pd.concat(test_frame, axis = 1)
                                # Inverse transforming true yValue
                                x = list(filter(lambda x: x in num_data.columns and x in xValues, X_test.columns))
                                x.append(yValue)
                                test_true_inv = transformation_inv(test_trueY[x],obj_t)
                                true_y = pd.DataFrame(columns = ['Actual'], index = range(y_test.shape[0]))
                                true_y['Actual']= test_true_inv[yValue]
                                ### decoding label encoder
                                if (cat_data.shape[1]>0):
                                    j=0
                                    for i in cat_data.columns:
                                            test_trueY[i] = lab[j].inverse_transform(test_trueY[i])
                                            j = j+1

                                ### For plotting transformed true Y data
                                true_y_trnf = y_test

                                # Inverse Transforming predicted yValue
                                test_trueY[yValue] = predicted
                                test_pred_inv = transformation_inv(test_trueY[x],obj_t)
                                pred_y = pd.DataFrame(columns = ['Predicted'], index = range(test_pred_inv.shape[0]))
                                pred_y['Predicted']=test_pred_inv[yValue]
                                ### For plotting transformed predicted Y data
                                pred_y_trnf = predicted

                                ### Final data Frame
                                act_pred=true_y['Actual'] - pred_y['Predicted']
                                Diff = pd.DataFrame(columns = ['Actual-Predicted'],index=range(len(act_pred)))
                                Diff['Actual-Predicted'] = act_pred
                                x.remove(yValue)
                                if (cat_data.shape[1]>0) and (num_data.shape[1]==1):
                                    test_cat_final = test_trueY[cat_data.columns]
                                    frame = [true_y, pred_y, Diff, test_cat_final]
                                elif (cat_data.shape[1]==0) and (num_data.shape[1]>0):
                                    frame = [true_y, pred_y, Diff, test_true_inv[x]]
                                else:
                                    test_cat_final = test_trueY[cat_data.columns]
                                    frame = [true_y, pred_y, Diff, test_cat_final, test_true_inv[x]]
                                Final_pred_data = pd.concat(frame, axis= 1)

                                ### Metrics
                                r2 = round(r2_score(true_y_trnf,pred_y_trnf),2)
                                if r2>0:
                                    r2 = [r2]
                                else:
                                    r2 = [0]
                                #rmse = sqrt(mean_squared_error(true_y_trnf,true_y_trnf))
                                Final_pred_data = Final_pred_data.to_dict(orient='records')
                                list6 = [variable_imp]

                                listed=[[r], r2, Final_pred_data, list6]
                                #return dt_out
                            else:
                                listed = ["Please select appropriate Split. If target is numerical, select 'mse'"]
                                #return listed
                            '''
                        else:
                            if Split == 'entropy':
                                print("Building the model based on the Categorical Y considering all values")
                                print("Starting Auto Tuning")
                                clf = DecisionTreeClassifier(random_state=0, criterion="gini", class_weight = "balanced")
                                #Hyper Parameters Set
                                if Tunelength in [5,6,7]:
                                    params = {'max_features': ['auto', 'sqrt', 'log2'], "min_samples_leaf": [2,3,5,6], "max_depth":[4,5,6,7], "min_samples_split":[0.2,0.4,0.6], "min_impurity_decrease" : [0.0001,0.001,0.01]}
                                else:
                                    params = {'max_features': ['auto', 'sqrt', 'log2'], "min_samples_leaf": [2,3,5,6], "max_depth":[4,5,6,7,int(Tunelength)], "min_samples_split":[0.2,0.4,0.6], "min_impurity_decrease" : [0.0001,0.001,0.01]}
                                
                                #Making models with hyper parameters sets
                                
                                model = GridSearchCV(clf, param_grid=params, n_jobs=-1, cv = Nfolds, verbose=1)
                                print("Started tuning")
                                model.fit(X_train,y_train)
                                print("Tuning completed")
                                bp_list = model.best_params_
                                print(bp_list)
                                '''
                                from sklearn.tree._tree import TREE_LEAF

                                def is_leaf(inner_tree, index):
                                    # Check whether node is leaf node
                                    return (inner_tree.children_left[index] == TREE_LEAF and 
                                            inner_tree.children_right[index] == TREE_LEAF)
                                
                                def prune_index(inner_tree, decisions, index=0):
                                    # Start pruning from the bottom - if we start from the top, we might miss
                                    # nodes that become leaves during pruning.
                                    # Do not use this directly - use prune_duplicate_leaves instead.
                                    if not is_leaf(inner_tree, inner_tree.children_left[index]):
                                        prune_index(inner_tree, decisions, inner_tree.children_left[index])
                                    if not is_leaf(inner_tree, inner_tree.children_right[index]):
                                        prune_index(inner_tree, decisions, inner_tree.children_right[index])
                                
                                    # Prune children if both children are leaves now and make the same decision:     
                                    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
                                        is_leaf(inner_tree, inner_tree.children_right[index]) and
                                        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
                                        (decisions[index] == decisions[inner_tree.children_right[index]])):
                                        # turn node into a leaf by "unlinking" its children
                                        inner_tree.children_left[index] = TREE_LEAF
                                        inner_tree.children_right[index] = TREE_LEAF
                                        ##print("Pruned {}".format(index))
                                
                                def prune_duplicate_leaves(mdl):
                                    # Remove leaves if both 
                                    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
                                    prune_index(mdl.tree_, decisions)
                                '''
                                
                                best_mod = DecisionTreeClassifier(max_features = bp_list['max_features'], min_samples_leaf = bp_list['min_samples_leaf'], max_depth = bp_list['max_depth'], min_samples_split = bp_list['min_samples_split'], min_impurity_decrease = bp_list['min_impurity_decrease'], random_state=0)
                                
                                best_mod.fit(X_train,y_train)
                                '''
                                r = export_text(best_mod, feature_names=xValues)
                                print(r)
                                '''
                                
                                temp_cat_lab = list(data2[yValue].unique())
                                temp_cat_lab.sort()
                                try:
                                    temp_cat_lab[0] = float(temp_cat_lab[0])
                                    for i in range(len(temp_cat_lab)):
                                        temp_cat_lab[i] = int(temp_cat_lab[i])
                                        temp_cat_lab[i] = str(temp_cat_lab[i])
                                except Exception:
                                    temp_cat_lab = temp_cat_lab
                                #code = tree_to_code_nlg(best_mod, xValues, yValue, temp_cat_lab)
                                code2 = tree_to_code(best_mod, xValues, temp_cat_lab, obj_t[0],trans_col)
                                print("Tree info Extracted")
                                ### Prediction
                                predicted = best_mod.predict(X_test)
                                print("Completed Predictions")
                                X_test.index = range(X_test.shape[0])
                                test_trueY = X_test
                                if (num_data.shape[1]>0):
                                        x = list(filter(lambda x: x in num_data.columns and x in xValues, X_test.columns))
                                        test_true_inv = transformation_inv(test_trueY[x],obj_t)
                                true_y = pd.DataFrame(columns = ['Actual'], index = range(y_test.shape[0]))
                                y_test.index = range(y_test.shape[0])
                                true_y['Actual']= y_test

                                pred_y = pd.DataFrame(columns = ['Predicted'], index = range(len(predicted)))
                                pred_y['Predicted']=predicted

                                ### Decoding label encoder
                                if (cat_data.shape[1]>0):
                                        j = 0
                                        for i in cat_data.columns:
                                                test_trueY[i] = lab[j].inverse_transform(test_trueY[i])
                                                j = j+1

                                test_cat_final = test_trueY[cat_data.columns]
                                try:
                                    test_cat_final = test_cat_final.drop(columns = [yValue])
                                except Exception:
                                    test_cat_final = test_cat_final
                                ## Final Dat Frame
                                print(num_data.shape[1])
                                print(cat_data.shape[1])
                                if (cat_data.shape[1]>0) and (num_data.shape[1]==0):
                                        frame = [true_y, pred_y, test_cat_final]
                                elif (cat_data.shape[1]==1) and (num_data.shape[1]>0):
                                        frame = [true_y, pred_y, test_true_inv[x]]
                                else:
                                        frame = [true_y, pred_y, test_cat_final, test_true_inv[x]]
                                Final_pred_data = pd.concat(frame, axis= 1)
                                Final_pred_data
                                
                                ## Metrics
                                #from sklearn.metrics import roc_curve, auc, roc_auc_score

                                #roc = roc_auc_score(pred_y, best_mod.predict_proba(X_test)[:,1])                        
                                cm = confusion_matrix(true_y['Actual'], pred_y['Predicted'])
                                print(len(data2[yValue].unique()))
                                conf_mat = pd.DataFrame(columns = ['Prediction','Reference','Freq'], index = range(len(data2[yValue].unique())**2))
                                ## Filling Values in Prediction
                                conf_mat['Prediction'] = len(data2[yValue].unique())*temp_cat_lab
                                ## Filling Values in Reference
                                ref_list = list(filter(lambda x: x in temp_cat_lab, temp_cat_lab))
                                ref_list_conf = []
                                for i in ref_list:
                                    for j in range(len(ref_list)):
                                        ref_list_conf.append(i)
                                
                                conf_mat['Reference'] = ref_list_conf
                                ## Filling Values in frequency
                                cm = pd.DataFrame(cm)
                                freq_list=[]
                                for i in range(len(cm)):
                                    freq_list.append(cm[i])
                                
                                cm  = pd.concat(freq_list, axis = 0)
                                print(len(conf_mat['Freq']));print(cm)
                                conf_mat['Freq'] = cm.values
                                
                                acc = accuracy_score(true_y['Actual'], pred_y['Predicted'])
                                ## Precision
                                prec = precision_score(true_y['Actual'], pred_y['Predicted'], average = None)
                                pre_score = pd.DataFrame(columns = ['Precision', 'Levels'], index = list(true_y['Actual'].unique()))
                                pre_score['Levels'] = list(true_y['Actual'].unique())
                                pre_score['Precision'] = prec

                                ## Recall
                                rec = recall_score(true_y['Actual'], pred_y['Predicted'], average = None)
                                rec_score = pd.DataFrame(columns = ['Recall', 'Levels'], index = list(true_y['Actual'].unique()))
                                rec_score['Levels'] = list(true_y['Actual'].unique())
                                rec_score['Recall'] = rec

                                list1 = [round(acc*10)]
                                list1_2 = pre_score.to_dict(orient='records')
                                list1_3 = rec_score.to_dict(orient='records')
                                #list2 = Final_pred_data.to_dict(orient='records')
                                conf_mat = conf_mat.to_dict(orient='records')
                                print("Actual_predicted")
                                #print(Actual_predicted)
                                #list3 = Actual_predicted.to_dict(orient='records')
                                #list4 = [ent_cor]
                                #list5 = [glm_model.coef()]
                                # Top 6 variable importance
                                try:
                                    Var_imp = Var_imp[0:6]
                                except Exception:
                                    Var_imp = Var_imp
                                list6 = Var_imp.to_dict(orient='records')
                                print(code2)
                                listed = [code2, list1_2, list1_3, list1, conf_mat, list6, '']
                                print("listed")
                            else:
                                    listed = [{"message": "Please select appropriate Split. If target is categorical, select 'entropy'", "error": "Error"}]
                    else:
                        listed = [{"message": "There are less than two columns for building model or the Response variable is constant. Please make sure that there are atleast 1 Input dependant variables and it is not a constant Value", "error": "Error"}]
                else:
                    listed = [{"message": "Selected Target Variable(Y-axis) is removed during preprocessing", "error": "Error"}]
            else:
                listed = [{"message": "There are less than two columns for building model. Few of the selected Variables have been removed during preprocessing", "error": "Error"}]
        else:
            listed = [{"message": "Target Variable (yValue) cannot be constant", "error": "Error"}]
    else:
         listed = [{"message": "Not enough data to perform modelling, need atleast 150 data points", "error": "Error"}]
    return listed

def tree_to_code(tree, feature_names, temp_cat_lab, obj,trans_col):
    tree_ = tree.tree_
    feature_name = []
    temp_feat = list(tree_.feature)
    for i in temp_feat:
        if i != _tree.TREE_UNDEFINED:
            feature_name.append(feature_names[i])
        else:
            feature_name.append("undefined!") 
    
    l = []; l.append("")
    
    def recurse(node, depth):
        #indent = "|  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            if tree_.feature[tree_.children_left[node]] == _tree.TREE_UNDEFINED:
                if '{}'.format(temp_cat_lab[np.argmax(tree_.value[node])])+"\"}" in l[-1]:
                    l.append(",{"+"\"name\": \""+ str(node)+"\",\"rule\":\""+'{} <= {}'.format(name, round(dt_transformation_inv(threshold, obj,trans_col,name),2))+"\",\"value\":\"")
                else:
                    l.append("{"+"\"name\": \""+ str(node)+"\",\"rule\":\""+'{} <= {}'.format(name, round(dt_transformation_inv(threshold, obj,trans_col,name),2))+"\",\"value\":\"")
            else:
                if '{}'.format(temp_cat_lab[np.argmax(tree_.value[node])])+"\"}" in l[-1]:
                    l.append(",{"+"\"name\": \""+ str(node)+"\",\"value\":\""+""+"\",\"rule\":\""+'{} <= {}'.format(name, round(dt_transformation_inv(threshold, obj,trans_col,name),2))+"\", \"children\": [")
                else:
                    if node ==0:
                        l.append("{"+"\"name\": \""+ str(node)+"\",\"value\":\"Root@@Root= 1\""+",\"rule\":\""+'{} <= {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\", \"children\": [")
                    else:
                        l.append("{"+"\"name\": \""+ str(node)+"\",\"value\":\""+""+"\",\"rule\":\""+'{} <= {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\", \"children\": [")
            #print("{}if {} <= {}:".format(indent, name, round(dt_transformation_inv(threshold,obj,trans_col,name),2)))
            recurse(tree_.children_left[node], depth + 1)
            
            if tree_.children_left[node] == 0 or (node ==0 and "]}" in l[-2]) or (node ==0 and "]}" in l[-1]):
                l.pop()
                
            if tree_.feature[tree_.children_right[node]] == _tree.TREE_UNDEFINED:
                if "\", \"children\": [" in l[-1]:
                    print('here')
                    print(l[-1])
                    l.append("{"+"\"name\": \""+ str(node)+"\",\"rule\":\""+'{} > {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\",\"value\":\"")
                
                else:
                    l.append(",{"+"\"name\": \""+ str(node)+"\",\"rule\":\""+'{} > {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\",\"value\":\"")
            else:
                #print(l[-1])
                if "\"}" in l[-1] and tree_.feature[tree_.children_left[node]] == _tree.TREE_UNDEFINED and node ==0:
                    l[-1] = l[-1].replace('}', '')
                    l.append(", \"children\": [")
                    l.append("{"+"\"name\": \""+ str(node)+"\",\"value\":\""+""+"\",\"rule\":\""+'{} > {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\", \"children\": [")
                elif ("\"}" in l[-1] or "]}" in l[-1]):
                    l.append(",{"+"\"name\": \""+ str(node)+"\",\"value\":\""+""+"\",\"rule\":\""+'{} > {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\", \"children\": [")
                else:
                    l.append("{"+"\"name\": \""+ str(node)+"\",\"value\":\""+""+"\",\"rule\":\""+'{} > {}'.format(name, round(dt_transformation_inv(threshold,obj,trans_col,name),2))+"\", \"children\": [")
            #print(l[-2])
            #print("{}else: if {} > {}".format(indent, name, round(dt_transformation_inv(threshold,obj,trans_col,name),2)))
            recurse(tree_.children_right[node], depth + 1)
            l.append("]}")
            
        else:
            l.append('{}'.format(temp_cat_lab[np.argmax(tree_.value[node])])+"\"}")
            #print("{} {}".format(indent, temp_cat_lab[np.argmax(tree_.value[node])]))

    recurse(0, 1)
    
    k = ''.join(l)
    return k


def tree_to_code_nlg(tree, feature_names,yValue, temp_cat_lab):

    tree_ = tree.tree_
    feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	#print("def tree({}):".format(", ".join(feature_names)))
    l1 = []; l2 = []
    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if len(l2)==0:
                l2.append("if {} is less than or equal to {} ".format(name, round(threshold,2)))
            else:
                l2.append("and if {} is less than or equal to {} ".format(name, round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            
            for i in range(len(l2)):
                if "if {} is less than or equal to {} ".format(name, round(threshold,2)) in l2[i]:
                    for j in range(i, len(l2)):
                        x=i
                        l2.remove(l2[x])
                    break
            if len(l2)==0:
                l2.append("if {} is greater than {} ".format(name, round(threshold,2)))
            else:
                l2.append("and if {} is greater than {} ".format(name, round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            l2.append("then {} is {}".format(yValue, np.argmax(tree_.value[node][0])))
            print(tree_.value[node])
            l1.append([''.join(l2)])
            l2.remove(l2[-1])
            l2.remove(l2[-1])
            
    recurse(0, 1)
    return l1
    

#### Envotting


@app.route("/envotting", methods = ['POST'])

def vottingclassifer(request):
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.args.get('columnsArray')
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    xValues = request.args.get('xValues')
    print(xValues)
    xValues = ast.literal_eval(xValues)
    print(xValues)
    yValue =request.args.get('yValue')
#    yValue = ast.literal_eval(yValue)
#    yValue = yValue[0]
    pool = ThreadPool()

    async_result = pool.apply_async(VottingclassiferThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue))
    return_val = async_result.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr



@csrf_exempt
def VottingclassiferThread(query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue):
    #data = pd.read_excel("C:/Users/Akhilesh/Downloads/BoC/Ensemble/Data1.xlsx",encoding="latin")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(yValue)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    if data.shape[0]>150 :
        ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub, drp_clmns = correlations(data, columnsArray=columnsArray_e,no_rem_col='none',Val=data.columns,xVal=xValues,yVal=yValue)
        if data[yValue].dtypes != 'float' and data[yValue].dtypes != 'int':
            if yValue not in rm_cols and yValue not in drp_clmns:

                print('######### Data after correlations #########################')
                print(data.head())
                num_data = data.select_dtypes(include=['number']).copy()
                print('######### Num data #########################')
                print(num_data.head())

                if data.shape[1]>1 and len(data[yValue].unique())>1:
                    cols = data.columns
                    num_cols = data._get_numeric_data().columns
                    cat_cols = list(set(cols) - set(num_cols))
                    try:
                        cat_cols.remove(yValue)
                    except ValueError:
                        cat_cols = cat_cols
                    y_new = data[yValue]
                    cols_y = yValue
                    le = preprocessing.LabelEncoder()
                    le.fit(y_new)
                    y_new = le.transform(y_new)
                    y_new = y_new.astype('str')
                    y_new= pd.DataFrame(y_new,columns = [cols_y])
                    data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
				#    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)
                    x = data_n.drop(yValue,axis=1)
                    y = y_new

                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)
		#    knn = KNeighborsClassifier()
		#    params_knn = {'n_neighbors': np.arange(1, 25)}
		#    knn_gs = GridSearchCV(knn, params_knn, cv=5)
		#    knn_gs.fit(X_train, y_train)
		#    knn_best = knn_gs.best_estimator_
		#    print(knn_gs.best_params_)
		#    knn_accuracy = knn_best.score(X_test, y_test)

                    clf1=SVC(C=1,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape='ovr',degree=3,gamma=0.1,kernel='rbf',max_iter=-1,probability=True,random_state=None,shrinking=True,tol=0.001,verbose=False)
                    clf1.fit(X_train,y_train)
                    y_pred1 = clf1.predict(X_test)
                    SVM_acc = accuracy_score(y_test,y_pred1)

                    clf2 = LogisticRegression()
                    clf2.fit(X_train, y_train)
                    y_pred2 = clf2.predict(X_test)
                    logistic_accuracy = accuracy_score(y_test,y_pred2)

                    clf3 = DecisionTreeClassifier(criterion='gini',max_depth = 5)
                    clf3 = clf3.fit(X_train,y_train)
                    y_pred3 = clf3.predict(X_test)
                    decision_tree = metrics.accuracy_score(y_test, y_pred3)
                    estimators=[('SVM_acc', clf1), ('log_reg', clf2), ('decision_tree', clf3)]
                    ensemble = VotingClassifier(estimators, voting='soft')
                    ensemble.fit(X_train, y_train)
                    y_pred_f=ensemble.predict(X_test)
                    Voting_accuracy = accuracy_score(y_test,y_pred_f)
                    cm = confusion_matrix(y_test, y_pred_f)
                    print("Classification Report")
                    print(classification_report(y_test, y_pred_f))
                    print("Confusion Matrix\n",cm)

                    Accuracy = [SVM_acc,logistic_accuracy,decision_tree, Voting_accuracy]
                    print("Accuracy of SVM Classifier is",SVM_acc)
                    print("Accuracy of Logistic Regression is",logistic_accuracy)
                    print("Accuracy of Decision Tree Classifier is",decision_tree)
                    print("Accuracy of Voting Classifier is",Voting_accuracy)

                    y_true = pd.DataFrame(y_test.values.copy(),columns=['Actual'])
                    y_pred_f = pd.DataFrame(y_pred_f.copy(),columns=['Predicted'])
                    X_test = pd.DataFrame(X_test)
                    X_test.index = range(len(X_test))
                    cols = [X_test,y_true,y_pred_f]

                    final = pd.concat(cols,axis=1)
                    df1 = final.to_dict(orient='records')
                    listed = [Accuracy, df1]

			#predicting class prob for all classifiers
                    #probas = [c.fit(x, y).predict_proba(x) for c in (clf1, clf2, clf3, ensemble)]
                    #class1_1 = [pr[0, 0] for pr in probas]
                    #class2_1 = [pr[0, 1] for pr in probas]
                    return listed
                else:
                    listed = ['There are less than two coloumns to build the model few of the variables are removed during preprocessing']
            else:
                listed = ['yValue removed while preprocessing! Please select appropriate yValue as the Target variable.']
        else:
            listed = ['yValue is Numerical. Logistic Regression works on Categorical Binary yValue.']
    else:
        listed = ("Not enough data to perform modelling need atleast 150 observations to perform modelling")
    return listed

#### Forecasting

@app.route("/forecasting", methods = ['POST'])

def Forecasting():
    print("in Forecasting")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    print(columnList)
    xValues = request.form.get('xValues')
    xValues = ast.literal_eval(xValues)
    xValues = xValues[0]
    print(xValues)
    print(type(xValues))
    yValue = request.form.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    print(yValue)
    parametersObj=request.form.get('parametersObj')
    pool = ThreadPool()
    async_result = pool.apply_async(ForecastingThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue,parametersObj))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)


def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

def ForecastingThread(query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue, parametersObj):
    print("in ForecastingThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    parametersObj = json.loads(parametersObj)
    #print(query_result)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(df)
    #data, columnsArray_e = lp_f.columns_data_type(df, columnsArray)
    data, columnsArray_e = lp_f.columns_data_type(df, json.loads(columnsArray))
    nan_list = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"]
    for i in nan_list:
        data = data.replace(to_replace = i,value = np.nan)
    print(data)
    ##temp
    #data = data[data['Product Category']=='Tops']
    #data = data[data['Consumer Group']=='Total Mens']
    #data = data[data['Ship To Customer']=='SCU - GRAND INDONESIA - JAKARTA PU (0020014703)']
    if data.shape[0]>150:
        y_len = len(data[yValue].unique())
        if len(data[yValue].unique())>30:
            ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t, sub, drop_columns = lp_f.correlations(data, columnsArray=columnsArray_e, Val = data.columns, xVal = xValues, yVal =yValue,no_rem_col = yValue)
            print(rm_cols);print(miss_cols);print(drop_columns)
            print('######### Data after correlations ##############')
            print(data.head())
            num_data = data.select_dtypes(include=['number', 'float', 'int']).copy()
            print('######### Num data ##########')
            print(num_data.head())
            
            if len(data[yValue].unique())>30:
                if data[yValue].dtypes == 'datetime64[ns]':
                    output = forecasting_model(data,variable_col = xValues,date_col = yValue, model = str(parametersObj['forecastingfamily']), independentVariables='',test_split = 0.25, yValue = yValue, xValues = xValues, Objt = obj_t, no_of_periods_to_forecast=int(parametersObj['no_of_periods_to_forecast']), columnsArray_e = columnsArray_e)
                    try:
                        output[0] = output[0].to_dict(orient='records')
                        output[6] = output[6].to_dict(orient='records')
                        output[8] = output[8].to_dict(orient='records')
                        output[9] = output[9].to_dict(orient='records')
                        
                        ## Removing nan from decomposed data
                        trial = output[0]
                        for i in range(len(trial)):
                            for key, values in trial[i].items():
                                if key == 'trend':
                                    if math.isnan(trial[i][key]):
                                        #print(key)
                                        del trial[i][key]
                                        break
    
                        for i in range(len(trial)):
                            for key, values in trial[i].items():
                                if key == 'randomness':
                                    if math.isnan(trial[i][key]):
                                        #print(key)
                                        del trial[i][key]
                                        break
    
                        ## Removing NAN from forcasted data
                        trial = output[8]
                        for i in range(len(trial)):
                            for key, values in trial[i].items():
                                if key == 'Value':
                                    if math.isnan(trial[i][key]):
                                        del trial[i][key]
                                        break
    
                        for i in range(len(trial)):
                            for key, values in trial[i].items():
                                if key == 'fitted':
                                    if math.isnan(trial[i][key]):
                                        del trial[i][key]
                                        break
    
                        for i in range(len(trial)):
                            for key, values in trial[i].items():
                                if key == 'Point.Forecast':
                                    if math.isnan(trial[i][key]):
                                        del trial[i][key]
                                        break
    
                    except Exception:
                        output[0] = list(output[0])
    
                    print(type(output))
                else:
                    output = [{"message": "Target Variable is not in Date time format", "error": "Error"}]
            else:
                output = [{"message": "Target Variable must have more than 30 days to make predictions", "error": "Error"}]
        else:
            if y_len == 1:
                output = [{"message": "Target Variable has only 1 day and must have more than 30 days to make predictions", "error": "Error"}]
            else:
                output = [{"message": "Target Variable has "+ str(y_len)+ " days and must have more than 30 days to make predictions", "error": "Error"}]
    else:
        output = [{"message": "Not enough data is given", "error": "Error"}]
    return output

'''
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def nnetar(time_ser, n_steps, no_of_periods_to_forecast):
    print('in NNETAR model')
    x, y = split_sequence(time_ser, n_steps)
    X = x.reshape(x.shape[0], 1, x.shape[1])
    neu = int(((2/3)*(len(time_ser)))-1)
    # define model
    print('defining model')
    model = Sequential()
    model.add(LSTM(neu, dropout = 0.3, batch_input_shape=(1, X.shape[1], X.shape[2]), stateful=True))
    neu = int(neu/2)
    model.add(Dense(neu))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    print(len(time_ser))
    if len(time_ser)>150:
        model.fit(X, y, epochs=500, batch_size=1, verbose=1, shuffle=False)
    else:
        model.fit(X, y, epochs=300, batch_size=1, verbose=1, shuffle=False)

    ###Train predictions
    x_input_index = list(time_ser.index[-n_steps:])
    x_input = array(list(time_ser[x_input_index]))

    forecasted = []
    for i in range(no_of_periods_to_forecast):
        x_input = x_input.reshape(1, 1, n_steps)
        pred = model.predict(x_input)
        forecasted.append(pred)

        x_input = x_input.reshape(1, n_steps)
        x_input = np.append(x_input, pred, axis=1)
        x_input = x_input[0][-n_steps:]
        
    print('NNETAR forecast completed')
    return forecasted
'''

def forecasting_model(data_initial,variable_col,date_col,model,independentVariables,test_split,yValue, xValues, Objt,no_of_periods_to_forecast, columnsArray_e):
    
    if date_col == '':
        index = data_initial.index
    else:
        index = pd.DatetimeIndex(data_initial[date_col])
    print("done")
    ts = pd.Series(data_initial[variable_col])
    ts.index = index 
    ts = ts.sort_index()
    test_ind = round(len(ts)*test_split)
    ts_train = ts[:len(ts)-test_ind]
    ts_test = ts[-test_ind:]
    no_of_periods_to_forecast = no_of_periods_to_forecast
    #ts_train_df = lp_f.transformation_inv(pd.DataFrame(ts_train), Objt)
    ts_test_df = lp_f.transformation_inv(pd.DataFrame(ts_test), Objt)
    
    #ts.plot()
    
    ########## resampling with time frame monthly, quarterly and annual          
    try:
        print('Resampling data')
        # Resampling to monthly frequency
        ts_monthly = ts_train.resample('M').mean()
        #print(type(ts_monthly))
        temp_ts_monthly = ts_monthly
        ts_monthly = pd.DataFrame(columns =[yValue, xValues], index = range(len(temp_ts_monthly)))
        ts_monthly[yValue] = temp_ts_monthly.index
        ts_monthly[xValues] = temp_ts_monthly.values
        print(ts_monthly)
        # Resampling to quarterly frequency
        ts_quarterly = ts_train.resample('Q-DEC').mean()
        temp_ts_quarterly = ts_quarterly
        ts_quarterly = pd.DataFrame(columns =[yValue, xValues], index = range(len(ts_quarterly)))
        ts_quarterly[yValue] = temp_ts_quarterly.index
        ts_quarterly[xValues] = temp_ts_quarterly.values
        print(ts_quarterly)
        #Resampling to annual frequency
        ts_annual = ts_train.resample('A-DEC').mean()
        temp_ts_annual = ts_annual
        ts_annual = pd.DataFrame(columns =[yValue, xValues], index = range(len(ts_annual)))
        ts_annual[yValue] = temp_ts_annual.index
        ts_annual[xValues] = temp_ts_annual.values
        print(ts_annual)
        
        
    #    ts_monthly.plot()
    #    ts_quarterly.plot()
    #    ts_annual.plot()
    
    except:
        print('You need specify the date column')
    
    days_available = len(data_initial.dropna()) 
    months_available = len(ts_monthly.dropna())
    years_available= len(ts_annual.dropna())
    print('done till')
    
    if  days_available+months_available+years_available == 0:
        error = "Insufficient data for time series analysis with "+days_available+" Days, "+months_available+" Months and "+years_available+" Years"
        print(error)
    #    return 1
    else:
        error = ''
    
    try:
        print('Decomposing data')
        result = seasonal_decompose(ts_train, model='additive', freq=12)
        #result.plot()
        decomp_overall = pd.DataFrame([list(result.observed), list(result.trend), list(result.seasonal), list(result.resid), list(ts_train.index)])
        decomp_overall = decomp_overall.transpose()
        decomp_overall.columns = ["actual","trend","seasonality","randomness","date"]
        count = decomp_overall['seasonality'].value_counts() 
        j= min(count)
        #print('number of cycles':j)
        period = len(decomp_overall)/j
        print(period)
    except Exception as e:
        error = "Unable to Decompose time series"
        print(e)
       
    Percentage_Variance_Explained_by_trend = (decomp_overall['trend'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    Percentage_Variance_Explained_by_seasonality = (decomp_overall['seasonality'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    Percentage_Variance_Explained_by_randomness = (decomp_overall['randomness'].dropna().var())/(decomp_overall['actual'].dropna().var())*100
    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(list(decomp_overall['actual']), range(len(decomp_overall)))
    slope_with_time = gradient
    
    var_temp = variable_col
    for i in range(len(columnsArray_e)):
        if variable_col == columnsArray_e['changedColumnName'].iloc[i]:
            var_temp = columnsArray_e['columnName'].iloc[i]
            
        
    slope_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> The slope of time for <font style=\"color: #078ea0;\">"+var_temp+" </font> is <b>"+str(round(slope_with_time,4))+"</b> , For every one unit change in time <font style=\"color: #078ea0;\">"+var_temp+" </font> is effected by <b>"+str(round(slope_with_time,4))+"</b> units </span></li>"
     
    time_plot = "<li class=\"nlg-item\"><span class=\"nlgtext\"> The change of "+var_temp+" over time </span></li>"
    
    
    #model = 'NNETAR'
    future=ts
#    future=future.reset_index()
    mon=future
#    future = pd.DataFrame(future)
#    mon=future
    mon=mon.index+pd.DateOffset(months=int(no_of_periods_to_forecast))
    future_dates = mon[-int(no_of_periods_to_forecast):]
    #future = future.set_index('columnname2')
    newDf = pd.DataFrame(index=future_dates, columns=[variable_col])
    future = pd.concat([future,newDf])
    future[variable_col] = future[0]
    future = future[variable_col]
    
    try:  
        if (model == "Holtwinters"):
            print('Running Holtwinters Model')
            forecast_model = ExponentialSmoothing(np.asarray(decomp_overall['actual']),seasonal_periods=int(period) ,trend='add', seasonal='add').fit()
            forecasted = forecast_model.predict(start=len(ts_train)-1, end=len(ts_train)-1+len(ts_test)-1)
            future.iloc[-int(no_of_periods_to_forecast):] = forecast_model.forecast(no_of_periods_to_forecast)
            forecasted = pd.Series(forecasted)
            future = pd.Series(future)
            print('Holtwinters done')
        elif (model == "ARIMA"):
            # evaluate an ARIMA model for a given order (p,d,q)
            '''
                def evaluate_arima_model(arima_order):
                	# make predictions
                	model = ARIMA(ts_train, order=arima_order)
                	model_fit = model.fit(disp=0)
                	forecasted = model_fit.predict(start=len(ts_train)-1, end=len(ts_train)-1+len(ts_test)-1)
                	# calculate out of sample error
                	error = mean_squared_error(ts_test, pd.Series(forecasted))
                	return error
     
                # evaluate combinations of p, d and q values for an ARIMA model
                def evaluate_models(p_values, d_values, q_values):
                    
                	best_score, best_cfg = float("inf"), None
                	for p in p_values:
                		for d in d_values:
                			for q in q_values:
                				order = (p,d,q)
                				try:
                					mse = evaluate_arima_model(order)
                					if mse < best_score:
                						best_score, best_cfg = mse, order
                					print('ARIMA%s MSE=%.3f' % (order,mse))
                				except:
                					continue
                	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
                	return best_cfg
                
                p_values = [0,1,2,3,4]
                d_values = [0,1]
                q_values = [0,1,2,3,4]
                ord_det = evaluate_models(p_values, d_values, q_values)
                print(ts_test_df)
                forecast_model = ARIMA(ts_train, order=ord_det)
                
                print('here')
                try:
                    model_fit = forecast_model.fit(disp=-1)
                    forecasted = model_fit.predict(start=len(ts_train_df)-1, end=len(ts_train_df)-1+len(ts_test_df)-1)
                    print(forecasted)
                    future.iloc[-int(no_of_periods_to_forecast):] = model_fit.forecast(steps=no_of_periods_to_forecast)[0]
                    forecasted = pd.Series(forecasted)
                    future = pd.Series(future)
                except Exception as e:
                    print(e)
            '''
            ## Testing SARIMAX
            
            print('Running ARIMA Model')
            smodel = pm.auto_arima(ts_train, start_p=1, start_q=1,test='adf',max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=None, D=1, trace=True,error_action='ignore', suppress_warnings=True, stepwise=True)
            fitted, confint = smodel.predict(n_periods=len(ts_test_df), return_conf_int=True)
            forecasted = pd.Series(fitted)
            #Future values
            fitted_future, confint = smodel.predict(n_periods=len(ts_test_df)-1+no_of_periods_to_forecast, return_conf_int=True)
            future.iloc[-int(no_of_periods_to_forecast):] = fitted_future[-no_of_periods_to_forecast:]
            future = pd.Series(future)    
            '''          
                elif (model == "NNETAR"):
                    print(' in NNETAR')
                    try:
                        forecasted = nnetar(ts,7,(len(ts_test)+int(no_of_periods_to_forecast)))
                        #future.iloc[-int(no_of_periods_to_forecast):] = nnetar(ts,7,int(no_of_periods_to_forecast))
                    
                    except Exception as e:
                        msg = str(e)
                        print(msg)
                    
                    tem_forec = []
                    for i in range(len(forecasted)):
                        tem_forec.append(forecasted[i][0][0])
                    forecasted = pd.Series(tem_forec)
                    
                    future.iloc[-int(no_of_periods_to_forecast):] = forecasted[-int(no_of_periods_to_forecast):].values
                    forecasted = forecasted[:-int(no_of_periods_to_forecast)]
                    print('future forecast')
                    
                    future = pd.Series(future)
        #            forecast_model = ARIMA(ts_train, order=(7,0,1))
        #            model_fit = forecast_model.fit(disp=0)
        #            forecasted = model_fit.predict(start=len(ts_train)-1, end=len(ts_train)-1+no_of_periods_to_forecast-1)
            '''
        else:
            print('Model method not specified')   
        
        #### Error metrics evaluate
        forecasted.index = ts_test.index
        
        ##Inverse transforming forecasted data
        forec= pd.DataFrame(columns = ['Forecasted'], index = forecasted.index)
        forec['Forecasted'] = forecasted.values
        forecasted= lp_f.transformation_inv(forec, Objt)
        forecasted.index = ts_test.index
        
        ## Inverse transforming original ts data
        tss = pd.DataFrame(columns = ['Original'], index = forecasted.index)
        tss['Original'] = ts_test.values
        tss_test = lp_f.transformation_inv(tss, Objt)
        tss_test.index = forecasted.index
        dropping_index = []
        for i in range(len(tss_test)):
            if tss_test['Original'].iloc[i] == 0 or tss_test['Original'].iloc[i] == np.nan:
                dropping_index.append(tss_test['Original'].index[i])
        
        for i in dropping_index:
            tss_test.drop(index = i, inplace = True)
            forecasted.drop(index = i, inplace = True)
        
        tss_test = pd.Series(tss_test['Original'])
        forecasted = pd.Series(forecasted['Forecasted'])
        #print('forecasted')
        '''
        tss_t = pd.DataFrame(columns = ['Original'], index = range(len(ts_train)))
        tss_t['Original'] = ts_train.values
        tss_train = transformation_inv(tss_t, Objt)
        tss_train.index = ts_train.index
        tss_train = pd.Series(tss_train['Original'])
        '''
        MAPE = np.mean((np.abs(forecasted - tss_test)/np.abs(tss_test))*100)
        MSE = np.mean((forecasted - tss_test)**2)
        ME = np.mean(forecasted - tss_test)
        MAE = np.mean(np.abs(forecasted - tss_test))
        
        print('Applying Inverse transformation on data')
        # NLG with model description
        
        Percentage_Variance_Explained_by_trend_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Trend </font> is <b>"+str(round(Percentage_Variance_Explained_by_trend,4))+"</b> </span></li>"
        Percentage_Variance_Explained_by_seasonality_text = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Seasonality </font>is <b>"+str(round(Percentage_Variance_Explained_by_seasonality,4))+ "period of each seasonal cycle is <b>"+str(period)+"</b> </span></li>"
        Percentage_Variance_Explained_by_randomness_text  = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #078ea0;\"> Randomness </font> is <b>"+str(round(Percentage_Variance_Explained_by_randomness,4))+"</b> </span></li>"
        
        seasonal_text = "The amount of "+var_temp+" is effected due to seasonality with time"
        seasonal_component = pd.concat([decomp_overall["date"],decomp_overall["seasonality"]], axis=1)
        model_text = "Model has Forecasted the "+ var_temp+" for next "+str(no_of_periods_to_forecast)+" periods"
        
        forecasted_temp = pd.DataFrame(columns = ['Date'], index = forecasted.index)
        '''
        tss_test_temp = pd.DataFrame(columns = ['Original'], index = forecasted.index)
        tss_test_temp['Original'] = tss_test.values
        '''
        forecasted_temp['Date'] = forecasted_temp.index
        forecasted_temp['Value'] = tss_test.values
        forecasted_temp['fitted'] = forecasted.values
        forecasted_temp.index = range(len(forecasted_temp))
        
        future_foc = pd.DataFrame(columns = ['Date', str(variable_col)+' Forecast values'], index = range(int(no_of_periods_to_forecast)))
        future_foc['Date']= future[-int(no_of_periods_to_forecast):].index
        future_foc[str(variable_col)+' Forecast values'] = future[-int(no_of_periods_to_forecast):].values
        
        ## Inverse transforming future data
        
        try:
            fut_forec= pd.DataFrame(columns = ['Future'], index = future[-int(no_of_periods_to_forecast):].index)
            fut_forec['Future'] = future_foc[str(variable_col)+' Forecast values'].values
            forecasted= lp_f.transformation_inv(fut_forec, Objt)
            print('transinf done')
        
            future_foc = pd.DataFrame(columns = ['Date', str(variable_col)+' Forecast values'], index = range(int(no_of_periods_to_forecast)))
            future_foc['Date']= future[-int(no_of_periods_to_forecast):].index
            future_foc[str(variable_col)+' Forecast values']  = forecasted.values
            print('future vals')
            
            forecasted_temp['Point.Forecast'] = np.nan
            merge_forv = future_foc
            merge_forv.columns = ['Date', 'Point.Forecast']
            forecasted_temp = pd.concat([forecasted_temp, merge_forv], axis = 0)
            forecasted_temp.index = range(len(forecasted_temp))
            print('merge_future')
        except Exception as e:
            print(str(e))
        ## Inverse transformation on ts_train
        ts_inv = pd.DataFrame(columns = ['Value'],index=range(len(ts_train)))
        ts_inv['Value'] =ts_train.values
        ts_inv = lp_f.transformation_inv(ts_inv, Objt)
        
        ## Final df with ts_train, ts_test and future prediction
        total = pd.DataFrame(columns = ['Date','Value','fitted','Point.Forecast'],index=range(len(ts_train)))
        total['Date']= ts_train.index
        total['Value'] = ts_inv['Value'].values
        total_df = pd.concat([total,forecasted_temp],axis=0)
        
        print(MAPE)
        output = [decomp_overall, slope_text, Percentage_Variance_Explained_by_trend_text,Percentage_Variance_Explained_by_seasonality_text,Percentage_Variance_Explained_by_randomness_text,seasonal_text,seasonal_component,model_text,total_df,future_foc,[MAPE],[MSE],[ME],[MAE], time_plot] 
        #print(output)
    except Exception as e:
        if 'constant' in str(e):
            output = [{"message": "Data to predict cannot be a constant value.", "error": "Error"}]
        else:
            output = [{"message": "Encountered Exception :" + str(e), "error": "Error"}]
    
    return output    

#### MLP

#@csrf_exempt
@app.route("/MLP", methods = ['POST'])

def mlpclassifier(request):
    query = request.args.get('query')

    dbName = request.args.get('dbName')
    password = request.args.get('password')
    userName = request.args.get('userName')
    columnsArray = request.args.get('columnsArray')
    dbHost = request.args.get('dbHost')
    columnList =request.args.get('columnList')
    xValues = request.args.get('xValues')
    print(xValues)
    xValues = ast.literal_eval(xValues)
    print(xValues)
    yValue =request.args.get('yValue')
#    yValue = ast.literal_eval(yValue)
#    yValue = yValue[0]
    pool = ThreadPool()

    async_result = pool.apply_async(MLPThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue))
    return_val = async_result.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr

def MLPThread(query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue):
    #data = pd.read_excel("C:/Users/Akhilesh/Downloads/BoC/Ensemble/Data1.xlsx",encoding="latin")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(yValue)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    if data.shape[0]>150 :
         ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub, drp_clmns = correlations(data, columnsArray=columnsArray_e,no_rem_col='none',Val=data.columns,xVal=xValues,yVal=yValue)
         if data[yValue].dtypes != 'float' and data[yValue].dtypes != 'int':
             if yValue not in rm_cols and yValue not in drp_clmns:
                 print('######### Data after correlations #########################')
                 print(data.head())
                 num_data = data.select_dtypes(include=['number']).copy()
                 print('######### Num data #########################')
                 print(num_data.head())

                 if data.shape[1]>1 and len(data[yValue].unique())>1:

                     cols = data.columns
                     num_cols = data._get_numeric_data().columns
                     cat_cols = list(set(cols) - set(num_cols))
                     try:
                         cat_cols.remove(yValue)

                     except ValueError:
                         cat_cols = cat_cols

                     y_new = data[yValue]
                     cols_y = yValue
                     le = preprocessing.LabelEncoder()
                     le.fit(y_new)
                     y_new = le.transform(y_new)
                     y_new = y_new.astype('str')
                     y_new= pd.DataFrame(y_new,columns = [cols_y])
                     data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                              #    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)

                     x = data_n.drop(yValue,axis=1)
                     y = y_new
                     if(data[yValue].dtypes != 'float') and (data[yValue].dtypes != 'int'):
                         X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)
                         
                         mlp = MLPClassifier()

                         parameter_space = {
                         'hidden_layer_sizes': [(10,20,10),(50,100,50)],
                         'activation': ['tanh', 'relu'],
                         'solver': ['sgd', 'adam'],
                         'alpha' : [0.0001, 0.05],
                         'learning_rate': ['constant','adaptive'],
                         'max_iter':[100,200,500]
                             }


                         clf = GridSearchCV(mlp, parameter_space, n_jobs=4, cv=5,verbose=1)
                         clf.fit(X_train, y_train)
                         print('Best parameters found:\n', clf.best_params_)
                         model = clf.best_estimator_
                         print("the best model and parameters are the following: {} ".format(model))


                         means = clf.cv_results_['mean_test_score']
                         stds = clf.cv_results_['std_test_score']
                         for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                             print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

                         y_true, y_pred = y_test , clf.predict(X_test)

                         y_true = pd.DataFrame(y_test.values.copy(),columns=['Actual'])
                         y_pred = pd.DataFrame(y_pred.copy(),columns=['Predicted'])
                         X_test = pd.DataFrame(X_test)
                         X_test.index = range(len(X_test))
                         cols = [X_test,y_true,y_pred]

                         final = pd.concat(cols,axis=1)

                         print('Results on the test set:')
                         print(classification_report(y_true, y_pred))

                         print("Confusion Matrix:")
                         print(confusion_matrix(y_test, y_pred))
                         accuracy = accuracy_score(y_test, y_pred)
                         print(accuracy)

                         #loss_value = model.loss_curve_

                         df1 = final.to_dict(orient='records')
                         listed = [accuracy, df1]
                     else:
                        listed = ("Incorrect Yvalue datatype is selected. Yvalue must be in string")
                 else:
                     listed = ['yValue removed while preprocessing! Please select appropriate yValue as the Target variable.']

         else:
             listed = ['yValue is Numerical. MLP works on Categorical Binary yValue.']

    else:
        listed = ("Not enough data to perform modelling need atleast 150 observations to perform modelling")


#    plt.plot(loss_value)
#    plt.show()

    return listed

#### Random Forest

#@csrf_exempt
@app.route("/random_forest", methods = ['POST'])

def Randomforest():
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    yValue =request.form.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    parametersObj= request.args.get('parametersObj')
    print(parametersObj)
    pool = ThreadPool()
    async_result = pool.apply_async(RandomForestThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,yValue,parametersObj))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def RandomForestThread(query,dbName,password,userName,columnsArray,dbHost,columnList,yValue, parametersObj):
    print("in RandomForestThread")
    parametersObj = json.loads(parametersObj)
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    nan_list = [".",",","@","#","$","%","^","&","*","(",")","-","+","=","?"]
    for i in nan_list:
        data = data.replace(to_replace = i,value = np.nan)
    if data.shape[0]>150 and data.shape[1]>1:
        print(data.columns)
        print(data)
        if len(data[yValue].unique())>1:
            ent_cor,data,rm_cols, miss_cols, obj_t, sub, drop_columns,trans_col = correlations(data, columnsArray=columnsArray_e, no_rem_col='none', Val = data.columns)
            data = data.loc[:,~data.columns.duplicated()]
            if data.shape[1]>1:
                xValues = list(filter(lambda x: x not in rm_cols and x not in yValue, data.columns))
                data_x_unique = data[xValues]
                for i in data_x_unique.columns:
                    if len(data_x_unique[i].unique())>1:
                        continue
                    else:
                        if data[i].dtypes != 'float' and  data[i].dtypes != 'int':
                            print('dropping ' + str(i))
                            data = data.drop(columns = [i])
                            drop_columns.append(i)
                        else:
                            if data_x_unique.shape[1]>1:
                                continue
                            else:
                               data = data.drop(columns = [i])
                               drop_columns.append(i)
            else:
                listed = ['There is only one column after preprocessing. Please try giving different inputs or different dataset as the dataset might not be applicable for this model.']
                return listed
            if data.shape[1]>1:
                if yValue not in list(rm_cols) and yValue not in drop_columns:
                    xValues = list(filter(lambda x: x not in rm_cols and x in data.columns, xValues))
                    try:
                            variable_imp = variable_importance_h2o(data, xValues, yValue)
                    except Exception as e:
                            msg = str(e)
                            if 'Response cannot be constant' in msg:
                                listed = ['Target Variable cannot be a constant value.']
                            else:
                                listed = [msg]
                            return listed
                        #labels = xValues
                    var_imp = []; var_name=[]
                    for i in range(len(variable_imp)):
                            for j in range(len(variable_imp[i])):
                                var_imp.append(variable_imp[i][j][2])
                                var_name.append(variable_imp[i][j][0])
                    Var_imp  = pd.DataFrame(columns = ['y'], index = range(len(var_imp)))
                    Var_imp['y'] = var_imp
                    Var_name = pd.DataFrame(columns = ['x'], index = range(len(var_name)))
                    Var_name['x'] = var_name
                    frame_var = [Var_name, Var_imp]
                    Var_imp = pd.concat(frame_var, axis = 1)

                    vif_frame = [data[xValues], data[yValue]]
                    data_vif = pd.concat(vif_frame, axis = 1)
                    num_data = data_vif.select_dtypes(include=['number', 'int', 'float']).copy()
                    cat_data = data_vif.select_dtypes(include=['object', 'category']).copy()
                    #vif_var = vif(num_data, yValue, 10)
                    ### Data preperation
                    data_x = data[xValues]
                    # Categorical Y value

                    data_y = pd.DataFrame(data[yValue], columns = [yValue])
                    if yValue in cat_data.columns:
                        # Converting YValue number to word so regression will not be performed.
                        try:
                            data_y[yValue] = pd.to_numeric(data_y[yValue])
                            for i in range(data_y[yValue].shape[0]):
                                data_y[yValue][i] = nw.convert(data_y[yValue].iloc[i])
                                print("Im here")
                        except Exception:
                            data_y[yValue]=data_y[yValue].astype("category")
                            data_y = data_y

                    frames= [data_x, data_y]
                    data2 = pd.concat(frames, axis=1)
                    #sub = data2.columns
                    print(data2)
                    ## Model Building ##
                    try:
                        hf = h2o.H2OFrame(data2)
                        if  hf.shape[1]<1:
                            try:
                                    for i in data2.columns:
                                        if data2[i].dtypes == 'float':
                                            continue
                                        else:
                                             data2[i] = data2[i].str.encode('utf-8')
                            except Exception:
                                    data2 = data2
                            hf = h2o.H2OFrame(data2)
                        else:
                            hf = hf
                    except Exception as e:
                        msg = str(e)
                        listed = [msg]
                        return listed

                    train, valid, test = hf.split_frame(ratios=[.7, .1])
                    Nfolds = int(parametersObj['Nfolds'])
                    bc = bool(parametersObj['balance_classes'])
                    fold = str(parametersObj['fold_type'])
                    md = int(parametersObj['max_depth'])

                    if data2.shape[1]>1 and len(data2[yValue].unique())>1:

                        if(data[yValue].dtypes == 'float') or (data[yValue].dtypes == 'int64'):
                            grid_search_rf = H2ORandomForestEstimator(nfolds = Nfolds, stopping_rounds = 20, max_depth = md, stopping_metric = "rmse", sample_rate = 0.65)
                            try:
                                hyper_params = {'mtries':[2,3,4], 'ntrees':[100, 200, 300]}
                                grid = H2OGridSearch(grid_search_rf, hyper_params, search_criteria={'strategy': "Cartesian"})
                                #Train grid search
                                grid.train(xValues, yValue, training_frame= train, validation_frame=valid)

                                grid_sorted = grid.get_grid(sort_by='rmse',decreasing=False)
                                print(grid_sorted)
                                best_mod = grid_sorted.models[0]
                            except Exception:
                                rf = H2ORandomForestEstimator()
                                rf.train(xValues, yValue, training_frame= train, validation_frame=valid)

                                best_mod = rf
                            ## Prediction
                            predicted = best_mod.predict(test_data=test)

                            ### Changing the transformed numericals back to original values
                            print(test.columns)
                            test_trueY = h2o.as_list(test)

                            print(test_trueY)
                            x = list(filter(lambda x: x in num_data.columns and x in xValues, test.columns))
                            if yValue not in x:
                                x.append(yValue)
                                
                            temp = list(filter(lambda x: x in obj_t, obj_t)) 
                            test_true_inv = transformation_inv(test_trueY[x],obj_t)
                            true_y = test_true_inv[yValue]
                            ### For plotting transformed true Y data
                            #true_y_trnf = test_trueY[yValue]

                            #test_inv = transformation_inv(test.as_data_frame(),obj_t)
                            test[yValue] = predicted
                            test_predY = h2o.as_list(test)
                            test_pred_inv = transformation_inv(test_predY[x],temp)
                            pred_y = pd.DataFrame(columns = ['Predicted'+' '+ str(yValue)], index = range(test_pred_inv.shape[0]))
                            pred_y['Predicted'+' '+ str(yValue)] = test_pred_inv[yValue]
                            ### For plotting transformed predicted Y data
                            #pred_y_trnf = test_predY[yValue]

                            ## Final Data Frame
                            act_pred=true_y - pred_y['Predicted'+' '+ str(yValue)]
                            Diff = pd.DataFrame(columns = ['Actual-Predicted'],index=range(len(act_pred)))
                            Diff['Actual-Predicted'] = act_pred
                            test_truecat = test_trueY[cat_data.columns]
                            x.remove(yValue)
                            print(test_truecat.shape[1])
                            print(num_data.shape[1])
                            if (test_truecat.shape[1]>0) and (num_data.shape[1]==1):
                                frame = [true_y, pred_y, Diff, test_truecat]
                            elif (test_truecat.shape[1]==0) and (num_data.shape[1]>0):
                                frame = [true_y, pred_y, Diff, test_true_inv[x]]
                            else:
                                frame = [true_y, pred_y, Diff, test_truecat, test_true_inv[x]]
                            Final_pred_data = pd.concat(frame, axis= 1)
                            print(Final_pred_data)
                            for i in Final_pred_data.columns:
                                if Final_pred_data[i].dtype == 'float' or Final_pred_data[i].dtypes=='int':
                                    for j in range(Final_pred_data[i].shape[0]):
                                        if np.isnan(Final_pred_data[i].iloc[j]):
                                            Final_pred_data[i].iloc[j]=0

                            ## Plotting variable importance
                            #rf_model.varimp_plot()

                            ## Variable importance
                            #var_imp = best_mod.varimp()
                            list2 = Final_pred_data.to_dict(orient='records')
                            # Top 6 variable importance
                            try:
                                    Var_imp = Var_imp[0:6]
                            except Exception:
                                    Var_imp = Var_imp
                            list3 = Var_imp.to_dict(orient='records')
                            r2 = round(best_mod.r2(),2)
                            if r2 >0:
                                r2 = [r2]
                            else:
                                r2 = [0]
                            listed = [r2, list3, list2]
                            #return rf_output

                        else:
                            print("Building Random Forest Model for categorical Y by considering all variables in data")
                            try:
                                grid_search_rf = H2ORandomForestEstimator(nfolds = Nfolds, stopping_rounds = 10, max_depth = md, stopping_metric = "logloss", sample_rate = 0.65, balance_classes = bc, fold_assignment = fold)
                                try:
                                    hyper_params = {'mtries':[2,3,4], 'ntrees':[100, 200, 300]}
                                    grid = H2OGridSearch(grid_search_rf, hyper_params, search_criteria={'strategy': "Cartesian"})
                                    #Train grid search
                                    grid.train(xValues, yValue, training_frame= train, validation_frame=valid)
                                except Exception as e:
                                    print(e)
                                    '''
                                    if len(hyper_params['mtries'])>0:
                                        for i in range(len(hyper_params['mtries'])):
                                            try:
                                                hyper_params['mtries'].pop()
                                                grid = H2OGridSearch(grid_search_rf, hyper_params, search_criteria={'strategy': "Cartesian"})
                                                #Train grid search
                                                grid.train(xValues, yValue, training_frame= train, validation_frame=valid)
                                                break
                                            except Exception:
                                                continue
                                    '''
                                    

                                grid_sorted = grid.get_grid(sort_by='logloss',decreasing=False)
                                print(grid_sorted)

                                best_mod = grid_sorted.models[0]
                            except Exception:
                                rf = H2ORandomForestEstimator(nfolds = Nfolds, stopping_rounds = 10, max_depth = md, stopping_metric = "logloss", sample_rate = 0.65, balance_classes = bc, fold_assignment = fold)
                                rf.train(xValues, yValue, training_frame= train, validation_frame=valid)
                                best_mod = rf
                                
                            # Prediction
                            predict = best_mod.predict(test_data = test) 
                            predict = h2o.as_list(predict)
                            pred = pd.DataFrame(columns = ['Predicted ' +str(yValue)], index = range(predict['predict'].shape[0]))
                            try:
                                for i in range(predict.shape[0]):
                                        pred['Predicted ' +str(yValue)][i] = w2n.word_to_num(predict['predict'].iloc[i])
                            except Exception:
                                pred['Predicted ' +str(yValue)]= predict['predict']
                            pred['Predicted ' +str(yValue)]=pred['Predicted ' +str(yValue)].astype("category")

                            ## true y
                            test = h2o.as_list(test)

                            print("Converting word to true numbers")
                            true_y = pd.DataFrame(columns = ['Actual'], index = range(test.shape[0]))
                            try:
                                for i in range(true_y['Actual'].shape[0]):
                                        true_y['Actual'][i] = w2n.word_to_num(test[yValue].iloc[i])
                            except Exception:
                                true_y['Actual'] = test[yValue]
                            true_y['Actual']= true_y['Actual'].astype('category')

                            ### test data frame - Inverse transformation
                            if (num_data.shape[1]>0):
                                x = list(filter(lambda x: x in num_data.columns and x in xValues, test.columns))
                                test_true_inv = transformation_inv(test[x],obj_t)
                                test_true_inv_x = test_true_inv[x]
                            else:
                                test_true_inv_x= []

                            ## Metrics
                            cm = confusion_matrix(true_y['Actual'], pred['Predicted ' +str(yValue)])
                            try:
                                col = list(data[yValue].unique())                                    ### classes of yValue
                                col.sort()
                                conf_mat = pd.DataFrame(cm, columns = col, index = col)
                            except Exception:
                                conf_mat = pd.DataFrame(cm)
                            acc = accuracy_score(true_y['Actual'], pred['Predicted ' +str(yValue)])

                            ## Variable importance
                            #var_imp = best_mod.varimp()

                            ## Final Dat Frame
                            test_cat = test.select_dtypes(include=['object', 'category']).copy()
                            try:
                                test_cat = test_cat.drop(columns = [yValue])
                            except Exception:
                                test_cat = test_cat


                            if (cat_data.shape[1]>1) and (num_data.shape[1]==0):
                                frame = [true_y, pred, test_cat]
                            elif (cat_data.shape[1]==1) and (num_data.shape[1]>0):
                                frame = [true_y, pred, test_true_inv_x]
                            else:
                                frame = [true_y, pred, test_cat, test_true_inv_x]
                            Final_pred_data = pd.concat(frame, axis= 1)
                            for i in Final_pred_data.columns:
                                if Final_pred_data[i].dtype == 'float' or Final_pred_data[i].dtypes=='int':
                                    print(i)
                                    for j in range(Final_pred_data[i].shape[0]):
                                        if np.isnan(Final_pred_data[i].iloc[j]):
                                            Final_pred_data[i].iloc[j]=0

                            list1 = [round(acc*10)]
                            list2 = Final_pred_data.to_dict(orient='records')
                            conf_mat = conf_mat.to_dict(orient='records')
                            print("Actual_predicted")
                            # Top 6 variable importance
                            try:
                                    Var_imp = Var_imp[0:6]
                            except Exception:
                                    Var_imp = Var_imp
                            list3 = Var_imp.to_dict(orient='records')
                            #print(Actual_predicted)
                            #list3 = Actual_predicted.to_dict(orient='records')
                            #list4 = [ent_cor]
                            #list5 = [glm_model.coef()]
                            #list6 = [variable_imp]
                            listed = [list1, list3, conf_mat]
                            print("listed")
                    else:
                        listed = ["There is less than two column for building model or the Response variable is constant. Please make sure that there are atleast 1 Input dependant variables and it is not a constant Value"]
                else:
                    listed = ["Selected Target Variable(yValue) is removed during preprocessing"]
            else:
                listed = ["There are less than two columns for building model. Few of the selected Variables have been removed during preprocessing"]
        else:
            listed = ["Target Variable (yValue) cannot be constant"]
    else:
        listed = ["Dataset is too small to make predictions. Please make sure that the datset has 150 rows (atleast) and 2 columns(atleast)."]
    return listed

#### Sentiment Analysis

#@csrf_exempt
@app.route("/sentiment_analysis", methods = ['POST'])

def sentiment_analysis():
    print("in Sentimentanalysis")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    col =request.form.get('col')
    col = ast.literal_eval(col)
    if len(col)>1:
        col1 = col[0]
        col2 = col[1]
        pool = ThreadPool()
        async_result1 = pool.apply_async(sentimentThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,col1))
        async_result2 = pool.apply_async(sentimentThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,col2))
        return_val1 = async_result1.get()
        return_val2 = async_result2.get()
        return_val = [return_val1, return_val2]
    else:
        col1 = col[0]
        pool = ThreadPool()
        async_result1 = pool.apply_async(sentimentThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,col1))
        return_val = async_result1.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr

def sentimentThread(query,dbName,password,userName,columnsArray,dbHost,columnList,col,sentiment=False):
    print("in sentimentThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    labelled_data=pd.DataFrame()
    processed_data =[]
    processed_data=df[col].str.replace(r'0-9[^\w\s]', '')
    ## Removing Numbers
    def remove(list): 
        pattern = '[0-9]'
        list = [re.sub(pattern, '', i) for i in list] 
        return list
    processed_data = remove(processed_data)
    df[col] = remove(df[col])
    ## Removing special characters
    bad_chars = [';', ':', '!', "*",'(',')','{', '}','\n', '|',',','[',']','{','}','?','\"','$','>','<','-','+','=','_','&', '%', '@','.',"'",'/'] 
    for i in range(len(processed_data)):
        processed_data[i] = ''.join(filter(lambda i: i not in bad_chars, processed_data[i]))
        df[col].iloc[i] = ''.join(filter(lambda i: i not in bad_chars, df[col].iloc[i]) )
    
    labelled_data['processed_text']=processed_data
    print(processed_data)
    labelled_data['sentiment'] = 0
    '''
    labelled_data['processed_text'].apply(sentiment_analyzer_scores)
    print(labelled_data['sentiment'])
    labelled_data['pos_neg'] = ['positive' if x =='pos' else 'negative' for x in labelled_data['sentiment']]
    '''
    #labelled_data[labelled_data['sentiment']==0]]
    
    text=pd.DataFrame(labelled_data[['processed_text','sentiment']])
    text['processed_text']=text['processed_text'].str.lower()
    text['processed_text'] = [re.sub('\s+', ' ', sent) for sent in text['processed_text']]
    text['processed_text'] = [re.sub("\'", "", sent) for sent in text['processed_text']]
    data_words = list(sent_to_words(text['processed_text']))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    trigram_mod
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    pos_words = []
    neg_words = []
    nlp =spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams,nlp, allowed_argsags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print(data_lemmatized)
    #print(data_words_bigrams)
    ## Finding Sentiment
    vader = SentimentIntensityAnalyzer()
    print('Finding Positive and Negative words')
    for i in data_words_bigrams:
        for j in i:
             score = vader.polarity_scores(j)
             num1 = score['neg']
             num2 = score['pos']
             num3 = score['neu']
             if (num1 >= num2) and (num1 >= num3):
                 if j in neg_words:
                     continue
                 else:
                     neg_words.append(j)
             elif (num2 >= num1) and (num2 >= num3):
                 if j in pos_words:
                     continue
                 else:
                     pos_words.append(j)
             else:
               largest = 'neu'
             '''
             if sent_words[0] > 0.5:
                pos_words.append(j)
             elif sent_words[0] < 0:
                neg_words.append(j)
             '''
    
    #data_vec=[' '.join(data_lemmatized[i]) for i in range(0,len(data_lemmatized))]
    #vectorizer = CountVectorizer(analyzer='word', min_df=10, stop_words='english',             lowercase=True,token_pattern='[a-zA-Z0-9]{3,}' )
    #data_vectorized = vectorizer.fit_transform(data_vec)
    '''
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    corpus
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    print('Building Model')
    lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(data_vectorized)
    GridSearchCV(cv=None, error_score='raise', estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None, evaluate_every=-1,     learning_decay=0.7, learning_method=None, learning_offset=10.0, max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001, n_components=10, n_jobs=1, perp_tol=0.1, random_state=None, topic_word_prior=None, total_samples=1000000.0, verbose=0), iid=True, n_jobs=1, param_grid={ 'learning_decay': [0.5, 0.7, 0.9]}, pre_dispatch='2*n_jobs', refit=True, return_train_score='warn', scoring=None, verbose=0)
    best_lda_model = model.best_estimator_
    best_lda_model.perplexity(data_vectorized)
    lda_output = best_lda_model.transform(data_vectorized)
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
    docnames = ["Doc" + str(i) for i in range(len(df))]
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames
    df_topic_keywords.head()
    
    # Topic - Keywords Dataframe
    df = df.reset_index(drop=True)
    positive=df.loc[text['sentiment']=='pos']
    positive = positive.reset_index(drop=True)
    #pos_wordlist = positive['description_x'].split()
    pos_words=[]
    print('Finding Positive and Negative Words')
    for i in range(0,len(positive[col])):
        lis=positive[col][i].split()
        pos_words.append(lis)
        flat_list = [item for sublist in pos_words for item in sublist]
        line = [i for i in flat_list if len(i) > 1]
        pos_wordfreq = Counter(line)
    negative=df.loc[text['sentiment']=='neg']
    negative = negative.reset_index(drop=True)
    neg_words=[]
    for i in range(0,len(negative[col])):
            lis=negative[col][i].split()
            neg_words.append(lis)
    neg_list = [item for sublist in neg_words for item in sublist]
    neg_line = [i for i in neg_list if len(i) > 1]
    neg_wordfreq = Counter(neg_line)
        #df_document_topic_json=df_document_topic.to_dict(orient='records')
        #df_topic_keywords_json=df_topic_keywords.to_dict(orient='records')
    ## Finding Word Freq
    '''
    print('Finding Word Frequency')
    wordsforcloud=[]
    for i in range(0,len(data_lemmatized)):
            lis=data_lemmatized[i]
            wordsforcloud.append(lis)
    flat_list = [item for sublist in wordsforcloud for item in sublist]
    line = [i for i in flat_list if len(i) > 1]
    wordfreq = Counter(line)
    listed=[col,neg_words,pos_words,wordfreq]
    return listed
    
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
 score = analyser.polarity_scores(sentence)
 num1 = score['neg']
 num2 = score['pos']
 num3 = score['neu']
 if (num1 >= num2) and (num1 >= num3):
   largest = 'neg'
 elif (num2 >= num1) and (num2 >= num3):
   largest = 'pos'
 else:
   largest = 'neu'
 return largest

'''
#### XGboost


@csrf_exempt
def xgbfunc(request):
    print("in xgbfunc")
    query = request.args.get('query')
    dbName = request.args.get('dbName')
    password = request.args.get('password')
    userName = request.args.get('userName')
    columnsArray = request.args.get('columnsArray')
    dbHost = request.args.get('dbHost')
    columnList =request.args.get('columnList')
    xValues = request.args.get('xValues')
    xValues = ast.literal_eval(xValues)
    depth = request.args.get('depth')
    print(xValues)
    print(type(xValues))
    yValue =request.args.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    pool = ThreadPool()
    async_result = pool.apply_async(XGBoostThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue,depth))
    return_val = async_result.get()
    jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json")
    return jsonstr

def XGBoostThread(query,dbName,password,userName,columnsArray,dbHost,columnList,xValues,yValue,depth):
    print("in XGBoostThread")
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    #print(yValue)
    data, columnsArray_e = columns_data_type(df, json.loads(columnsArray))
    data = data.replace(to_replace = [".",",","@","#","$","%","^","&","*","(",")","-","+","=", "?"],value = np.nan)
    if data.shape[0]>150 :
        ent_cor,data,rm_cols, miss_cols, obj_t,sub, drp_clmns = correlations(data, columnsArray=columnsArray_e,no_rem_col='none',Val=data.columns)
        if yValue not in rm_cols and yValue not in drp_clmns:
            print('######### Data after correlations #########################')
            print(data.head())
            num_data = data.select_dtypes(include=['number']).copy()
            print('######### Num data #########################')
            print(num_data.head())

            if data.shape[1]>1:
                cols = data.columns
                num_cols = data._get_numeric_data().columns
                cat_cols = list(set(cols) - set(num_cols))
                try:
                    cat_cols.remove(yValue)

                except ValueError:
                    cat_cols = cat_cols

                y_new = data[yValue]
                cols_y = yValue
                le = preprocessing.LabelEncoder()
                le.fit(y_new)
                y_new = le.transform(y_new)
                #y_new = y_new.astype('str')
                y_new= pd.DataFrame(y_new,columns = [cols_y])
                data_n = pd.get_dummies(data, prefix_sep="__",columns=cat_cols)
                'here'
                data_ncols = list(data_n.columns)
                datan_df = pd.DataFrame(columns = ['colname'], index = range(len(data_ncols)))
                datan_df['colname'] = data_ncols
                print(datan_df[100:])
                datan_df = datan_df.replace(r'!@#$%^&*\]\[>\<\(\)', "", regex=True)
                data_n.columns = datan_df['colname'].values
                'here'
                
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                data_n.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data_n.columns.values]
                #    variable_imp = variable_importance_h2o(data, list(num_data.columns), yValue)
                
                x = data_n.drop(yValue,axis=1)
                y = y_new
                print(x.columns)
                depth=int(depth)

                if(data[yValue].dtypes != 'float') and (data[yValue].dtypes != 'int'):
                    data_dmatrix = xgb.DMatrix(data=x,label=y)
                    model = XGBClassifier()
                    if data.shape[0] < 1500:
                        n_esti = [100, 200,500]
                        lr_rate = [0.01, 0.05,0.1]

                    else:
                        n_esti = [100, 200,500]
                        lr_rate = [0.01, 0.05]




        #    gam = [0.0,0.1]
        #    alp=[0, 0.005]


                    param_grid=dict(learning_rate=lr_rate,n_estimators=n_esti)
                    try:
                        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

                        grid_search=GridSearchCV(model,param_grid,scoring="neg_log_loss",n_jobs=-1,cv=kfold,verbose=3)
                        grid_result = grid_search.fit(x, y)
                # summarize results
                        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
             #   best_learning_rate.value, best_n_estimators.value = grid_result.best_params_
                        means = grid_result.cv_results_['mean_test_score']
                        stds = grid_result.cv_results_['std_test_score']
                        params = grid_result.cv_results_['params']
                        for mean, stdev, param in zip(means, stds, params):
                            print("%f (%f) with: %r" % (mean, stdev, param))
                    # plot results
            #    scores = np.array(means).reshape(len(lr_rate), len(n_esti),len(depth))
            #    for i, value in enumerate(lr_rate):
            #        pyplot.plot(n_esti, scores[i], label='lr_rate: ' + str(value))
            #    pyplot.legend()
            #    pyplot.xlabel('n_esti')
            #    pyplot.ylabel('Log Loss')
            #    pyplot.savefig('n_estimators_vs_learning_rate.png')

                        grid_result = pd.DataFrame(grid_result.best_params_ , index=[0])


                        params={
                                "objective":"reg:squarederror",
                                'learning_rate':grid_result['learning_rate'].iloc[0],
                                'n_estimators':grid_result['n_estimators'].iloc[0],
                                'max_depth':depth
                                }

                        cv_results=xgb.cv(dtrain=data_dmatrix,params=params,nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

                        cv_results.head()
                        print((cv_results["test-rmse-mean"]).tail(1))

                        xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
                        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)

                        xgb_model=XGBClassifier(objective="binary:logistic",random_state=42,eval_metric="auc",learning_rate = grid_result['learning_rate'].iloc[0],n_estimators =grid_result['n_estimators'].iloc[0])

                        xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

                        y_pred = xgb_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print("Confusion Matrix:")
                        print(confusion_matrix(y_test, y_pred))
            #    plot_importance(xgb_model)
            #    pyplot.show()
                        y_true = pd.DataFrame(y_test.values.copy(),columns=['Actual'])
                        y_pred = pd.DataFrame(y_pred.copy(),columns=['Predicted'])
                        X_test = pd.DataFrame(X_test)
                        X_test.index = range(len(X_test))
                        cols = [X_test,y_true,y_pred]

                        final = pd.concat(cols,axis=1)

                        df1 = final.to_dict(orient='records')
                        df2 = cv_results.to_dict(orient='records')
                        listed = [accuracy, df1,df2]
                        print(accuracy)

                    except Exception:
                        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y)
                        xgb_model = XGBClassifier()
                        xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])
                        y_pred = xgb_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print("Confusion Matrix:")
                        print(confusion_matrix(y_test, y_pred))
                        y_true = pd.DataFrame(y_test.values.copy(),columns=['Actual'])
                        y_pred = pd.DataFrame(y_pred.copy(),columns=['Predicted'])
                        X_test = pd.DataFrame(X_test)
                        X_test.index = range(len(X_test))
                        cols = [X_test,y_true,y_pred]

                        final = pd.concat(cols,axis=1)
                        df1 = final.to_dict(orient='records')
                        listed = [accuracy, df1]
                        print(accuracy)
                else:
                    listed = ["Incorrect Yvalue datatype is selected. Yvalue must be in string"]
            else:
                listed = ['There are less than two coloumns to build the model few of the variables are removed during preprocessing']
        else:
           listed = ['Target is Numerical. XGB works on Categorical Binary Target variable.']
    else:
        listed = ["Not enough data to perform modelling need atleast 150 observations to perform modelling"]
    return listed
'''

### RESTART H2O 
#(Untick both the comments(both ticks), then run file to stop h2o and comment the statements again (undo the changes of removing ticks) to restart it)
    
'''
try:
    h2o.init()
    j=0
except Exception:
    h2o.cluster().shutdown()

if j == 0:
    h2o.cluster().shutdown()
''' 
### Smart Insights

@app.route("/smart_insights", methods = ['POST'])

def Smart_in():
    print("in Smart_in")
    query = request.form.get('query')
    dbName = request.form.get('dbName')
    password = request.form.get('password')
    userName = request.form.get('userName')
    columnsArray = request.form.getlist('columnsArray')
    columnsArray = columnsArray[0]
    columnsArray = str(columnsArray)
    dbHost = request.form.get('dbHost')
    columnList =request.form.getlist('columnList')
    columnList = columnList[0]
    yValue =request.form.get('yValue')
    yValue = ast.literal_eval(yValue)
    yValue = yValue[0]
    pool = ThreadPool()
    async_result = pool.apply_async(Smart_inThread, (query,dbName,password,userName,columnsArray,dbHost,columnList,yValue))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def Smart_inThread(query,dbName,password,userName,columnsArray,dbHost,columnList,yValue):
    client = Client(dbHost,user=userName,password=password,database=dbName)
    print("in  Smart_inThread")
    print(query)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    print(df)
    data, columnsArray_e = smi.columns_data_type(df, json.loads(columnsArray))
    for i in range(len(columnsArray_e)):
        if yValue == columnsArray_e['changedColumnName'].iloc[i]:
            yValue = columnsArray_e['columnName'].iloc[i]
        elif yValue == '':
            if columnsArray_e['tableDisplayType'].iloc[i] == 'number' and 'ID' not in columnsArray_e['columnName'].iloc[i] and 'id' not in columnsArray_e['columnName'].iloc[i]:
                yValue = columnsArray_e['columnName'].iloc[i]
                if len(list(data[yValue].unique()))<2:
                    continue
                else:
                    break
        else:
            continue
    if yValue == '':
        listed = [{"message": "There is no numerical target variable in the data.", "error": "Error"}]
        return listed
    dat_lis = []
    for i in data.columns:
        for j in range(len(columnsArray_e['changedColumnName'])):
            if i == columnsArray_e['changedColumnName'].iloc[j]:
                dat_lis.append(columnsArray_e['columnName'].iloc[j])
    try:
        data.columns = dat_lis
    except Exception:
        data.columns = list(data.columns)
    if data.shape[0]>150:
            print(data[yValue])
            if len(data[yValue].unique())>1:
                ent_cor,data,rm_cols, miss_cols, obj_t, sub, drop_columns,trans_col = correlations(data, columnsArray=columnsArray_e, no_rem_col='none', Val = data.columns)
                data = data.loc[:,~data.columns.duplicated()]
                print(data)
                if data.shape[1]>1:
                    xValues = list(filter(lambda x: x not in rm_cols and x in data.columns, data.columns))
                    data_x_unique = data[xValues]
                    for i in data_x_unique.columns:
                        if len(data_x_unique[i].unique())>1:
                            continue
                        else:
                            if data[i].dtypes != 'float' and  data[i].dtypes != 'int':
                                print('dropping ' + str(i))
                                data = data.drop(columns = [i])
                                drop_columns.append(i)
                            else:
                                if data_x_unique.shape[1]>1:
                                    continue
                                else:
                                   data = data.drop(columns = [i])
                                   drop_columns.append(i)
                else:
                    listed = [{"message": "There is only one column after preprocessing. Please try giving different inputs or different dataset as the dataset might not be applicable for this model.", "error": "Error"}]
                    return listed
                if data.shape[1]>1:
                    print(data.shape[1])
                    if yValue not in list(rm_cols) and yValue not in drop_columns:
                        xValues = list(filter(lambda x: x not in rm_cols and x not in miss_cols and x not in drop_columns, data.columns))
                        try:
                            variable_imp = smi.variable_importance_h2o(data, xValues, yValue)
                        except Exception as e:
                            msg = str(e)
                            if 'Response cannot be constant' in msg:
                                listed = [{"message": "Target Variable cannot be a constant value", "error": "Error"}]
                            else:
                                listed = [{"message": str(msg), "error": "Error"}]
                            return listed
                        var_imp = []; var_name=[]
                        for i in range(len(variable_imp)):
                            for j in range(len(variable_imp[i])):
                                var_imp.append(variable_imp[i][j][2])
                                var_name.append(variable_imp[i][j][0])
                        Var_imp  = pd.DataFrame(columns = ['y'], index = range(len(var_imp)))
                        Var_imp['y'] = var_imp
                        Var_name = pd.DataFrame(columns = ['variable'], index = range(len(var_name)))
                        Var_name['variable'] = var_name
                        frame_var = [Var_name, Var_imp]
                        Var_imp = pd.concat(frame_var, axis = 1)
                        #variable_imp = variable_importance_h2o(data, xValues, yValue)
                        vif_frame = [data[xValues], data[yValue]]
                        data_vif = pd.concat(vif_frame, axis = 1)
                        data_vif = data_vif.loc[:,~data_vif.columns.duplicated()]
                        num_data = data_vif.select_dtypes(include=['int', 'float', 'number']).copy()

                        #vif_var = vif(num_data, yValue, 10)
                        Nfolds = 6
                        ###NLG
                        temp1 = list(filter(lambda x: x in obj_t, obj_t))
                        print(num_data)
                        if (num_data.shape[1]>0):
                            num_data_inv = smi.transformation_inv(num_data,temp1)
                            labels = list(num_data_inv.columns)
                            print(num_data_inv)
                        else:
                            num_data_inv = []
                        num_x = list(filter(lambda x: x in num_data, xValues))
                        len_num = len(num_x)
                        if (len_num>1):
                            x_mean = num_data_inv[num_x].mean()
                            x_min = num_data_inv[num_x].min()
                            x_max = num_data_inv[num_x].max()
                            x_quant25 = num_data_inv[num_x].quantile(0.25)
                            x_quant50 = num_data_inv[num_x].quantile(0.5)
                            x_quant75 = num_data_inv[num_x].quantile(0.75)
                            x_skew = scipy.stats.skew(num_data_inv[num_x])
                        elif (len_num>0):
                            x_mean = num_data_inv[num_x].mean()
                            print(x_mean)
                            x_min = num_data_inv[num_x].min()
                            x_max = num_data_inv[num_x].max()
                            x_quant25 = []
                            x_quant50 = []
                            x_quant75 = []
                            x_skew = []
                        else:
                            x_mean = []
                            x_min = []
                            x_max = []
                            x_quant25 = []
                            x_quant50 = []
                            x_quant75 = []
                            x_skew = []
                        
                        ### Data preperation

                        #ind = range(data.shape[0])
                        data_x = data[xValues]
                        cat_data = data_x.select_dtypes(include=['object', 'category']).copy()
                        '''
                        if (cat_data.shape[1]>0):
                            print("Categorical Encoding started")
                            n_data = data_x.select_dtypes(include=['int', 'float', 'number']).copy()
                            # label_encoder object knows how to understand word labels.
                            lab = []
                            for i in range(cat_data.shape[1]):
                                lab.append(preprocessing.LabelEncoder())
                            for j in range(len(lab)):
                                for i in cat_data.columns:
                                    try:
                                        data_x[i] = data_x[i].str.encode('utf-8')
                                        data_x[i] = lab[j].fit_transform(data_x[i])
                                        break
                                    except Exception:
                                        continue
                            print(data_x)
                            data_x_category = []
                            for i in range(cat_data.shape[1]):
                                data_x_cat = pd.get_dummies(data_x[cat_data.columns[i]])
                                print('here')
                                data_x_cat= data_x_cat.astype('Int64')
                                col_val= []
                                for j in range(data_x_cat.shape[1]):
                                    col_val.append(cat_data.columns[i]+str(j))
                                dataframe = pd.DataFrame(columns = col_val, index = range(len(data_x_cat)))
                                for j in range(data_x_cat.shape[1]):
                                    dataframe[col_val[j]] = data_x_cat[j]
                                data_x_category.append(dataframe)

                            data_x_category.append(data_x[n_data.columns])
                            final_datax = pd.concat(data_x_category, axis = 1)
                            print("Categorical Encoding completed")
                        else:
                            final_datax = data_x

                        data_y = pd.DataFrame(data = data[yValue], columns = [yValue], index = ind)
                        frames= [final_datax, data_y]
                        data2 = pd.concat(frames, axis=1)
                        '''
                        data2 = data_vif
                        family = 'gaussian'
                        data2 = data2.loc[:,~data2.columns.duplicated()]
                        #print(data2[yValue].unique())
                        if data2.shape[1]>1 and len(data2[yValue].unique())>1:
                            if(data[yValue].dtypes == 'float') or (data[yValue].dtypes == 'int64'):
                                try:
                                    hf = h2o.H2OFrame(data2)
                                    if  hf.shape[1]<1:
                                        try:
                                                for i in data2.columns:
                                                    if data2[i].dtypes == 'float':
                                                        continue
                                                    else:
                                                         data2[i] = data2[i].str.encode('utf-8')
                                        except Exception:
                                                data2 = data2
                                        hf = h2o.H2OFrame(data2)
                                    else:
                                        hf = hf
                                except Exception as e:
                                    msg = str(e)
                                    listed = [msg]
                                    return listed
                                print("Finding variable importance by taking given numeric variable as a dependent variable")
                                train, valid, test = hf.split_frame(ratios=[.7, .1])
                                x_Values= list(data2.columns)
                                x_Values.remove(yValue)
                                glm_model = H2OGeneralizedLinearEstimator(family = family, nfolds= Nfolds)
                                glm_model.train(x_Values, yValue, training_frame= train, validation_frame=valid)
                                print('Model Training Completed')
                                ## Selecting best model from the cross validation score
                                mod = glm_model.cross_validation_models()
                                rmse_list=[]
                                for i in range(len(mod)):
                                    rmse_list.append(mod[i].rmse())
                                rmse_list.sort()
                                for i in range(len(mod)):
                                    if (mod[i].rmse() == rmse_list[0]):
                                        best_mod = mod[i]
                                        break
                                    else:
                                        continue
                                predicted = best_mod.predict(test_data=test)
                                print('Predicted data')
                                #best_mod.coef()
                                ### Changing the transformed numericals back to original values
                                print('Performing Inverse transformations')
                                test_trueY = h2o.as_list(test)
                                x = list(filter(lambda x: x in num_data.columns and x in xValues and x not in drop_columns, test_trueY.columns))
                                if yValue not in x:
                                    x.append(yValue)
                                try:
                                    temp = list(filter(lambda x: x in obj_t, obj_t))
                                    temp1 = list(filter(lambda x: x in obj_t, obj_t))
                                    test_true_inv = transformation_inv(pd.DataFrame(test_trueY[x]),obj_t)
                                    test_true_inv.index = range(len(test_true_inv))
                                    true_y = pd.DataFrame(columns = ['Actual'], index = range(test_true_inv.shape[0]))
                                except Exception:
                                    return ['Gone']
                                true_y['Actual']= test_true_inv[yValue]
                                true_y.index = range(len(true_y))
                                ### For plotting transformed true Y data
                                #true_y_trnf = test_trueY[yValue]
                                
                                #Inverse Transforming yValue
                                test[yValue] = predicted
                                test_predY = h2o.as_list(test)
                                print('2nd inverse')
                                test_pred_inv = smi.transformation_inv(pd.DataFrame(test_predY[x]),temp)
                                pred_y = pd.DataFrame(columns = ['Predicted_values'], index = range(test_pred_inv.shape[0]))
                                pred_y['Predicted_values']=test_pred_inv[yValue]
                                pred_y.index = range(len(pred_y))
                                print('Inverse transformation completed')
                                ### For plotting transformed predicted Y data
                                #pred_y_trnf = test_predY[yValue]

                                ### Converting encoded categoricals to original values
                                ## Extracting individual categorical column
                                '''
                                if (cat_data.shape[1]>0):
                                    print("Decoding Categorical features")

                                    for i in n_data.columns:
                                        try:
                                            x_Values.remove(i)
                                        except Exception:
                                            continue
                                    test_cat = test_trueY[x_Values]
                                    test_cat_frame=[]
                                    data_x_category.pop(len(data_x_category)-1)
                                    for i in range(cat_data.shape[1]):
                                        l_col = []; l_colname = []
                                        for j in range(data_x_category[i].shape[1]):
                                            l_col.append(test_cat[test_cat.columns[j]])
                                            l_colname.append(test_cat.columns[j])
                                        test_cat = test_cat.drop(columns = l_colname)
                                        categ_col = pd.concat(l_col, axis = 1)
                                        test_cat_frame.append(categ_col)

                                    ## decoding from keras encoder to label encoded data
                                    label_col = []
                                    for i in range(len(lab)):
                                        label= []
                                        for j in range(test_cat_frame[i].shape[0]):
                                            label.append(argmax(np.array(test_cat_frame[i].iloc[j])))
                                        temp_df = pd.DataFrame(label, columns = [cat_data.columns[i]],index = range(len(label)))
                                        label_col.append(temp_df)

                                    ## decoding label encoded data to original message
                                    for i in range(len(lab)):
                                        label_col[i] = lab[i].inverse_transform(label_col[i])
                                        label_col[i] = pd.Series(label_col[i])
                                        label_col[i] = label_col[i].str.decode(encoding ='utf-8', errors= 'ignore')
                                        label_col[i]=pd.DataFrame(label_col[i], index = range(len(label_col[0])))

                                    test_cat_final = pd.concat(label_col, axis = 1)
                                    test_cat_final.columns = cat_data.columns
                                    print(test_cat_final)
                                    print("Categorical features Decoded")
                                else:
                                    test_cat_final = []
                                '''
                                ## Final Dat Frame
                                print('Preparing final Outputs')
                                act_pred=true_y['Actual'] - pred_y['Predicted_values']
                                Diff = pd.DataFrame(columns = ['difference_Actual_predicted'],index=range(len(act_pred)))
                                Diff['difference_Actual_predicted'] = act_pred
                                x.remove(yValue)
                                print(cat_data.shape[1])
                                print(num_data.shape[1])
                                cat_cols = list(cat_data.columns)
                                test_cat_final = test_predY[cat_cols]
                                test_cat_final.index = range(len(test_cat_final))
                                if (cat_data.shape[1]>0) and (num_data.shape[1]==1):
                                    frame = [true_y, pred_y, Diff, test_cat_final]
                                elif (cat_data.shape[1]==0) and (num_data.shape[1]>0):
                                    frame = [true_y, pred_y, Diff, test_true_inv[x]]
                                else:
                                    frame = [true_y, pred_y, Diff, test_cat_final, test_true_inv[x]]
                                Final_pred_data = pd.concat(frame, axis= 1)
                                Actual_predicted = pd.concat([true_y, pred_y], axis = 1)
                                act_pred_cols =  ['dataframe.Actual', 'dataframe.Predicted_values']
                                Actual_predicted.columns = act_pred_cols
                                print(Final_pred_data)
                                '''
                                Actual_predicted = pd.DataFrame(columns = ['Actual', 'Predicted_values'], index = range(len(true_y)))
                                Actual_predicted['Actual'] = true_y
                                Actual_predicted['Predicted_values'] = pred_y
                                #linear_regr = [glm_model.r2(),dataframeNewSet,Actual_predicted, ent_cor,glm_model.coef(),variable_imp,vif_var]
                                '''
                                linear_regr = [ent_cor, [x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]]
                                list1 = [round(glm_model.r2()*10)]
                                '''
                                final_df_columns= []
                                for i in range(len(Final_pred_data.columns)):
                                    for j in range(len(columnsArray_e)):
                                        if Final_pred_data.columns[i] == columnsArray_e['columnDisplayName'][j]:
                                            final_df_columns.append(columnsArray_e['changedColumnName'][j])

                                final_list = ['Actual', 'Predicted_values', 'difference_Actual_predicted']
                                for i in final_df_columns:
                                    final_list.append(i)
                                Final_pred_data.columns = final_list
                                '''
                                print('Limiting the decimal Value')
                                for i in Final_pred_data.columns:
                                    if Final_pred_data[i].dtypes=='float' or Final_pred_data[i].dtypes=='int':
                                        for j in range(len(Final_pred_data[i])):
                                            Final_pred_data[i].iloc[j] = round(Final_pred_data[i].iloc[j], 2)
                                print('Completed')
                                list2 = Final_pred_data.to_dict(orient='records')
                                #print("Actual_predicted")
                                #print(Actual_predicted)
                                for i in Actual_predicted.columns:
                                    if Actual_predicted[i].dtypes=='float' or Actual_predicted[i].dtypes=='int':
                                        for j in range(len(Actual_predicted[i])):
                                            Actual_predicted[i].iloc[j] = round(Actual_predicted[i].iloc[j], 2)
                                            
                                list3 = Actual_predicted.to_dict(orient='records')
                                #list3_1 = list(Actual_predicted[Actual_predicted.columns[0]])
                                #list3_2 = list(Actual_predicted[Actual_predicted.columns[1]])
                                list4 = [best_mod.rmse()]
                                temp_dict = best_mod.coef()
                                temp_dataf = pd.DataFrame.from_dict(temp_dict, orient = 'index')
                                for i in list(temp_dataf.index):
                                    if temp_dataf[0][i]> 0:
                                        continue
                                    elif temp_dataf[0][i]< 0:
                                        continue
                                    else:
                                        temp_dataf.drop(list(temp_dataf.index[temp_dataf.index==i]), inplace = True)
                                temp_dataf['left'] = 0
                                temp_dataf['right'] = 0
                                temp_dataf['right h2o.coef.model.gaussian.'] = temp_dataf[0]
                                temp_dataf = temp_dataf.drop(columns = [0])
                                for i in range(len(temp_dataf)):
                                    for j in range(len(columnsArray_e)):
                                        ab_coeffname = temp_dataf.index[i].split('.')
                                        if ab_coeffname[0] == columnsArray_e['columnDisplayName'][j]:
                                            temp_dataf['left'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                            try:
                                                temp_dataf['right'].iloc[i] = ab_coeffname[1]
                                            except Exception:
                                                temp_dataf['right'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                            break
                                temp_dataf['left'].iloc[0] = temp_dataf.index[0]
                                temp_dataf['right'].iloc[0] = temp_dataf.index[0]
                                temp_dataf.index = range(len(temp_dataf))

                                list5 = temp_dataf.to_dict(orient='records')
                                #list6 = [variable_imp]
                                print(linear_regr)
                                listed7 = NLG(linear_regr, yValue, labels, len_num)
                                for i in range(len(Var_imp)):
                                    for j in range(len(columnsArray_e)):
                                        if Var_imp['variable'].iloc[i] == columnsArray_e['columnName'].iloc[j]:
                                             Var_imp['variable'].iloc[i] = columnsArray_e['changedColumnName'].iloc[j]
                                             break
                                
                                # Top 6 variable importance
                                try:
                                    Var_imp = Var_imp[0:6]
                                except Exception:
                                    Var_imp = Var_imp
                                
                                listed8 = Var_imp.to_dict(orient='records')
                                col_namechange = pd.DataFrame(json.loads(columnsArray))
                                ##Correlation
                                if len(ent_cor)>0:
                                    for i in ent_cor.columns:
                                        for j in range(len(ent_cor[i])):
                                            ent_cor[i].iloc[j] = round(ent_cor[i].iloc[j],3)
                                            
                                    l1 = ent_cor.to_dict(orient='records')
                                    names=ent_cor.columns
                                    print("Done till checkpoint 1")
                                    for i in range(0,len(ent_cor.columns)):
                                        l1[i].update([('_row',names[i])])
                                    for k in [1,2]:
                                        for i in range(len(l1)):
                                            for key, value in l1[i].items():
                                                for j in range(len(col_namechange)):
                                                    if key == col_namechange['columnDisplayName'][j]:
                                                        temp_val = l1[i][key]
                                                        del l1[i][key]
                                                        l1[i][col_namechange['changedColumnName'][j]] = temp_val
                            
                                    for i in range(len(l1)):
                                        tem_keys = list(l1[i].keys())
                                        tem_keys.remove(tem_keys[0])
                                        for key, value in l1[i].items():
                                            if key == '_row':
                                                #tem_row = l1[i]['_row']
                                                del l1[i]['_row']
                                                l1[i].update([('_row',tem_keys[i])])
                                                break
                                    listed9 = l1
                                elif (len(ent_cor)==0 and num_data.shape[1] <1):
                                    l1 = 'No Numerical data'
                                    listed9 = [l1]
                                elif num_data.shape[1] == 1:
                                    l1 = 'Only one Numeric column'
                                    listed9 = [l1]
                                    
                                ### Simulation list of ctegories
                                    
                                len_cat = len(columnsArray_e)
                                main_dict = []
                                print('Preparing simulation')
                                for i in range(len_cat):
                                    try:
                                        if columnsArray_e['tableDisplayType'].iloc[i] == 'string':
                                            d = {'data': list(data2[str(columnsArray_e['columnName'].iloc[i])].unique())}
                                            d1 = {'dataType':'string'}
                                            d2 = {'name': columnsArray_e['columnName'].iloc[i]}
                                            d3 = {"columnName": columnsArray_e['columnName'].iloc[i]}
                                            d.update(d1);d.update(d2);d.update(d3)
                                            print('updating')
                                            main_dict.append(d)
                                        else:
                                            d = {'data': [data2[str(columnsArray_e['columnName'].iloc[i])].iloc[-1]]}
                                            d1 = {'dataType':'number'}
                                            d2 = {'name': columnsArray_e['columnName'].iloc[i]}
                                            d3 = {"columnName": columnsArray_e['columnName'].iloc[i]}
                                            d.update(d1);d.update(d2);d.update(d3)
                                            print('updating')
                                            main_dict.append(d)
                                    except Exception:
                                        continue
                                
                                print('Simulation completed') 
                                #simu_target = dt_transformation_inv(data2[yValue].iloc[-1], temp1[0], [yValue], yValue)
                                simu_target = data2[yValue].iloc[-1]
                                Diff = pd.DataFrame(columns = ['difference_Actual_predicted'],index=range(len(act_pred)))
                                Diff['difference_Actual_predicted'] = act_pred

                                Diff["Deviation"]=Diff['difference_Actual_predicted']/true_y['Actual']

                                outliers=Diff[Diff["Deviation"]>=0.7]
                                out1 = Diff[Diff["Deviation"]<=-0.7]
                                outliers = pd.concat([outliers, out1], axis = 0)
                                outliers = outliers.to_dict(orient='records')
                                listed = [list1, list2, list3, list4, [glm_model.r2()], listed7, listed8, list5, listed9, main_dict, simu_target, yValue, outliers]
                            else:
                                listed = [{"message": "Target Variable (Y-axis) is Categorical data and only works if it is Numerical", "error": "Error"}]

                        else:
                            listed = [{"message": "There are less than two columns for building model or the Response variable is constant. Please make sure that there are atleast 1 Input dependant variables and it is not a constant Value", "error": "Error"}]

                    else:
                        listed = [{"message": "Selected Target Variable(yValue) is removed during preprocessing", "error": "Error"}]

                else:
                    listed = [{"message": "There are less than two columns for building model. Few of the selected Variables have been removed during preprocessing", "error": "Error"}]

            else:
                listed = [{"message": "Target Variable (yValue) cannot be constant", "error": "Error"}]

    else:
            listed = [{"message": "Not enough data to perform modelling, need atleast 150 data points", "error": "Error"}]

    return listed


def NLG(linear_regr, yValue,labels, length_num):
    # correlated variables
    corr = linear_regr[0]
    if len(corr)>0:
        corr = corr.drop(yValue)
        mask1 = corr[yValue] >= 0.3
        mask2 = corr[yValue] <= -0.2
        mask3 = (corr[yValue] >= 0.2) & (corr[yValue] <= 0.3)
        corr1 = []
        corr_val1 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between <strong>' + (corr[yValue][mask1].index) + '</strong> and <strong>' + yValue + '</strong>, With every unit of increase in '+ (corr[yValue][mask1].index) + ' there is an increase in ' + yValue + '</span></li>'
        for i in range(len(corr_val1)):
            corr1.append(corr_val1[i])
        corr2 = []
        corr_val2 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between<strong> ' + (corr[yValue][mask2].index) + '</strong> and <strong>' + yValue + '</strong>, With every unit of decrease in '+ (corr[yValue][mask2].index) + ' there is an increase in ' + yValue + '</span></li>'
        for i in range(len(corr_val2)):
            corr2.append(corr_val2[i])
        corr3 = []
        corr_val3 = '<li class=\"nlg-item\"><span class=\"nlgtext\">There is a little or no relationship between <strong> ' + (corr[yValue][mask3].index) + '</strong> and <strong>' + yValue + '</strong></span></li>'
        for i in range(len(corr_val3)):
            corr3.append(corr_val3[i])
    else:
        corr1 = []
        corr2 = []
        corr3 = []
    # quantile and min, max values
    #linear_regr[8] : [x_mean,x_min,x_max,x_quant25,x_quant50,x_quant75,x_skew]
    for i in range(len(linear_regr[1])):
        for j in range(len(linear_regr[1][i])):
            a = round(linear_regr[1][i][j],2)
            linear_regr[1][i][j] = a

    if length_num>0:
        var_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The average value of <strong>'+', '.join(labels)+'</strong> is <strong>'+', '.join(map(str, linear_regr[1][0]))+'</strong></span>''</li>'
        if length_num>1:
            quantile_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>'+', '.join(labels)+'</strong> is <strong>' +', '.join(map(str, linear_regr[1][1]))+'</strong> whereas <strong>25%</strong> of data lies below the value <strong>'+ str(', '.join(map(str,linear_regr[1][3])))+ '</strong> the median of <strong>'+ str(', '.join(labels))+ '</strong> is <strong>'+', '.join(map(str, linear_regr[1][4]))+'</strong> and the <strong>75%</strong> of data lies below the value <strong>'+', '.join(map(str, linear_regr[1][5]))+'</strong> and the max value is <strong>'+', '.join(map(str, linear_regr[1][2]))+'</strong></span></li>'
        else:
            quantile_val='<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>'+str(', '.join(labels))+'</strong> is <strong>' + str(', '.join(map(str, linear_regr[1][1])))+'</strong> and the max value is <strong>'+ str(', '.join(map(str, linear_regr[1][2])))+'</strong></span></li>'
    else:
        quantile_val=[]
        var_val = []
    listed = [corr1,corr2,corr3,var_val,quantile_val]
    return listed

'''
test = [['Second Class', 'Consumer',  'Kentucky','South','Furniture','Chairs',-1.26989, -1.03631,-0.959713]]

test = pd.DataFrame(test, columns = ['Ship Mode ', 'Segment','State','Region','Category','Sub-Category','Sales','Quantity','Discount']) 
'''

def update_type(t1, t2, dropna=False):
    return t1.map(t2).dropna() if dropna else t1.map(t2).fillna(t1)

### Simulation
    
@app.route("/simulation", methods = ['POST'])

def simulation():
    '''
    print("in linearRegression")
    query = request.args.get('query')
    dbName = request.args.get('dbName')
    password = request.args.get('password')
    userName = request.args.get('userName')
    
    dbHost = request.args.get('dbHost')
    columnList =request.args.get('columnList')
    yValue =request.args.get('yValue')
    yValue = ast.literal_eval(yValue)
    print(type(yValue))
    '''
    columnsArray_e = request.form.get('columnsArray')
    columnsArray_e = ast.literal_eval(columnsArray_e)
    test = request.form.get('test')
    test = ast.literal_eval(test)
    simulationdata =request.form.get('simulationdata')
    simulationdata = ast.literal_eval(simulationdata)
    print(simulationdata)
    pool = ThreadPool()
    async_result = pool.apply_async(simulationThread, (test,simulationdata, columnsArray_e))
    return_val = async_result.get()
    print(return_val)
    #jsonstr = HttpResponse(json.dumps(return_val), content_type="application/json/html")
    return jsonify(return_val)

def simulationThread(test,simulationdata, columnsArray_e):
    
    
    print('Running simulation on data')
    columnsArray_e = pd.DataFrame.from_records(columnsArray_e)
    test=pd.DataFrame.from_records(test)
    coefficients=[]
    coefficients_col = []
    coeff=pd.DataFrame.from_records(simulationdata)
    updated = update_type(coeff.right, columnsArray_e.set_index('changedColumnName').columnDisplayName)
    coeff['right']=updated
    ## Changing the names of left in coeff
    for i in range(len(coeff)):
        for j in range(len(columnsArray_e)):
            if coeff['left'].iloc[i] == columnsArray_e['changedColumnName'].iloc[j]:
                coeff['left'].iloc[i] = columnsArray_e['columnName'].iloc[j]
    ## Multiplying test values with coefficients
    for i in range(len(test)):
        for j in range(len(coeff)):
            if test['name'].iloc[i] == coeff['left'].iloc[j]:
                try:
                    value1 = float(test['value'].iloc[i])
                    coefficients.append(value1*coeff['right h2o.coef.model.gaussian.'].iloc[j])
                    coefficients_col.append(coeff['left'].iloc[j])
                except Exception:
                    if test['value'].iloc[i] == coeff['right'].iloc[j]:
                        coefficients.append(coeff['right h2o.coef.model.gaussian.'].iloc[j])
                        coefficients_col.append(coeff['left'].iloc[j])
    chart = pd.DataFrame(columns = ['Name', 'Coeff'], index = range(len(coefficients)))
    chart['Name'] = coefficients_col
    chart['Coeff'] = coefficients
    test['Coeff']= 0
    for i in range(len(test)):
        for j in range(len(chart)):
            if test['name'].iloc[i] == chart['Name'].iloc[j]:
                test['Coeff'].iloc[i] = chart['Coeff'].iloc[j]
    test = test[['name', 'Coeff']]
    print('Simulation Completed')
    listed = test.to_dict(orient = 'records')
    coefficients.append(coeff.iloc[0]['right h2o.coef.model.gaussian.'])
    tot_val=sum(coefficients)
    return [tot_val,listed]



if __name__ == "__main__":
    app.run(host='0.0.0.0')

