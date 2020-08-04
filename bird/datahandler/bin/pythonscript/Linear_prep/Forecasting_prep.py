import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import numpy as np
import pandas as pd # must be 0.24.0
import sklearn
from statistics import mean
from pandas.io.json import json_normalize
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import itertools
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
import statistics
from dateutil.parser import parse
import datetime
import calendar
from statsmodels.stats.outliers_influence import variance_inflation_factor

####################################################################
#import json
#try:
#    columnsArray = json.loads(columnsArray)
#except:
#    print('json format not valid')
#columnsArray = pd.DataFrame(columnsArray)
#columnsArray = columnsArray.replace(r'[^a-zA-Z0-9 -]', "", regex=True)
#
#data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv",dtype=str)
##data = pd.DataFrame(np.genfromtxt('E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv', dtype=str))
#data.info()
#
#
#data, columnsArray_e = columns_data_type(data[0:100], columnsArray)
#
#ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t = correlations(data, columnsArray=columnsArray_e, method='predict')



##GBM_impute(data, columnsArray, rm_cols)
#
#columnsArray_ind = []
#for i in columnsArray['columnName']:
#    if i in rm_cols:
#        columnsArray_ind.append(list(columnsArray[columnsArray['columnName']==i].index)[0])
#columnsArray_ind1 = set(columnsArray.index)-set(columnsArray_ind)
#print(columnsArray_ind1)
#columnsArray_edit = columnsArray.iloc[list(columnsArray_ind1)]        
#
## select observations without NA's
#data_clean = data.dropna()
## creating H2O Frame and splitting for model train
#data_clean.info()
#hf = h2o.H2OFrame(data_clean)
#train, valid, test = hf.split_frame(ratios=[.8, .1])
#
## select observations with NA's
#data_na_index = [i for i in (set(list(data.index)) - set(list(data_clean.index))) ]
#data_na = data.iloc[data_na_index]
#
#model_accuracy = []
#for i in range(len(data_na)):
#    # select features with NA's in current row
#    print('Index in data_na_index')
#    print(i)
#    y_set = set(data_na.iloc[i].index) - set(data_na.iloc[i].dropna().index)
#    gbm = H2OGradientBoostingEstimator()
#    xValues = set(data_na.columns)-y_set
#    print(xValues)
#    
#    for yValue in y_set:
#        print('yValue from y_set for current index')
#        print(yValue)
#        print('GBM model training')
#        gbm.train(xValues, yValue, training_frame=train, validation_frame=valid)
#        model_accuracy.append(gbm.r2())
##        test_na = data_na.iloc[i].drop(y_set)
#        test_na = data_na.iloc[i]
#        test_na = pd.DataFrame(test_na).transpose()
#        
#        print('Missing value prediction with GBM model')
#        
#        test_na, columnsArray_edit = columns_data_type(test_na, columnsArray_edit)
#        print(test_na)
#        
##        test_na = test_na.drop(xValues,axis=1)
#        test_na = test_na.drop(yValue,axis=1)
#        print(test_na.info())
#        test_na = h2o.H2OFrame(test_na)
#        predicted = gbm.predict(test_na)
#        predicted = predicted.as_data_frame()
#        predicted_val = list(predicted['predict'])[0]
#        print("Predicted value")
#        print(predicted_val)
#        data_na[yValue].iloc[i] = predicted_val
#
#acc = np.mean(model_accuracy)
#frames = [data_clean, data_na]
#df = pd.concat(frames, axis=0)
#df.info()
##return df, acc 

class my_dictionary(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 
  


####################################################################

"""
Return whether the string can be interpreted as a date.

 input:
     string - value to check on date format
 output:
     boolean - true - date, false - not date
"""

def is_date(string, fuzzy=False):
	if string is not None:
		try:
			float(string)
			return False
		except ValueError:
	#        return False
			if (str(string)=='nan'):
				return False
			if str(string).isnumeric():
				return False
			try: 
				parse(str(string), fuzzy=fuzzy)
				return True
		
			except:
				return False

"""
Converting date feature to timestamp format

 input:
     data - raw data frame before columns_data_type()
 output:
     data - changed data frame with timestamp instead date string
"""

def date2stamp(data):
    for i in list(data.columns):
        if data[i].dtypes !='datetime64[ns]':                       ### Changes Made
            for j in range(len(data)):
                if is_date(data[i].iloc[j]):
                    
    #                date = data[i].iloc[j]
    #                date = datetime.datetime.strptime(date, "%d/%m/%y")
    #                data[i].iloc[j] = str(date)
                    try:
                        data[i] = data[i].astype('datetime64[ns]')
                        break
                    except:
                        continue
    #                data[i].iloc[j] = calendar.timegm(date.utctimetuple())
        else:
            continue
    return data


"""
 Changing names of the features.
 Set types of features: numeric or categorical.
 Replacing special characters in column names.
 input:
     df - data frame
     columnsArray - data frame with names and types of columns
 output:
     df - changed data frame
"""
def columns_data_type(df, columnsArray = ""):
    
    df = date2stamp(df)
    columnsArray = pd.DataFrame(columnsArray)
    columnsArray = columnsArray.replace(r'!@#$%^&*.', " ", regex=True)
    df.columns = columnsArray['changedColumnName'] 
    df = df.replace(r'^\s*$', np.nan, regex=True)

    for i in columnsArray['changedColumnName']:
        sub = columnsArray[columnsArray['changedColumnName'] == i]
        if (sub['tableDisplayType'].values[0] == 'number'):
            df[i] = pd.to_numeric(df[i].values, errors ='coerce')
        if (sub['tableDisplayType'].values[0] == 'string'):
#            df[i] = df[i].astype(str)
            #df[i] =  pd.Categorical(df[i]).codes
            
            #for j in range(df[i].shape[0]):
             #   df[i].iloc[j] = str(df[i].iloc[j])
                
            df[i] = df[i].astype('category')
            
    return df,columnsArray

"""
 Imputation or removing of missing values using prediction model.
 Replacing blanks with NA's.
 input:
     df - data frame after preprocessing
     columnsArray - 
     rm_cols - removed columns by remove_col() 
 output:
     df - changed data frame
"""
def GBM_impute(data, columnsArray, rm_cols):

    columnsArray_ind = []
    c = list(filter(lambda x: x not in list(data.columns), list(columnsArray['columnDisplayName'].replace(r'[^a-zA-Z0-9 -]', "", regex=True))))
    for j in c:
        a=columnsArray.index[columnsArray['columnName'].replace(r'[^a-zA-Z0-9 -]', "", regex=True)==j].values
        columnsArray.drop(columnsArray.index[a], inplace = True)
        columnsArray.index = range(columnsArray.shape[0])
        
    for i in columnsArray['columnName']:
        if i in rm_cols:
            columnsArray_ind.append(list(columnsArray[columnsArray['columnName']==i].index)[0])
    columnsArray_ind1 = set(columnsArray.index)-set(columnsArray_ind)
#    print(columnsArray_ind1)
    columnsArray_edit = columnsArray.iloc[list(columnsArray_ind1)]        
    
    # select observations without NA's
    data_clean = data.dropna()
    # creating H2O Frame and splitting for model train
    #data_clean.info()
    hf = h2o.H2OFrame(data_clean)
    train, valid, test = hf.split_frame(ratios=[.8, .1])
    # select observations with NA's
    data_na_index = [i for i in (set(list(data.index)) - set(list(data_clean.index))) ]
    data_na = data.iloc[data_na_index]

    model_accuracy = []
    print("Number of missing values : " + str(len(data_na)))
    gbm = H2OGradientBoostingEstimator()
    for i in range(len(data_na)):
        y_set = set(data_na.iloc[i].index) - set(data_na.iloc[i].dropna().index)
        
        for yValue in y_set:
            xValues = set(data_na.columns)-y_set
            gbm.train(xValues, yValue, training_frame=train, validation_frame=valid)
            model_accuracy.append(gbm.r2())
            print(yValue)
            test_na = data_na
    #            print('Missing value prediction with GBM model')
            test_na, columnsArray_edit = columns_data_type(test_na, columnsArray_edit)
    #            print(test_na)
                
        #        test_na = test_na.drop(xValues,axis=1)
            test_na = test_na.drop(yValue,axis=1)
                #print(i)
            test_na = h2o.H2OFrame(test_na)
            predicted = gbm.predict(test_na)
            predicted = predicted.as_data_frame()
            for j in range(data_na.shape[0]):
                if np.isnan(data_na[yValue].iloc[j]):
                    data_na[yValue].iloc[j] = predicted['predict'][j]
                else:
                    continue
        
    acc = np.mean(model_accuracy)
    frames = [data_clean, data_na]
    df = pd.concat(frames, axis=0)
    df.sort_index(axis = 0, inplace = True)
    return df, acc 


"""
 Imputation or removing of missing values using mean an mode.
 Replacing blanks with NA's.
 input:
     df - data frame
     method
         drop - drop all NA's
         impute - mean for numeric and mode - for categorical
         predict - impute missing val. with prediction model
     columnsArray - 
     rm_cols - removed columns by remove_col()
 output:
     df - changed data frame
"""
def missing_val_impute(df, columnsArray, rm_cols):
    
    try:
        miss_count = pd.DataFrame(df.isna().sum())
        miss_count.columns = ['miss_count']
        
        cat_data = df.select_dtypes(include=['category']).copy()
        num_data = df.select_dtypes(include=['number']).copy()
        length = len(df)
        drop_col = []
        for i in miss_count.index.values:
            if miss_count['miss_count'][i] >=0.7*length:
                df = df.drop(columns = [i]) 
                drop_col.append(i)
            else:
                continue
            
        miss_count = pd.DataFrame(df.isna().sum())
        miss_count.columns = ['miss_count']
        
        if sum(miss_count['miss_count'])>0:
            if 0<sum(miss_count['miss_count'])<(0.3*length):
                df = df.dropna()
            elif (0.3*length)<sum(miss_count['miss_count'])<(0.5*length):
                num_data = num_data.fillna(num_data.mean())
                for i in list(cat_data.columns):
                    cat_data[i] = cat_data[i].fillna((cat_data[i].mode(dropna=True))[0])
                frames = [cat_data,num_data, df]
                df = pd.concat(frames, axis=1)
                df = df.loc[:,~df.columns.duplicated()]
            elif sum(miss_count['miss_count'])>(0.5*length):
                df, acc = GBM_impute(df, columnsArray, rm_cols) 
            else:
                print("Imputation method not specify") 
        else:
            print("No missing values in the data")
            df = df
    except:
        print("Imputation method doesn't meet the data")
        df = df.dropna(axis=0)
    return df, miss_count, drop_col

"""
 Removing columns that contains huge number of levels
 Removing zero variance column
 input:
     df - data frame
     ratio - ratio observations to levels
     no_rem_col - columns should not be deleted
 output:
     df - changed data frame
     removed_cols - list of removed columns
"""
def remove_col(df, ratio, no_rem_col):
    transformed_data = df
    removed_cols = []
    try:
        cat_data = df.select_dtypes(include=['category']).copy()
        num_data = df.select_dtypes(include=['number']).copy()
        if no_rem_col!='none':
            no_rem_data = df[no_rem_col].copy()

        num_level_cat = []
        removed_cols = []
        if (cat_data.shape[1]>0):
            for i in list(cat_data.columns):
                cat_list = list(cat_data[i].unique())
                num_obs = cat_data[i].count() 
                for j in cat_list:
                    num_level_cat.append([i,j,cat_data[i][cat_data[i]== j].count(),num_obs])
            num_level_cat = pd.DataFrame(num_level_cat)
            num_level_cat.columns = ['category','level','count_level','count_observ']
            
            for i in list(num_level_cat['category'].unique()):
                if (len(cat_data) / num_level_cat['level'][num_level_cat['category']==i].count() < ratio):
                    cat_data = cat_data.drop(i, 1)
                    removed_cols.append(i)
#Removing zero variance column
        var = pd.DataFrame(num_data.var())
        for i in list(var.index):
            if list(var[var.index==i][0])[0] == 0:
                num_data = num_data.drop(i, 1)
                removed_cols.append(i)
        if no_rem_col!='none':
            frames = [cat_data,num_data,no_rem_data]
            
        else:
            frames = [cat_data,num_data]
        transformed_data = pd.concat(frames, axis=1)
        transformed_data = transformed_data.loc[:,~transformed_data.columns.duplicated()]
    except:
        print("Exception in removing columns")
    return transformed_data,removed_cols



"""
 chi square test for correlation between categorical features
 input:
     cat_data - data frame with categorical features, encoded
 output:
     data frame with chi square metrics
"""
def ch_sq_test(cat_data):
    
    ncol = len(cat_data.columns)
    test = []
    if ncol>1:
        combos = list(itertools.combinations(range(0,ncol), 2))
        combos = pd.DataFrame(combos)
        ind1 = list(combos[0])
        ind2 = list(combos[1])
        #try:
         #   print(list(cat_data[cat_data.columns[ind1[1]]]))
        #except Exception:
         #   print(list(cat_data[cat_data.columns[ind1]]))
        for i in range(len(ind1)):
            try:
                test.append(chi2_contingency([list(cat_data[cat_data.columns[ind1[i]]]),list(cat_data[cat_data.columns[ind2[i]]])]))
            except:
                continue
        test = pd.DataFrame(test)
        #test.columns = ['stat', 'p-val', 'dof', 'expected']
        return test    
    elif ncol == 1:
        print("There is only one category field")
    else:
        print("No category field exists")
     

"""
 Transformation method implements 
 YeoJohnson transformation for numeric features 
 (power transform featurewise to make data more Gaussian-like)
 input:
     df - data frame 
 output:
     transformed data frame
     PowerTransformer() object for invert transformation
"""
def transformation(data):
    cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()
    pt = PowerTransformer(method = 'box-cox') 
    if(num_data.shape[1]>0):
        for i in num_data.columns:
            temp_i = num_data[i]**2
            temp_i = temp_i+1
            num_data[i] = temp_i
        print(num_data)
        transformed = pt.fit(num_data).transform(num_data)
        transformed = pd.DataFrame(transformed)
        transformed.columns = num_data.columns
    
        frames = [cat_data,transformed]
        transformed_data = pd.concat(frames, axis=1)
    else:
        transformed_data = data
    return transformed_data, pt     

"""
 Transformation invert method get the origin values back after
 YeoJohnson transformation
 input:
     df - data frame 
 output:
     original values data frame
"""
def transformation_inv(data, obj):
#    cat_data = data.select_dtypes(include=['category']).copy()
#    num_data = data.select_dtypes(include=['number']).copy() 
    num_data = data
    transformed = obj.inverse_transform(num_data)
    transformed = pd.DataFrame(transformed)
    transformed.columns = num_data.columns
    for i in transformed.columns:
        temp_trans = transformed[i]-1
        temp_trans = np.sqrt(temp_trans)
        transformed[i] = temp_trans

#    frames = [cat_data,transformed]
#    transformed_data = pd.concat(frames, axis=1)
    return transformed


"""
 Correlations in the data
 Pearson test for numerical
 Chi squared test for categorical
 input:
     data - data frame outputed be columns_data_type()
     columnsArray - 
     method - method for missing val. imput (drop, impute, predict)
     no_rem_col - columns should not be deleted
 output:
     corr matrix between numeric features
     chi squared method result for categorical features
     preprocessed data
     list of exclude columns
     list of missing values amount for each columns 
"""
def correlations(data, columnsArray, no_rem_col, Val, xVal, yVal):
    # data type conversion and deleting missing values 
    chisq_dependency = []
    ent_cor = []    
    cols = data.columns                                                           ### Changes Made
    data, rm_cols = remove_col(data, ratio=3, no_rem_col=no_rem_col)
    print(rm_cols)
    col = list(filter(lambda x: x not in rm_cols and x in data.columns, cols))
    data = data.reindex(columns = col)
    data, miss_cols, drop_columns = missing_val_impute(data, columnsArray=columnsArray, rm_cols=rm_cols)
    print("Data info after missing_val_impute()")
    print(data.info())
    data.index = range(data.shape[0])
    sub= list(filter(lambda x: x not in rm_cols and x not in drop_columns and x in xVal and x in data.columns, Val))
    sub.append(yVal)
    cat_data = data.select_dtypes(include=['category']).copy()
    num_data = data.select_dtypes(include=['number']).copy()
    if (num_data.shape[1]>0):
        temp_dat = data.reindex(columns=sub)
        try:
            sub.remove(yVal)
        except Exception:
            sub = sub
        agg_dic = my_dictionary()
        for i in range(len(sub)):
            agg_dic.add(str(sub[i]), sum)
        temp_dat = temp_dat.groupby(yVal).agg(agg_dic)
        print(temp_dat)
        try: 
            temp_dat = temp_dat.resample("D", how='sum')
            if len(temp_dat)>600:
                temp_dat = temp_dat.resample("M", how='sum')
            else:
                temp_dat = temp_dat
        except Exception as e:
            print(e)
        
        temp_dat[yVal] = temp_dat.index
        temp_dat.index = range(len(temp_dat))
        print(temp_dat)
        data1, obj_t = transformation(temp_dat)
        data2 = data1
    else:
        data2 = data
        obj_t = "No Numerical"
    try:
        frm = [data2,temp_dat[yVal]]
        data = pd.concat(frm, axis = 1)
    except Exception:
        data = data2
    print("Data info after missing_val_impute()")
    print(data.info())
    if (len(num_data.columns)>1):
        ent_cor = num_data.corr()
    else:
        print("There is only one feature exists. You need at least two to analyse")
    if (len(cat_data.columns)>1):
        chisq_dependency = ch_sq_test(cat_data)
    else:
        print("There is only one feature exists. You need at least two to analyse")
#    frames = [cat_data, num_data]
#    data = pd.concat(frames, axis=1)
        
    return ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t, sub,drop_columns

"""
 Determine variable importance method implements 
 h2o.glm and h2o.gbm models for further using h2o.varimp 
 
 input:
     df - data frame 
     variable - dependent variable (y)
 output:
     matrix of variable importance
"""
def variable_importance_h2o(data_tempor, predictors, response_col):
    
        
        if(data_tempor[response_col].dtypes == 'float') or (data_tempor[response_col].dtypes == 'int'):
            print("Finding variable importance by taking given numeric variable as a dependent variable")
            try:
                hf = h2o.H2OFrame(data_tempor)
                temp_j = 0
                if hf.shape[1]<2:
                    temp_j = 1
                    for i in data_tempor.columns:
                        if data_tempor[i].dtypes == 'float' or data_tempor[i].dtypes == 'int':
                                        continue
                        else:
                                        try:
                                            data_tempor[i] = data_tempor[i].str.encode('utf-8')
                                        except Exception:
                                            data_tempor[i] = data_tempor[i].astype('int')
                                            for j in range(data_tempor[i].shape[0]):
                                                temp_df = str(data_tempor[i].iloc[j])
                                                data_tempor[i].iloc[j] = temp_df
                                            data_tempor[i] = data_tempor[i].astype('category')
                                            data_tempor[i] = data_tempor[i].str.encode('utf-8')
                                            
            except Exception:
               data_tempor = data_tempor
               hf = h2o.H2OFrame(data_tempor)
                 
            train, valid, test = hf.split_frame(ratios=[.8, .1]) 
            gbm = H2OGradientBoostingEstimator()
            
            gbm.train(predictors, response_col, training_frame= train, validation_frame=valid) 
            var_imp1 = gbm.varimp()
            if temp_j == 1:
                try:
                        for i in data_tempor.columns:
                            if data_tempor[i].dtypes == 'float':
                                continue
                            else:
                                 data_tempor[i] = data_tempor[i].str.decode('utf-8', errors= 'ignore')
                except Exception:
                        data_tempor = data_tempor
                
            
            Fin_imp_var = [var_imp1]
            return Fin_imp_var
        else:
            print("Finding variable importance by taking categorical variables as dependent variable")
            try:
                hf = h2o.H2OFrame(data_tempor)
                temp_j = 0
                if hf.shape[1]<2:
                    temp_j = 1
                    for i in data_tempor.columns:
                        if data_tempor[i].dtypes == 'float' or data_tempor[i].dtypes == 'int':
                            continue
                        else:
                            try:
                                data_tempor[i] = data_tempor[i].str.encode('utf-8')
                            except Exception:
                                data_tempor[i] = data_tempor[i].astype('int')
                                for j in range(data_tempor[i].shape[0]):
                                    temp_df = str(data_tempor[i].iloc[j])
                                    data_tempor[i].iloc[j] = temp_df
                                data_tempor[i] = data_tempor[i].astype('category')
                                data_tempor[i] = data_tempor[i].str.decode('utf-8', errors= 'ignore')
            except Exception:
                    data_tempor = data_tempor
                    hf = h2o.H2OFrame(data_tempor)
            
            train, valid, test = hf.split_frame(ratios=[.8, .1])    
            gbm = H2OGradientBoostingEstimator()
            gbm.train(predictors, response_col, training_frame= train, validation_frame=valid)
            if temp_j == 1:
                try:
                        for i in data_tempor.columns:
                            if data_tempor[i].dtypes == 'float':
                                continue
                            else:
                                 data_tempor[i] = data_tempor[i].str.decode('utf-8', errors= 'ignore')
                except Exception:
                        data_tempor = data_tempor
    #        print(gbm)
            var_imp2 = gbm.varimp()
            Fin_imp_var = [var_imp2]
            
        return Fin_imp_var
           
        
    
def vif(num_data, y, thresh):
    
    try:
        num_data = num_data.drop([y], axis=1)
    except KeyError:
        num_data = num_data
    vif = pd.DataFrame()
    if (num_data.shape[1]>0):
        if len(num_data.columns)>1:
            vif["VIF Factor"] = [variance_inflation_factor(num_data.values, i) for i in range(num_data.shape[1])]
            vif["features"] = num_data.columns
            return list(vif["features"][vif["VIF Factor"]>thresh])
        else:
            ab = ["There is only one Numerical column"]
    else:
        ab = ['No numerical Data']
    return ab
 
