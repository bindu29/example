import sys
### change to the current script folder
import h2o
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans
from math import sqrt
import numpy as np
from Linear_prep.R2P_cluster_dataprep import variable_importance_h2o, transformation_inv
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from clickhouse_driver import Client
from multiprocessing.pool import ThreadPool
from django.http import HttpResponse

############################################################
############################################################


"""
 k-means model for clustering problem 
 
 input:
     df - preprocessed data frame
 output:
     cluster column
     principle components columns
     principle components metrics
     centroid
     inter cluster similarity value for each cluster
     intra cluster similarity value
     size of each cluster

"""


def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    return wcss


def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances)) + 2

def kmeans_model(df, Vals):
    
        new_dum=pd.get_dummies(df)
        #print(new_dum)
        #hf = h2o.H2OFrame(new_dum)
        #train, valid, test = hf.split_frame(ratios=[.8, .1])    
        # kmeans model'
        #X_test.fillna(X_train.mean(), inplace=True)
        scaler = MinMaxScaler(feature_range=[0, 1])
        data_rescaled = scaler.fit_transform(new_dum)
        if len(new_dum.columns)>10:
            pca = PCA().fit(data_rescaled)
            cum_sum=np.cumsum(pca.explained_variance_ratio_)
            n_comp = len(cum_sum[cum_sum <= 0.90])
            if n_comp>=2:
                pca = PCA(n_components=n_comp)
            else:
                pca = PCA(n_components=3)
            dataset = pca.fit_transform(data_rescaled)
        else:
            dataset=data_rescaled
        sum_of_squares = calculate_wcss(dataset)
    # calculating the optimal number of clusters
        n = optimal_number_of_clusters(sum_of_squares)
        if n < new_dum.shape[0]:
             # running kmeans to our optimal number of clusters
             kmeans = KMeans(n_clusters=n)
             clusters = kmeans.fit_predict(dataset)
             df['cluster']=clusters
    #This is how you can compute the withinss, betweenss and totss by yourself
             inter_cluster_error = 0
    # Within Cluster Sum-of-Square Error
             intra_cluster_error = 0
    # Centroids
             centroids = kmeans.cluster_centers_
    # Size of clusters
             cluster_size = kmeans.n_clusters
    #cluster_column.columns = ['cluster']
    #frames = [df,cluster_column]
    #transformed_data = pd.concat(frames, axis=1)
             df = df.loc[:,~df.columns.duplicated()]
             output = [df,centroids, inter_cluster_error, intra_cluster_error, cluster_size]
        else:
            output = ['Data contains less number of rows']
        return output

"""
 Cluster profiling method 
 
 input:
     data - data frame after kmeans_model()
     xValues  
 output:
     important variables based on cluster column
     cluster wise mean and median
     cluster wise mode
"""
def cluster_profiling(data, Vals):
    ######### cluster_profiling
    
    # variable importance for cluster - categorical feature
    try:
        Vals.remove('cluster')
    except Exception:
        Vals = Vals
    variable_imp = variable_importance_h2o(data, Vals, 'cluster')
    variable_im_df = pd.DataFrame(columns = ['x','y'], index = range(len(variable_imp[0])))
    
    for i in range(len(variable_imp[0])):
        variable_im_df['x'].iloc[i] = variable_imp[0][i][0]
        variable_im_df['y'].iloc[i] = variable_imp[0][i][2]
    ### cluster wise mean and median for numeric data, mode for categorical
    
    clusters = list(data['cluster'].unique())
    clusters.sort()
    
    cat_data = data.select_dtypes(include=['category', 'object']).copy()
    cat_data = pd.concat([cat_data,data['cluster']], axis=1)
    cat_data = cat_data.loc[:,~cat_data.columns.duplicated()]
    cat_data_cols = list(cat_data.columns)
    
    #cat_data = pd.concat([cat_data,data['cluster']], axis=1)
    num_data = data.select_dtypes(include=['number']).copy()
    #print(num_data)
    num_data_cols = list(num_data.columns)
    print("Preparing cluster statistics")
    num_data = pd.concat([num_data,data['cluster']], axis=1)
    num_data = num_data.loc[:,~num_data.columns.duplicated()]
    mean_clust = pd.DataFrame(columns=num_data_cols, index=clusters)
    median_clust = pd.DataFrame(columns=num_data_cols, index=clusters)
    mode_clust = pd.DataFrame(columns=cat_data_cols, index=clusters)
    for i in range(len(clusters)):
        print('Cluster : '+str(i))
        for j in num_data_cols:
            #print(j)
            mean_clust[j].iloc[i] = round(num_data[num_data['cluster'] == clusters[i]][j].mean(),2)
            median_clust[j].iloc[i] = round(num_data[num_data['cluster'] == clusters[i]][j].median(),2)
    
    for i in range(len(clusters)):
        for j in cat_data_cols:
            mode_clust[j].iloc[i] = cat_data[cat_data['cluster'] == clusters[i]][j].mode()[0]
            
           
    output = [variable_im_df, mean_clust, median_clust, mode_clust]
    print('Cluster statistics done')
    return output


"""
 k-means method for clustering problem 
 
 input:
    #* @param dbHost 
    #* @param dbPort 
    #* @param userName 
    #* @param password
    #* @param dbName 
    #* @param query
    #* @param yValue  e.g. response_col = 'target_var' 
    #* @param xValues e.g. predictors = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost']
    #* @param parametersObj
    #* @param columnsArray
 output:
     cluster column
     principle components
     centroid
     inter cluster similarity value for each cluster
     intra cluster similarity value
     size of each cluster

"""
'''
def clustering(dbHost="",dbPort="",userName="",password="",dbName="",query="",xValues="",parametersObj="",columnsArray="",columnList=""):
    ##############
    # connecting to BD
    ##############
    client = Client(dbHost,user=userName,password=password,database=dbName)
    query_result = client.execute(query)
    df = pd.DataFrame(query_result, columns = eval(columnList))
    print(df.head())
    df = df.dropna(axis=1, how='all')
    #print(df.head())
#    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/50000_Sales_Records_Dataset_e.csv")
#    data = pd.read_excel("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/quine.xlsx")
#    data = pd.read_excel("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/airfares.xlsx")
#    data = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/AAPL.csv")
    data, columnsArray_e = columns_data_type(df[0:100], columnsArray)
    ent_cor,chisq_dependency,data,rm_cols, miss_cols, obj_t,sub = correlations_cluster(data, columnsArray=columnsArray_e, no_rem_col='none',Val=data.columns)
    print('Data info before clustering')
    
    output = kmeans_model(data[xValues],xValues)
    return output
'''

"""
 Description of the clusters similarity
 based on the inter and intra metrics 
 input:
     inter_cluster_error, 
     intra_cluster_error  
 output:
     list of text descriptions
"""
def cluster_nlg(inter_cluster_error, intra_cluster_error):
    
    min_withiness = intra_cluster_error.index(min(intra_cluster_error))    
    withiness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> <strong> Cluster"+str(min_withiness)+"</strong> has lowest value of withiness. The points within the <strong> Cluster"+str(min_withiness)+"</strong> are more homogenous </span></li>"
    betweeness = inter_cluster_error
    if(betweeness>=10000):
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have reasonably high value of betweeness  <strong>"+str(round(betweeness))+"  </strong> indicating that the heterogeneity among them is high </span></li>"
    elif (betweeness>5000 and betweeness<10000):
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have medium level of betweeness  <strong>"+str(round(betweeness))+" </strong>indicating that the heterogeneity among them is not very high </span></li>"
    elif (betweeness<5000): 
        betweeness_desc = "<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have low level of betweeness <strong>"+str(round(betweeness))+"  </strong> indicating that the heterogeneity among them is low </span></li>"
    
    return [withiness_desc, betweeness_desc]

"""
 Description of the statistics by clusters  
 input:
     variable_imp - important variables based on cluster column
     mean_clust, median_clust - cluster wise mean and median
     mode_clust output - cluster wise mode
 output:
     list of text descriptions
"""


def profiling_nlg(variable_imp, mean_clust, median_clust, mode_clust):
    print(mean_clust)
    print(median_clust)
    print(mode_clust)
    max_mean = []
    min_mean = []
    freq_info = []
    if len(list(mean_clust.columns))>0:
        for i in list(mean_clust.columns):
            max_mean.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong> "+ str(i) +"</strong> in <strong> Cluster "+str(list(mean_clust[i]).index(max(list(mean_clust[i]))))+"</strong> is highest across other clusters"+"</span></li>")
            min_mean.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong>"+ str(i)+ "</strong> in <strong> Cluster "+str(list(mean_clust[i]).index(min(list(mean_clust[i]))))+"</strong> is the lowest across other clusters"+"</span></li>")
    if len(list(mode_clust.columns))>0:
        for i in list(mode_clust.columns):
            for j in list(mode_clust.index): 
                freq_info.append("<li class=\"nlg-item\"><span class=\"nlgtext\"> The frequency of <strong>"+ str(i)+"</strong> - <strong> "+str(mode_clust[i].iloc[int(j)])+" </strong> is the most repeated level in <strong> Cluster "+str(j)+"</strong></span></li>")
    return [max_mean, min_mean, freq_info]
 
### Kmeans using H2o
    

def kMeans_model(data_temp, Vals, parametersObj, obj_t):
    
    print('Performing KMeans modelling')
    hf = h2o.H2OFrame(data_temp)
    
    predictors = list(data_temp.columns)
    
    # split into train and validation sets
    train, valid = hf.split_frame(ratios = [.8], seed = 1234)
    num_data = data_temp.select_dtypes(include=['number', 'int', 'float']).copy()
    # try using the `k` parameter:
    # build the model with three clusters
    # initialize the estimator then train the model
    try:
        hf_kmeans = H2OKMeansEstimator(max_iterations = int(parametersObj['max_iterations']), score_each_iteration= bool(parametersObj['score_each_iteration']), ignore_const_cols= bool(parametersObj['ignore_const_cols']), k = int(parametersObj['kvalue']), max_runtime_secs = int(parametersObj['max_runtime_secs']), categorical_encoding = str(parametersObj['categoricalencoding']), standardize= bool(parametersObj['standardize']), estimate_k= bool(parametersObj['estimate_k']))
        hf_kmeans.train(x = predictors, training_frame = train, validation_frame=valid)
    
    except Exception:   
        hf_kmeans = H2OKMeansEstimator(max_iterations = int(parametersObj['max_iterations']), score_each_iteration= bool(parametersObj['score_each_iteration']), ignore_const_cols= bool(parametersObj['ignore_const_cols']), k = int(parametersObj['kvalue']), max_runtime_secs = int(parametersObj['max_runtime_secs']), categorical_encoding = str(parametersObj['categoricalencoding']), standardize= bool(parametersObj['standardize']))
        hf_kmeans.train(x = predictors, training_frame = train, validation_frame=valid)
    
    clusters = hf_kmeans.predict(hf)
    data_temp['cluster']=clusters.as_data_frame()
    data_temp['cluster'] = data_temp['cluster']+1
    
    #k = int(parametersObj['kvalue'])
    ab_t = hf_kmeans.centroid_stats()['within_cluster_sum_of_squares']
    within = pd.DataFrame(columns = ['withiness'], index = range(len(ab_t)))
    #print(len(within))
    within['withiness']= hf_kmeans.centroid_stats()['within_cluster_sum_of_squares']
    #print(len(hf_kmeans.centroid_stats()['within_cluster_sum_of_squares']))
    
    tss = hf_kmeans.totss()
    betweenss = hf_kmeans.betweenss()
    #center = hf_kmeans.centers(train = True)
    size = pd.DataFrame(columns = ['cluster_size'], index = range(len(ab_t)))
    size['cluster_size'] = hf_kmeans.centroid_stats()['size']
    ## PCA    
    
    pca = H2OPrincipalComponentAnalysisEstimator(k = 2, transform = "STANDARDIZE", pca_method="GLRM",use_all_factor_levels=True, impute_missing=False, max_iterations = 300)
    pca.train(training_frame=hf)
    cords = pca.rotation().as_data_frame()
    cords = cords[['pc1', 'pc2']]
    cords.columns = ['PC1', 'PC2']
    print('here')
    num_data = data_temp.select_dtypes(include=['int', 'float', 'number']).copy()
    num_data_cols = list(num_data.columns)
    print('here1')
    
    cat_cols = list(filter(lambda x: x not in num_data_cols, data_temp.columns))
    data_temp1 = data_temp[cat_cols]
    try:
        num_data_cols.remove('cluster')
    except Exception:
        num_data_cols = num_data_cols
    if num_data.shape[1]>1:
        num_data = num_data[num_data_cols]
        num_data_temp = transformation_inv(num_data,obj_t)
        frames = [num_data_temp, data_temp1,data_temp['cluster']]
        data_temp = pd.concat(frames, axis = 1)
        for i in num_data.columns:
               if data_temp[i].dtypes=='float' or data_temp[i].dtypes=='int':
                      for j in range(len(data_temp[i])):
                          data_temp[i].iloc[j] = round(data_temp[i].iloc[j], 2)
    else:
        frames = [data_temp1,num_data['cluster']]
        data_temp = pd.concat(frames, axis = 1)
    print(data_temp.columns)
    print('KMeans done')
    listed = [data_temp, cords, [], within, tss, betweenss, size]
    return listed

    
    
    
    
