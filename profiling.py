import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd
import statistics
import collections
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


def getUniqueness(feature_vector, column):
    column_values= copy.copy(feature_vector[column])
    distinct_values = column_values.nunique()
    non_null_values = column_values.count()
    if non_null_values==0: uniqueness=0
    else : uniqueness = round(float(distinct_values)/non_null_values,2)
    return uniqueness

def getAvgLength(feature_vector, column, mode):
    column_values= copy.copy(feature_vector[column])
    column_values.fillna('nan')
    lengths = []       
    for i in column_values.values:
        if i!='nan': 
            if mode == 'tokens' : lengths.append(len(str(i)) )
            elif mode == 'words' : lengths.append(len(str(i).split()))

    avg = 0 if len(lengths) == 0 else round(float(sum(lengths) / len(lengths)),2)
    return avg

def getFeatureCorr(feature_vector, path='', show=True):
    print("*****Correlation between features and target variable "+path+"*****")
    if ('source_id' in feature_vector.columns):
        correlations = feature_vector[feature_vector.columns[3:]].corr()['label'][:].to_frame().abs()
    else: correlations = feature_vector[feature_vector.columns].corr()['label'][:].to_frame().abs()
    correlations = correlations.sort_values(by='label', ascending=False)
    if show:
        plt.subplots(figsize=(5,15))
        sns.heatmap(correlations, 
        xticklabels=correlations.columns,
        yticklabels=correlations.index)
        plt.show()
    return correlations

def getFeatureDensities(feature_vector, common_attributes, show=False):
    feature_vector_with_nulls = copy.copy(feature_vector)
    feature_vector_with_nulls = feature_vector_with_nulls.replace({-1: None})
    non_null_values = feature_vector_with_nulls.count()
    density = round(feature_vector_with_nulls.count()/len(feature_vector_with_nulls.index),2)
    visited = []
    overall_density = []
    if show:
        print("*****Feature densities*****")
    for feat in feature_vector_with_nulls.columns:
        if (feat not in ['source_id', 'target_id', 'pair_id', 'label']):
            for common_attr in common_attributes:
                if (feat.startswith(common_attr) and common_attr not in visited):
                    visited.append(common_attr)
                    overall_density.append(density[feat])
                    if show: print(common_attr+": "+str(density[feat]))
    return statistics.mean(overall_density)

def getTopCorrelatedAttr(correlations, topOne=True):
    print("*****Highly correlated attributes (>0.6)*****")

    print(correlations[(correlations['label']>0.6) & (correlations['label']!=1.0)])
    if topOne:
        top_corr_feature = correlations.iloc[1].name
    else: top_corr_feature = correlations[(correlations['label']>0.6) & (correlations['label']!=1.0)].index.tolist()
    
    return top_corr_feature


def getCornerCaseswithOptimalThreshold(feature_vector, attributes):
    positives = copy.copy(feature_vector[feature_vector['label']==True])
    negatives = copy.copy(feature_vector[feature_vector['label']==False])
    
    positives = positives.replace(-1, 0)
    negatives = negatives.replace(-1, 0)
    
    positive_values = positives[attributes].mean(axis=1).values
    negative_values = negatives[attributes].mean(axis=1).values
    
    thresholds = []
    fp_fn = []
    for t in np.arange(0.0, 1.0, 0.01):
        fn = len(np.where(positive_values<t)[0])
        fp = len(np.where(negative_values>=t)[0])
        thresholds.append(t)
        fp_fn.append(fn+fp)
    optimal_threshold = thresholds[fp_fn.index(min(fp_fn))]
    hard_cases = min(fp_fn)
    groups_positives = positives[attributes].groupby(attributes).size().reset_index()
   
    return hard_cases/len(positive_values),groups_positives.shape[0]/len(positive_values)
        

def getCornerCasesByKmeansDiscretization(feature_vector, attribute):
    discr_pos = KBinsDiscretizer(n_bins=4, encode = 'ordinal', strategy='kmeans')

    positives = copy.copy(feature_vector[feature_vector['label']==True])
    negatives = copy.copy(feature_vector[feature_vector['label']==False])
    
    positives[attribute].replace(-1, np.nan, inplace=True)
    negatives[attribute].replace(-1, np.nan, inplace=True)
    
    positive_values = positives[attribute].mean(axis=1).values
    negative_values = negatives[attribute].mean(axis=1).values
    
    #remove the missing values
    positive_scores = np.delete(positive_values, np.argwhere(positive_values == np.nan)).reshape(-1,1)
    positive_scores_binned = discr_pos.fit_transform(positive_scores)
    if len(collections.Counter(positive_scores_binned.flat))==1:
        ratio_pos_corner_cases=0
        pos_corner_cases_range=1 #all positives have a score of 1.0
    else:
        ratio_pos_corner_cases = collections.Counter(positive_scores_binned.flat).get(0.0)/len(positives.index)
        pos_corner_cases_range = discr_pos.bin_edges_[0][1]
        
    #remove the missing values
    discr_neg = KBinsDiscretizer(n_bins=4, encode = 'ordinal', strategy='kmeans')
    negative_scores = np.delete(negative_values, np.argwhere(negative_values == np.nan)).reshape(-1,1)
    negative_scores_binned = discr_neg.fit_transform(negative_scores)
    
    if len(collections.Counter(negative_scores_binned.flat))==1:
        ratio_neg_corner_cases=0
        neg_corner_cases_range=0 #all negatives have a score of 0.0
    else:
        created_bins = len(collections.Counter(negative_scores_binned.flat)) #might create less than 4 bins
        ratio_neg_corner_cases = collections.Counter(negative_scores_binned.flat).get(created_bins-1)/len(negatives.index)
        neg_corner_cases_range = discr_neg.bin_edges_[0][created_bins-1]
    
    #if the ratio of positives = 0 then the neg.corner cases are only the ones having a sim.score= max(positive_scores) and the other way around
    if ratio_pos_corner_cases==0: 
        ratio_neg_corner_cases = negatives[negatives[attribute] == 1].shape[0]/negatives.shape[0]
        neg_corner_cases_range = 1
    
    if ratio_neg_corner_cases == 0:
        ratio_pos_corner_cases = positives[positives[attribute] == 0].shape[0]/positives.shape[0]
        pos_corner_cases_range = 0
    return ratio_pos_corner_cases, pos_corner_cases_range, ratio_neg_corner_cases, neg_corner_cases_range

def getCornerCases(feature_vector, attribute, getHardRatios=False):
    positives = feature_vector[feature_vector['label']==True]
    negatives = feature_vector[feature_vector['label']==False]

    bins = [-1.0,0,0.2,0.4,0.6,0.8,1.0]
    bin_positives = pd.cut(positives[attribute], bins=bins).value_counts(sort=False)
    bin_negatives = pd.cut(negatives[attribute], bins=bins).value_counts(sort=False)
    ratio_hard_neg = float(len(negatives[negatives[attribute]>0.8].index))/float(len(feature_vector.index))
    ratio_hard_pos = float(len(positives[positives[attribute]<0.2].index))/float(len(feature_vector.index))
    
    if (getHardRatios): return bin_positives, bin_negatives,ratio_hard_neg,ratio_hard_pos
    else: return bin_positives, bin_negatives
    
def get_cor_attribute(common_attributes, pairwiseatt):
    for c_att in common_attributes:
        if  pairwiseatt.startswith(c_att): return c_att
        if c_att.startswith('cosine_tfidf'): return 'all'