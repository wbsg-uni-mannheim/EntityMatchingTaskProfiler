from __future__ import division
from similarity.levenshtein import Levenshtein
from similarity.jaccard import Jaccard
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
import re
from datetime import tzinfo
from dateutil.parser import parse
import pytz
from numpy import trapz
import re
from scipy.ndimage import gaussian_filter
from numpy import matlib
from copy import deepcopy
import nltk
from nltk import ngrams
import matplotlib.pyplot as plt
from datautils import *


    
def get_date_type_(date_str):
    try:
        date_ = parse(date_str, fuzzy=True, default=datetime(1, 1, 1, 1, 1, tzinfo=tzoffset(None, 18000)))
        return date_
    except:
        import pdb
        pdb.set_trace()
        display('Could not parse %s' % date_str)
        return


def get_date_type(date_str):
    separator = ''
    if '.' in date_str:
        separator = '.'
    elif '\\' in date_str:
        separator = '\\'
    elif '/' in date_str:
        separator = '/'
    elif '-' in date_str:
        separator = '-'
    else:
        return None
    try:
        date_parts = [ d.strip() for d in date_str.split(separator) ]
        if re.match('\\d{4}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            return datetime.datetime.strptime(date_str, '%Y' + separator + '%m' + separator + '%d').date()
        if re.match('\\d{1,2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{4}', date_str):
            return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%Y').date()
        if re.match('\\d{2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            p = re.compile('\\d+')
            splitted_date = p.findall(date_str)
            if int(splitted_date[0]) < 32 and int(splitted_date[1]) < 13:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            if int(splitted_date[0]) > 32:
                return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
            try:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            except:
                try:
                    return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
                except:
                    display('Unknown pattern or invalid date: %s' % date_str)
                    return None

        else:
            return parse(date_str, fuzzy=True)
    except:
        f = open('unparseddates.txt', 'a')
        f.write(date_str + '\n')
        f.close()
        return None


def get_day_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan' or date1 == '' or date2 == '':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.day - date2_.day
    return abs(delta)


def get_month_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan' or date1 == '' or date2 == '':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.month - date2_.month
    return abs(delta)


def get_year_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan' or date1 == '' or date2 == '':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    difference = abs(date1_.year - date2_.year)
    if len(date1) != len(date2) and difference % 100 == 0:
        difference = 0
    return difference


def get_num_equal(num1, num2):
    if num1 == 'nan' or num2 == 'nan' or num1 == '' or num2 == '':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        if num1_ == num2_:
            return 1.0
        return 0.0
    except:
        return -1


def get_abs_diff(num1, num2):
    if num1 == 'nan' or num2 == 'nan' or  num1 == '' or num2 == '':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        return abs(num1_ - num2_)
    except:
        return -1

def get_jaccard_token_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        return 1-nltk.jaccard_distance(set(str1), set(str2))
    
    
def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        return float(len(c)) / float(len(a) + len(b) - len(c))


def get_relaxed_jaccard_sim(str1, str2, n_grams=1):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '' :
        return -1.0
    a = set(str1.split())
    b = set(str2.split())
    if not a or not b: return -1
    c = []
    for a_ in a:
        for b_ in b:
            if get_levenshtein_sim(a_, b_) > 0.7:
                c.append(a_)
    intersection = len(c)
    min_length = min(len(a), len(b))
    if intersection > min_length:
        intersection = min_length
    return float(intersection) / float(len(a) + len(b) - intersection)


def get_containment_sim(str1, str2):
    #it's not really a long string necessarily but it does not make sense to do word based containment
    if (len(set(str1.split()))>1 and len(set(str2.split()))>1): 
        a = set(str1.split())
        b = set(str2.split())
    else: #for single words we consider the tokens
        a = set(str1)
        b = set(str2)
        
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif len(a) == 0 or len(b) == 0:
        return -1.0
    else:
        return float(len(c)) / float(min(len(a), len(b)))


def get_levenshtein_sim(str1, str2):
    levenshtein = Levenshtein()
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    else:
        max_length = max(len(str1), len(str2))
        return 1.0 - levenshtein.distance(str1, str2) / max_length


def get_missing(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '' :
        return 1.0
    else:
        return 0.0


def get_overlap_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif str1 == str2:
        return 1.0
    else:
        return 0.0


def get_cosine_word2vec(str1, str2, model):
    if str1 == 'nan' or str2 == 'nan' or str1 == '' or str2 == '':
        return -1.0
    elif str1.replace(' ', '') in model.vocab and str2.replace(' ', '') in model.vocab:
        return model.similarity(str1.replace(' ', ''), str2.replace(' ', ''))
    else:
        return 0.0


def get_cosine_tfidf(tfidf_scores_ids, sourceID, targetID):
    source_index = np.where(tfidf_scores_ids['ids'] == sourceID)
    target_index = np.where(tfidf_scores_ids['ids'] == targetID)
    score = cosine_similarity(tfidf_scores_ids['scores'][source_index].todense(), tfidf_scores_ids['scores'][target_index].todense())
    return score[0][0]
    

def calculateTFIDF(records, grams=1): 
    try:
        records_data = records['data']
        concat_records = []
        for row in records_data:
            if (isinstance(row,np.ndarray)): # tfidf based on  more that one features
                concat_row = ''
                for value in row:
                    if not pd.isnull(value):
                        if type(value) is str:
                            if value.lower() != 'nan':
                                value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value)) #think of product model names e.g. ak-123
                                concat_row += ' ' + value
                        else: # tfidf based on one feature 
                            value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                            concat_row += ' ' + str(value)

                concat_records.append(concat_row.lower())
            else: 
                if pd.isnull(row):
                    concat_records.append("")
                else:
                    value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row))
                    concat_records.append(value.lower())

        tf_idfscores = TfidfVectorizer(encoding='latin-1', ngram_range=(grams,grams)).fit_transform(concat_records)
        tf_idf = dict()
        tf_idf['ids'] = records['ids']
        tf_idf['scores'] = tf_idfscores
    except:
        import pdb;pdb.set_trace();
    return tf_idf


def aggregate_score(frame,labelattr):
    #consider only columns of density > 0.1
    frame = frame.drop(['source_id','target_id','pair_id','label'], axis=1)
    #frame = frame[[labelattr]] #if you want to consider only the label
    tobedropped=[]
    for c in frame.columns:
        empty_values = len(frame[frame[c] == -1])
        per= float(empty_values)/float(len(frame[c]))
        density = 1-per 
        if (density<0.1):tobedropped.append(c)
   
    display("To be dropped:"+str(len(tobedropped)))
    frame = frame.drop(tobedropped, axis=1)
    
    cosine_tfidf_column = frame['cosine_tfidf']
    other_columns  = frame.drop(['cosine_tfidf'], axis=1)
    other_columns = other_columns.replace(-1.0,np.nan)
    
    #calculate column weights
    column_weights = []
    for c in other_columns:
        nan_values = other_columns[c].isna().sum()
        ratio = float(nan_values)/float(len(other_columns[c]))
        column_weights.append(1.0-ratio)

    weighted_columns = other_columns*column_weights
    #weighted_columns = other_columns # do not weight
    other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
    other_columns_mean = other_columns_sum/len(other_columns.columns)
    
    #rescale 
    other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))
    cosine_tfidf_column = np.interp(cosine_tfidf_column, (cosine_tfidf_column.min(), cosine_tfidf_column.max()), (0, +1))
    
    weighted_cosine = cosine_tfidf_column*0.5
    weighted_other_columns = other_columns_mean*0.5
    sum_weighted_similarity = weighted_other_columns+weighted_cosine
    
    #just take the mean of all columns, no weighting, cosine tfidf is not more important
    #sum_weighted_similarity  = other_columns_mean
    
    return sum_weighted_similarity
    
def elbow_threshold(similarities):        
    similarities.sort()

    sim_list = list(similarities)
    nPoints = len(sim_list)
    allCoord = np.vstack((range(nPoints), sim_list)).T
    
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel    
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    
    
    print ("Knee of the curve is at index =",idxOfBestPoint)
    print ("Knee value =", similarities[idxOfBestPoint])
       
    plt.plot(np.arange(len(similarities)), similarities)
    plt.show()
    return similarities[idxOfBestPoint]

def valley_threshold(aggr_scores):
    aggr_scores = [round(i,2) for i in aggr_scores if i != -1]
    
    aggr_scores.sort()
    similarities = deepcopy(aggr_scores)
   
    similarities.reverse()
    hist, _ = np.histogram(similarities, bins=len(similarities), range=(0.0, 1.0))
    hist = 1.0 * hist / np.sum(hist)
    plt.hist(hist)
    plt.show()
    
    import pdb; pdb.set_trace();

    val_max = -999
    thr = -1
    #print_progress(1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
    float_list = [round(elem, 2) for elem in similarities]
    #normalizes by occurrences of most frequent value
    fre_occur = float_list.count(max(float_list,key=float_list.count))
    for t in range(1, len(similarities) - 1):
        #print_progress(t + 1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        if q1 != 0 and q2 != 0:
            m1 = np.sum(np.array([ i for i in range(t) ]) * hist[:t]) / q1
            m2 = np.sum(np.array([ i for i in range(t, len(similarities)) ]) * hist[t:]) / q2

            val = (1.0-float(float_list.count(round(similarities[t],2)))/float(fre_occur))*(q1 * (1.0 - q1) * np.power(m1 - m2, 2))
            if val_max < val:
                val_max = val
                thr = similarities[t]
    
    
    print ("Threshold defined with valley threshold method: %f " % thr)
    return thr