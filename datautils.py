from sklearn.datasets import dump_svmlight_file
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from similarityutils import *
from gensim.models import Word2Vec, KeyedVectors
import re
import sys
from dateutil.parser import parse
from datetime import datetime, tzinfo 
import pytz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import pint
from pint import UnitRegistry
from collections import Counter
import math
from joblib import Parallel, delayed


ureg = UnitRegistry()

def getLabelledDataFromFile(fileName, rescale=False, showPlots=False):
    data = pd.read_csv(fileName, ',')

    if rescale:
        print("Rescale values")
        data.replace(-1, np.nan, inplace=True)
        for c in data.columns.drop(['source_id','target_id','pair_id','label']):
            data[c] -= data[c].min()
            data[c] /= data[c].max()
            if ('diff' in c):
                #reverse
                data[c] = 1- data[c]
                data.rename(columns={c: c.replace("diff", "sim")}, inplace=True)
        data.replace(np.nan,-1, inplace=True)
    
    #set features
    data_feature_names = list(set(data.columns.values) - {'label','source_id','target_id','pair_id',''})
    # remove the type mproperty if existing
    data_feature_names = [x for x in data_feature_names if not x.startswith('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')]
    data_feature_values = data[data_feature_names]
    
    print("Nan values in input labelled data: ", data.isnull().sum().sum())
    print("Replace all Nan values with -1")
    data = data.replace(np.NaN,-1)   
    
    #set label
    data_labels = data['label']
    #encode the labels
    label = LabelEncoder()
    data_labels = label.fit_transform(data_labels)
    label_names=[True, False]
    label_order=label.transform(label_names)
    
    if showPlots:
        sns.countplot(data_labels, label="count")
        plt.show()
    labeled_data = dict()
    labeled_data['feature_values'] = data_feature_values
    labeled_data['feature_names'] = data_feature_names
    labeled_data['labels'] = data_labels
    labeled_data['ids'] = data['pair_id']
    
    return labeled_data

def readData(directory):
    source = pd.read_csv('%s/source_.csv' % directory, sep='\|\|')
    print("Source file records:",len(source))
    target = pd.read_csv('%s/target_.csv'  % directory, sep='\|\|')
    print("Target file records:",len(target))
    pool = pd.read_csv('%s/pool.csv'  % directory, sep=';')
    print("Correspondences in the pool:",len(pool))
    validation = pd.read_csv('%s/validation.csv'  % directory, sep=';')
    print("Correspondences in the validation set:",len(validation))
    data = dict()
    data['source'] = source
    data['target'] = target
    data['pool'] = pool 
    data['validation'] = validation
    return data

def getTypesofData(data):
    dict_types = dict()
    #return dictionary
    for column in data:
        column_values = data[column].dropna()
       
        type_list=list(set(column_values.map(type).tolist()))
        #import pdb;pdb.set_trace();
        
        if len(type_list) == 0: 
            "No type could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        elif len(type_list) >1: 
            "More than one types could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        else:            
            if str in type_list:   
                types_of_column = []
                length = 0 
                for value in column_values:
                    length = length + len(value.split())
                    if re.match(r'.?\d{2,4}[-\.\\]\d{2}[-\.\\]\d{2,4}.?', value):
                        #check if it can be really converted
                        date_value = get_date_type(value)
                        if date_value != None:types_of_column.append('date')
                
                avg_length = length/len(column_values)
                
                if (avg_length>6): types_of_column.append('long_str')
                if len(set(types_of_column)) <= 1:                  
                    if ('date' in types_of_column and (types_of_column.count('date')> (len(column_values)/2))):
                        # assign date if you found the date type for the majority of thev alues
                        dict_types[column] = 'date'
                    elif ('long_str' in types_of_column):
                        dict_types[column] = 'long_str'
                    else : dict_types[column] = 'str'
                else: 
                    print("More than one types could be detected. Default (string) will be assigned.")
                    dict_types[column] = 'str'
            else: # else it must be numeric
                dict_types[column] = 'numeric'
    return dict_types

def is_date(string, fuzzy=True):
    try: 
        parse(string, fuzzy=fuzzy, default= datetime(1, 1, 1, tzinfo=pytz.UTC))        
        return True

    except ValueError:
        return False

def tryconvert(value, unit):
    converted_value=value
    if  type(value) != str and math.isnan(value): converted_value= value
    else: 
        value=value.strip()
       
        Q_ = ureg.Quantity
        try:
            converted = Q_(value).to(unit)
            converted_value = float(str(converted).split(' ')[0])
        except :
            try: 
                converted_value =float(str(value).split(' ')[0])
            except:  converted_value= float('NaN')
    return converted_value

def getUnit(value):    
    
    try:
        unit_name = str(ureg.Quantity(value).units)
        return unit_name
    except:
        return 'dimensionless'
    
def normalizeData(frame):
    try:
        numeric_attr_names = ['weight', 'length', 'width', 'height', 'price', 'number','volume', 'cost']
        for column in frame:
            column_values = frame[column]
            type_list=list(set(column_values.map(type).tolist()))

            #convert if it should be numeric
            if str in type_list and column_values.name in numeric_attr_names:
                frame[column] =frame[column].replace({'"':' inches', '$':''}, regex=True)
                column_values = frame[column]
                print("Unit convertion for column:"+column)
                # first check the units of measurement
                non_nan_column_values = frame[column].dropna().values
                units = list(map(lambda x : getUnit(x), non_nan_column_values))
                units = filter(lambda x:x!='dimensionless',units)
                units_counter = Counter(units)
                if (units_counter != Counter()): #if the counter is not empty and there are values to be converted
                    most_common_unit = units_counter.most_common(1)[0][0]
                else: most_common_unit='dimensionless'
                
                converted_values = list(map(lambda x, y=most_common_unit : tryconvert(x, y), column_values.values))
                frame[column] = converted_values
    except:
        import pdb;pdb.set_trace();

    return frame


            
    
def createFeatureVectorFile(source,target,pool, keyfeature='subject_id', embeddings = True, predefinedTypes = dict(), printProgress=False, saveFile=False, tfidf_grams=1, threads=10):    
    try:
        source_headers = source.columns.values
        target_headers = target.columns.values

        source = normalizeData(source)
        target = normalizeData(target)
        #print("Get types of data")
        dict_types_source = getTypesofData(source)
        #display(dict_types_source)
        dict_types_target = getTypesofData(target)
        #display(dict_types_target)

        common_elements = list(set(source_headers) & set(target_headers) - {keyfeature})
        common_elements_types = dict()
        for common_element in common_elements:
            if (dict_types_source[common_element] is dict_types_target[common_element]):
                common_elements_types[common_element] = dict_types_source[common_element]
            else:
                if (dict_types_source[common_element]=='long_str' or dict_types_target[common_element]=='long_str'):
                    #print("Different data types in source and target for element %s. Will assign long string" % common_element)
                    common_elements_types[common_element] = 'long_str'
                else: 
                    #print("Different data types in source and target for element %s. Will assign string" % common_element)
                    common_elements_types[common_element] = 'str'


        #calculate tfidf vectors
        #print ("Calculate tfidf scores")   
        records = dict()
        records['data'] = np.concatenate((source[common_elements].values, target[common_elements].values), axis=0)
        records['ids'] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)
        
        tfidfvector_ids = calculateTFIDF(records) 

        #print("Create similarity based features from",len(common_elements),"common elements")

        similarity_metrics={
            'str':['lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine', 'containment', 'token_jaccard'],
            'numeric':['abs_diff', 'num_equal'],
            'date':['day_diff', 'month_diff','year_diff'],
            'long_str':['cosine','lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine_tfidf', 'containment']
        }

        if not embeddings:
            similarity_metrics['str'].remove('cosine')
            similarity_metrics['long_str'].remove('cosine')


        #features = []
        
        #fix headers
        header_row = []
        header_row.append('source_id')
        header_row.append('target_id')
        header_row.append('pair_id')
        header_row.append('label')
        header_row.append('cosine_tfidf')
        for f in common_elements:
            for sim_metric in similarity_metrics[common_elements_types[f]]:
                header_row.append(f+"_"+sim_metric)       

        
        
        word2vec=None
        if embeddings :
            print("Load pre-trained word2vec embeddings")
            filename = '../../GoogleNews-vectors-negative300.bin'
            word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
            print("Pre-trained embeddings loaded")

        #create long feature specific tfidf vectors
        tfidfvector_perlongfeature = dict()
        if 'long_str' in common_elements_types.values():
            for feature in common_elements_types:
                if common_elements_types[feature] == 'long_str':             
                    records_feature = dict()
                    records_feature["data"] = np.concatenate((source[feature].values, target[feature].values), axis=0)
                    records_feature["ids"] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)
                    tfidfvector_perlongfeature[feature] = calculateTFIDF(records_feature)
        
        #get a list of stopwords
        cachedStopWords = stopwords.words("english")

        features_df = pd.DataFrame(columns=header_row, index=np.arange(len(pool)))
        #features_df = features_df.fillna(None)

        #if printProgress:
           # print_progress(0, len(pool), prefix = 'Create Features:', suffix = 'Complete')
        def typeSpecificSimilarities(data_type, valuea, valueb, f,frame_index, tfidfvector=None, r_source_id=None, r_target_id=None):
            #similarity-based features
            for sim_metric in similarity_metrics[data_type]:
                if sim_metric=='lev':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_levenshtein_sim(valuea,valueb)
                elif sim_metric=='jaccard':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_jaccard_sim(valuea,valueb)
                elif sim_metric=='token_jaccard':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_jaccard_token_sim(valuea,valueb)
                elif sim_metric=='relaxed_jaccard':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_relaxed_jaccard_sim(valuea,valueb)
                elif sim_metric=='overlap':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_overlap_sim(valuea,valueb)
                elif sim_metric=='containment':           
                    features_df.at[frame_index, f+'_'+sim_metric]=get_containment_sim(valuea,valueb)
                elif sim_metric=='cosine':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_cosine_word2vec(valuea,valueb,word2vec)
                elif sim_metric=='cosine_tfidf':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_cosine_tfidf(tfidfvector, r_source_id, r_target_id)
                elif sim_metric=='abs_diff':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_abs_diff(valuea,valueb)
                elif sim_metric=='num_equal':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_num_equal(valuea,valueb)
                elif sim_metric=='day_diff':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_day_diff(valuea,valueb) 
                elif sim_metric=='month_diff':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_month_diff(valuea,valueb)
                elif sim_metric=='year_diff':
                    features_df.at[frame_index, f+'_'+sim_metric]=get_year_diff(valuea,valueb)
                else: print("Unknown similarity metric %s" % sim_metric)

        
        def featureVectorPerPair(pair):
            #for i in range(len(pool)):
                #if printProgress: print_progress(i + 1, len(pool), prefix = 'Create Features:', suffix = 'Complete')
                #metadata
            r_source_id = pool['source_id'].loc[pair]
            r_target_id = pool['target_id'].loc[pair]
            features_df.at[pair, 'source_id']=r_source_id
            features_df.at[pair, 'target_id']=r_target_id
            features_df.at[pair, 'pair_id']=r_source_id+'-'+r_target_id
            features_df.at[pair, 'label']=pool['matching'].loc[pair]

            features_df.at[pair,'cosine_tfidf']=get_cosine_tfidf(tfidfvector_ids, r_source_id, r_target_id)

            for f in common_elements:
                fvalue_source = str(source.loc[source[keyfeature] == r_source_id][f].values[0])
                fvalue_target = str(target.loc[target[keyfeature] == r_target_id][f].values[0])

                if common_elements_types[f] is 'str' or common_elements_types[f] is 'long_str' :
                    fvalue_source = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_source.lower())).strip()
                    fvalue_target = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_target.lower())).strip()
                ## if long str remove stopwords
                if common_elements_types[f] is 'long_str':
                    fvalue_source = ' '.join([word for word in fvalue_source.split() if word not in cachedStopWords])
                    fvalue_target = ' '.join([word for word in fvalue_target.split() if word not in cachedStopWords])

                if f in tfidfvector_perlongfeature:                    
                    typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, f, pair, tfidfvector_perlongfeature[f], r_source_id, r_target_id)

                else: 
                    typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, f, pair)

                 
        #parallel execution 
        Parallel(n_jobs=threads, verbose=5, backend="threading")(map(delayed(featureVectorPerPair),np.arange(len(pool))))    
        #now rescale to 0-1
        print("Rescale [-1,1]")
        features_df.replace(-1, np.nan, inplace=True)
        for c in features_df.columns.drop(['source_id','target_id','pair_id','label']):
            features_df[c] -= features_df[c].min()
            features_df[c] /= features_df[c].max()
            if ('diff' in c):
                #reverse
                features_df[c] = 1- features_df[c]
                features_df.rename(columns={c: c.replace("abs_diff", "sim")}, inplace=True)
        features_df.replace(np.nan,-1, inplace=True)

        if saveFile:
            features_df.to_csv(saveFile, index=False)
    
    except Exception as e: 
        print(e)
        import pdb;pdb.set_trace();
    return features_df

    
    

    
def writeDataAsLibSVM(X,y,fileName, query_ids_):
    dump_svmlight_file(X,y,fileName,zero_based=True,multilabel=False, query_id=query_ids_)
    
def duplicatesinTrainingData(c,y):
    alreadyAdded = False
    dupl_c = dict()
    sorted_ind_c = sorted(range(len(c)), key=lambda x: c[x]) # sort incoming list but save the indexes of sorted items
 
    for i in xrange(len(c) - 1): # loop over indexes of sorted items
        if c[sorted_ind_c[i]] == c[sorted_ind_c[i+1]]: # if two consecutive indexes point to the same value, add it to the duplicates
            if not alreadyAdded:
                dupl_c[ c[sorted_ind_c[i]] ] = [sorted_ind_c[i], sorted_ind_c[i+1]]
                alreadyAdded = True
            else:
                dupl_c[ c[sorted_ind_c[i]] ].append( sorted_ind_c[i+1] )
        else:
            alreadyAdded = False
    return dupl_c

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

    
def showFeatureImportances(column_names, model, classifierName, dir_):
      
    importances = model.feature_importances_
       
    column_names = [c.replace('<http://schema.org/Product/', '').replace('>','') for c in column_names]
    sorted_zipped = sorted(list(zip(column_names, importances)), key = lambda x: x[1], reverse=True)[:50]
    #for feature in sorted(zipped, key = lambda x: x[1]):
        #print(feature)
    plt.figure(figsize=(18,5))
    plt.title('Feature importances for classifier %s (max. top 50 features)' % classifierName)
    plt.bar(range(len(sorted_zipped)), [val[1] for val in sorted_zipped], align='center', width = 0.8)
    plt.xticks(range(len(sorted_zipped)), [val[0] for val in sorted_zipped])
    plt.xticks(rotation=90)
    plt.savefig(dir_, bbox_inches='tight')
    plt.show() 
