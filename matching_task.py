import copy
from profiling import *
from learningutils import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_validate,cross_val_predict, StratifiedShuffleSplit, GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from collections import Counter
from similarityutils import * 
from sklearn import tree
class MatchingTask(object):

    def __init__(self, source_dt, target_dt, gs, gs_feature_vector, metadata_mt, common_attributes_names, feature_vector_train,feature_vector_test,feature_vector_validation):
        
        self.source_dt = copy.copy(source_dt)
        self.target_dt = copy.copy(target_dt)
        self.gs = copy.copy(gs)
        self.gs_feature_vector = copy.copy(gs_feature_vector)
        
        self.metadata_mt = metadata_mt
        self.common_attributes_names = common_attributes_names
               
        self.important_attr_names = None
        self.important_features = None
        self.label_attr = self.metadata_mt.get("primattr")
        self.source_subset=source_dt[source_dt['subject_id'].isin(gs['source_id'].values)]
        self.target_subset=target_dt[target_dt['subject_id'].isin(gs['target_id'].values)]
        self.source_subset_match=source_dt[source_dt['subject_id'].isin(gs[gs['matching']==True]['source_id'].values)]
        self.target_subset_match=target_dt[target_dt['subject_id'].isin(gs[gs['matching']==True]['target_id'].values)]
        self.gs_prediction_scores = None
        
        self.dict_summary = dict()
        self.dict_important_features_profiling = dict()
        self.dict_profiling_features = dict()
        self.dict_linear_results = dict()
        self.dict_non_linear_results = dict()
        self.dict_unsupervised_results = dict()
    
        #in case they are provided
        self.feature_vector_train =feature_vector_train
        self.feature_vector_test = feature_vector_test
        self.feature_vector_validation = feature_vector_validation
    
    
    #returns profiling features for matching task - considers only the source and target subset that appears in the correspondences     
    def getSummaryFeatures(self):
        try:
            #general
            self.dict_summary['#records_source'] = self.source_dt.shape[0]
            self.dict_summary['#records_target'] = self.target_dt.shape[0]
            count_record_pairs = len(self.gs_feature_vector.index)
            self.dict_summary['count_record_pairs'] = count_record_pairs
            self.dict_summary['count_attr'] = len(self.common_attributes_names) 
            self.dict_summary['#match'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==True])
            self.dict_summary['#non-match'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==False])
            self.dict_summary['ratio_pos'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==True])/count_record_pairs
            self.dict_summary['ratio_neg'] = len(self.gs_feature_vector[self.gs_feature_vector['label']==False])/count_record_pairs
            short_string_attr = []
            long_string_attr = []
            num_attr = []
            date_attr = []
            for attr in self.common_attributes_names:
                if attr+"_token_jaccard" in self.gs_feature_vector.columns : short_string_attr.append(attr)
                elif attr+"_cosine_tfidf"  in self.gs_feature_vector.columns : long_string_attr.append(attr) 
                elif attr+"_num_equal" in self.gs_feature_vector.columns : num_attr.append(attr)
                elif attr+"_year_sim" in self.gs_feature_vector.columns : date_attr.append(attr)
            
            self.dict_summary['#short_string_attr'] = len(short_string_attr)
            self.dict_summary['#long_string_attr'] = len(long_string_attr)
            self.dict_summary['#numeric_attr'] = len(num_attr)
            self.dict_summary['#date_attr'] = len(date_attr)

            #density features
            self.dict_summary['avg_density_all'] = getFeatureDensities(self.gs_feature_vector, self.common_attributes_names)
            self.dict_summary['density_label'] = getFeatureDensities(self.gs_feature_vector, [self.label_attr])
           
        except:  import pdb; pdb.set_trace();
        
    def getProfilingFeatures(self):
        
        #get identifying features
        #we drop the cosine_tfidf because we want to get single attribute related features (and importances)
        X = self.gs_feature_vector.drop(['source_id', 'target_id', 'pair_id', 'label','cosine_tfidf'], axis=1)
        y =  self.gs_feature_vector['label'].values
        
        clf = RandomForestClassifier(random_state=1, min_samples_leaf=2)
        model = clf.fit(X,y)     
        features_in_order, feature_weights = showFeatureImportances(X.columns.values,model,'rf') 
        # all features that are relevant for the matching
        self.dict_profiling_features['matching_relevant_features'] = []
        matching_relevant_attributes = []
        for feat, weight in zip(features_in_order, feature_weights):
            if weight>0: 
                self.dict_profiling_features['matching_relevant_features'].append(feat)
                matching_relevant_attributes.append(get_cor_attribute(self.common_attributes_names,feat))
        
        self.dict_profiling_features['matching_relevant_attributes_datatypes'] = self.getAttributeDistinctDatatypes(matching_relevant_attributes)
        self.dict_profiling_features['matching_relevant_attributes'] = set(matching_relevant_attributes)
        self.dict_profiling_features['matching_relevant_attributes_count'] = len(set(matching_relevant_attributes))
        self.dict_profiling_features['matching_relevant_attributes_density'] = round(getFeatureDensities(self.gs_feature_vector, self.dict_profiling_features['matching_relevant_features']),2)
        
        #max results
        xval_scoring = {'f1_score' : make_scorer(f1_score)}       
        max_result = cross_validate(clf, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring, n_jobs=-1)
        max_f1_score = round(np.mean(max_result['test_f1_score']),2)
        
        #gather features that are relevant for 95% of the max f1 score
        sub_result = 0.0
        for i in range(1,len(features_in_order)+1):
            results_subvector = cross_validate(clf, X[features_in_order[:i]], y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring, n_jobs=-1)
            sub_result = round(np.mean(results_subvector ['test_f1_score']),2)
            if (sub_result>0.95*max_f1_score): break;
        
        important_features = features_in_order[:i]

        self.dict_profiling_features['top_matching_relevant_features_count'] = len(important_features)
        self.dict_profiling_features['F1_xval_max'] = max_f1_score
        self.dict_profiling_features['F1_xval_top_matching_relevant_features'] = sub_result
        self.dict_profiling_features['top_matching_relevant_features'] = important_features
        mapped_ident_features = []
        for attr in important_features:  mapped_ident_features.append(get_cor_attribute(self.common_attributes_names,attr))
        self.dict_profiling_features['top_relevant_attributes'] =  set(mapped_ident_features)
        self.dict_profiling_features['top_relevant_attributes_datatypes'] =  self.getAttributeDistinctDatatypes(set(mapped_ident_features))
        self.dict_profiling_features['top_relevant_attributes_count'] = len(self.dict_profiling_features['top_relevant_attributes'])
        self.dict_profiling_features['top_relevant_attributes_density']=round(getFeatureDensities(self.gs_feature_vector, important_features),2)
        
        avg_length_tokens_ident_feature = []
        avg_length_words_ident_feature = []
        for attr in self.dict_profiling_features['top_relevant_attributes']: 
            #check if it is string
            if (attr+'_containment' in self.gs_feature_vector.columns):
                avg_length_tokens_ident_feature.append(np.mean([getAvgLength(self.source_subset, attr, 'tokens'), getAvgLength(self.target_subset, attr, 'tokens')]))
                avg_length_words_ident_feature.append(np.mean([getAvgLength(self.source_subset, attr, 'words'), getAvgLength(self.target_subset, attr, 'words')]))
        
        self.dict_profiling_features['avg_length_tokens_top_relevant_attributes'] = round(sum(avg_length_tokens_ident_feature),2)
        self.dict_profiling_features['avg_length_words_top_relevant_attributes'] = round(sum(avg_length_words_ident_feature),2)

        
        #corner cases
        interstingness,uniqueness = getCornerCaseswithOptimalThreshold(self.gs_feature_vector,important_features)
        self.dict_profiling_features['corner_cases_top_matching_relevant_features'] = round(interstingness,2)
        self.dict_profiling_features['avg_uniqueness_top_matching_relevant_features'] = round(uniqueness,2)
        
    def getAttributeDistinctDatatypes(self, attr_list):
        datatypes=[]
        for att in attr_list:
            if (att+"_day_diff" in self.gs_feature_vector.columns): datatypes.append('date')
            if (att+"_num_equal" in self.gs_feature_vector.columns): datatypes.append('numeric')
            if (att+"_containment" in self.gs_feature_vector.columns):
                if (att+"_token_jaccard" in self.gs_feature_vector.columns):
                    datatypes.append('short string')
                else:
                    datatypes.append('long string')
        return set(datatypes)
    
    
    def getNestedXValidationResults(self, model):
        X = self.gs_feature_vector.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
        y = self.gs_feature_vector['label'].values
        f1_scorer = make_scorer(f1_score)
        
        if model=="non-linear":
            grid_values = {'n_estimators' : [10, 100, 500], 'max_depth' : [10, 50, None], 'min_samples_leaf' : [1, 3, 5]}
            clf = RandomForestClassifier(random_state=1)
            results_dict = self.dict_non_linear_results
            
        elif model=="linear":
            grid_values_1 = { 'kernel' : ['linear'], 'max_iter':[100000]}
            grid_values_2 = {'C' : np.logspace(-2, 5, 10), 'max_iter':[100000], 'kernel' : ['rbf'], 'gamma': [1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+01, 'scale']}
            grid_values= []
            grid_values.append(grid_values_1)
            grid_values.append(grid_values_2)
            clf = SVC(random_state=1, probability=True)
            results_dict = self.dict_linear_results
        
        clf_gs = GridSearchCV(clf, grid_values, scoring=f1_scorer, cv=StratifiedShuffleSplit(n_splits=4,random_state =1), verbose=10, n_jobs=20)
        xval_scoring = {'precision' : make_scorer(precision_score),
                               'recall' : make_scorer(recall_score), 'f1_score' : make_scorer(f1_score)}

        results = cross_validate(clf_gs, X, y, cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring)
        prediction_probs = cross_val_predict(clf_gs, X, y, method='predict_proba')
        pred_scores = list(map(lambda x:max(x),prediction_probs))

        results_dict['precision'] =round(np.mean(results['test_precision']),2)
        results_dict['recall'] =round(np.mean(results['test_precision']),2)
        results_dict['f1'] =round(np.mean(results['test_f1_score']),2)
        results_dict['f1_std'] =round(np.std(results['test_f1_score']),2)

        results_dict['proba_scores'] =round(np.mean(pred_scores),2)
        results_dict['proba_scores_std'] =round(np.std(pred_scores),2)
        
    def getSplitValidationResults(self, model):
        
        if (self.feature_vector_train is None): #there is no split do it with the ratio 70-20-10
            X = self.gs_feature_vector.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
            y = self.gs_feature_vector['label'].values
            sssplits_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
            train_index, val_test_index = next(sssplits_train_test.split(X,y))
            X_train, X_val_test, y_train, y_val_test = X.loc[train_index], X.loc[val_test_index],y[train_index],y[val_test_index]
            X_val_test.reset_index(inplace=True)
            X_train.reset_index(inplace=True)   
            sssplits_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=1)
            val_index, test_index = next(sssplits_val_test.split(X_val_test, y_val_test))
            X_val, X_test, y_val, y_test = X_val_test.loc[val_index], X_val_test.loc[test_index], y_val_test[val_index], y_val_test[test_index]
        else:
            print("Using provided train-test-validation files")
            X_train = self.feature_vector_train.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
            y_train = self.feature_vector_train['label'].values
            X_test = self.feature_vector_test.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
            y_test = self.feature_vector_test['label'].values
            X_val = self.feature_vector_validation.drop(['source_id', 'target_id', 'pair_id', 'label'], axis=1)
            y_val = self.feature_vector_validation['label'].values
            
        f1_scorer = make_scorer(f1_score)
        # source:https://www.wellformedness.com/blog/using-a-fixed-training-development-test-split-in-sklearn/
        x_train_val = np.concatenate([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        
        val_fold = np.concatenate([
            np.full(X_train.shape[0], -1 ,dtype=np.int8),
            # The development data.
            np.zeros(X_val.shape[0], dtype=np.int8)
        ])
        print("Predefined validation fold")
        display(Counter(val_fold))
        cv = PredefinedSplit(val_fold)
        
        if model=="non-linear":
            grid_values = {'n_estimators' : [10, 100, 500], 'max_depth' : [5, 10, 50, None], 'min_samples_leaf' : [1, 3, 5]}
            clf = RandomForestClassifier(random_state=1)
            results_dict = self.dict_non_linear_results
            
        elif model=="linear":
            grid_values_1 = { 'kernel' : ['linear'], 'max_iter':[100000]}
            grid_values_2 = {'C' : np.logspace(-2, 5, 10), 'max_iter':[100000], 'kernel' : ['rbf'], 'gamma': [1.e-06, 1.e-04, 1.e-02, 1.e+00, 1.e+01, 'scale']}
            grid_values= []
            grid_values.append(grid_values_1)
            grid_values.append(grid_values_2)
            clf = SVC(random_state=1, probability=True)
            results_dict = self.dict_linear_results

        model = GridSearchCV(clf, grid_values, scoring=f1_scorer, cv=cv,verbose=10, n_jobs=-1) #parallel jobs
        model.fit(x_train_val,y_train_val)
        predictions = model.predict(X_test)
        prediction_probs = model.predict_proba(X_test)
        pred_scores = list(map(lambda x:max(x),prediction_probs))
        prec, recall, fscore, support  = precision_recall_fscore_support(y_test, predictions, average='binary')
        
        results_dict['precision'] =round(prec,2)
        results_dict['recall'] =round(recall,2)
        results_dict['f1'] =round(fscore,2)
        results_dict['f1_std'] =0.0

        results_dict['proba_scores'] =round(np.mean(pred_scores),2)
        results_dict['proba_scores_std'] =round(np.std(pred_scores),2)
        
        #and now calculate the x-val results using the best model from the grid search
        xval_scoring = {'f1_score' : make_scorer(f1_score)}
        results = cross_validate(model.best_estimator_, pd.concat([X_train,X_val,X_test], ignore_index=True), np.concatenate((y_train,y_val,y_test)), cv=StratifiedShuffleSplit(n_splits=4,random_state =1),  scoring=xval_scoring,n_jobs=-1)
        results_dict['x-val f1'] =round(np.mean(results['test_f1_score']),2)
        results_dict['x-val f1 sigma'] =round(np.std(results['test_f1_score']),2)
     
    