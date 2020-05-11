from sklearn.tree import _tree
import matplotlib.pyplot as plt
import numpy as np

def printTreeRules(feature_names, tree):
    tree_model = []
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))
    
    recurse(0, 1)
    #print (''.join(tree_model))
    
def get_model_importances(model,classifierName=None):
 
    if classifierName == 'logr':
        importances = model.coef_.ravel()
    elif classifierName == 'svm':
        if model.kernel != 'linear':
            display("Cannot print feature importances without a linear kernel")
            return
        else: importances = model.coef_.ravel()
    else:
        importances = model.feature_importances_
    
    return importances

def showFeatureImportances(column_names, model, classifierName):
      
    importances = get_model_importances(model, classifierName)
       
    column_names = [c.replace('<http://schema.org/Product/', '').replace('>','') for c in column_names]
    sorted_zipped = sorted(list(zip(column_names, importances)), key = lambda x: x[1], reverse=True)[:50]
   
    plt.figure(figsize=(18,3))
    plt.title('Feature importances for classifier %s (max. top 50 features)' % classifierName)
    plt.bar(range(len(sorted_zipped)), [val[1] for val in sorted_zipped], align='center', width = 0.8)
    plt.xticks(range(len(sorted_zipped)), [val[0] for val in sorted_zipped])
    plt.xticks(rotation=90)
    plt.show() 
    features_in_order = [val[0] for val in sorted_zipped]
    feature_weights_in_order = [round(val[1],2) for val in sorted_zipped]
    return features_in_order,feature_weights_in_order