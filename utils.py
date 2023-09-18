import pandas as pd
import numpy as np

import seaborn as sns

from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate



def tree_to_code(tree, feature_names):
    if isinstance(tree, DecisionTreeClassifier):
        model = 'clf'
    elif isinstance(tree, DecisionTreeRegressor):
        model = 'reg'
    else:
        raise ValueError('Need Regression or Classification Tree')
        
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(indent, f'if {name} <= {threshold:.2%}')
            recurse(tree_.children_left[node], depth + 1)
            print(indent, f'else:  # if {name} > {threshold:.2%}')
            recurse(tree_.children_right[node], depth + 1)
        else:
            pred = tree_.value[node][0]
            val = pred[1]/sum(pred) if model == 'clf' else pred[0]
            print(indent, f'return {val:.2%}')
    recurse(0, 1)
    
    
def stack_cv_results(scores, metrics):
    """ Stack the results from cross validation """
    
    columns = pd.MultiIndex.from_tuples(
        [tuple(m.split('_', 1)) for m in scores.keys()],
        names=['Dataset', 'Metric'])
    
    data = np.array(list(scores.values())).T                          # Scores for each metric
    df = pd.DataFrame(data=data, columns=columns).iloc[:, 2:]         # Drop the fit and score times
    
    results = pd.melt(df, value_name='Value')                         # Melt dataframe for Value
    results.Metric = results.Metric.apply(lambda x: metrics.get(x))   # Get titles for metrics
    results.Dataset = results.Dataset.str.capitalize()                # Capitalise train and test
    
    return results


def plot_cv_results(df, metrics, model=None):
    """ Plot the score from the cross validation"""
    m = list(metrics.values())
    palette = {"Train": "lightblue", "Test": "lightcoral"}
    g = sns.catplot(data=df,
                    x='Dataset', 
                    y='Value', 
                    hue='Dataset', 
                    col='Metric',
                    palette=palette,
                    col_order=m,
                    order=['Train', 'Test'],
                    kind="box", 
                    col_wrap=3,
                    sharey=False,
                    height=4, aspect=1.2)
    
    df = df.groupby(['Metric', 'Dataset']).Value.mean().unstack().loc[m]
    for i, ax in enumerate(g.axes.flat):
        s = f"Train: {df.loc[m[i], 'Train']:>7.4f}\nTest:  {df.loc[m[i], 'Test'] :>7.4f}"
        ax.text(0.75, 0.85, s, fontsize=10, transform=ax.transAxes, 
                bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))
        
    g.fig.suptitle(model, fontsize=16)
    g.fig.subplots_adjust(top=.9)
    

def run_cross_validation(estimator, X, y, metrics, cv=3, verbose=False, model=None):
    """"""
    scores = cross_validate(estimator=estimator,
                            X=X, 
                            y=np.ravel(y),
                            scoring=list(metrics.keys()),
                            cv=cv,
                            return_train_score=True,
                            n_jobs=-1,
                            verbose=verbose)
    
    results = stack_cv_results(scores, metrics)
    plot_cv_results(results, metrics, model)
    
    return results
    
    
    