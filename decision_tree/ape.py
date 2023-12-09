#pandas
#scikit-learn 
#gensim 
#openpyxl

from sklearn import tree 
import pandas as pd 
import matplotlib.pyplot as plt
import math

baseDir = '/home/martin/Documents/projects/machine_learning_UTSA/decision_tree/'
fname = 'DT_RF_Example_Data.xlsx'

def attachDummies(df,col,retainColumnsIfMoreThanTwo=True):
    '''
    Takes in a pandas dataframe and returns
    a dataframe with the passed in column
    converted to a binary column for the unique values.
    If the unique values is 2, then one of the columns
    is dropped.
    '''
    uniqueVals = df[col].unique()
    df2 = pd.get_dummies(df[col])
    if len(uniqueVals) == 2:
        colToAdd = uniqueVals[0]
        df = pd.concat((df,df2[colToAdd]),axis=1)
        colToAdd = [colToAdd]
    elif len(uniqueVals) > 2 and not retainColumnsIfMoreThanTwo:
        colToAdd = uniqueVals[1:]
        df = pd.concat((df,df2[colToAdd]),axis=1)
    df.drop(col,axis=1,inplace=True)
    for addedCol in colToAdd:
        df.rename(columns={addedCol:f"{col}_is{addedCol}"},inplace=True)
    return df

def getEntropy(df,col):
    '''
    This function calculates the entropy of the passed in col (column)
    for the passed in df (dataframe/pandas)
    
    To do this you need to:
    1 - Identify all of the unique states (values) - for dataframes we can all df[col].unique()
    2 - For each unique value, you need to calculate it's proportion (all proportions should sum to 1)
    3 - To obtain entropy of the total system, do the following
    4 - Multiply the proportion by log2 proportion
    5 - Do the above (step 4), for each of the unique values (states)
    6 - Sum the results of the above two steps (4,5)
    7 - Negate that sum
    '''
    totalCount = df.shape[0]
    uniqueVals = df[col].unique()
    entropy = 0 
    for uniqueVal in uniqueVals:
        valCount = len(df[(df[col]==uniqueVal)])
        entropy += (valCount/totalCount)*math.log2((valCount/totalCount))
    entropy *= -1
    return entropy 

def getInformationGain(df, splitcol, target): 
    '''
    This function returns the information gain value that would be obtained
    by splitting on the passed in splitcol (the column to use as a decision branch)
    This function necessarily calls the getEntropy function
    
    The bigger the returned value of getInformationGain the better. 
    You want to reduce entropy. And larger numbers returned from this fuction indicate a bigger reduction in entropy
    '''
    startingEntropy = getEntropy(df,target)
    uniqueVals = df[splitcol].unique()
    splits = []
    for uniqueVal in uniqueVals:
        split = df[df[splitcol] == uniqueVal]
        splits.append(split)
    
    newEntropy = 0
    for split in splits:
        prob = (split.shape[0] / df.shape[0])
        newEntropy += prob * getEntropy(split, target)
    infoGain = startingEntropy - newEntropy
    return infoGain

data = pd.read_excel(f"{baseDir}{fname}")
ycol = 'Target'

tmp = getEntropy(data,ycol)
print(f"Entropy of data is {tmp}")

for col in data.columns:
    if col != ycol:
        infoGain = getInformationGain(data, col, ycol)
        print(f"{col} info gain is {infoGain}")

# allcols = data.columns
# for col in allcols:
#     if col == ycol:
#         continue
#     data = attachDummies(data, col)

# xcols = []
# for col in data.columns:
#     xcols.append(col)

# data.to_csv(f"{baseDir}/output/remapped.csv")

# X, y = data[xcols], data[ycol]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)

# plt.figure()
# tree.plot_tree(clf,filled=True)
# plt.savefig(f'{baseDir}/output/tree.png',bbox_inches = "tight")
