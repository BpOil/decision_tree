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

tmp = getEntropy(data,ycol)
print(tmp)

tmp1 = getInformationGain(data,'Heady','Target')
print(tmp1)