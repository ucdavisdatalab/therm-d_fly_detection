import pandas as pd
import numpy as np

def get_matrix(path):
    ''' Turns excel sheet into np array '''
    df = pd.read_excel(path)
    ind = df[df.iloc[:,0] == 'temperature probe'].index[0]
    df = df[ind:ind + 3].dropna(axis = 1)
    df.columns = df.iloc[0]
    df = df[1:3]
    df = df.set_index('temperature probe')
    df = df.to_numpy()
    return(df)

def extropolate(X, mat):
    ''' Piecewise extroploation function where X is a distance input and the outputs gives a guess for the temperature'''
    if np.min(mat[0]) > X or np.max(mat[0]) < X:
        return(None)
    for i in range(mat.shape[1]):
        if X == mat[0,i]:
            return(mat[1,i])
        else:
            pt = np.searchsorted(mat[0], X)
            return((mat[1, pt] - mat[1, pt - 1]) / (mat[0, pt] - mat[0, pt -1]) * (X - mat[0, pt - 1]) + mat[1, pt - 1])