import csv
import numpy as np

def tfidf(in_array,divide_by_total=False):
    out = np.zeros(in_array.shape)
    m,n = in_array.shape
    row_sum = np.sum(in_array,axis=0)
    for i in range(m):
        df = len([a for a in in_array[i,:] if a>0])
        for j in range(n):
            if in_array[i,j]>0:
                if divide_by_total:
                    tf = in_array[i,j] / row_sum[j]
                else:
                    tf = in_array[i,j]
                idf = np.log(d / df)
                out[i,j] = tf * idf
    return out

with open('NIPS_1987-2015.csv','r') as f:
    content = csv.reader(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_NONNUMERIC)
    vocab = []
    temp = []
    for row in content:
        vocab.append(row[0])
        temp.append(row[1:])

vocab = vocab[1:]
X_old = np.asarray(temp[1:])
non_zero_indices = [j for j in range(X_old.shape[1]) if np.sum(X_old[:,j]) != 0]
docs = []
for i in non_zero_indices:
    docs.append(temp[0][i])

X = X_old[:,non_zero_indices]
Xt = np.transpose(X)
t,d = X.shape
Y = tfidf(X)
