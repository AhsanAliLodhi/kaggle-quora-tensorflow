import numpy as np
import pandas as pd


def class_distribution_analysis(labels,class_indices):
    c = sorted(class_indices.items(), key=lambda kv: kv[1])
    c = [item[0] for item in c ]
    df = pd.DataFrame(labels,columns=c)
    count = df.sum()
    percent = count/count.sum()*100
    weight = count.sum()/count
    weight = weight/weight.min()
    weight = weight/weight.sum()
    final = pd.DataFrame({'class_name':count.index, 'count':count.values, 'percent':percent.values, 'proposed_class_weights': weight.values})
    final = final.append(pd.Series({'class_name':'Total'}).append(final.sum(numeric_only=True)), ignore_index=True)
    return final

def batch_iterator(data,labels,batch_size,epochs,shuffle = True):
    s_data = np.array(data)
    if labels is not None:
        s_labels = np.array(labels)
    num_records = len(data)
    batches = int((len(data) - 1) / batch_size) + 1
    
    indices = np.random.permutation(np.arange(num_records))
    if shuffle:
        s_data = s_data[indices]
        if labels is not None:
            s_labels = s_labels[indices]
    
    for epoch in range(epochs):
        for batch in range(batches):
            start = batch * batch_size
            end = min(batch_size * (batch + 1), num_records)
            if labels is not None:
                yield s_data[start:end],s_labels[start:end],start,end
            else:
                yield s_data[start:end],start,end

def confusion_matrix(true_labels, preds, classes):
    cf_mat = pd.DataFrame(0, index=np.arange(classes), columns=np.arange(classes))
    i = 0
    for label in true_labels:
        true_class = np.nonzero(label)[0][0]
        pred_class = np.nonzero(preds[i])[0][0]
        cf_mat[true_class][pred_class]+=1
        i+=1
    tp = pd.Series(np.diag(cf_mat), index=[cf_mat.index, cf_mat.columns])
    fp = cf_mat.sum(0).subtract(tp,1)
    fn = cf_mat.sum(1).subtract(tp,1)
    p = tp.divide(tp.add(fp))
    r = tp.divide(tp.add(fn))
    f = (2*p.multiply(r).divide(p.add(r))).mean()
    return cf_mat,f

def one_hot_encode(size,index):
    vec = np.zeros(size,dtype=int)
    vec[index] = 1
    return vec

def read_data(dir,x_col,y_col=None,sample_percent = 1):
    data = pd.read_csv(dir)
    X = data[x_col]
    if y_col is not None:
        classes =  data[y_col].unique()
        y = [one_hot_encode(len(classes),label) for label in data[y_col]]
        y = np.array(y)
    X = np.array(X)
    if sample_percent < 1:
        idx = np.random.randint(len(X),size = int(len(X)*sample_percent))
        X = X[idx]
        if y_col is not None:
            y = y[idx]
        print("After sampling ")
    if y_col is not None:
        print("Data shape ",X.shape,y.shape)
        return X,y
    else:
        return X

def split_data(X,y,split):
    # shuffle data before splitting
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    #split by selecting last rows
    split_index = -1 * int(split * float(len(y)))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    print("After splitting ",X_train.shape,y_train.shape,X_val.shape,y_val.shape)
    return X_train,y_train,X_val,y_val