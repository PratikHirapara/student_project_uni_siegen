import pandas as pd
import numpy as np

def preprocessing():

    df = pd.DataFrame()

    # List of all csv files
    Data = ['move object in fixed position-002.csv',
                'reach fixed object both hand.csv',
                'reach fixed object both rhand-001.csv',
                'reach where the other hand is resting-001.csv',
                'turning 180.csv']


    # Reading all the files and appending on data frame
    for i in range(len(Data)):
        sub_data = pd.read_csv('E:/project_data/{}'.format((Data[i])))
        # store sub data
        df = df.append(sub_data)

    # adding no. of blocks according to label
    df['act_block'] = ((df['labels'].shift(1) != df['labels'])).astype(int).cumsum()
    numblocks = df['act_block'].max()

    # frame preparation
    Fs = 240
    frame_size = 240*1

    windows = []
    for block in range(1,numblocks+1):
        a = np.array(df[df['act_block'] == block])
        
        i = 0
        while True:
            if len(a[i:i+frame_size,:])<240:
                break
            windows.append(a[i:i+frame_size,:])
            i = i + frame_size

    windows = np.array(windows)
    windows = windows[:, :, :-1]
    #windows.shape  (2771, 240, 153)

    X = windows[:,:,:-1]
    y = windows[:,:,-1]
    Y = np.array([y[i][0] for i in range(len(windows))])

    return X,Y

