import pandas as pd
import torch

def get_datafile():
    motion = 'walk1_subject1_shortened.csv'
    return 'LAFAN1_Retargeting_Dataset/g1/' + motion

def data_to_tensor(filepath):
    df = pd.read_csv(filepath)
    for i, row in df.iteritems():
        data = torch.from_numpy(row[1].to_numpy()).float()
    