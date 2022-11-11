import sys
import os
import glob
import pandas as pd

dir_path = sys.argv[1]
label_idx = sorted(os.listdir(f'{dir_path}/train'))

def to_label(pharse):
    train_files = glob.glob(f'{dir_path}/{pharse}/*/*')
    train_label = []
    for f in train_files:
        l = f.split('/')[2]
        train_label.append(label_idx.index(l))
    df = pd.DataFrame({
        'path': train_files,
        'label': train_label
    })
    df.to_csv(f'{pharse}.csv', index=False)

to_label('train')
to_label('val')