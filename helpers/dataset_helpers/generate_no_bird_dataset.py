import pandas as pd
import math

# CSV file with annotations from PyHa converted into 5 second chunks
df = pd.read_csv('data_labels_264n.csv')
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
audio_files = df['IN FILE'].unique() #Get all audio files present in the dataset

columns = []
for col in df.columns:
    columns.append(col)

print(df.dtypes)
step = 5
temp_df = pd.DataFrame()
# OFFSET float64
# Label int64
# Manual ID object 

for a in audio_files:
    a_df = df.loc[df['IN FILE'] == a]
    clip_length = a_df['CLIP LENGTH'].iloc[0]
    maxstep = math.floor(clip_length//5)*5
    for i in range(0, maxstep, step):
        if (i not in a_df['OFFSET'].values):
            t_df = a_df.iloc[0,:].copy(True)
            t_df['OFFSET'] = float(i)
            t_df['MANUAL ID'] = "no bird"
            t_df['Label'] = int(264)
            t_df = t_df.transpose()
            temp_df = pd.concat([temp_df, t_df], ignore_index=True, axis= 1)

temp_df.transpose().to_csv("no_bird.csv",index=True)
