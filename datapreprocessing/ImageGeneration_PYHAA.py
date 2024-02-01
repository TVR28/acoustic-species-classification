from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *
from PyHa.annotation_post_processing import *
import pandas as pd
import os
import librosa.display
import numpy as np
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt

isolation_parameters = {
     "model" :          "microfaune",
     "technique" :       "steinberg",
     "threshold_type" :  "median",
     "threshold_const" : 2.0,
     "threshold_min" :   0.0,
     "window_size" :     2.0,
     "chunk_size" :      5.0,
     "verbose"     :     True
}


image_path = "./kaggle_dataset/"
path_folders = os.listdir(image_path)

for folder in path_folders:
    try:
        automated_df = generate_automated_labels(image_path + folder + '/',isolation_parameters);
        df = annotation_chunker_no_duplicates(automated_df, 5)
    except Exception as e:
        print("Failed ", folder)
        continue
    print("Annotation done for ", folder)
    for index, row in df.iterrows():
        file_name = row["IN FILE"]
        folder_name = row["FOLDER"]
        
        path = folder_name+file_name
        offset = float(row["OFFSET"])
        duratiom = float(row["DURATION"])
        normalized_sample_rate=44100
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(path, offset=offset, duration=duratiom, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except BaseException:
            print("Failed to load" + path)
                
        # Resample the audio if it isn't the normalized sample rate
        try:
            if SAMPLE_RATE != normalized_sample_rate:
                rate_ratio = normalized_sample_rate / SAMPLE_RATE
                SIGNAL = scipy_signal.resample(
                SIGNAL, int(len(SIGNAL) * rate_ratio))
                SAMPLE_RATE = normalized_sample_rate
        except Exception as e:
            print("Failed to Downsample" + path + str(e))
                
        # convert stereo to mono if needed
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2

        D = np.abs(librosa.stft(SIGNAL))**2
        S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate)
        
        # save s directly -> as an image preferably
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=normalized_sample_rate,fmax=normalized_sample_rate//2, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        folder_name = folder_name[:-1]
        bird_name = folder_name[13:]
        bird_path = "./IMAGES_HighPassFilter/"+bird_name+"/"
        isExist = os.path.exists(bird_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(bird_path)
        plt.savefig(bird_path+str(index)+".png")
        plt.close()
print(automated_df)