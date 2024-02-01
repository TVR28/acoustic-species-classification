
import librosa.display
import numpy as np
import scipy.signal as scipy_signal
import matplotlib.pyplot as plt
import numpy as np
import os


path = r"C:\Users\aahire\Documents\CSS581\Midterm\acoustic-species-identification\datapreprocessing\kaggle_dataset\train_audio"
path_folders = os.listdir(path)
completed_path = r"C:\Users\aahire\Documents\CSS581\Midterm\acoustic-species-identification\datapreprocessing\kaggle_dataset\train_audio\Without_any_preprocessing"
completed_path_folders = os.listdir(completed_path)
print(path_folders)
for folder in path_folders:
    if folder not in completed_path:
        # Construct the full path for the new folder
        new_folder_path = os.path.join(completed_path, folder)

        # Create the new folder
        os.makedirs(new_folder_path)

        print(f"Folder '{folder}' created under '{completed_path}'.")
    else:
        print(f"Folder '{folder}' already exists under '{completed_path}'.")
        continue

    for file in os.listdir(path + "\\" +folder):
     
        normalized_sample_rate=44100
        try:
            SIGNAL, SAMPLE_RATE = librosa.load(path + "\\" + folder + "\\" + file, offset=0, duration=5, sr=None, mono=True)
            SIGNAL = SIGNAL * 32768
        except BaseException as e:
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
        # Might want to compare to just taking the first set of data.
        if len(SIGNAL.shape) == 2:
            SIGNAL = SIGNAL.sum(axis=1) / 2
        SIGNAL = librosa.effects.pitch_shift(SIGNAL, sr = normalized_sample_rate, n_steps=4)
        D = np.abs(librosa.stft(SIGNAL))**2
        S = librosa.feature.melspectrogram(S=D, sr=normalized_sample_rate,fmax=normalized_sample_rate/2)
        # save s directly -> as an image preferably
        fig, ax = plt.subplots()


        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=normalized_sample_rate,fmax=normalized_sample_rate/2, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        plt.savefig(completed_path + "\\" + folder + "\\" + file[:-4] + ".png")
        plt.close()
