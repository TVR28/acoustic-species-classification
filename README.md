# Acoustic Species Classification

The **Acoustic Species Classification** project aims to identify bird species from audio recordings using deep learning. This approach leverages passive acoustic monitoring and multi-class classification to facilitate biodiversity assessments and conservation strategies by automating species identification over large spatial scales.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Introduction

Birds are crucial indicators of environmental changes due to their mobility and diverse habitat needs. This project aims to automate bird species identification using audio recordings, providing a scalable solution for biodiversity assessments. By employing machine learning techniques, we can improve the accuracy and efficiency of monitoring bird populations over large areas.

## Requirements

```txt
python>=3.8
torch
librosa
numpy
pandas
sklearn
PyHa
```

## Installation
```bash
git clone https://github.com/username/acoustic-species-classification
cd acoustic-species-classification
pip install -r requirements.txt
```

## Dataset

The dataset consists of audio recordings from the Xeno-canto database, featuring 16,900 recordings of 264 bird species from Kenya, Africa. The test set includes approximately 200 ten-minute soundscapes. Data preprocessing involves converting audio files into Mel spectrograms for model input.

## Model Architecture

The project utilizes the EfficientNet-B0 architecture, known for its balance between performance and computational efficiency. This model is particularly suited for image classification tasks and has been adapted here for audio spectrogram classification.

## Data Preprocessing

The audio data processing pipeline consists of several critical stages to prepare the data for model training:
Audio Preprocessing
- Audio recordings are standardized to 32 kHz sampling rate and converted to OGG format.
- Noise reduction and filtering techniques are applied to remove unwanted background interference.
- Audio signals are normalized to maintain consistent amplitude levels across the dataset.

### Feature Extraction
Raw audio is converted into mel spectrograms using Short-Time Fourier Transform3
Configuration parameters:

```python
MelSpectrogram(
    num_mel_bins=128,
    sampling_rate=32000, 
    fft_length=2048,
    sequence_stride=512,
    window="hann"
)
```
### Data Augmentation

- Noise injection: Adding gaussian noise to improve robustness.
- Pitch shifting: Randomly altering the pitch.
- Time stretching: Modifying the audio speed.
- Time and frequency masking on spectrograms.

## Training

The model is trained using a supervised learning approach with the following settings:
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy
- **Batch Size:** 10
- **Epochs:** 10
- **Learning Rate:** 0.0001


## Results

EfficientNet-B0 outperformed other models like ResNet50 and VGG16 in terms of validation accuracy and generalization capability. The final model achieved a validation accuracy of 88.51%, demonstrating its effectiveness in classifying bird species from audio data.

| Model | Initial Accuracy | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|-----------------|---------------|-----------------|-------------------|-------------------|
| ResNet50 | 73.02% | 0.17 | 0.63 | 95.00% | 85.94% |
| VGG16 | 77.08% | 0.20 | 0.77 | 94.55% | 83.14% |
| EfficientNet B0 | 78.12% | 0.10 | 0.47 | 97.65% | 88.51% |


## Future Work

Future enhancements include:
- Developing an ensemble model to improve classification accuracy.
- Implementing advanced data augmentation techniques.
- Exploring pre-training strategies on diverse bird datasets.
- Extending the model to support multi-label classification for recordings with multiple bird species.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
