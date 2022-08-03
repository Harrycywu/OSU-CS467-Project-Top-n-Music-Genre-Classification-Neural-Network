import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('--feature_type', default='wavelet', help="Meaningful features extracted from audio files: wavelet, spectrogram, mel_spectrogram, mfcc, etc.")
parser.add_argument('--sr', default=22050, type=int, help="Sampling rate")
parser.add_argument('--n_fft', default=2048, type=int, help="Length of the FFT window (Frame)")
parser.add_argument('--hop_length', default=512, type=int, help="Number of samples between successive frames")
parser.add_argument('--not_image', action='store_true', default=False, help="Whether process the data as image files?")


# Data Preprocessing
def data_preprocess(feature_type, sr, n_fft, hop_length, not_image):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    data_path = 'Datasets/GTZAN/genres/'
    save_path = 'Datasets/GTZAN/processed/'
    file_type = '.au'
    features = []
    labels = []

    # Process audio files of each music genre (total=10(genre)*100(files)=1000)
    progress = tqdm(total=1000, desc="Processing Data")

    for genre in genres:
        for num in range(100):
            file_path = data_path + genre + '/' + genre + '.000' + f'{num:02}' + file_type

            # Get the audio time series data
            x, _sr = librosa.load(file_path, sr=sr)

            # Check which type of feature to preprocess & Do the corresponding preprocessing
            # Wavelet
            if feature_type == 'wavelet':
                feature = x

            # Spectrogram
            elif feature_type == 'spectrogram':
                # Short Time Fourier Transform (STFT) returns a complex-valued matrix D: np.abs(D(f,t)) -> Amplitude; np.angle(D(f,t)) -> Phase
                X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

                # Convert an amplitude spectrogram to dB-scaled spectrogram
                Xdb = librosa.amplitude_to_db(np.abs(X))
                feature = Xdb

            # Mel Spectrogram
            elif feature_type == 'mel_spectrogram':
                feature = librosa.feature.melspectrogram(y=x, sr=_sr, n_fft=n_fft, hop_length=hop_length)

                # Convert to dB-scaled
                feature = librosa.power_to_db(feature)

            # MFCCs
            elif feature_type == 'mfcc':
                feature = librosa.feature.mfcc(y=x, sr=_sr, n_fft=n_fft, hop_length=hop_length)

            # Check if saving as images
            if not not_image:
                # Check if wavelet
                if feature_type == 'wavelet':
                    librosa.display.waveshow(feature, sr=_sr)

                # Show spectrogram
                else:
                    librosa.display.specshow(feature, sr=_sr)

                # Save the images generated
                plt.savefig(save_path + feature_type + '/' + genre + '_' + f'{num:02}' + '.png')
                plt.close()

            # If not, store feature & corresponding label data
            else:
                # Store the feature got & corresponding label
                features.append(feature)
                labels.append(genre)

            # Update the progress
            progress.update(1)

    # If not processing as images, return feature & corresponding label data
    if not_image:
        return features, labels


if __name__ == '__main__':
    # Load the parameters
    args = parser.parse_args()

    # Data preprocessing
    # Process the data as image files and store the files in the processed folder
    data_preprocess(args.feature_type, args.sr, args.n_fft, args.hop_length, args.not_image)

    # If not processing as images: add argument --not_image
    # feas, labels = data_preprocess(args.feature_type, args.sr, args.n_fft, args.hop_length, args.not_image)
    # print(feas[0].shape, labels[0])    # Check data
