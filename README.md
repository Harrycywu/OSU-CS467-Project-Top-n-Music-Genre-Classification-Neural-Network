# OSU CS 467 Online Capstone Project

**Project Topic**: Top-n Music Genre Classification Neural Network

**Term**: Summer 2022

**Team Members**: Cheng Ying Wu & Sophie Zhao

**Project Description:**

We developed a program that allows users to enter a song clip and receive a top-n list of this audio's most likely music genre sorted by probability in descending order. The prediction is made by the pre-trained Convolutional Neural Network (CNN) model designed by our project team.

# Environment Setup

We use PyTorch to develop our CNN model:

* Python 3.8
* PyTorch 1.12.0

Other software libraries:

* librosa
* matplotlib
* numpy
* opencv (cv2)
* tkinter
* tqdm

# Program File Structure

Please follow the file structure below:
```
. 
├── Datasets
│   ├── GTZAN
│   │   ├── genres (10 music genre files with 100 audio files in each folder)
│   │   │   ├── blues
│   │   │   │   ├── blues.00000.au
│   │   │   │   ├── blues.00001.au
│   │   │   │   ├── ...
│   │   │   ├── classical
│   │   │   ├── ...
│   │   ├── processed (must create the following 4 empty folders)
│   │   │   ├── mel_spectrogram
│   │   │   ├── mfcc
│   │   │   ├── spectrogram
│   │   │   ├── wavelet 
├── Images
├── Results
├── Run_Scripts
│   ├── run.sh
│   ├── run_fea.sh
├── Train_Log
├── Trained_Model
├── data_overview.ipynb
├── data_preprocess.py
├── GUI.py
├── model.py
├── split_data.py
└── train.py
```

# Dataset

We use the **GTZAN Genre Collection**, created by Tzanetakis and Cook ("Musical genre classification of audio signals," *IEEE Transactions on speech and audio processing*, vol. 10, no. 5, pp. 293-302, 2002), as our dataset.
* GTZAN Genre Collection: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

We have done some simple data exploration for this dataset and gotten familiar with using Librosa to process these audio files. 
* Refer to **data_overview.ipynb** file

GTZAN dataset comprises 10 genres, including Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, and Rock. Each genre contains 100 songs, and each audio file is 30 seconds long in WAVE format with 22,050 Hz.
![image](https://user-images.githubusercontent.com/57364511/182638427-71315f60-ff9f-46f1-8c9e-5d186c26c053.png)

## Data Download & Preprocessing

**Data Download**: Go to the GTZAN Genre Collection website on Kaggle provided above (Dataset section) and download the zip file. Unzip the file and make the unzipped files named and structured as the **Program File Structure** described above.

**Data Preprocessing**

Execute the **data_preprocess.py** with specified feature type argument as follows:

Note: There are four types of features, including wavelet, spectrogram, mel_spectrogram, mfcc.
```
$ python data_preprocess.py --feature_type=mel_spectrogram
```
After executing, you will get the corresponding image files.

Here are the sample processed image files of each music genre and feature type:

**Wavelet**
![image](https://user-images.githubusercontent.com/57364511/182647816-9a54302b-93e7-416a-9c65-43b6d25ac216.png)

**Spectrogram**
![image](https://user-images.githubusercontent.com/57364511/182648113-3222068d-81fd-466c-9304-e59f89943e44.png)

**Mel Spectrogram**
![image](https://user-images.githubusercontent.com/57364511/182648197-5dc37d29-2460-4a68-bf62-b642aec0bbd1.png)

**MFCC**
![image](https://user-images.githubusercontent.com/57364511/182648163-0fb6659b-59ce-49b3-865e-481de05a5f52.png)

# How to Run the Program

We use tkinter to develop a GUI that allows users to enter a song clip and receive a top-n list of this audio's most likely music genre. Run the **GUI.py** file as follows:
```
$ python GUI.py
```

Then, follow the instructions on the GUI:
![image](https://user-images.githubusercontent.com/57364511/182654387-d7eeff9a-4d51-40bb-9947-65e04bd5ba00.png)

## How does the Program Work?

We will first process the inputted audio file into the image file (e.g., MFCC), and then the trained CNN model will take this image file as input and predict the most likely music genre of this audio file.
![image](https://user-images.githubusercontent.com/57364511/182657103-1be772fe-d78a-41e0-accf-ddb399997dc9.png)

# How We Develop the CNN Model for Music Genre Classification Task

**Model Architecture**

**Feature Selection**

**Model Selection**

# References

