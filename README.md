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
. OSU-CS467-Project-Top-n-Music-Genre-Classification-Neural-Network
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
│   ├── run.sh (the bash script to run the experiments with different batch sizes & learning rates)
│   ├── run_fea.sh (the bash script to run the experiments with different features)
├── Train_Log
├── Trained_Model (the place to store the trained model if adding argument --save_model)
├── data_overview.ipynb (the Jupyter Notebook with the exploration of data (GTZAN Genre Collection) & getting familiar with using Librosa)
├── data_preprocess.py (used to preprocess the GTZAN raw audio data as the image data of different features)
├── GUI.py (our main program)
├── model.py (model architecture)
├── split_data.py (used to split the data into sub-datasets like training, validation, and testing sets)
└── train.py (used to train models with different settings)
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
![image](https://user-images.githubusercontent.com/57364511/182648163-0fb6659b-59ce-49b3-865e-481de05a5f52.png)

**MFCC**
![image](https://user-images.githubusercontent.com/57364511/182648197-5dc37d29-2460-4a68-bf62-b642aec0bbd1.png)

# How to Run the Program

Follow the procedures described below:

(1.) Git Clone this Repo & Set up the environment --> (2.) Download GTZAN data (saved as the file structure described above) --> (3.) Preprocess the raw GTZAN audio data --> (4.) Train the model with the provided parameters --> (5.) Run the GUI program

--------------------

**(1.) Git Clone this Repo & Set up the environment:**
```
$ git clone https://github.com/Harrycywu/OSU-CS467-Project-Top-n-Music-Genre-Classification-Neural-Network.git
```

**(2.) & (3.) Data download & Preprocessing:** Please refer to the "Data Download & Preprocessing" part above.

**(4.) Train the model using the following command:**
```
$ python train.py --feature_type=mel_spectrogram --save_model
```
Note: This model has about **62.67%** accuracy on the testing data.

**(5.) Run the GUI program:**

We use tkinter to develop a GUI that allows users to enter a song clip and receive a top-n list of this audio's most likely music genre. Run the **GUI.py** file as follows:
```
$ python GUI.py
```

Then, follow the instructions on the GUI:
![image](https://user-images.githubusercontent.com/57364511/183475749-b19258bf-28ea-4cf6-a95b-fccca63f8e73.png)

## How does the Program Work?

The program will first process the inputted audio file into the image file (e.g., MFCC or Mel Spectrogram). Then the trained CNN model will take this image file as input, predict the most likely music genre of this audio file, and finally show the prediction result.

![image](https://user-images.githubusercontent.com/57364511/183298567-a0acab58-80da-41ca-b53f-b9e13f6d93d6.png)

# How We Develop the CNN Model for Music Genre Classification Task

**Implementation**

First, we use Librosa and OpenCV to process the audio files. Second, we use PyTorch to build a framework that can train CNN models (used to predict a top-n list of inputted audio's most likely music genre) by using different settings, such as different batch sizes & learning rates. Here, we use Cross Entropy Loss since we are doing a classification task. Meanwhile, we use tqdm to keep track of the running progress of each part of the program (e.g., loading data & training progress).

Using our developed framework, we can train CNN models with the best accuracy of around 60~70% using the preprocessed GTZAN dataset. After getting the pre-trained model, our application allows users to select an audio file and receive a plot, created by matplotlib, showing a top-n list of this audio's most likely music genre predicted by the pre-trained CNN model.

**Model Architecture**

![image](https://user-images.githubusercontent.com/57364511/183299362-2005e345-2e4a-41ff-b4eb-b0d3282a9c84.png)

The first layer of our model is Batch Normalization Layer, which standardizes the inputs to the network. The Batch Normalization Layer is followed by a three-layer CNN with ReLU and max-pooling, and the dimensions of inputs & outputs of each layer are developed based on research and tuning. We added a Dropout layer to prevent over-fitting and flattened the data to connect to the final fully connected layers (ReLU & Softmax).

**Feature Selection**

We ran the designed CNN model with different features by running the **run_fea.sh** (batch size=32 & learning rate=0.00001). And we find that the **Mel Spectrogram** outperforms other features. Thus, we choose to preprocess the raw GTZAN data as the Mel Spectrogram to be the input for our CNN model.

The following are the loss & accuracy plots:

***Wavelet***: Accuracy: ~27%

![image](https://user-images.githubusercontent.com/57364511/183299908-adf60c41-4dd5-4e10-8e2b-c2116a60aee9.png)
![image](https://user-images.githubusercontent.com/57364511/183299794-1a768867-5bc9-4957-b1a0-9a59b11ba919.png)

***Spectrogram***: Accuracy: ~57%

![image](https://user-images.githubusercontent.com/57364511/183299922-c9e92add-b1ba-4861-91e7-fba866367265.png)
![image](https://user-images.githubusercontent.com/57364511/183299931-32ec6464-41c6-4360-9ad2-99f885a952bd.png)

***Mel Spectrogram***: Accuracy: ~59%

![image](https://user-images.githubusercontent.com/57364511/183299945-6971c418-98e1-4989-8335-e0a78c54dcb8.png)
![image](https://user-images.githubusercontent.com/57364511/183299953-391e2e7f-564a-4d2d-8215-a0be83e96ded.png)

***MFCC***: Accuracy: ~45%

![image](https://user-images.githubusercontent.com/57364511/183299959-4e7929e3-007f-462e-9bd2-76317e418871.png)
![image](https://user-images.githubusercontent.com/57364511/183299967-05fd66b0-adcb-49dd-b70c-df574cdb38a4.png)

**Model Selection**

Due to time insufficiency, we only ran the designed CNN model (feature = Mel Spectrogram) with different batch sizes & learning rates by running the **run.sh**. And we find that using **batch size = 32** & **learning rate = 0.001** outperforms other settings. Therefore, we finally chose to train the model using Mel Spectrogram as features, batch size = 32, and learning rate = 0.001, and used this pre-trained model for our application.

The loss & accuracy plot of ***batch size = 32 & learning rate = 0.001***: Accuracy: ~62.67%

![image](https://user-images.githubusercontent.com/57364511/183482699-b737513e-ba82-4fca-a89a-68b69203737c.png)
![image](https://user-images.githubusercontent.com/57364511/183482567-faa54528-a073-4bec-a857-cd58734afe4a.png)

# Future Work

(1.) Fix the bug that cannot load the audio file with mp3 format.

(2.) Do feature engineering and data augmentation for the GTZAN dataset to better train the model.

(3.) Develop the Confusion Matrix to do a deeper analysis of different models.

# References

**Paper & Journal**

(1.) G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," ***IEEE Transactions on speech and audio processing***, vol. 10, no. 5, pp. 293-302, 2002.

(2.) Hareesh Bahuleyan, "Music genre classification using machine learning techniques", 2018.

(3.) Y.-H. Cheng, P.-C. Chang and C.-N. Kuo, "Convolutional Neural Networks Approach for Music Genre Classification", ***2020 International Symposium on Computer Consumer and Control (IS3C)***, pp. 399-403, 2020.

**Article**

(1.) Parul Pandey, "Music Genre Classification with Python", https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8

(2.) Sawan Rai, "Music Genres Classification using Deep learning techniques", https://www.analyticsvidhya.com/blog/2021/06/music-genres-classification-using-deep-learning-techniques/

**GitHub Repo**

(1.) sawan16, "Genre-Classification-using-Deep-learning", https://github.com/sawan16/Genre-Classification-using-Deep-learning

(2.) CNN-MNIST Example, https://hackmd.io/@lido2370/SJMPbNnKN?type=view

(3.) HareeshBahuleyan, "music-genre-classification", https://github.com/HareeshBahuleyan/music-genre-classification
