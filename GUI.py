# ---------- Graphical User Interface ----------
from multiprocessing.connection import wait
import tkinter as tk
import torch
import torch.nn.functional as F
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from model import *
from tkinter import filedialog
from tkinter import messagebox as tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------- Variables ----------
genre_map = {0: 'Blues', 1: 'Classical', 2: 'Country', 3:'Disco', 4: 'Hiphop', 5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}
file_path = ''


# ---------- Functions ----------
def select():
    global file_path
    file_path = filedialog.askopenfilename(initialdir = "/", title = "Select an audio file", filetypes=[("Audio Files", ".wav .au .mp3")])

    # Check if select a file
    if file_path:
        # Show the go button
        go_image = tk.PhotoImage(file="Images/Go_Button.png")
        go_button.configure(image=go_image)
        go_button.image = go_image

        # Show the name of the selected file
        file_name = file_path.split('/')[-1]
        entry1_text.set(file_name)
    else:
        tkMessageBox.showerror(title="Error", message="You have not chosen the audio file. Please select!")

def go():
    # Clean frame: div1
    for widget in div1.winfo_children():
        widget.destroy()

    # Check the existence of the file_path
    if not file_path:
        tkMessageBox.showerror(title="Error", message="You have not chosen the audio file. Please select!")
        return

    # Data Preprocessing -> Mel Spectrogram -> Spectrogram Image
    x, _sr = librosa.load(file_path, sr=22050)
    x_mel = librosa.feature.melspectrogram(y=x, sr=_sr, n_fft=2048, hop_length=512)
    # Convert to dB-scaled
    x_mel = librosa.power_to_db(x_mel)
    librosa.display.specshow(x_mel, sr=_sr)
    plt.savefig("Images/mel_spec.png")
    plt.close()

    # Read the image & Process as a tensor
    image = cv2.imread("Images/mel_spec.png")[...,::-1]
    # Resize the image -> (256, 256, 3)
    resized_image = cv2.resize(image, (256, 256))
    # Change to tensor
    image_tensor = torch.tensor(resized_image, dtype = torch.float32)

    # Load the trained model & Put the input data into our trained model to make predictions
    model = CNNModel(kernel_size=3, stride=1, dropout_rate=0.5, pooling_stride=2, padding=0)
    model.load_state_dict(torch.load("Trained_Model/GTZAN_CNN.pt"))
    model.eval()
    # Change data shape
    data = image_tensor.view((-1, 3, 256, 256))
    # Make predictions for this data
    prediction = model(data)
    # Apply a softmax function to get probabilities
    prob = F.softmax(prediction.data, dim=1)
    prob = prob.numpy()[0]

    # Get the result
    result = {}
    for num in range(10):
        # Genre: probability
        result[genre_map[num]] = prob[num]

    # Sort in descending order
    sorted_result = sorted(result.items(), key=lambda x: x[1])

    # Store the results
    genres = []
    probabilities = []
    for res in sorted_result:
        genres.append(res[0])
        probabilities.append(float("{:.2f}".format(res[1]*100)))

    # Show the figure result
    fig = plt.figure(figsize=(6, 4), dpi=75)
    plt.barh(genres, probabilities)
    plt.title("Most Likely Music Genre")
    plt.xlabel('Probability(%)')
    for i, val in enumerate(probabilities):
        plt.text(val + 3, i, str(val)+'%', color = 'black', fontweight = 'bold')
    canvas = FigureCanvasTkAgg(fig, master=div1)
    canvas.get_tk_widget().grid()
    plt.close()

    # Reset
    go_button.configure(image=button_image)
    go_button.image = button_image
    entry1_text.set('')

def quit_message():
    res = tkMessageBox.askyesno(title="Exit", message="Are you sure you want to exit?")

    # If receiving yes, then leave the program
    if res:
        window.quit()


# ---------- GUI ----------
window = tk.Tk()
window.title("CS 467 Online Capstone Project")

label1 = tk.Label(window, text = "Selected Audio File: ", font=("Futura", 16))
label1.grid(row = 0, column = 0)

entry1_text = tk.StringVar()
entry1 = tk.Entry(window, textvariable=entry1_text)
entry1.grid(row = 0, column = 1)

# Button to select the audio file
select_button = tk.Button(window, text='Select', command=select)
select_button.grid(row = 0, column = 2)

# Button to show the instructions & start predicting
button_image = tk.PhotoImage(file="Images/Select_Button.png")
go_button = tk.Button(window, image=button_image, command=go)
go_button.grid(row = 1, column = 0, columnspan=3)

# Division for results
div1 = tk.Frame(window,  width=600, height=450, bg='white')
div1.grid(row = 2, column = 0, columnspan=3)

# Exit message
window.protocol("WM_DELETE_WINDOW", quit_message)

window.mainloop()
