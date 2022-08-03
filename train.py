import torch
import argparse
import logging
import time
import cv2
import matplotlib.pyplot as plt
from split_data import *
from model import *
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader


# Global variables
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
genre_map = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
val_key_map = {0: 'blues', 1: 'classical', 2: 'country', 3:'disco', 4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, type=int, help="Training ID")
parser.add_argument('--no_validation', action='store_true', default=False, help="Whether using a validation set?")
parser.add_argument('--random_seed', default=1, type=int, help="Random Seed")
parser.add_argument('--feature_type', default='mfcc', help="Meaningful features extracted from audio files: wavelet, spectrogram, mel_spectrogram, mfcc, etc.")
parser.add_argument('--image_size', default=256, type=int, help="Resize images to?")
parser.add_argument('--model', default='CNN', help="Which model? CNN")
parser.add_argument('--kernel_size', default=3, type=int, help="Kernel size of the CNN model")
parser.add_argument('--stride', default=1, type=int, help="Stride length of the CNN model")
parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout Rate")
parser.add_argument('--pooling_stride', default=2, type=int, help="Stride in pooling layer")
parser.add_argument('--padding', default=0, type=int, help="Padding in CNN model")
parser.add_argument('--lr', default=0.001, type=float, help="Learning Rate")
parser.add_argument('--n_iters', default=30000, type=int, help="Number of Iterations")
parser.add_argument('--batch_size', default=32, type=int, help="Batch Size")
parser.add_argument('--save_model', action='store_true', default=False, help='Whether saving the current model')
parser.add_argument('--device', default='cpu', help="CUDA_DEVICE")


# Setup a PyTorch Dataset for GTZAN
class GTZANDataset(Dataset):
    def __init__(self, features, labels, device):
        self._features = features
        self._labels = labels
        self._device = device

    def __getitem__(self, index):
        # Normalization: RGB (0~256) -> 0~1
        x = self._features[index].astype('float32') / 255
        x = torch.tensor(x, dtype = torch.float32).to(self._device)
        y = torch.tensor(self._labels[index], dtype = torch.int64).to(self._device)
        return x, y

    def __len__(self):
        return len(self._features)


def load_data(is_training, data, feature_type, image_size, batch_size, device='cpu'):
    if is_training:
        desc = "Loading Training Data"
    else:
        desc = "Loading Testing Data"

    # data: [[image, genre], ..., [image, genre]]
    # Processed data path
    data_path = "Datasets/GTZAN/processed/"

    # Read the corresponding processed data
    data_path = data_path + feature_type + '/'
    data_type = '.png'

    # Get & Store the data
    features = []
    labels = []
    progress = tqdm(total=len(data), desc=desc)

    for image_name, genre in data:
        # Specify the image path
        image_path = data_path + image_name + data_type

        # Read images using cv2.imread: row (height) x column (width) x color (3 -> BGR)
        # [...,::-1]: Convert BGR to RGB
        # Original image dimension: (480, 640, 3)
        image = cv2.imread(image_path)[...,::-1]

        # Resize the image -> (256, 256, 3)
        resized_image = cv2.resize(image, (image_size, image_size))

        # Add the processed image data to features & labels
        features.append(resized_image)

        # Encode the labels by using genre_map
        label = genre_map[genre]
        labels.append(label)

        # Update the progress
        progress.update(1)

    # Call PyTorch Data Loader
    dataset = GTZANDataset(features=features, labels=labels, device=device)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=True if is_training else False)
    return iterator


def train(model, train_data, loss_func, optimizer, input_shape):
    model.train()
    
    # Record the total loss
    total_loss = 0

    # Used to calculate the accuracy
    correct = 0
    total = 0

    # Set the progress bar
    t = trange(len(train_data), desc='Training')
    for index, (data, label) in zip(t, train_data):
        # Change data shape
        data = data.view(input_shape)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        prediction = model(data)

        # Calculate the loss and its gradients
        loss = loss_func(prediction, label)
        loss.backward()

        # Update parameters
        optimizer.step()

        # Calculate the performance: accuracy
        # Get the predicted class from the maximum value
        predicted_class = torch.max(prediction.data, dim=1)[1]

        # Print Information for Debugging
        # print("Predicted Probabilities: {}, Predicted Genre: {}, Label: {}".format(prediction.data, val_key_map[predicted_class.item()], val_key_map[label.item()]))

        # Get total number of labels
        total += len(label)

        # Calculate the total correct predictions
        correct += (predicted_class == label).float().sum()

        # Record total loss
        loss = loss.item()
        total_loss += loss

        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(total_loss / total))
    return total_loss, correct, total


def evaluate(model, test_data, loss_func, input_shape):
    model.eval()

    # Record the total loss
    total_loss = 0

    # Used to calculate the accuracy
    correct = 0
    total = 0

    t = trange(len(test_data), desc='Evaluating')
    for index, (data, label) in zip(t, test_data):
        # Change data shape
        data = data.view(input_shape)

        # Make predictions for this batch
        prediction = model(data)

        # Calculate the loss and its gradients
        loss = loss_func(prediction, label)
        loss = loss.item()
        total_loss += loss

        # Calculate the performance
        # Get the predicted class from the maximum value
        predicted_class = torch.max(prediction.data, dim=1)[1]

        # Get total number of labels
        total += len(label)

        # Calculate the total correct predictions
        correct += (predicted_class == label).float().sum()
    return total_loss, correct, total


def visualization(num_epochs, train_losses, train_accuracies, val_losses, val_accuracies, test_losses, test_accuracies, id):
    # Loss Plot
    plt.plot(range(num_epochs), train_losses, 'b-', label="Training Loss")
    if val_losses:
        plt.plot(range(num_epochs), val_losses, 'g-', label="Validation Loss")
    plt.plot(range(num_epochs), test_losses, 'r-', label="Testing Loss")
    plt.title("Loss Plot")
    plt.xlabel("Number of Epochs")
    plt.ylabel('Loss')
    plt.legend()
    # Save the image
    plt.savefig("Results/Loss_plot_{}.png".format(id))
    # Clear the plot
    plt.clf()

    # Performance (Accuracy) Plot
    plt.plot(range(num_epochs), train_accuracies, 'b-', label="Training Accuracy")
    if val_accuracies:
        plt.plot(range(num_epochs), val_accuracies, 'g-', label="Validation Accuracy")
    plt.plot(range(num_epochs), test_accuracies, 'r-', label="Testing Accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Number of Epochs")
    plt.ylabel('Accuracy(%)')
    plt.legend()
    # Save the image
    plt.savefig("Results/Acc_plot_{}.png".format(id))


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__ == '__main__':
    # Set the timer
    start_time = time.time()

    # Load the parameters
    args = parser.parse_args()

    # Set the logger
    set_logger("Train_Log/train.log")
    
    # Log the information about the current training setting
    logging.info("{}. {} Model: feature type={}, lr={}, batch size={}, random seed={}".format(args.id, args.model, args.feature_type, args.lr, args.batch_size, args.random_seed))

    # ---------- Get Data ----------
    # Split the data first
    # Not using the validation set
    if args.no_validation:
        training_set, validation_set, testing_set = split_data(False, args.random_seed)
    # Default to use the validation set
    else:
        training_set, validation_set, testing_set = split_data(True, args.random_seed)

    # Call the load data function to load the corresponding data
    train_data = load_data(True, training_set, args.feature_type, args.image_size, args.batch_size)
    if not args.no_validation:
        val_data = load_data(False, validation_set, args.feature_type, args.image_size, args.batch_size)
    test_data = load_data(False, testing_set, args.feature_type, args.image_size, args.batch_size)

    # ---------- Training & Evaluation ----------
    # Calculate the number of epochs
    num_epochs = args.n_iters / (len(training_set) / args.batch_size)
    num_epochs = int(num_epochs)

    # Get the model & Specify other parameters
    if args.model == 'CNN':
        model = CNNModel(kernel_size=args.kernel_size, stride=args.stride, dropout_rate=args.dropout_rate, pooling_stride=args.pooling_stride, padding=args.padding).to(args.device)
    # print(model)

    loss_func = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    input_shape = (-1, 3, args.image_size, args.image_size)

    # Store the loss & performance metrics
    train_losses = []
    train_accuracies = []
    if not args.no_validation:
        val_losses = []
        val_accuracies = []
    test_losses = []
    test_accuracies = []

    best_epoch = 0
    best_test_acc = 0.0
    if not args.no_validation:
        best_val_acc = 0.0
    best_model = None

    # Used to perform the early stopping
    early_stop = 0
    last_epoch = 0

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}:".format(epoch, num_epochs))

        # ----- Training -----
        train_loss, train_correct, train_total = train(model, train_data, loss_func, optimizer, input_shape)

        # Store train_acc / epoch
        train_accuracy = 100 * train_correct / float(train_total)
        train_accuracies.append(train_accuracy)

        # Store loss / epoch
        train_losses.append(train_loss / train_total)

        # ----- Validation & Testing -----
        if not args.no_validation:
            val_loss, val_correct, val_total = evaluate(model, val_data, loss_func, input_shape)
            test_loss, test_correct, test_total = evaluate(model, test_data, loss_func, input_shape)

            # Store accuracy / epoch
            val_accuracy = 100 * val_correct / float(val_total)
            val_accuracies.append(val_accuracy)
            test_accuracy = 100 * test_correct / float(test_total)
            test_accuracies.append(test_accuracy)

            # Store loss / epoch
            val_losses.append(val_loss / val_total)
            test_losses.append(test_loss / test_total)
            print("Train Epoch {}/{}: Train_Loss: {:.6f} Train_Acc: {:.2f}% Val_Loss: {:.6f} Val_Acc: {:.2f}% Test_Loss: {:.6f} Test_Acc: {:.2f}%".format(epoch, num_epochs, train_loss / train_total, train_accuracy, val_loss / val_total, val_accuracy, test_loss / test_total, test_accuracy))

            # Record the new best accuracy & epoch & model
            # Use the validation set to choose the best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch
                best_model = model
                best_test_acc = test_accuracy

                # Reset the early stopping
                early_stop = 0

            last_epoch = epoch

            # Check if early stopping
            early_stop += 1
            if early_stop >= 50:
                print("Early Stopping!")
                break

        # ----- Testing -----
        else:
            test_loss, test_correct, test_total = evaluate(model, test_data, loss_func, input_shape)

            # Store accuracy / epoch
            test_accuracy = 100 * test_correct / float(test_total)
            test_accuracies.append(test_accuracy)

            # Store loss / epoch
            test_losses.append(test_loss / test_total)
            print("Train Epoch {}/{}: Train_Loss: {:.6f} Train_Acc: {:.2f}% Test_Loss: {:.6f} Test_Acc: {:.2f}%".format(epoch, num_epochs, train_loss / train_total, train_accuracy, test_loss / test_total, test_accuracy))

            # Record the new best accuracy & epoch & model
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                best_epoch = epoch
                best_model = model

                # Reset the early stopping
                early_stop = 0

            last_epoch = epoch

            # Check if early stopping
            early_stop += 1
            if early_stop >= 50:
                print("Early Stopping!")
                break

    # Record the time used
    time_elapsed = time.time() - start_time

    # Log the result: best testing accuracy
    logging.info("Result: Best Accuracy = {:.2f}%, epoch={}/{}, time elapsed={:.2f}(s)".format(best_test_acc, best_epoch, last_epoch, time_elapsed))
    logging.info("#"*50)

    # Whether saving the model
    if args.save_model:
        torch.save(best_model.state_dict(), "Trained_Model/GTZAN_{}.pt".format(args.model))

    # Visualize the losses & performance
    if not args.no_validation:
        visualization(last_epoch, train_losses, train_accuracies, val_losses, val_accuracies, test_losses, test_accuracies, args.id)
    else:
        visualization(last_epoch, train_losses, train_accuracies, None, None, test_losses, test_accuracies, args.id)
