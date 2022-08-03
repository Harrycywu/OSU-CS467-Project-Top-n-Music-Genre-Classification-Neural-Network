import torch
import random
import numpy as np


# Global variables
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


# Function to set the random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Split data
def split_data(validation, random_seed):
    # Each genre: 00-99 (100 files)
    num_lst = []
    for num in range(100):
        num_lst.append(f'{num:02}')

    # Get all data
    # [Name of the Image, label]
    total_set = []
    for genre in genres:
        for num in num_lst:
            total_set.append([genre+'_'+num, genre])
    
    # Random shuffle
    # Set a random seed
    set_seed(random_seed)
    random.shuffle(total_set)

    # Get the complete training, validation, and testing sets
    training_set = []
    validation_set = []
    testing_set = []

    # Check if using a validation set
    if validation:
        # With a validation set: 70%/15%/15%
        training_set, validation_set, testing_set = total_set[:700], total_set[700:850], total_set[850:]
    else:
        # Without a validation set: 70%/0%/30%
        training_set, validation_set, testing_set = total_set[:700], [], total_set[700:]

    return training_set, validation_set, testing_set


# The function that extracts the same number of data from each genre
def split_data_norm(validation, random_seed):
    # Each genre: 00-99 (100 files)
    num_lst = []
    for num in range(100):
        num_lst.append(f'{num:02}')

    # Random shuffle
    # Set a random seed
    set_seed(random_seed)
    random.shuffle(num_lst)
    
    # Get the complete training, validation, and testing sets
    training_set = []
    validation_set = []
    testing_set = []
    
    total_set = []
    for genre in genres:
        for num in num_lst:
            total_set.append([genre+'_'+num, genre])

    # Check if using a validation set
    if validation:
        # With a validation set: 70%/15%/15%
        train_num, val_num, test_num = num_lst[:70], num_lst[70:85], num_lst[85:]
    else:
        # Without a validation set: 70%/0%/30%
        train_num, val_num, test_num = num_lst[:70], [], num_lst[70:]

    # Get each data subset
    # Format: [Name of the Image, label]
    # Training
    for genre in genres:
        for num in train_num:
            training_set.append([genre+'_'+num, genre])

    # Validation
    for genre in genres:
        for num in val_num:
            validation_set.append([genre+'_'+num, genre])

    # Testing
    for genre in genres:
        for num in test_num:
            testing_set.append([genre+'_'+num, genre])

    return training_set, validation_set, testing_set


if __name__ == '__main__':
    # Sample Usage
    random_seed = 1

    # # With a validation set
    # training_set, validation_set, testing_set = split_data(True, random_seed)
    # print(len(training_set), len(validation_set), len(testing_set))
    # print(testing_set)

    # Without a validation set
    training_set, validation_set, testing_set = split_data_norm(True, random_seed)
    print("Training data: {}, Validation data: {}, Testing data: {}".format(len(training_set), len(validation_set), len(testing_set)))
    print(testing_set)
