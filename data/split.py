import json

# File paths
input_file = 'belle_chat_ramdon_10k.json'  # Input file containing the data
test_output_file = 'test.json'  # Output file for the test dataset
train_output_file = 'train.json'  # Output file for the training dataset

# Reading the data
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)  # Load data from the JSON file

# Ensure the data is a list and has more than 1000 elements
if isinstance(data, list) and len(data) > 1000:
    # Splitting the data
    test_set = data[:1000]  # First 1000 entries for the test set
    train_set = data[1000:]  # Remaining entries for the training set

    # Saving the test dataset
    with open(test_output_file, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)  # Write the test data to a JSON file
    
    # Saving the training dataset
    with open(train_output_file, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)  # Write the training data to a JSON file
else:
    # Print an error message if the data isn't a list or doesn't have enough elements
    print("The data is not a list or has fewer than 1000 elements.")

