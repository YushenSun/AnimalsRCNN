# Define the list of target class labels
classes = ['elephant', 'lion', 'giraffe', 'zebra', 'rhino', 'non-animal']

# Assign unique integer codes to each target class
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# You can define more target classes and codes as needed

# Print the class labels and their corresponding integer codes
for cls, idx in class_to_idx.items():
    print(f'{cls}: {idx}')