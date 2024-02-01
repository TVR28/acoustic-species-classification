import os
import random
import shutil

# This file has been used to select 30 classes out of 264 classes for the small dataset 
# as model traning on 264 classes was taking too long.

def get_random_subdirectories(parent_directory, num_subdirectories):
    all_subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    random_subdirectories = random.sample(all_subdirectories, min(num_subdirectories, len(all_subdirectories)))
    return [os.path.join(parent_directory, subdirectory) for subdirectory in random_subdirectories]

def copy_directories(source_directories, destination_directory):
    for source_directory in source_directories:
        destination_path = os.path.join(destination_directory, os.path.basename(source_directory))
        shutil.copytree(source_directory, destination_path)

parent_directory = r"C:\Users\aahire\Documents\CSS581\Midterm\Without_any_preprocessing"
num_subdirectories_to_select = 30
destination_directory = r"C:\Users\aahire\Documents\CSS581\Midterm\Without_any_preprocessing_small"

random_subdirectories = get_random_subdirectories(parent_directory, num_subdirectories_to_select)

print(f"Random Subdirectories (selected from {len(os.listdir(parent_directory))} total subdirectories):")
for subdirectory in random_subdirectories:
    print(subdirectory)

# Copy selected subdirectories to the destination directory
copy_directories(random_subdirectories, destination_directory)
