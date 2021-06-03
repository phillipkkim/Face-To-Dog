import os
from shutil import copy

name = "results/face2dog_baseline"
path = name + "/test_latest/images"

new_path = name + "/test_latest/fake"

for filename in os.listdir(path):
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    if filename.endswith("fake.png"):
        copy(os.path.join(path, filename), os.path.join(new_path, filename))
