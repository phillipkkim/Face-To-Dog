#Arda Mavi

from keras.models import Sequential
from keras.models import model_from_json
import numpy as np
from .get_dataset import get_img

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = 'cat' if Y[0] == 0 else 'dog'
    return Y
    
def prediction_score(model, X):
    Y = model.predict(X)
    #print(Y)
    return Y[0, 1]
    Y = np.argmax(Y, axis=1)
    Y = 1 if Y[0] == 0 else 0 # Returns a score of 0 if it's a dog
    #print('What score is predict file generating', Y)
    return Y

def generateScore(fake_img):
    import sys
#    img_dir = sys.argv[1]
    #from get_dataset import get_img
    fake_img = fake_img.detach().cpu().numpy()  # Turn tensor into a numpy array
    img = get_img(fake_img) # Helen's edits
    import numpy as np
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = open(dir_path + '/Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights(dir_path + "/Data/Model/weights.h5")
    score = prediction_score(model, X)
    return score


if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    from get_dataset import get_img
    img = get_img(img_dir)
    import numpy as np
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    print('It is a ' + Y + ' !')
