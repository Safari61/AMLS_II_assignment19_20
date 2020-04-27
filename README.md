# README

# THE MOST IMPORTANT THINGS:
#   1:
#       Please change the file paths into your own paths, when running the codes!!!
#   2:
#       Please first run the data pre-processing codes, to generate input data for training the model.
#       The files are too big to upload to Github, so you have to run the codes 
#       and genenrate them first, then run the "main.py". You CAN'T directly run the "main.py"!!!
# Visualization of training process:
#   You can load the tensorboard logs generated during the training process to visualize the training process, and the these figures are used in my report.


1. Organization of my project
    there are two parts of each task, 
    the first part is to read data from the given dataset, preprocess and save them in pickle files. The codes in this part is named as "data_preprocessing.py".
    the second part is model implementation, in detail, to build the model and train it using data from the first part. The codes of this part are saved altogether in the "main.py".
    
2. The role of each file
    The codes in "data_preprocessing" are used to pre-process and generate data.
    The codes in "main.py" are used to build and train the model.
    
3. The packages required to run your code:
    tensorflow,
    tensorflow.keras,
    tensorflow_model_optimization,
    tempfile,
    sklearn.model_selection,
    numpy,
    pickle,
    os,
    random,
    cv2