# field-recording-segmentation
Python and Matlab code for segmentation of field recordings

## Tensorflow
The folder contains Python code for training and using the deep learning models for labelling field recordings into a set of classes (e.g. speech, singing, instrumental etc.)
* generate_features.py processes audio files, splits them into frames, calculates FFT and stores a them into Tensorflow .tfr files to be read during training. Input to the program is a list of folders, each should include a "sample labels XXX.txt" text file, which contains a list of audio files in the folder XXX and their corresponding class labels. Both, the FFT ans the labels are stored in the tfr files
* train_and_test.py trains and test a deep residual model. It reads .tfr files, converts it into mel spectra and trains/tests models with cross valdation. Several parameters can be set in the default.ini file.
* label_file.py takes an already trained model and labels an arbitrary file with the corresponding labels. The sample rate for labeling is read from defaults.ini.
