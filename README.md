# field-recording-segmentation
Python and Matlab code for segmentation of field recordings. Please cite the following paper if you use the code in your research:
* M. Marolt, C. Bohak, A. Kavčič, and M. Pesek, [Automatic segmentation of ethnomusicological field recordings](https://www.mdpi.com/2076-3417/9/3/439/pdf), Applied sciences, vol. 9, iss. 3, pp. 1-12, 2019.

## Tensorflow
The folder contains Python code for training and using the deep learning models for labelling field recordings into a set of classes (e.g. speech, singing, instrumental etc.)
* generate_features.py processes audio files, splits them into frames, calculates FFT and stores a them into Tensorflow .tfr files to be read during training. Input to the program is a list of folders, each should include a "sample labels XXX.txt" text file, which contains a list of audio files in the folder XXX and their corresponding class labels. Both, the FFT ans the labels are stored in the tfr files
* train_and_test.py trains and test a deep residual model. It reads .tfr files, converts it into mel spectra and trains/tests models with cross valdation. Several parameters can be set in the default.ini file.
* label_file.py takes an already trained model and labels an arbitrary file with the corresponding labels. The sample rate for labeling is read from defaults.ini.

## Matlab
The folder contains Matlab code for probabilistic segmentation of field recordings based on energy and classification into a set of classes (e.g. speech, singing, instrumental etc.)
* segmentRecordingDeep.m is the main function that takes an audio file (field recording) and probabilities of classification of the file into a set of classes (as returned e.g. by the tensorflow model) and returns the segment boundaries and segment labels.

## MIREX 2015, 2018
The folders contains our submissions to [MIREX 2015 Music/Speech Classification and Detection task](https://www.music-ir.org/mirex/wiki/2015:Music/Speech_Classification_and_Detection_Results), as well as [MIREX 2018 Music and or Speech Detection task](https://www.music-ir.org/mirex/wiki/2018:Music_and_or_Speech_Detection_Results). See the enclosed READMEs for usage.

## Tensorflow model
The folder contains a trained tensorflow model export that labels 2 second audio fragments as solo singing, choir singing, instrumental or speech. Input to the model consists of 513x140 blocks of magnitude spectrogram values (as calculated e.g. by librosa stft) with window size 1024 and step size 315 at 22050 sampling rate. Output are the probabilities of the four target classes for each block.
The model can be used in code as:
```python
  Df=# Nx513x140x1 sized array of magnitude fft blocks
  exportpath=PATH_TO_EXPORT_FOLDER
  with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["scoring-tag"], exportpath)
    predictions = tf.get_default_graph().get_tensor_by_name("predictions:0")
    probabilities = sess.run([predictions], feed_dict={'xinput:0': Df})
```
