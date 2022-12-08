# Sleep Stage Prediction using CNN and Attention

## Dataset

We used the Sleep-EDF Database Expanded which contains 197 whole-night PolySomnoGraphic (PSG) sleep recordings, containing EEG, EOGn and EMG markers.
For the project, we used the data of 22 patients. Each patient had data available for two nights, each night lasting around 10 hours. The signals were recorded with a frequency of 100Hz.

## Preprocessing

We used the MNE library (a package used to extract neurophysiological data like EEG) to extract polysomnography (PSG) and hypnogram. First, we extracted the three relevant channels and the hypnogram label from the dataset. Then, we limited the sleep stage W to the last 30 minutes before the first sleep occurrence and the first 30 minutes after the last sleep occurrence to limit the impact of class imbalance.
We used PySpark to preprocess and transform it before training our model.

Once we loaded the signal and label datasets into a PySpark data frame, we created a primary key combining patient ID and timstep which was then used to join the PSG signals and hypnogram labels. Then, we converted the string labels into numerical values and combined the Stages 3 and 4 into a single label to reduce class imbalance. We also removed any unknown stages and removed unnecessary columns. Finally, we divided the channel data into 30-second epochs.

## Model and Training
We used convolution neural networks (CNN) to extract the features and transformers to predict the hypnogram scoring. We trained the model both on a single machine and multiple machines (disitributed training using Elephas).

Learning curves can be found in the /plots/ folder.

## Results
The model achieved an accuracy of 83.36% on the train set and an accuracy of 81.15% on the test set.

The model was also tested on two subjects who were not present in the train, validation, or test sets.

Detailed results can be found in the /results/ folder.
