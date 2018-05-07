# Implementing 3D CNN for Lung Cancer Detection

•	Download and install CUDA such that GPU can be utilized for processing on data and this speeds up training by a considerate amount of time. Also Download CUDNN and copy the contents of the folder to the respective contents in the CUDA folder

•	Install anaconda with python 3.5

•	Create a conda environment in command prompt and name it as tensorflow gpu. Follow instructions in this page to setup tensorflow gpu for the system: https://www.tensorflow.org/install/install_windows 

•	Activate the environment

•	Import necessary libraries specified below 

•	OpenCV, Dicom, pandas, tensorflow, numpy, os, matplotlib, scikit-learn

•	After import of packages is complete, make sure that the indentation is followed precisely as that can cause multiple errors

•	Open Jupyter Notebook from within the activated environment

•	LungCancer3DCNN.ipynb has the 3D CNN model to be trained and contains model to detect individual patient's tumor

•	After the necessary parameters are specified for each layer of the network, the model will be created, to which the training data should be passed and the trained model is processed as output

•	On the trained model, we then run the test data, for which we will receive an accuracy, if the accuracy is stagnant, it means that the model has considerable amount of overfitting and datasets available are insufficient. We are able to predict the patients and whether they have cancer or not based on the trained model

•	The above steps were necessary for a person to run the code and achieve the best possible solution set accurately
