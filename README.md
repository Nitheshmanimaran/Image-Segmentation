README.md

Running the Streamlit code

1. Make sure you have Streamlit, TensorFlow, and other required packages installed.
2. Download or clone the code from the repository.
3. In the first line of the code, change the MODEL_PATH variable to the path of the model you want to use.
4. Open a terminal in the folder where the code is located and run streamlit run app.py (assuming the code file is named "app.py").
5. The Streamlit app should now be running on your localhost.
6. Follow the instructions on the app to upload an image and view the segmentation result.

Changing the Model

To use a different model, replace the model file at the path specified in the MODEL_PATH variable with your desired model file. Make sure the model is in the correct format and that it uses the UpdatedMeanIoU metric as specified in the code.