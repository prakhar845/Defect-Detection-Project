Automated Defect Detection in Manufacturing
🏭 Project Overview
This project implements a deep learning solution for automatically identifying and classifying common surface defects on metal sheets. Using a computer vision model, this system can help automate quality control in a manufacturing setting, leading to increased efficiency and accuracy. The project leverages transfer learning from a pre-trained model to achieve high performance with a relatively small dataset.

🛠️ Technical Skills
Languages & Libraries: Python, TensorFlow, Keras, Matplotlib, Seaborn, Gradio

Techniques: Computer Vision, Deep Learning, Transfer Learning, Convolutional Neural Networks (CNNs), Data Augmentation, Data Pipelines (tf.data)

Models: MobileNetV2 (pre-trained)

Deployment: Gradio for creating an interactive web UI

📂 Project Structure
Automated_Defect_Detection/
|
|-- venv/                       # Virtual environment (ignored by Git)
|-- data/                       # Image dataset folder (ignored by Git)
|-- defect_classifier_model.h5  # Trained Keras model (ignored by Git)
|-- analysis.ipynb              # Jupyter Notebook with model training and evaluation
|-- app.py                      # Python script for the Gradio web application
|-- .gitignore                  # Specifies files to be ignored by Git
|-- README.md                   # This file

📋 Methodology
Data Preparation: The NEU Metal Surface Defects Database, containing 1,800 images across six defect categories, was organized into a class-specific folder structure required by Keras.

Data Pipeline: An efficient data pipeline was built using tf.data and image_dataset_from_directory to load, preprocess, and batch the images for training.

Data Augmentation: A Keras preprocessing layer was created to apply random transformations (flips, rotations, zooms) to the training images in real-time. This technique artificially expands the dataset and helps the model generalize better.

Transfer Learning: A MobileNetV2 model, pre-trained on the ImageNet dataset, was used as the convolutional base. The base model's weights were frozen, and new, trainable classification layers were added on top.

Training & Evaluation: The model was trained for 20 epochs, achieving high accuracy on the validation set. Performance was visualized with accuracy/loss plots and a detailed confusion matrix to analyze class-specific results.

Deployment: The final trained model was saved and deployed into an interactive web application using Gradio, allowing users to upload an image and receive a real-time classification of the defect.

🚀 How to Run
Clone the repository:

git clone https://github.com/prakhar845/Defect-Detection-Project.git
cd Defect-Detection-Project

Set up the environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

(Note: Create a requirements.txt file by running pip freeze > requirements.txt in your activated environment.)

Download the data: Download the "NEU Metal Surface Defects Database" from Kaggle. Unzip it and organize the images into a data/ folder with subdirectories for each class as described in the notebook.

Train the model: Open and run the analysis.ipynb Jupyter Notebook to train the model and save the defect_classifier_model.h5 file.

Launch the app:

python app.py

Open the local URL provided in your browser to use the application.