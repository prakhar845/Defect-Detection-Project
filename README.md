## Automated Defect Detection in Manufacturing

### Project Overview
In modern manufacturing, ensuring the quality of products is paramount. This project introduces a deep learning solution designed to automate the critical process of quality control. By leveraging computer vision, this system automatically identifies and classifies six common types of surface defects on metal sheets, moving beyond the limitations of traditional manual inspection.

The goal is to provide a tool that increases the efficiency, accuracy, and consistency of the quality control pipeline. This project implements an end-to-end workflow, from processing raw image data to deploying a user-friendly web application that delivers real-time predictions. The core of the solution is a Convolutional Neural Network (CNN) that utilizes transfer learning from the highly efficient MobileNetV2 architecture to achieve high performance.


### Technology Stack 
1. **Languages & Libraries:** Python, TensorFlow, Keras, Matplotlib, Seaborn, Gradio
2. **Techniques:** Computer Vision, Deep Learning, Transfer Learning, Convolutional Neural Networks (CNNs), Data Augmentation, Data Pipelines (tf.data)
3. **Models:** MobileNetV2 (pre-trained)
4. **Deployment:** Gradio for creating an interactive web UI

### Project Structure
Automated Defect Detection in Manufacturing/
|
|-- add_env/                    # Virtual environment (ignored by Git)
|-- data/                       # Image dataset folder (ignored by Git)
|-- defect_classifier_model.h5  # Trained Keras model (ignored by Git)
|-- analysis.ipynb              # Jupyter Notebook with model training and evaluation
|-- app.py                      # Python script for the Gradio web application
|-- .gitignore                  # Specifies files to be ignored by Git
|-- README.md                   # This file
|-- requirements.txt            # Project Dependencies


### Methodology
1. **Data Preparation:** The model was trained on the NEU Metal Surface Defects Database, which contains 1,800 grayscale images categorized into six classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches. The data was organized into a class-specific folder structure (data/train/Crazing, data/train/Inclusion, etc.), a standard practice for image classification tasks with Keras.

2. **Building an Efficient Data Pipeline** To handle the image data efficiently and prevent I/O bottlenecks during training, a robust data pipeline was constructed using TensorFlow's tf.data API. The image_dataset_from_directory utility was used to load images from the folders, automatically inferring labels from the directory structure and creating batches of preprocessed tensors ready for the model.

3. **Data Augmentation for Robustness** Deep learning models perform best with large, diverse datasets. To improve the model's ability to generalize and prevent overfitting on the relatively small dataset, data augmentation was implemented. A Keras preprocessing layer was integrated directly into the model to apply random transformations—such as horizontal/vertical flips, rotations, and zooms—to the training images in real-time. This artificially expands the dataset and ensures the model learns the core features of the defects rather than just memorizing the training images.

4. **Transfer Learning with MobileNetV2** Instead of training a CNN from scratch, this project leverages the power of transfer learning. The MobileNetV2 model, pre-trained on the massive ImageNet dataset, was used as the feature extraction base. By freezing the weights of these pre-trained layers, we retain their powerful, generalized knowledge of shapes, textures, and patterns. A new, trainable classification "head" was then added on top of the frozen base, consisting of dense layers that were specifically trained to recognize the unique patterns of the six metal defect categories

5. **Model Training and Evaluation** The model was trained for 20 epochs, and its performance was monitored on a separate validation set. The model achieved high accuracy, demonstrating its effectiveness in distinguishing between the different defect classes. The training process was visualized with accuracy and loss plots to ensure the model was learning effectively without overfitting. Finally, a confusion matrix was generated to provide a detailed, class-by-class analysis of the model's predictive performance.

6. **Deployment as an Interactive Web App** A model's true value is in its application. The final trained model was saved as an .h5 file and deployed into an interactive web application using the Gradio library. This user-friendly interface allows anyone to test the model by simply uploading an image of a metal surface and receiving an instant classification of the defect type, along with a confidence score. This demonstrates a complete, end-to-end workflow from data to a practical, deployed solution.

### How to Run
**Clone the repository:**

git clone https://github.com/prakhar845/Defect-Detection-Project.git
cd Defect-Detection-Project

**Set up the environment:**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

**Download the data:** Download the "NEU Metal Surface Defects Database" from Kaggle. Unzip it and organize the images into a data/ folder with subdirectories for each class as described in the notebook.

**Train the model:** Open and run the analysis.ipynb Jupyter Notebook to train the model and save the defect_classifier_model.h5 file.

**Launch the app:**

python app.py

Open the local URL provided in your browser to use the application.

## Contributing
Contributions are welcome! If you have suggestions for improvements, new analysis ideas, or bug fixes, please:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes and commit them (git commit -m 'Add new feature').
4. Push to the branch (git push origin feature/your-feature-name).
5. Open a Pull Request.
