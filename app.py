import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('defect_classifier_model.h5')

class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

def predict_defect(image):
    """
    Takes an image as a NumPy array, preprocesses it, and returns
    a dictionary of predicted labels and their probabilities.
    """
    img_array = np.expand_dims(image, axis=0)
    
    predictions = model.predict(img_array)
    
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return confidences

iface = gr.Interface(
    fn=predict_defect,
    inputs=gr.Image(height=200, width=200, label="Upload a Defect Image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="🏭 Metal Surface Defect Detector",
    description="Upload an image of a metal surface to classify its defect type. The model will predict the most likely defects.",
    examples=[
        ['data/scratches/scratches_200.jpg'],
        ['data/patches/patches_200.jpg'],
        ['data/crazing/crazing_200.jpg']
    ]
)

iface.launch()