import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_detection_model.h5')

def predict_image(img):
    resized_img = cv2.resize(img, (32, 32))
    normalized_img = resized_img.astype('float32') / 255.0
    processed_img = np.expand_dims(normalized_img, axis=0)
    prediction = model.predict(processed_img)
    if prediction < 0.5:
        return "No"
    else:
        return "Yes"

def classify_image(input_image):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    prediction = predict_image(img)
    return prediction

# Define the input and output components for the Gradio interface
image_input = gr.inputs.Image()
label_output = gr.outputs.Label()

# Create the Gradio interface
app =gr.Interface(fn=classify_image, inputs=image_input, outputs=label_output, title='Brain Tumor Detection').launch()
app.launch()