import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pickle import load
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import os

# Set page config
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📸",
    layout="wide"
)

# Title and description
st.title("📸 Image Caption Generator")
st.markdown("Upload an image and get an AI-generated caption!")

@st.cache_resource
def load_models():
    """Load the pre-trained models and tokenizer"""
    try:
        # Load tokenizer
        tokenizer = load(open("tokenizer.p", "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 32
        
        # Define model architecture
        def define_model(vocab_size, max_length):
            # features from the CNN model squeezed from 2048 to 256 nodes
            inputs1 = Input(shape=(2048,), name='input_1')
            fe1 = Dropout(0.5)(inputs1)
            fe2 = Dense(256, activation='relu')(fe1)

            # LSTM sequence model
            inputs2 = Input(shape=(max_length,), name='input_2')
            se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
            se2 = Dropout(0.5)(se1)
            se3 = LSTM(256)(se2)

            # Merging both models
            decoder1 = add([fe2, se3])
            decoder2 = Dense(256, activation='relu')(decoder1)
            outputs = Dense(vocab_size, activation='softmax')(decoder2)

            # tie it together [image, seq] [word]
            model = Model(inputs=[inputs1, inputs2], outputs=outputs)
            model.compile(loss='categorical_crossentropy', optimizer='adam')
            return model
        
        # Load the caption model
        caption_model = define_model(vocab_size, max_length)
        caption_model.load_weights('models2/model_9.h5')
        
        # Load Xception model for feature extraction
        xception_model = Xception(include_top=False, pooling="avg")
        
        return tokenizer, caption_model, xception_model, vocab_size, max_length
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please make sure the following files exist in your project directory:")
        st.info("- tokenizer.p")
        st.info("- models2/model_8.h5")
        return None, None, None, None, None

def extract_features(image, model):
    """Extract features from image using Xception model"""
    try:
        # Resize image to 299x299 (Xception input size)
        image = image.resize((299, 299))
        image = np.array(image)
        
        # Handle images with 4 channels (RGBA)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[..., :3]
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Preprocess image
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        
        # Extract features
        feature = model.predict(image, verbose=0)
        return feature
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def word_for_id(integer, tokenizer):
    """Get word for given integer ID"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    """Generate caption for the image"""
    try:
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo, sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        
        # Clean up the caption
        caption = in_text.replace('start', '').replace('end', '').strip()
        return caption
    
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"

# Load models
with st.spinner("Loading models... This may take a moment."):
    tokenizer, caption_model, xception_model, vocab_size, max_length = load_models()

if tokenizer is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    help="Upload an image file to generate a caption"
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        
        # Generate caption button
        if st.button("Generate Caption", type="primary"):
            with st.spinner("Generating caption..."):
                # Extract features
                features = extract_features(image, xception_model)
                
                if features is not None:
                    # Generate caption
                    caption = generate_caption(caption_model, tokenizer, features, max_length)
                    
                    # Display the caption
                    st.success("Caption generated successfully!")
                    st.markdown(f"**Caption:** {caption}")
                    
                    # Option to copy caption
                    st.code(caption, language=None)
                else:
                    st.error("Failed to extract features from the image.")

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a deep learning model to generate captions for images. "
    "The model combines a pre-trained Xception CNN for image feature extraction "
    "with an LSTM network for caption generation."
)

st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload an image using the file uploader
2. Click 'Generate Caption' to get an AI-generated description
3. The caption will appear on the right side
""")

st.sidebar.title("Supported Formats")
st.sidebar.markdown("""
- JPG/JPEG
- PNG
- BMP
- TIFF
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")