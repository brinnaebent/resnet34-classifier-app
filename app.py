import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objects as go

# Load the pre-trained ResNet34 model
@st.cache_resource
def load_model():
    model = resnet34(pretrained=True)
    model.eval()
    return model

# Load ImageNet labels
@st.cache_resource
def load_labels():
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(labels_url)
    labels = response.json()
    return labels

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Predict the top 5 classes of the image
def predict_top5(model, image, labels):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return [(labels[catid.item()], prob.item()) for catid, prob in zip(top5_catid, top5_prob)]

# Create a bar plot of top 5 results
def plot_top5(results):
    labels, probs = zip(*results)
    fig = go.Figure(data=[go.Bar(x=probs, y=labels, orientation='h')])
    fig.update_layout(
        title='Top 5 Classification Results',
        xaxis_title='Probability',
        yaxis_title='Class',
        height=400,
    )
    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Image Classifier", layout="wide")
    
    st.title("Is your adversarial patch working?")
    
    model = load_model()
    labels = load_labels()
    
    st.write("Take a picture with your camera or upload a photo. Get a classification with the ResNet34 model. Remember, patches may be size and orientation dependent!")
    
    # Camera input
    camera_image = st.camera_input("Take a picture", key="camera")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption='Captured Image', use_column_width=True)
        
        if st.button('Classify Image'):
            with st.spinner('Classifying...'):
                processed_image = preprocess_image(image)
                results = predict_top5(model, processed_image, labels)
            
            # Display results
            st.success(f"Top classification: {results[0][0]}")
            fig = plot_top5(results)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.write("Or upload an image from your gallery:")
    
    # File uploader as a secondary option
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Classify Uploaded Image'):
            with st.spinner('Classifying...'):
                processed_image = preprocess_image(image)
                results = predict_top5(model, processed_image, labels)
            
            # Display results
            st.success(f"Top classification: {results[0][0]}")
            fig = plot_top5(results)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
