import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import json
import base64

# Read the background image file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg.jpg')

loaded_model = load_model('model_medicinal.h5')
img_height = 180
img_width = 180

classes = ['Aloevera',
 'Amla',
 'Amruthaballi',
 'Arali',
 'Astma_weed',
 'Badipala',
 'Balloon_Vine',
 'Bamboo',
 'Beans',
 'Betel',
 'Bhrami',
 'Bringaraja',
 'Caricature',
 'Castor',
 'Catharanthus',
 'Chakte',
 'Chilly',
 'Citron lime (herelikai)',
 'Coffee',
 'Common rue(naagdalli)',
 'Coriender',
 'Curry',
 'Doddpathre',
 'Drumstick',
 'Ekka',
 'Eucalyptus',
 'Ganigale',
 'Ganike',
 'Gasagase',
 'Ginger',
 'Globe Amarnath',
 'Guava',
 'Henna',
 'Hibiscus',
 'Honge',
 'Insulin',
 'Jackfruit',
 'Jasmine',
 'Kambajala',
 'Kasambruga',
 'Kohlrabi',
 'Lantana',
 'Lemon',
 'Lemongrass',
 'Malabar_Nut',
 'Malabar_Spinach',
 'Mango',
 'Marigold',
 'Mint',
 'Neem',
 'Nelavembu',
 'Nerale',
 'Nooni',
 'Onion',
 'Padri',
 'Palak(Spinach)',
 'Papaya',
 'Parijatha',
 'Pea',
 'Pepper',
 'Pomoegranate',
 'Pumpkin',
 'Raddish',
 'Rose',
 'Sampige',
 'Sapota',
 'Seethaashoka',
 'Seethapala',
 'Spinach1',
 'Tamarind',
 'Taro',
 'Tecoma',
 'Thumbe',
 'Tomato',
 'Tulsi',
 'Turmeric',
 'ashoka',
 'camphor',
 'kamakasturi',
 'kepala']

# def get_medicinal_benefits(plant_name):
#     # Send a GET request to Google
#     url = f"https://www.google.com/search?q={plant_name}+medicinal+benefits"
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(url, headers=headers)

#     # Parse the HTML content of the page with BeautifulSoup
#     soup = BeautifulSoup(response.content, 'html.parser')

#     # Find the first few <div> elements that contain the search results
#     divs = soup.find_all('div', limit=5)

#     # Extract and return the text from these <div> elements
#     benefits = ' '.join(div.text for div in divs)
#     print(benefits)
#     return benefits

file_name = 'ben.txt'
def get_medicinal_benefits(plant_name):
    # Open the file in read mode
    with open(file_name, 'r') as f:
        # Load the dictionary from the file
        plants = json.load(f)

    # Check if the plant is in the dictionary
    if plant_name in plants:
        return plants[plant_name]
    else:
        return "No information available for this plant."

# Example usage:
# print(get_medicinal_benefits("Aloe Vera", "plants.txt"))


# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array

# Function to predict plant name and get medicinal benefits
def predict_and_get_benefits(image_path):
    img_array = preprocess_image(image_path)
    preds = loaded_model.predict(img_array)
    plant_name = classes[preds.argmax()]
    benefits = get_medicinal_benefits(plant_name)
    return plant_name, benefits

# Title for the Streamlit app
st.title('Medicinal Plant Recognition and Benefits')

# File uploader to upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if st.button('Predict Plant Name and Medicinal Benefits'):
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    plant_name, benefits = predict_and_get_benefits(img)
    st.write(f"Predicted plant: {plant_name}")
    st.write(f"Medicinal benefits: {benefits}")
    # st.write(f"Medicinal benefits: {benefits[283:]}")