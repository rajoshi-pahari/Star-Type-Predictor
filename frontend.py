import streamlit as st
import pandas as pd
import requests
from io import StringIO
import random

# Set Page Config
st.set_page_config(
    page_title="Star Type Predictor 💫",
    page_icon="💫",
)

# List of 5 background image URLs (ensure they are direct image links)
background_urls = [
    "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTR8fGdhbGF4eXxlbnwwfDB8MHx8fDA%3D",  # Example image 1
    "https://images.unsplash.com/photo-1467685790346-20bfe73a81f0?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjd8fGdhbGF4eXxlbnwwfDB8MHx8fDA%3D",  # Example image 2
    "https://images.unsplash.com/photo-1487235829740-e0ac5a286e1c?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NDh8fGdhbGF4eXxlbnwwfDB8MHx8fDA%3D",  # Example image 3
    "https://images.unsplash.com/photo-1726828581304-1bd8a2b90246?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjN8fG5lYnVsYXxlbnwwfDB8MHx8fDA%3D",  # Example image 4
    "https://images.unsplash.com/photo-1503264116251-35a269479413?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTl8fG91dGVyJTIwc3BhY2V8ZW58MHwwfDB8fHww",  # Example image 5
]

# Randomly select one background image URL
selected_bg = random.choice(background_urls)

# Inject custom CSS with the randomly selected background
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("{selected_bg}") no-repeat center center fixed;
    background-size: cover;
    padding-bottom: 60px; 
    min-height: 100vh;  
    overflow-y: auto;
}}

[data-testid="stSidebar"] {{
    background: rgba(50, 50, 50, 0.8); /* Dark grey with transparency */
}}

footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    z-index: 10;  /* Ensure footer is above other content */
}}

h1, h2, h3, h4, h5, h6, p {{
    color: white;  /* Set header and paragraph text color to white */
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Now you can proceed with your app content
st.title("Welcome to Star Type Predictor 💫")

st.markdown("""
    <footer>
        🌟 This project is developed by <b>Rajoshi Pahari</b> as part of the <b>ML4A Training Program</b> at Spartificial. 🚀
    </footer>
""", unsafe_allow_html=True)

button_style = """
<style>
button {
    background-color: #808080; /* Grey */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 12px;
}
.stButton>button:hover {
        background-color: #6e6e6e;  /* Darker gray when hovered */
        transform: scale(1.1);  /* Slightly enlarge button on hover */
    }
    
    .stButton>button:active {
        background-color: #555555;  /* Even darker gray when clicked */
        transform: scale(0.95);  /* Shrink button when clicked */
    }
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# FastAPI endpoints
API_URL_SINGLE = "http://127.0.0.1:8000/predict-single/"
API_URL_MULTIPLE = "http://127.0.0.1:8000/predict-multiple/"

# Title and description
st.title("Star Type Predictor 💫")
st.sidebar.title("Choose Prediction Mode")
mode = st.sidebar.radio("Select mode", ["Single Star Predictor", "Multiple Star Predictor"])

if mode == "Single Star Predictor":
    st.header("Single Star Predictor")
    st.write("Enter the details of a single star:")

    # Input fields
    temperature = st.number_input("Temperature (K)", min_value=0, step=1, format="%d")
    luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0, step=0.1, format="%.1f")
    radius = st.number_input("Radius (R/Ro)", min_value=0.0, step=0.1, format="%.1f")
    absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", step=0.1, format="%.1f")

    # Predict button
    if st.button("Predict"):
        payload = {
            "Temperature (K)": temperature,
            "Luminosity(L/Lo)": luminosity,
            "Radius(R/Ro)": radius,
            "Absolute magnitude(Mv)": absolute_magnitude,
        }

        try:
            response = requests.post(API_URL_SINGLE, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Star Type: {result['predicted_type']}")
            else:
                st.error(f"Error: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

elif mode == "Multiple Star Predictor":
    st.header("Multiple Star Predictor")
    st.write("Upload a CSV file with the following columns:")
    st.code("""
    Temperature (K), 
    Luminosity(L/Lo), 
    Radius(R/Ro), 
    Absolute magnitude(Mv)
    """, language="plaintext")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the file into a DataFrame
        input_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV:")
        st.dataframe(input_df)

        # Predict button
        if st.button("Predict for Multiple Stars"):
            try:
                # Convert the DataFrame to CSV format for sending
                csv_buffer = StringIO()
                input_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                response = requests.post(
                    API_URL_MULTIPLE,
                    files={"file": ("uploaded_file.csv", csv_buffer.read(), "text/csv")}
                )

                if response.status_code == 200:
                    predictions = response.json().get("predictions", [])

                    # Display predictions as a DataFrame
                    result_df = pd.DataFrame(predictions)
                    st.dataframe(result_df)

                    # Provide an option to download results
                    csv_download = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv_download,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
