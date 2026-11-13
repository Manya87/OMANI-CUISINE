
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="OMANI CUISINE | AI FOOD RECOGNITION",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "omani_cuisine_model_improved.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names_improved.json"


# --------------------------------------------------
# 2. PREMIUM EMERALD AND GOLD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 85% 5%, #174d3d 0%, transparent 30%),
        linear-gradient(135deg, #071e18, #0b2d23 55%, #102f26);
    color: #f5ead3;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061b15, #103629);
    border-right: 1px solid #b99a54;
}

[data-testid="stSidebar"] * {
    color: #f5ead3 !important;
}

h1, h2, h3 {
    color: #e4c477 !important;
    font-family: Georgia, serif;
}

p, label, span {
    color: #f2e8d5;
}

.hero {
    padding: 45px 35px;
    border: 1px solid #b99a54;
    border-radius: 20px;
    background: linear-gradient(120deg, #123c2e, #09251d);
    text-align: center;
    margin-bottom: 25px;
}

.hero-title {
    font-family: Georgia, serif;
    color: #e4c477;
    font-size: 42px;
    font-weight: bold;
    letter-spacing: 2px;
}

.hero-subtitle {
    color: #e7dcc6;
    font-size: 17px;
}

.gold-card {
    background: rgba(22, 65, 49, 0.85);
    border: 1px solid #9e844b;
    border-radius: 16px;
    padding: 23px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #174b38, #0d3025);
    border: 1px solid #d0ad5e;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.gold-text {
    color: #e4c477;
}

.small-note {
    color: #c6bda9;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(90deg, #c7a65b, #e4c477);
    color: #10271d;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    padding: 10px 25px;
}

div.stButton > button:hover {
    background: #f0d995;
    color: #10271d;
}

[data-testid="stFileUploader"] {
    background: rgba(21, 62, 47, 0.8);
    border: 1px dashed #c7a65b;
    border-radius: 15px;
    padding: 15px;
}

hr {
    border-color: #806d43;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD MODEL AND CLASS LABELS
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def get_model():
    try:
        return load_model()
    except Exception as error:
        st.error(f"Could not load the trained model: {error}")
        st.stop()


# --------------------------------------------------
# 4. IMAGE PREPROCESSING AND PREDICTION
# --------------------------------------------------

def predict_food(image, model, class_names):

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocessing is already included in the improved model.
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    results = []

    for index in top_indices:
        label = class_names[int(index)]
        score = float(predictions[index])

        category, dish = label.split("/", 1)

        results.append({
            "label": label,
            "category": category,
            "dish": dish,
            "score": score
        })

    return results


# --------------------------------------------------
# 5. DISPLAY HELPERS
# --------------------------------------------------

def pretty_name(slug):
    return slug.replace("_", " ").replace("-", " ").title()


def show_hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def show_metric_card(number, heading):
    st.markdown(f"""
    <div class="gold-card" style="text-align:center;">
        <h2>{number}</h2>
        <p>{heading}</p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>OMANI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#d4b66e;'>"
        "CUISINE AI PROJECT</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Recognize Food",
            "Dataset Explorer",
            "About Project"
        ]
    )

    st.divider()

    st.markdown(
        "<p style='text-align:center;font-size:12px;'>"
        "AI POWERED FOOD RECOGNITION<br>"
        "50 OMANI FOOD CLASSES</p>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 7. HOME PAGE
# --------------------------------------------------

if page == "Home":

    show_hero(
        "OMANI CUISINE",
        "AI-POWERED FOOD RECOGNITION SYSTEM"
    )

    st.markdown(
        "<h3 style='text-align:center;'>"
        "Discover the Flavours of Oman</h3>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <p style="text-align:center;max-width:800px;margin:auto;">
    An intelligent image classification system designed to
    recognize traditional Omani cuisine using deep learning.
    Upload a food image and let artificial intelligence
    identify the dish and its food category.
    </p>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric_card("50", "Food Classes")

    with col2:
        show_metric_card("25", "Vegetarian Classes")

    with col3:
        show_metric_card("25", "Non-Vegetarian Classes")

    st.write("")
    st.markdown("### Explore the Project")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="gold-card">
            <h3>🍽️ Food Recognition</h3>
            <p>Upload a food photograph and receive
            the model's top three predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="gold-card">
            <h3>📚 Dataset Explorer</h3>
            <p>Explore the 50 food categories used
            in the classification system.</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Academic project demonstration | Omani Cuisine Recognition"
    )


# --------------------------------------------------
# 8. FOOD RECOGNITION PAGE
# --------------------------------------------------

elif page == "Recognize Food":

    show_hero(
        "FOOD RECOGNITION",
        "Upload an image to identify an Omani dish"
    )

    uploaded_file = st.file_uploader(
        "Choose a food image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        left, right = st.columns([1, 1.2])

        with left:
            st.markdown("### Uploaded Image")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("### AI Analysis")

            if st.button("Recognize This Dish", use_container_width=True):

                with st.spinner("Analyzing your food image..."):

                    model, class_names = get_model()

                    results = predict_food(
                        image,
                        model,
                        class_names
                    )

                best = results[0]

                st.markdown(
                    "<div class='result-card'>",
                    unsafe_allow_html=True
                )

                st.markdown("### <span class='gold-text'>"
                            "Predicted Dish</span>",
                            unsafe_allow_html=True)

                st.markdown(
                    f"## {pretty_name(best['dish'])}"
                )

                st.markdown(
                    f"**Category:** {pretty_name(best['category'])}"
                )

                st.markdown(
                    f"**Model score:** {best['score'] * 100:.2f}%"
                )

                st.markdown("</div>", unsafe_allow_html=True)

                st.write("")
                st.markdown("### Top 3 Predictions")

                for position, result in enumerate(results, start=1):

                    st.markdown(
                        f"**{position}. {pretty_name(result['dish'])}**"
                        f" — {result['score'] * 100:.2f}%"
                    )

                    st.progress(
                        min(max(result["score"], 0.0), 1.0)
                    )

                st.caption(
                    "Note: Model scores are outputs of the classifier "
                    "and are not guaranteed real-world probabilities."
                )

    else:
        st.info("Upload a food photograph to begin recognition.")


# --------------------------------------------------
# 9. DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    show_hero(
        "DATASET EXPLORER",
        "Explore the food categories in the trained model"
    )

    if CLASS_PATH.exists():

        with open(CLASS_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        vegetarian = [
            item for item in class_names
            if item.startswith("vegetarian/")
        ]

        non_vegetarian = [
            item for item in class_names
            if item.startswith("non_vegetarian/")
        ]

        category_filter = st.selectbox(
            "Select food category",
            ["All", "Vegetarian", "Non-Vegetarian"]
        )

        if category_filter == "Vegetarian":
            selected = vegetarian
        elif category_filter == "Non-Vegetarian":
            selected = non_vegetarian
        else:
            selected = class_names

        st.write(f"### {len(selected)} Food Classes")

        cols = st.columns(2)

        for index, item in enumerate(selected):

            category, dish = item.split("/", 1)

            with cols[index % 2]:
                st.markdown(f"""
                <div class="gold-card">
                    <h4>{pretty_name(dish)}</h4>
                    <p>{pretty_name(category)}</p>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.error("Class names file could not be found.")


# --------------------------------------------------
# 10. ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    show_hero(
        "ABOUT THE PROJECT",
        "Deep Learning for Traditional Omani Cuisine"
    )

    st.markdown("### Project Overview")

    st.write("""
    This project presents an image-based food recognition system
    designed to classify traditional Omani dishes into vegetarian
    and non-vegetarian categories.

    The system uses a trained deep learning model to analyze
    uploaded food photographs and predict the corresponding
    food class.
    """)

    st.markdown("### Technology Stack")

    technologies = [
        ("Programming Language", "Python"),
        ("Deep Learning Framework", "TensorFlow / Keras"),
        ("Model Architecture", "MobileNetV2"),
        ("Image Processing", "Pillow and NumPy"),
        ("Web Framework", "Streamlit"),
        ("Classification Classes", "50")
    ]

    for technology, value in technologies:
        st.markdown(f"""
        <div class="gold-card">
            <b>{technology}</b>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### System Workflow")

    st.markdown("""
    1. Collect and organize Omani food images.
    2. Prepare and split the dataset.
    3. Train the deep learning classification model.
    4. Evaluate and select the trained model.
    5. Upload a food image through the website.
    6. Generate and display the top predictions.
    """)

    st.markdown("### Model Information")

    st.write("""
    The website uses the improved trained model and its matching
    class-label file. The model performs image classification
    on 224 × 224 RGB images.
    """)

    st.caption(
        "The website does not claim an accuracy value unless "
        "supported by the project's actual evaluation results."
    )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<p style="text-align:center;color:#c5ad79;font-size:13px;">
OMANI CUISINE FOOD RECOGNITION SYSTEM<br>
AI • DEEP LEARNING • FOOD CLASSIFICATION
</p>
""", unsafe_allow_html=True)
