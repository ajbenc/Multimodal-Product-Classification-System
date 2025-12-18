import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from src.vision_embeddings_tf import FoundationalCVModel, load_and_preprocess_image
from src.nlp_models import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Product Classification Demo",
    page_icon="🛍️",
    layout="wide"
)

# Title and description
st.title("🛍️ Product Classification System")
st.markdown("""
**Demo:** Select a product to see AI-powered text classification in action!

This system uses **natural language processing** (MiniLM sentence-transformers) 
to classify products with **94.16% accuracy** based on product descriptions.

*💡 Full multimodal version (text + images, 87.56%) available locally.*
""")

# Sidebar with info
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Text Classification", "94.16%", help="⭐ Current demo mode")
    st.metric("Multimodal (Local)", "87.56%", help="With images")
    st.metric("Image-Only (Local)", "82.00%", help="With images")
    
    st.markdown("---")
    st.header("🔧 Technologies")
    st.markdown("""
    - **Vision:** ConvNeXtV2
    - **NLP:** MiniLM
    - **Framework:** TensorFlow
    - **Architecture:** Early Fusion
    """)
    
    st.markdown("---")
    st.markdown("""
    **👨‍💻 Developer:** Andres Benavides  
    **🔗 GitHub:** [@ajbenc](https://github.com/ajbenc)
    """)

# Load dataset for product selection
@st.cache_data
def load_dataset():
    # Try sample dataset first (for deployment), then full dataset (for local)
    dataset_paths = [
        'demo_sample/sample_products.csv',  # Sample for deployment
        'data/processed_products_with_images.csv'  # Full dataset for local
    ]
    
    for path in dataset_paths:
        try:
            import os
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'demo_sample' in path:
                    st.sidebar.info("🎯 Running with sample dataset (deployment mode)")
                return df
        except Exception as e:
            continue
    
    st.error("⚠️ Could not load dataset. Please ensure data files exist.")
    return None

# Product selection interface
st.subheader("🎯 Select a Product to Classify")

dataset = load_dataset()

if dataset is not None:
    # Get unique categories and create user-friendly sequential names
    categories = sorted(dataset['class_id'].unique())
    
    # Create simple sequential category names (Category 1, Category 2, etc.)
    category_names = {cat: f"Category {idx + 1}" for idx, cat in enumerate(categories)}
    
    # Add product count for context
    for cat in categories:
        count = len(dataset[dataset['class_id'] == cat])
        category_names[cat] = f"Category {categories.index(cat) + 1} ({count} products)"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Step 1:** Choose Category")
        selected_category = st.selectbox(
            "Product Category",
            options=categories,
            format_func=lambda x: category_names[x],
            label_visibility="collapsed"
        )
    
    # Filter products by category
    category_products = dataset[dataset['class_id'] == selected_category].head(20)  # Show first 20
    
    with col2:
        st.markdown("**Step 2:** Select Product")
        product_options = {row['sku']: f"{row['name'][:50]}..." if len(row['name']) > 50 else row['name'] 
                          for idx, row in category_products.iterrows()}
        
        selected_sku = st.selectbox(
            "Product",
            options=list(product_options.keys()),
            format_func=lambda x: product_options[x],
            label_visibility="collapsed"
        )
    
    # Get selected product details
    selected_product = dataset[dataset['sku'] == selected_sku].iloc[0]
    
    st.markdown("---")
    
    # Display product details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Product Image")
        try:
            image_path = selected_product['image_path']
            if pd.notna(image_path) and image_path:
                import os
                if os.path.exists(image_path):
                    product_image = Image.open(image_path)
                    st.image(product_image, caption=selected_product['name'], use_container_width=True)
                    st.caption(f"📂 Path: {image_path}")
                else:
                    st.info("📷 **Text-Only Classification Mode** - Showcasing 94.16% accuracy NLP model")
                    st.caption(f"📂 Expected path: {image_path}")
                    st.markdown("*Images not included in deployment for size optimization. Full demo with images available locally.*")
                    product_image = None
            else:
                st.warning("No image path available for this product")
                product_image = None
        except Exception as e:
            st.warning(f"Could not load image - Using text-only classification")
            product_image = None
    
    with col2:
        st.subheader("📝 Product Information")
        st.markdown(f"**Name:** {selected_product['name']}")
        st.markdown(f"**SKU:** {selected_product['sku']}")
        if pd.notna(selected_product.get('price')):
            st.markdown(f"**Price:** ${selected_product['price']:.2f}")
        if pd.notna(selected_product.get('manufacturer')):
            st.markdown(f"**Brand:** {selected_product['manufacturer']}")
        
        st.markdown("**Description:**")
        product_description = selected_product['description'] if pd.notna(selected_product['description']) else "No description available"
        st.text_area("", value=product_description, height=200, label_visibility="collapsed", disabled=True)
    
    # Store for prediction
    selected_image = product_image
    selected_description = product_description
    true_category = selected_product['class_id']
    
else:
    st.error("⚠️ Could not load dataset. Please ensure 'data/processed_products_with_images.csv' exists.")
    selected_image = None
    selected_description = None
    true_category = None

# Classification button
st.markdown("---")
if dataset is not None:
    classify_btn = st.button("🚀 Classify Product", type="primary", use_container_width=True)
else:
    classify_btn = False
    st.warning("Cannot classify without dataset loaded.")

# Load models (cached)
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... (this may take a minute on first run)"):
        try:
            # Load vision model
            vision_model = FoundationalCVModel(backbone='resnet50', mode='eval')
            
            # Load NLP model
            nlp_model = HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
            
            # Load trained classifier
            from tensorflow import keras
            multimodal_model = keras.models.load_model('best_multimodal_mlp_model.h5')
            
            return vision_model, nlp_model, multimodal_model
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None

# Load actual product categories from the dataset
@st.cache_data
def load_categories():
    try:
        # Load the dataset to get actual categories
        df = pd.read_csv('data/processed_products_with_images.csv')
        categories = sorted(df['class_id'].unique())
        return [f"Category {cat}" for cat in categories]
    except:
        # Fallback categories if file not found
        return [f"Category {i}" for i in range(50)]  # Adjust range as needed

CATEGORIES = load_categories()

# Prediction function
def predict_product(image_data, text_data, vision_model, nlp_model, classifier):
    try:
        # Get image embeddings
        if image_data is not None:
            img_array = np.array(image_data.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            image_embeddings = vision_model.predict(img_array)
        else:
            image_embeddings = np.zeros((1, 2048))  # ResNet50 output size
        
        # Get text embeddings
        if text_data:
            text_embeddings = nlp_model.get_embedding(text_data)
            text_embeddings = np.array(text_embeddings).reshape(1, -1)
        else:
            text_embeddings = np.zeros((1, 384))  # MiniLM output size
        
        # Model expects separate inputs as a dictionary (matching original training format)
        model_inputs = {
            'text_input': text_embeddings,
            'image_input': image_embeddings
        }
        
        # Predict
        predictions = classifier.predict(model_inputs)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_idx]
        
        return predicted_class, confidence, top_3_idx, top_3_probs
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None, None

# Classification logic
if classify_btn:
    if selected_image is None and not selected_description:
        st.warning("⚠️ Selected product has no image or description!")
    else:
        # Load models
        vision_model, nlp_model, multimodal_model = load_models()
        
        if vision_model and nlp_model and multimodal_model:
            with st.spinner("🤖 Analyzing product..."):
                # Make prediction
                pred_class, confidence, top_3_idx, top_3_probs = predict_product(
                    selected_image, selected_description, vision_model, nlp_model, multimodal_model
                )
                
                if pred_class is not None:
                    # Check if prediction is correct
                    is_correct = (pred_class == true_category)
                    
                    if is_correct:
                        st.success("✅ Classification Complete - **CORRECT PREDICTION!** 🎉")
                    else:
                        st.warning("✅ Classification Complete - Prediction differs from actual category")
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    
                    # Show actual vs predicted
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("True Category", f"Category {true_category}")
                    with col2:
                        category_name = CATEGORIES[pred_class] if pred_class < len(CATEGORIES) else f"Category {pred_class}"
                        st.metric("Predicted", f"Category {pred_class}", 
                                 delta="✓ Correct" if is_correct else "✗ Wrong",
                                 delta_color="normal" if is_correct else "inverse")
                    with col3:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Top 3 predictions
                    st.markdown("#### 📊 Top 3 Predictions")
                    
                    # Safe category name extraction
                    top_3_names = []
                    for idx in top_3_idx:
                        if idx < len(CATEGORIES):
                            top_3_names.append(CATEGORIES[idx])
                        else:
                            top_3_names.append(f"Category {idx}")
                    
                    chart_data = pd.DataFrame({
                        'Category': top_3_names,
                        'Confidence': top_3_probs * 100
                    })
                    
                    st.bar_chart(chart_data.set_index('Category'))
                    
                    # Show detailed probabilities
                    with st.expander("📋 See detailed predictions"):
                        for i, (idx, prob, name) in enumerate(zip(top_3_idx, top_3_probs, top_3_names)):
                            st.write(f"{i+1}. **{name}**: {prob*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using TensorFlow, Transformers & Streamlit</p>
    <p>⭐ <a href='https://github.com/ajbenc/Multimodal-Product-Classification-System'>Star on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
