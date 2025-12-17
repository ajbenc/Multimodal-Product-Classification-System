# 🚀 Deployment Guide - Streamlit Cloud

This guide will help you deploy your Product Classification demo app to Streamlit Cloud (FREE hosting!).

---

## 📋 Prerequisites

- ✅ GitHub repository pushed (already done!)
- ✅ Streamlit account (free - we'll create one)
- ✅ Your app.py file (created!)

---

## 🎯 Step-by-Step Deployment

### 1. **Create Streamlit Cloud Account**

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **"Sign up"** or **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub repositories

### 2. **Deploy Your App**

1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository:** `ajbenc/Multimodal-Product-Classification-System`
   - **Branch:** `main` (or `master`)
   - **Main file path:** `app.py`
   - **App URL:** Choose a custom name like `product-classifier-demo`

3. Click **"Deploy!"**

### 3. **Wait for Deployment** (5-10 minutes first time)

Streamlit will:
- Install all dependencies from `requirements_app.txt`
- Download model files
- Start your app

You'll see the build logs in real-time.

### 4. **Get Your Live URL**

Once deployed, you'll get a URL like:
```
https://product-classifier-demo.streamlit.app
```

---

## ⚙️ Configuration (Important!)

### Model Files Issue

Since model files (`.h5`) are too large for GitHub, you have 2 options:

#### **Option A: Streamlit Secrets (Recommended)**

1. Host your models on cloud storage (Google Drive, Dropbox, AWS S3)
2. Add download URLs to Streamlit secrets
3. App downloads models on first run

Update `app.py`:
```python
@st.cache_resource
def load_models():
    # Download models from cloud if not present
    if not os.path.exists('best_multimodal_mlp_model.h5'):
        model_url = st.secrets["MODEL_URL"]
        urllib.request.urlretrieve(model_url, 'best_multimodal_mlp_model.h5')
    
    multimodal_model = keras.models.load_model('best_multimodal_mlp_model.h5')
    return multimodal_model
```

#### **Option B: Use Smaller Models**

Train a compressed version specifically for demo:
```python
# In your training notebook
from tensorflow import keras

# Quantize the model to reduce size
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

#### **Option C: Demo Mode (Quick Solution)**

For immediate deployment, modify `app.py` to use pre-trained embeddings only:

```python
# Simplified demo without full model
def predict_category(image_emb, text_emb):
    # Use similarity matching instead of full model
    # This is a demo fallback
    pass
```

---

## 🔧 Advanced Configuration

### Add Secrets to Streamlit

1. Go to your app dashboard on Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Add your secrets in TOML format:

```toml
# .streamlit/secrets.toml
MODEL_URL = "https://drive.google.com/..."
HUGGINGFACE_TOKEN = "hf_..."
```

### Custom Requirements

If deployment fails due to dependencies, modify `requirements_app.txt`:

```txt
# Lighter versions for deployment
streamlit==1.29.0
tensorflow-cpu==2.15.0  # CPU version (smaller)
transformers==4.35.0
torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Resource Limits

Streamlit Cloud free tier:
- **RAM:** 1 GB
- **CPU:** 1 core
- **Storage:** Limited

**Tip:** Use model caching aggressively with `@st.cache_resource`

---

## 🐛 Troubleshooting

### Issue: "App is taking too long to load"

**Solution:**
```python
# Add progress indicator
with st.spinner("Loading models... (first run may take 2-3 minutes)"):
    models = load_models()
```

### Issue: "Out of memory"

**Solutions:**
1. Use `tensorflow-cpu` instead of full TensorFlow
2. Load models lazily (only when needed)
3. Use smaller backbone models (ResNet50 instead of ConvNeXtV2)

### Issue: "Module not found"

**Solution:** Ensure all dependencies are in `requirements_app.txt`

### Issue: "Model file not found"

**Solution:** Implement Option A (cloud storage) from above

---

## 📱 Alternative Deployment Options

### Hugging Face Spaces (Alternative)

1. Create account on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space (Gradio or Streamlit)
3. Push your code
4. Automatic deployment!

**Pros:**
- Better for ML models (more resources)
- Model hosting included
- GPU support available

### Render (Alternative)

1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Select "Web Service"
4. Deploy with Dockerfile

---

## ✅ Post-Deployment Checklist

After successful deployment:

1. **Update README.md** with live demo link:
   ```markdown
   🚀 **Live Demo:** [Try it here!](https://product-classifier-demo.streamlit.app)
   ```

2. **Test the app:**
   - Upload sample images
   - Enter product descriptions
   - Verify predictions work

3. **Share on LinkedIn:**
   ```
   🎉 Excited to share my ML project demo!
   
   Try my Multimodal Product Classification System:
   🔗 https://product-classifier-demo.streamlit.app
   
   Features:
   ✅ Real-time AI predictions
   ✅ 87.56% accuracy
   ✅ Combines computer vision + NLP
   
   #MachineLearning #AI #DeepLearning
   ```

4. **Pin repository** on GitHub profile

5. **Add to portfolio website**

---

## 🎨 Enhance Your Demo

### Add Sample Products

Create a "Try Sample" button:
```python
if st.button("🎲 Try Sample Product"):
    # Load pre-selected sample
    sample_image = Image.open("samples/laptop.jpg")
    sample_text = "High-performance gaming laptop with RTX 4080..."
```

### Add About Section

```python
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    This system demonstrates multimodal machine learning...
    
    **Technical Stack:**
    - Vision: ConvNeXtV2
    - NLP: MiniLM
    - Framework: TensorFlow
    """)
```

### Add Analytics

```python
# Track usage (optional)
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

st.session_state.prediction_count += 1
st.sidebar.metric("Predictions Made", st.session_state.prediction_count)
```

---

## 🔒 Security Best Practices

1. **Never commit API keys** - use Streamlit secrets
2. **Validate user inputs** - check file types and sizes
3. **Rate limiting** - add cooldown between predictions
4. **Sanitize inputs** - clean text before processing

---

## 📊 Monitor Your App

Streamlit Cloud provides:
- **Analytics:** View usage statistics
- **Logs:** Debug issues in real-time
- **Metrics:** Memory and CPU usage

Access via: App Settings → Analytics/Logs

---

## 💰 Cost

**Streamlit Cloud Free Tier:**
- ✅ 1 private app
- ✅ Unlimited public apps
- ✅ Community support
- ✅ No credit card required

**Perfect for portfolios!**

---

## 🎉 You're Ready!

Your app will be live at:
```
https://your-custom-name.streamlit.app
```

Share it with recruiters and on your resume! 🚀

---

**Questions?** Open an issue on GitHub or contact me!
