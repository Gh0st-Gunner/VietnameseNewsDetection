import streamlit as st
import joblib
import os
import pandas as pd
from pyvi import ViTokenizer
import re

# ================
# CONFIGURATION
# ================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ================
# VIETNAMESE STOPWORDS
# ================
VIETNAMESE_STOPWORDS = set([
    'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ',
    'chưa', 'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang',
    'được', 'do', 'đó', 'đây', 'để', 'đều', 'điều', 'đó', 'gì', 'hay', 'hoặc',
    'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một', 'này', 'nên',
    'nếu', 'ngay', 'như', 'những', 'phải', 'qua', 'ra', 'rằng', 'rất', 'rồi', 'sau',
    'sẽ', 'so', 'tại', 'theo', 'thì', 'trong', 'trên', 'và', 'vào', 'vậy', 'vì',
    'với', 'vừa', 'về'
])

# ================
# TEXT PREPROCESSING FUNCTIONS
# ================
def clean_text(text):
    """Advanced text cleaning for Vietnamese news."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_vi_enhanced(text):
    """Enhanced Vietnamese tokenization with stopword removal."""
    text = clean_text(text)
    tokenized = ViTokenizer.tokenize(text)
    words = tokenized.split()
    filtered = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]
    return ' '.join(filtered)

# ================
# LOAD MODELS AND ARTIFACTS
# ================
@st.cache_resource
def load_models():
    """Load all trained models and artifacts."""
    try:
        # Load models
        xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.pkl"))
        
        # Load vectorizer
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_enhanced.pkl"))
        
        # Load label encoder
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder_enhanced.pkl"))
        
        return xgb_model, lr_model, vectorizer, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.error("Make sure you've run train.py first to generate the models.")
        return None, None, None, None

# ================
# STREAMLIT APP
# ================
st.set_page_config(page_title="Vietnamese News Classifier", layout="wide")

st.title("🗞️ Vietnamese News Classifier")
st.markdown("Classify Vietnamese news articles using trained ML models")

# Load models
xgb_model, lr_model, vectorizer, label_encoder = load_models()

if xgb_model is None:
    st.stop()

# Sidebar for info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"""
    **Trained Models:**
    - XGBoost
    - Logistic Regression
    
    **Classes:** {len(label_encoder.classes_)}
    """)
    
    with st.expander("📋 Available Categories"):
        categories_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Label': range(len(label_encoder.classes_))
        })
        st.dataframe(categories_df, use_container_width=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Vietnamese News Content")
    user_input = st.text_area(
        "Paste or type Vietnamese news article:",
        placeholder="Nhập nội dung bài báo tiếng Việt tại đây...",
        height=200
    )

with col2:
    st.subheader("⚙️ Processing Options")
    show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
    show_probabilities = st.checkbox("Show prediction probabilities", value=True)
    apply_button = st.button("🔍 Apply Classification", use_container_width=True)

# Add JavaScript for Ctrl+Enter support
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        // Find and click the apply button
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (btn.innerText.includes('Apply Classification')) {
                btn.click();
            }
        });
    }
});
</script>
""", unsafe_allow_html=True)

# Make predictions
if (user_input.strip() and apply_button) or (user_input.strip() and "apply_button" not in st.session_state):
    st.markdown("---")
    
    # Preprocess text
    if show_preprocessing:
        with st.expander("🔍 Preprocessing Steps"):
            st.write("**Original text:**")
            st.text(user_input[:200] + "..." if len(user_input) > 200 else user_input)
            
            cleaned = clean_text(user_input)
            st.write("**After cleaning:**")
            st.text(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
            
            tokenized = tokenize_vi_enhanced(user_input)
            st.write("**After tokenization & stopword removal:**")
            st.text(tokenized[:200] + "..." if len(tokenized) > 200 else tokenized)
    
    # Process text
    processed_text = tokenize_vi_enhanced(user_input)
    
    if len(processed_text.split()) == 0:
        st.warning("⚠️ Text is empty after preprocessing. Please enter valid content.")
    else:
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Make predictions
        xgb_pred = xgb_model.predict(text_vector)[0]
        lr_pred = lr_model.predict(text_vector)[0]
        
        # Get probabilities
        xgb_proba = xgb_model.predict_proba(text_vector)[0]
        lr_proba = lr_model.predict_proba(text_vector)[0]
        
        # Decode predictions
        xgb_category = label_encoder.inverse_transform([xgb_pred])[0]
        lr_category = label_encoder.inverse_transform([lr_pred])[0]
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("### 🚀 XGBoost")
            st.metric(
                "Predicted Category",
                xgb_category,
                f"Confidence: {xgb_proba[xgb_pred]:.2%}"
            )
            if show_probabilities:
                with st.expander("Probabilities"):
                    proba_df = pd.DataFrame({
                        'Category': label_encoder.classes_,
                        'Probability': xgb_proba
                    }).sort_values('Probability', ascending=False)
                    st.bar_chart(proba_df.set_index('Category'))
        
        with result_col2:
            st.markdown("### 📊 Logistic Regression")
            st.metric(
                "Predicted Category",
                lr_category,
                f"Confidence: {lr_proba[lr_pred]:.2%}"
            )
            if show_probabilities:
                with st.expander("Probabilities"):
                    proba_df = pd.DataFrame({
                        'Category': label_encoder.classes_,
                        'Probability': lr_proba
                    }).sort_values('Probability', ascending=False)
                    st.bar_chart(proba_df.set_index('Category'))
        
        # Consensus
        st.markdown("---")
        predictions = [xgb_category, lr_category]
        
        st.subheader("🤝 Model Consensus")
        if len(set(predictions)) == 1:
            st.success(f"✓ Both models agree: **{predictions[0]}**")
        else:
            consensus_df = pd.DataFrame({
                'Model': ['XGBoost', 'Logistic Regression'],
                'Prediction': predictions
            })
            st.warning("⚠️ Models have different predictions:")
            st.dataframe(consensus_df, use_container_width=True)

else:
    st.info("👆 Enter some Vietnamese news content to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Vietnamese News Classification System | Powered by Streamlit, scikit-learn, XGBoost</small>
</div>
""", unsafe_allow_html=True)