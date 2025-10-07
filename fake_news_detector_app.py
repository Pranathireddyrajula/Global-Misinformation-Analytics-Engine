import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import warnings
import pandas as pd
import altair as alt
from collections import Counter
from textblob import TextBlob
import numpy as np

# Suppress NLTK's download warnings and pandas warnings
warnings.filterwarnings("ignore", message=".*download.*")
pd.options.mode.chained_assignment = None # Suppress SettingWithCopyWarning

# --- A. STREAMLIT PAGE CONFIG (MUST BE FIRST COMMAND) ---
st.set_page_config(
    page_title="Misinformation Analytics Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 1. CONFIGURATION AND ASSET LOADING ---

# Define paths. 
MODEL_PATH = './model_assets/model.pkl'
VECTORIZER_PATH = './model_assets/vectorizer.pkl'
DATA_PATH_TRUE = './data/DataSet_Misinfo_TRUE.csv'
DATA_PATH_FAKE = './data/DataSet_Misinfo_FAKE.csv'

# Initialize stemmer and stopwords 
ps = PorterStemmer()
with st.spinner("Downloading necessary NLTK resources..."):
    try:
        # Ensure NLTK resources are downloaded (quietly)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('taggers/averaged_perceptron_tagger', quiet=True) 
        all_stopwords = stopwords.words('english')
    except LookupError:
        st.error("NLTK resource download failed.")
        all_stopwords = []


# Load the Model and Vectorizer
@st.cache_resource
def load_ml_assets(model_path, vectorizer_path):
    """Loads the trained ML model and TF-IDF vectorizer."""
    try:
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found. Expected: '{model_path}'")
            return None, None
        if not os.path.exists(vectorizer_path):
            st.error(f"Error: Vectorizer file not found. Expected: '{vectorizer_path}'")
            return None, None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"An error occurred while loading assets: {e}")
        return None, None

# Load the Data for the Dashboard
@st.cache_data
def load_data(true_path, fake_path):
    """Loads and combines the original datasets for analysis, adding word count."""
    with st.spinner("Loading and preprocessing data for dashboard (this runs only once)..."):
        try:
            if not os.path.exists(true_path) or not os.path.exists(fake_path):
                st.error("Error: One or both data files not found. Check the './data' folder.")
                return None
                
            df_true = pd.read_csv(true_path)
            df_fake = pd.read_csv(fake_path)
            
            df_true['target'] = 1
            df_fake['target'] = 0
            
            # Combine, drop NaNs, and shuffle
            df = pd.concat([df_true, df_fake], axis=0).dropna(subset=['text']).sample(frac=1).reset_index(drop=True) 
            
            # <<< CRITICAL OPTIMIZATION: Limit data size for fast startup >>>
            MAX_ROWS_FOR_SPEED = 20000 
            if len(df) > MAX_ROWS_FOR_SPEED:
                df = df.head(MAX_ROWS_FOR_SPEED)

            df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
            df['News Type'] = df['target'].apply(lambda x: 'Real News' if x == 1 else 'Fake News')

            return df
        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")
            return None

# Load assets inside a spinner for initial startup visibility
with st.spinner("Loading Machine Learning Assets (Model and Vectorizer)..."):
    model, vectorizer = load_ml_assets(MODEL_PATH, VECTORIZER_PATH)
    df_full = load_data(DATA_PATH_TRUE, DATA_PATH_FAKE)


# --- 2. PREPROCESSING & PREDICTION ---

def preprocess_text(text):
    """Applies the same cleaning and stemming steps as during training."""
    if not isinstance(text, str):
        text = str(text)

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

def predict_news(text_input, model, vectorizer):
    """Cleans, vectorizes, and predicts the class of the input text."""
    if not text_input or model is None or vectorizer is None:
        return 0, [0.5, 0.5]

    cleaned_text = preprocess_text(text_input)
    try:
        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        return prediction, probability
    except Exception:
        return 0, [0.5, 0.5]


# --- 3. MISINFORMATION RISK INDEX (MRI) CALCULATION (CACHED FOR SPEED) ---
@st.cache_data(show_spinner=False)
def calculate_sentiment(text_input):
    """Calculates Polarity and Subjectivity for a given text."""
    if not text_input or not text_input.strip():
        return 0.0, 0.0
    blob = TextBlob(text_input)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def calculate_mri(text_input, prediction, probability, user_engagement):
    """
    Calculates the Misinformation Risk Index (MRI) based on 3 signals (0-100).
    """
    if text_input is None or not text_input.strip():
        return 0.0, 0.0, 0.0

    # Retrieve cached sentiment
    polarity, subjectivity = calculate_sentiment(text_input)

    # 1. CLASSIFIER RISK (70% Weight)
    classifier_risk = probability[0] * 100 
    
    # 2. ENGAGEMENT SIGNAL (20% Weight)
    engagement_score = user_engagement * 10.0
    
    # 3. SENTIMENT SIGNAL (10% Weight)
    sentiment_risk_raw = np.maximum(abs(polarity), subjectivity)
    sentiment_score = sentiment_risk_raw * 100.0
    
    if prediction == 0:
        mri_score = (
            (classifier_risk * 0.70) + 
            (engagement_score * 0.20) + 
            (sentiment_score * 0.10)
        )
    else:
        mri_score = (sentiment_score * 0.10) 
    
    mri_score = max(0, min(100, mri_score))
    
    return mri_score, polarity, subjectivity


# --- 4. ADVANCED ANALYTICS (EDA) FUNCTIONS ---

def get_top_discriminating_words(model, vectorizer, n_words=10):
    """Extracts the top N words contributing to REAL vs FAKE classification."""
    if model is None or vectorizer is None:
        return None, None
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    word_scores = pd.DataFrame({'Word': feature_names, 'Coefficient': coefficients})
    top_real = word_scores.nlargest(n_words, 'Coefficient')
    top_fake = word_scores.nsmallest(n_words, 'Coefficient')
    return top_real, top_fake

def render_article_length_analysis(df):
    """Renders histogram comparing word counts of real vs fake news using a downsampled set."""
    st.markdown("### Article Length Comparison")
    st.info("Distribution of article word counts. Real News tends to be longer on average.")
    
    # Since df is already sampled in load_data, we can use it directly
    df_chart = df.copy() 
    
    avg_counts = df.groupby('News Type')['word_count'].mean().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        real_avg = avg_counts[avg_counts['News Type'] == 'Real News']['word_count'].iloc[0]
        st.metric("Average Word Count (Real News)", f"{real_avg:.0f} words")
    with col2:
        fake_avg = avg_counts[avg_counts['News Type'] == 'Fake News']['word_count'].iloc[0]
        st.metric("Average Word Count (Fake News)", f"{fake_avg:.0f} words")

    chart = alt.Chart(df_chart).mark_bar(opacity=0.7).encode(
        x=alt.X('word_count', bin=alt.Bin(maxbins=50), title='Article Word Count (Bins)'),
        y=alt.Y('count()', title='Number of Articles (Sampled)'),
        color=alt.Color('News Type', scale=alt.Scale(domain=['Real News', 'Fake News'], range=['#34A853', '#EA4335'])),
        tooltip=['News Type', 'word_count', alt.Tooltip('count()', title='Count (Sampled)')]
    ).properties(
        height=300
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# Word analysis function remains cached for speed once calculated
@st.cache_data
def render_top_trending_words(df, top_n=20):
    """Displays a bar chart of the top N most frequent words in the entire dataset."""
    
    all_text = ' '.join(df['text'].dropna().apply(preprocess_text).tolist())
    words = all_text.split()
    word_counts = Counter(words)
    
    filtered_words = [(word, count) for word, count in word_counts.items() 
                      if len(word) > 2 and word not in all_stopwords and word not in ['said', 'will', 'new', 'year', 'could']] 
    
    top_words_df = pd.DataFrame(filtered_words, columns=['Word', 'Count']).nlargest(top_n, 'Count')

    st.markdown("### Top 20 Most Frequent Words (EDA)")
    chart = alt.Chart(top_words_df).mark_bar(color='#4285F4').encode(
        y=alt.Y('Word', sort='-x', title='Word'),
        x=alt.X('Count', title='Frequency'),
        tooltip=['Word', 'Count']
    ).properties(
        title=f'Top {top_n} Most Frequent Preprocessed Words',
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def render_sentiment_timeline_mockup():
    """Mock-up chart to show how sentiment tracking works over time."""
    st.markdown("### Mock Sentiment Timeline (Demonstration)")
    st.info("This is a mock-up demonstrating how real-world sentiment scores (Polarity) and Subjectivity trends are monitored over time to detect shifts in narrative.")

    months = pd.to_datetime(pd.date_range(start='2024-01-01', periods=12, freq='M'))
    np.random.seed(42)
    polarity = np.random.uniform(-0.1, 0.4, 12) + np.sin(np.linspace(0, 2*np.pi, 12)) * 0.2 
    subjectivity = np.random.uniform(0.3, 0.7, 12)

    df_time = pd.DataFrame({
        'Month': months,
        'Polarity (Emotional Tone)': polarity,
        'Subjectivity (Opinionated)': subjectivity
    })
    
    df_melted = df_time.melt('Month', var_name='Metric', value_name='Value')

    line_chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('Month', title='Time'),
        y=alt.Y('Value', title='Score (-1 to 1)'),
        color='Metric',
        tooltip=['Month', 'Metric', alt.Tooltip('Value', format='.2f')]
    ).properties(
        title='Simulated Monthly Sentiment Trend',
        height=300
    ).interactive()

    st.altair_chart(line_chart, use_container_width=True)


# --- 7. STREAMLIT UI LAYOUT ---

st.title("📰 Global Misinformation Analytics Engine")
st.markdown("A Machine Learning powered tool for classification, risk scoring, and data analytics.")
st.markdown("---")

# Create three tabs for Detector, Dashboard, and Word Analysis
detector_tab, dashboard_tab, analysis_tab = st.tabs(["🚀 News Detector", "📊 Data Dashboard", "🔍 Word Analysis"])

with detector_tab:
    st.header("Real-Time Article Analysis & Risk Scoring")
    
    if model is not None and vectorizer is not None:
        col_input, col_eng = st.columns([3, 1])

        if 'news_input' not in st.session_state:
            st.session_state['news_input'] = ""
        
        with col_input:
            st.markdown("##### Article Content")
            news_input = st.text_area(
                "Paste News Article Text Here:",
                key="text_input_key", 
                height=250,
                placeholder="e.g., 'The President signed an executive order today regarding climate change...' (More text improves accuracy!)",
                value=st.session_state['news_input']
            )
            st.session_state['news_input'] = news_input


        with col_eng:
            st.markdown("##### Simulation Signals (Mock)")
            user_engagement = st.slider(
                "Estimated Engagement/Shares (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key="engagement_slider"
            )
            st.markdown("---")
        
        # --- RESULTS ---
        if news_input.strip():
            
            prediction, probability = predict_news(news_input, model, vectorizer)
            mri_score, polarity, subjectivity = calculate_mri(
                news_input, prediction, probability, user_engagement
            )

            st.markdown("---")
            
            col_class, col_mri, col_sent = st.columns(3)

            with col_class:
                st.markdown("##### CLASSIFIER RESULT (Model Confidence)")
                if prediction == 1:
                    st.success("✅ REAL NEWS", icon="🟢")
                    st.markdown(f"**Confidence:** **{probability[1]*100:.2f}%**")
                else:
                    st.error("❌ FAKE NEWS", icon="🔴")
                    st.markdown(f"**Confidence:** **{probability[0]*100:.2f}%**")
            
            with col_mri:
                st.markdown("##### MISINFORMATION RISK INDEX (MRI)")
                
                # Use Streamlit's required delta_color values
                risk_color_streamlit = "inverse" if mri_score >= 60 else "normal"
                
                delta_text = "High Risk" if mri_score >= 60 else ("Medium Risk" if mri_score >= 30 else "Low Risk")

                st.metric(
                    label="Combined Risk Score (0-100)", 
                    value=f"{mri_score:.1f}", 
                    delta=delta_text,
                    delta_color=risk_color_streamlit
                )
            
            with col_sent:
                st.markdown("##### SENTIMENT & TONE")
                st.metric("Polarity (Emotional Tone)", f"{polarity:.2f}")
                st.metric("Subjectivity (Opinionated)", f"{subjectivity:.2f}")

            st.markdown("---")
            st.info(f"**MRI Breakdown:** Score is based 70% on **Classifier Confidence in Fake**, 20% on **Simulated Engagement** ({user_engagement}/10), and 10% on **Extreme Sentiment**.")
        

        else:
            st.info("Paste an article above and adjust the engagement slider to see the Misinformation Risk Index (MRI) score.")

    else:
        st.warning("The Detector is offline because the model assets could not be loaded. Please ensure `model.pkl` and `vectorizer.pkl` are in the `./model_assets` folder.")


with dashboard_tab:
    st.header("Training Data Analytics")
    
    if df_full is not None:
        # --- 7.1 Key Metrics ---
        total_articles = len(df_full)
        st.metric(label="Total Analyzed Articles (Training Data Sample)", value=f"{total_articles:,}") # Adjusted label
        # ... rest of the dashboard code remains the same
        st.markdown("### Training Data Class Distribution")
        
        counts = df_full['target'].value_counts()
        total = counts.sum()
        
        real_count = counts.get(1, 0)
        fake_count = counts.get(0, 0)
        
        real_percent = (real_count / total) * 100 if total > 0 else 0
        fake_percent = (fake_count / total) * 100 if total > 0 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Real News Articles (1)", value=f"{real_count:,}", delta=f"{real_percent:.1f}% of Total", delta_color="normal")
        with col2:
            st.metric(label="Fake News Articles (0)", value=f"{fake_count:,}", delta=f"{fake_percent:.1f}% of Total", delta_color="inverse")

        # --- 7.2 Article Length Histogram ---
        render_article_length_analysis(df_full)
        
        st.markdown("### Model Performance Summary")
        
        st.success("These are the actual performance metrics from our trained model.")
        
        perf_data = {
            'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            'Value': ['93.12%', '92.18%', '93.10%', '91.27%']
        }
        st.table(pd.DataFrame(perf_data))
        
    else:
        st.warning("The Data Dashboard is offline because the original CSV data files could not be loaded. Please ensure `DataSet_Misinfo_TRUE.csv` and `DataSet_Misinfo_FAKE.csv` are in the `./data` folder.")


with analysis_tab:
    st.header("Advanced Narrative & Word Analysis")
    
    if df_full is not None:
        col_word, col_time = st.columns(2)

        with col_word:
            render_top_trending_words(df_full)
        
        with col_time:
            render_sentiment_timeline_mockup()

        st.markdown("---")
        st.subheader("Model Feature Analysis: Top Discriminating Words")
        if model is not None and vectorizer is not None:
            top_real, top_fake = get_top_discriminating_words(model, vectorizer)
            
            st.markdown("""
            This analysis shows the top 10 words (after cleaning and stemming) that the Logistic Regression model 
            uses to classify news as Real or Fake.
            """)
            
            col_real, col_fake = st.columns(2)
            
            with col_real:
                st.subheader("Words Indicative of REAL News")
                st.table(top_real[['Word', 'Coefficient']].reset_index(drop=True).style.format({"Coefficient": "{:.4f}"}))
                
            with col_fake:
                st.subheader("Words Indicative of FAKE News")
                st.table(top_fake[['Word', 'Coefficient']].reset_index(drop=True).style.format({"Coefficient": "{:.4f}"}))
                
        else:
            st.warning("Word analysis is unavailable because the model or vectorizer could not be loaded.")
    else:
        st.warning("Advanced Analytics is unavailable because the original CSV data files could not be loaded.")
