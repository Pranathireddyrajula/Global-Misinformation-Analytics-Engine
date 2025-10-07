Global Misinformation Analytics Engine(NLP / Streamlit)

Project Overview
This project is a sophisticated, end-to-end Machine Learning web application designed to classify news articles as either Real (Credible) or Fake (Misinformation). It uses Natural Language Processing (NLP) techniques, specifically the TF-IDF Vectorizer and a Logistic Regression model, trained on a comprehensive dataset of labeled news articles.

The application is deployed as an interactive web dashboard using Streamlit, featuring:

News Detector: A real-time classification tool that provides a verdict (Real/Fake) and a confidence score.
.
Data Dashboard: Visual analytics showing the training data distribution and the model's performance metrics (Accuracy: 93.12%, F1-Score: 92.18%).

Word Analysis: Explainable AI (XAI) feature that displays the top words (features) the model uses to discriminate between Real and Fake news.

Key Technologies
Modeling: Scikit-learn (Logistic Regression, TF-IDF)

Deployment: Streamlit

NLP: NLTK

Visualization: Pandas, Altair

Getting Started
1. Repository Structure
Ensure your local project folder (D:\GMAE) is structured as follows before uploading to GitHub:

Fake_News_Detector/
├── fake_news_detector_app.py  <-- The main Streamlit application
├── model_assets/
│   ├── model.pkl              <-- The trained Logistic Regression model
│   └── vectorizer.pkl         <-- The fitted TF-IDF Vectorizer
├── data/
│   ├── DataSet_Misinfo_FAKE.csv  <-- Fake articles dataset
│   └── DataSet_Misinfo_TRUE.csv  <-- Real articles dataset
├── README.md                  <-- This file
├── requirements.txt           <-- List of required Python packages

[Datasets-kaggle
 DataSet_Misinfo_FAKE.csv  
│DataSet_Misinfo_TRUE.csv]
-->you can go to kaggle and find these datasets
    

2. Installation and Setup
Clone the Repository:

git clone https://github.com/Pranathireddyrajula/Global-Misinformation-Analytics-Engine.git
cd Fake_News_Detector

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows

Install Dependencies:

pip install -r requirements.txt

Download NLTK Data: The app requires NLTK's punkt and stopwords data. Run this command once:

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

3. Running the Application
Start the Streamlit application from the root directory:

streamlit run fake_news_detector_app.py



Model Performance
Metric

Score

Interpretation

Accuracy

93.12%

Overall percentage of correct predictions.

F1-Score

92.18%

Balanced measure of Precision and Recall.

Precision

93.10%

When the model predicts REAL, it is correct 93.10% of the time.

Recall

91.27%

The model captures 91.27% of all actual REAL articles.

