# 📰 Fake News Classifier

A machine learning project that classifies news articles as **real** or **fake** based on their textual content, built as part of a Machine Learning course project (BTech Data Science and Engineering, Second Year, Section B).

**Authors:** Pattabhi Rama (230968033), Prayas Gupta (230968236)

## Overview

This project trains and compares multiple machine learning models to detect fake news using TF-IDF vectorized text features. It also includes a simple [Streamlit](https://streamlit.io/) web app for interactive predictions.

## Project Workflow

1. **Dataset** — A labeled dataset (`test.csv`) containing article titles, text content, and a `fake` label (0 = real, 1 = fake).
2. **Preprocessing**
   - Dropped unnecessary/unnamed columns.
   - Combined `title` and `text` into a single `content` field.
   - Cleaned text by removing punctuation and numbers, and lowercasing.
3. **Train-Test Split** — 80% training / 20% testing.
4. **Vectorization** — Applied `TfidfVectorizer` (English stop words removed, `max_df=0.7`) to convert cleaned text into numerical features.
5. **Models Trained**
   - **Multinomial Naive Bayes**
   - **Logistic Regression**
   - **XGBoost Classifier**
6. **Evaluation** — Accuracy score, classification report, and confusion matrix for each model.
7. **Visualization** — Confusion matrix heatmap and class balance plot using Seaborn/Matplotlib.
8. **Model Persistence** — Trained model and TF-IDF vectorizer saved as `.pkl` files using `pickle`.
9. **Deployment** — A Streamlit app (`fake_news_model.pkl` + `tfidf_vectorizer.pkl`) that lets users paste in news text and get a real/fake prediction.

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- Seaborn / Matplotlib
- Streamlit
- pickle

## Repository Structure

```
├── news_classifier.ipynb      # Main notebook: preprocessing, training, evaluation
├── test.csv                   # Dataset (title, text, fake label)
├── fake_news_model.pkl        # Saved trained model
├── tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
└── README.md
```

## Getting Started

### 1. Install dependencies

```bash
pip install pandas scikit-learn xgboost seaborn matplotlib streamlit
```

### 2. Run the notebook

Open `news_classifier.ipynb` in Jupyter and run all cells to preprocess the data, train the models, and generate the saved `.pkl` files.

### 3. Run the Streamlit app

Once the model and vectorizer `.pkl` files are generated, launch the app:

```bash
streamlit run app.py
```

> Note: extract the Streamlit code section from the notebook into a standalone `app.py` file to run it outside the notebook.

Then paste any news article text into the input box and click **Predict** to see whether it's classified as real or fake.

## Results

The notebook compares the accuracy of three models:

| Model | Notes |
|---|---|
| Multinomial Naive Bayes | Baseline, fast to train |
| Logistic Regression | Strong performance and interpretability |
| XGBoost | Best accuracy, more computationally expensive |

Exact accuracy scores, classification reports, and confusion matrices for each model are printed in the notebook output.

## Future Improvements

- Hyperparameter tuning (e.g., GridSearch/RandomizedSearch) for each model.
- Cross-validation for more robust performance estimates.
- Experiment with word embeddings (Word2Vec, GloVe, or transformer-based embeddings) instead of TF-IDF.
- Expand the dataset for better generalization.

## License

This project was created for academic purposes as part of a Machine Learning coursework assignment.
