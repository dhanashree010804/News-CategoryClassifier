# 📰News Category Classifier

## 📖Project Overview
The **News Category Classifier** is a web application based on machine learning that classifies news articles or headlines into specified categories automatically.
It employs **TF-IDF vectorization** and several supervised learning techniques to determine if a given news belongs to **Business, Entertainment, Politics, Sports, or Tech**, etc.

The project handles:
- Developing strong text-classification models on the **BBC News Dataset** (training).
- Cross-dataset generalization testing on the **AG News Dataset** (testing).
- Having an interactive Streamlit web application for real-time classification and visualization.

---

## 🗂️ Dataset Overview

### **1️⃣Training Dataset — BBC News Dataset**
This dataset includes news articles scraped from the BBC website.
There is one label per article under one of the five broad categories:

| Category | Description | Number of Articles |
|-----------|--------------|--------------------|
| **Business** | Finance, trade, economy, and markets | 510 |
| **Entertainment** | Movies, music, TV, and arts | 386 |
| **Politics** | Government, policies, and elections | 417 |
| **Sport** | Matches, tournaments, and athletes | 511 |
| **Tech** | Innovations, gadgets, and IT news | 401 |
<img width="439" height="281" alt="image" src="https://github.com/user-attachments/assets/c86b1218-4532-482c-b722-323a75ac2aa8" />

**Key Preprocessing Steps:**
- Text cleaning (punctuation, stopwords removal)
- Tokenization and normalization
- TF-IDF feature extraction

📁 *Dataset Source:* [BBC Fulltext and Category Dataset (Kaggle)](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)

---

### **2️⃣Testing Dataset — AG News Dataset**
To test the ability of trained models to generalize, testing was conducted on the **AG News Classification Dataset**.
It has news titles and brief descriptions classified into four categories:

| Category | Description | Sample Size |
|-----------|--------------|-------------|
| **World** | International and global news | 30,000 |
| **Sports** | Sports events and news | 30,000 |
| **Business** | Corporate and market news | 30,000 |
| **Sci/Tech** | Technology and science updates | 30,000 |

This cross-dataset evaluation guarantees the classifier is working optimally outside the initial training space.

📁 *Dataset Source:* [AG News Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

---

## Models Used
The models listed below were trained and benchmarked:

| Model | Description | Training Accuracy |
|--------|--------------|------------------|
| **Logistic Regression** | Linear model well adapted to high-dimensional TF-IDF data | 96.18% |
| **Naive Bayes (MultinomialNB)** | Probabilistic bag-of-words text model | 96.18% |
| **Support Vector Machine (SVM)** | Good high-margin classifier with great generalization | 96.63% |

All models were dumped using `joblib` for usage within the Streamlit application.

##  Model Training and Evaluation

This chapter gives in-depth information about how the models were trained, validated, and tested on two datasets — the **BBC News Dataset** to train and the **AG News Dataset** to test.

---

###  Training Information (BBC Dataset)

####  NLP Preprocessing Pipeline

| Step | Description | Libraries Used |
|------|--------------|----------------|
| 1️⃣ Data Cleaning | Punctuation removed, digits removed, lowercased | `re`, `string` |
| 2️⃣ Tokenization | Tokenized text into words | `nltk.word_tokenize` |
| 3️⃣ Stopword Removal | Removed stop words (common non-informative words) | `nltk.corpus.stopwords` |
| 4️⃣ Lemmatization | Lematized words into root form | `WordNetLemmatizer` |
| 5️⃣ TF-IDF Vectorization | Transformed text into numerical features | `sklearn.feature_extraction.text.TfidfVectorizer(max_features=5000, stop_words='english')`

All models were trained on the TF-IDF features and cross-checked via a **train-test split (80%-20%)**.

---

### Training Results and Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|--------|-----------|-----------|----------|-----------|--------|
| Logistic Regression | 96.18% | 0.9626 | 0.9618 | 0.9618 | Strong linear baseline |
| Naive Bayes | 96.18% | 0.9631 | 0.9618 | 0.9616 | Fast and simple, slightly worse performance |
| **SVM (Best)** | **96.63%** | **0.9671** | **0.9663** | **0.9664** | Best performing model overall |

✅ **Observation:**
SVM outranked other models consistently in accuracy and generalization performance and was chosen as the ultimate deployment solution.

---
---  
## Testing and Cross-Dataset Validation (AG Dataset)

To make sure the trained models generalize effectively, they were tested on the **AG News Dataset** — a huge dataset of 120,000 news articles categorized into 4 categories.

---

### Testing Steps

1️⃣ Loaded trained models (`.joblib`) and TF-IDF vectorizer.
2️⃣ Preprocessed AG dataset with the same text cleaning pipeline.
3️⃣ Used the vectorizer for feature transformation.
4️⃣ Made predictions for classes using every trained model.
5️⃣ Checked model predictions against ground truth labels.
6️⃣ Computed accuracy, precision, recall, and F1-score.
7️⃣ Visualized confusion matrix and accuracy comparison chart.

---

### Test Results and Comparison

| Metric | Logistic Regression | Naive Bayes | SVM |
|---------|--------------------|--------------|------|
| **Accuracy** | 0.9519 | 0.9474 | 0.9471 |
| **Precision** | 0.9496 | 0.9694 | 0.9653 |
| **Recall** | 0.9544 | 0.9238 | 0.9276 |
| **F1-Score** | 0.9520 | 0.9461 | 0.9461 |
<img width="516" height="320" alt="image" src="https://github.com/user-attachments/assets/eb22564f-ecb9-4116-84a6-38f39075049f" />

> **Observation:** Logistic Regression achieved the best overall performance on the AG dataset, showing excellent adaptability and generalization from BBC to AG News data.

- The model trained on **BBC dataset** generalized effectively on **AG dataset**.
- **Logistic Regression** emerged as the most stable performer.
- Both **Naive Bayes** and **SVM** also showed high accuracy (~94%), confirming consistent classification capability across domains.

---

## 🌐 Streamlit Web Application

An interactive **Streamlit web app** was built to make the News Category Classifier accessible to users in real time.  
The app leverages the trained **TF-IDF vectorizer** and **SVM model** to classify news headlines into categories.

🟢 **Live App (Deployed Link):**  
👉 [https://news-categoryclassifier.streamlit.app/](https://news-categoryclassifier.streamlit.app/)

---

### 🧭 How the App Works

#### 📝 **Input:**
Users can enter **one or multiple news headlines**, separated by new lines.  
Each headline is treated as an independent input for classification.

**Example Input:**
Government announces new tax policy for startups.
Apple unveils the latest iPhone with AI features.
The national team wins the championship title.


---

#### ⚙️ **Processing Pipeline:**
1. Text inputs are preprocessed and transformed using the **saved TF-IDF vectorizer**.  
2. Each headline is passed through the **trained models** (Logistic Regression, Naive Bayes, and SVM).  
3. Predictions are combined into a structured table for comparison.  
4. Streamlit dynamically visualizes category distributions and model confidence.

---

#### 📋 **Output:**
- A **table** displaying each input headline with predicted categories from all models.  
- A **bar chart** summarizing how many headlines belong to each category.  
- Instant feedback and results rendered directly in the browser.

**Example Output Table:**

| News Headline | Logistic Regression | Naive Bayes | SVM |
|----------------|---------------------|--------------|-----|
| Government announces new tax policy for startups. | Business | Business | Business |
| Apple unveils the latest iPhone with AI features. | Tech | Tech | Tech |
| The national team wins the championship title. | Sports | Sports | Sports |

---

### 🖥️ How to Run the Streamlit App Locally

#### **Step 1️⃣ – Clone the Repository**
```bash
git clone https://github.com/dhanashree010804/News-CategoryClassifier.git
cd News-CategoryClassifier
```
#### **Step 2️⃣ – Install Dependencies**
```bash
#Make sure Python 3.8+ is installed, then run:
pip install -r requirements.txt
```
#### **Step 3️⃣ – Launch the App**
```bash
streamlit run app.py
```
---

## 👩‍💻 Author

**Dhanashree Giriya**  
🎓 *Data Science & Machine Learning Enthusiast*  
📍 *India*

🔗 **GitHub:** [@dhanashree010804](https://github.com/dhanashree010804)  
🔗 **Live Streamlit App:** [news-categoryclassifier.streamlit.app](https://news-categoryclassifier.streamlit.app/)  
🔗 **Project Repository:** [News-CategoryClassifier (GitHub)](https://github.com/dhanashree010804/News-CategoryClassifier)

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it for educational or research purposes, provided that proper credit is given to the author.

