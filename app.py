import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load Trained Models and Vectorizer
# -------------------------------
st.set_page_config(page_title="📰 News Category Classifier", layout="wide")
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    models = {
        "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
        "Naive Bayes": joblib.load("naive_bayes_model.joblib"),
        "SVM": joblib.load("svm_model.joblib")
    }
    return tfidf, models

tfidf, models = load_models()

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📰 News Classification Dashboard")
st.markdown("Classify multiple news headlines or articles using trained ML models.")

# Example input
st.sidebar.header("💡 Example Inputs")
st.sidebar.write("""
1. Government announces new education policy.  
2. Team wins the world cup after thrilling match.  
3. Stock markets hit record highs amid investor optimism.  
4. New smartphone released with advanced camera.  
""")

# User input
st.subheader("📝 Enter Multiple News Headlines")
news_input = st.text_area("Enter one news headline per line:", height=200)

if st.button("Classify News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter at least one news headline.")
    else:
        # Split into multiple lines
        news_list = [line.strip() for line in news_input.split("\n") if line.strip()]
        X = tfidf.transform(news_list)

        # Store results for all models
        results = {}
        for name, model in models.items():
            preds = model.predict(X)
            results[name] = preds

        # Combine results into a DataFrame
        df = pd.DataFrame({"News": news_list})
        for model_name, preds in results.items():
            df[model_name + " Prediction"] = preds

        st.success("✅ Classification Completed!")
        st.subheader("📊 Model Predictions")
        st.dataframe(df, use_container_width=True)

        # -------------------------------
        # Visualization
        # -------------------------------
        st.subheader("📈 Prediction Distribution")

        # Melt for visualization
        df_melt = df.melt(id_vars="News", var_name="Model", value_name="Category")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df_melt, x="Category", hue="Model", ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, and Joblib.")
