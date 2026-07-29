import streamlit as st
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1{
    color:#4CAF50;
}

div[data-testid="metric-container"]{
    border:1px solid #3d3d3d;
    padding:15px;
    border-radius:12px;
    background-color:#1f1f1f;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_products():

   products = pd.read_csv("products_clean.csv")

    text_columns = [
        "title",
        "brand_name",
        "breadcrumbs",
        "product_description"
    ]

    for col in text_columns:
        products[col] = products[col].fillna("")

    products["combined_features"] = (

        products["title"] + " " +

        products["brand_name"] + " " +

        products["breadcrumbs"] + " " +

        products["product_description"]

    )

    return products


products = load_products()

# ----------------------------------------------------
# BUILD RECOMMENDATION MODEL
# ----------------------------------------------------

@st.cache_resource
def build_model(df):

    tfidf = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = tfidf.fit_transform(
        df["combined_features"]
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    indices = pd.Series(
        df.index,
        index=df["title"]
    ).drop_duplicates()

    return cosine_sim, indices


cosine_sim, indices = build_model(products)

# ----------------------------------------------------
# LOAD SENTIMENT MODEL
# ----------------------------------------------------

with open("sentiment_model.pkl","rb") as f:
    sentiment_model = pickle.load(f)

with open("sentiment_tfidf.pkl","rb") as f:
    sentiment_tfidf = pickle.load(f)

# ----------------------------------------------------
# RECOMMENDATION FUNCTION
# ----------------------------------------------------

def recommend_products(product_name, top_n=5):

    idx = indices[product_name]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    product_indices = [
        i[0]
        for i in similarity_scores
    ]

    similarity = [
        round(i[1]*100,2)
        for i in similarity_scores
    ]

    recommendation = products.iloc[
        product_indices
    ][
        [
            "title",
            "brand_name",
            "price_value",
            "rating_stars",
            "rating_count"
        ]
    ].copy()

    recommendation["Similarity (%)"] = similarity

    return recommendation

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "🏠 Home",

        "🛒 Product Recommendation",

        "💬 Sentiment Analysis",

        "ℹ️ About Project"

    ]

)
# ----------------------------------------------------
# HOME PAGE
# ----------------------------------------------------

if page == "🏠 Home":

    st.title("🛍️ E-Commerce Recommendation System")

    st.write("""
Welcome to the **E-Commerce Recommendation System**.

This application helps users discover similar products and
analyze customer reviews using Machine Learning techniques.
""")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📦 Products",
        len(products)
    )

    col2.metric(
        "🏷️ Brands",
        products["brand_name"].nunique()
    )

    col3.metric(
        "⭐ Average Rating",
        round(products["rating_stars"].mean(),2)
    )

    col4.metric(
        "📝 Reviews",
        "6,327"
    )

    st.markdown("---")

    st.subheader("✨ Project Features")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
### 🛒 Product Recommendation

Find similar products using
Content-Based Recommendation
powered by TF-IDF and Cosine Similarity.
""")

        st.info("""
### 📦 Amazon Dataset

Built using Amazon product
and customer review datasets.
""")

    with col2:

        st.info("""
### 💬 Sentiment Analysis

Predict whether customer
reviews are Positive or Negative.
""")

        st.info("""
### ⚙ Machine Learning

Algorithms Used

• TF-IDF

• Cosine Similarity

• Logistic Regression
""")
# ----------------------------------------------------
# PRODUCT RECOMMENDATION PAGE
# ----------------------------------------------------

elif page == "🛒 Product Recommendation":

    st.title("🛒 Product Recommendation")

    st.write(
        "Select a product to receive similar product recommendations."
    )

    selected_product = st.selectbox(
        "Choose a Product",
        sorted(products["title"].unique())
    )

    if st.button("🔍 Recommend Products"):

        selected = products[
            products["title"] == selected_product
        ].iloc[0]

        st.markdown("---")

        st.subheader("Selected Product")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Brand", selected["brand_name"])

        with c2:
            st.metric(
                "Price",
                f"${selected['price_value']:.2f}"
            )

        with c3:
            st.metric(
                "Rating",
                selected["rating_stars"]
            )

        with c4:
            st.metric(
                "Reviews",
                int(selected["rating_count"])
            )

        st.markdown("---")

        st.subheader("Recommended Products")

        recommendations = recommend_products(
            selected_product
        )

        for i, row in recommendations.iterrows():

            with st.container():

                st.markdown("### 🛍️ " + row["title"])

                col1, col2, col3, col4 = st.columns(4)

                col1.write(f"**Brand**")
                col1.write(row["brand_name"])

                col2.write(f"**Price**")
                col2.write(f"${row['price_value']:.2f}")

                col3.write(f"**Rating**")
                col3.write(f"⭐ {row['rating_stars']}")

                col4.write(f"**Similarity**")
                col4.progress(
                    min(row["Similarity (%)"]/100, 1.0),
                    text=f"{row['Similarity (%)']}%"
                )

                st.divider()
 # ----------------------------------------------------
# SENTIMENT ANALYSIS PAGE
# ----------------------------------------------------

elif page == "💬 Sentiment Analysis":

    st.title("💬 Sentiment Analysis")

    st.write(
        "Enter a customer review to predict whether it is Positive or Negative."
    )

    review = st.text_area(
        "Customer Review",
        height=180
    )

    if st.button("Predict Sentiment"):

        if review.strip() == "":

            st.warning(
                "Please enter a review."
            )

        else:

            vector = sentiment_tfidf.transform(
                [review]
            )

            prediction = sentiment_model.predict(
                vector
            )[0]

            st.markdown("---")

            st.subheader("Prediction")

            if prediction == 1:

                st.success(
                    "😊 Positive Review"
                )

                st.info(
                    "The customer expresses a positive opinion about the product."
                )

            else:

                st.error(
                    "😞 Negative Review"
                )

                st.info(
                    "The customer expresses dissatisfaction with the product."
                ) 
# ----------------------------------------------------
# ABOUT PROJECT
# ----------------------------------------------------

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("""
### Project Name

**E-Commerce Recommendation System**

---

### Objective

To recommend similar products using Content-Based Recommendation
and analyze customer reviews using Sentiment Analysis.

---

### Dataset

- Amazon Products Dataset
- Amazon Customer Reviews Dataset

---

### Machine Learning Algorithms

- TF-IDF Vectorization
- Cosine Similarity
- Logistic Regression

---

### Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit

---
### Developed By

**Aman Tiwari**  
**Aman Saxena**  
**Akash**  


""") 
