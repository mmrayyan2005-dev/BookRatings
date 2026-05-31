import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ── Load artifacts ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "model_artifacts"
    return (
        joblib.load(f"{base}/best_model.joblib"),
        joblib.load(f"{base}/robust_scaler.joblib"),
        joblib.load(f"{base}/onehot_encoder.joblib"),
        joblib.load(f"{base}/X_train_columns.joblib"),
        joblib.load(f"{base}/category_stats.joblib"),
        joblib.load(f"{base}/genre_enrichment.joblib"),
    )

best_model, scaler, encoder, X_cols, cat_stats_df, genre_enrich_df = load_artifacts()

GENRE_TO_CATEGORIES = {
    'mystery':    ['Mystery', 'Crime', 'Thriller'],
    'romance':    ['Romance', 'Adult Fiction', 'Womens Fiction', 'New Adult'],
    'self-help':  ['Self Help', 'Health', 'Parenting', 'Psychology'],
    'humor':      ['Humor'],
    'philosophy': ['Philosophy', 'Religion', 'Spirituality'],
    'fiction':    ['Fiction', 'Historical Fiction', 'Science Fiction',
                   'Fantasy', 'Young Adult', 'Classics', 'Horror', 'Poetry'],
}
CAT_TO_GENRE = {cat: g for g, cats in GENRE_TO_CATEGORIES.items() for cat in cats}

genre_richness = dict(zip(genre_enrich_df['genre'],
                          genre_enrich_df['genre_tag_richness_score']))
genre_tagcount = dict(zip(genre_enrich_df['genre'],
                          genre_enrich_df['avg_tag_count']))

def extract_stock(s):
    if 'In stock' in str(s):
        m = re.search(r'(\d+)', str(s))
        return int(m.group(1)) if m else 1
    return 0

def preprocess_input(title, price, category, availability):
    row       = cat_stats_df[cat_stats_df['category'] == category]
    cat_avg   = float(row['category_avg_price'].values[0])  if len(row) else 35.0
    cat_count = float(row['category_book_count'].values[0]) if len(row) else 20.0

    title_len  = len(title)
    word_count = len(title.split())
    has_num    = int(bool(re.search(r'\d', title)))
    stock      = extract_stock(availability)
    log_p      = np.log1p(price)
    ppc        = price / max(title_len, 1)
    genre      = CAT_TO_GENRE.get(category, 'general')
    richness   = genre_richness.get(genre, genre_richness.get('general', 3.0))
    tagcount   = genre_tagcount.get(genre,  genre_tagcount.get('general', 3.0))

    num_df = pd.DataFrame(
        [[price, title_len, stock, word_count, ppc,
          cat_avg, cat_count, log_p, richness, tagcount]],
        columns=['price', 'title_length', 'stock_count', 'word_count_title',
                 'price_per_title_char', 'category_avg_price',
                 'category_book_count', 'log_price',
                 'genre_tag_richness_score', 'avg_tag_count_for_genre']
    )
    num_scaled = pd.DataFrame(scaler.transform(num_df), columns=num_df.columns)

    # Use exactly the columns the encoder was fitted on
    encoder_cols = list(encoder.feature_names_in_)
    cat_row = {}
    for col in encoder_cols:
        if col == 'category':    cat_row[col] = category
        elif col == 'availability': cat_row[col] = availability
        else:                    cat_row[col] = ''
    cat_df  = pd.DataFrame([cat_row], columns=encoder_cols)
    cat_enc = encoder.transform(cat_df)
    cat_encoded_df = pd.DataFrame(
        cat_enc, columns=encoder.get_feature_names_out(encoder_cols))

    row_assembled = pd.concat(
        [num_scaled,
         pd.DataFrame([[has_num]], columns=['has_number_in_title']),
         cat_encoded_df], axis=1)

    final = pd.DataFrame(0.0, index=[0], columns=X_cols)
    for c in row_assembled.columns:
        if c in final.columns:
            final[c] = row_assembled[c].values
    return final

# ── UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Book Rating Predictor", page_icon="📚")
st.title("📚 Book Rating Predictor")
st.markdown("Enter book details to predict its **star rating (1–5)**.")

CATEGORIES = sorted([
    'Travel','Mystery','Historical Fiction','Sequential Art','Classics',
    'Philosophy','Romance','Womens Fiction','Fiction','Childrens',
    'Religion','Nonfiction','Music','Science Fiction','Fantasy',
    'New Adult','Young Adult','Science','Poetry','Horror','Art',
    'Psychology','Autobiography','Parenting','Adult Fiction','Humor',
    'Spirituality','Christian Fiction','Business','Historical',
    'Contemporary','Self Help','Politics','Health','Thriller','Crime',
])

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        title    = st.text_input("Book Title", "The Hitchhiker\'s Guide to the Galaxy")
        price    = st.number_input("Price (GBP)", min_value=0.01, max_value=500.0,
                                   value=15.99, step=0.01)
    with col2:
        category = st.selectbox("Category", CATEGORIES)
        availability = st.selectbox("Availability", [
            'In stock (20 available)', 'In stock (5 available)',
            'In stock (1 available)', 'In stock', 'Out of stock'])
    submitted = st.form_submit_button("🔮 Predict Rating", use_container_width=True)

if submitted:
    errors = []
    if not title.strip():
        errors.append("Book title cannot be empty.")
    if len(title.strip()) < 2:
        errors.append("Book title must be at least 2 characters.")
    if price <= 0:
        errors.append("Price must be greater than 0.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        try:
            with st.spinner("Predicting…"):
                processed = preprocess_input(title.strip(), price, category, availability)
                pred      = best_model.predict(processed)[0]
                proba     = best_model.predict_proba(processed)[0]
                conf      = float(np.max(proba)) * 100

            st.success(f"⭐ Predicted Rating: **{pred} / 5 Stars**")
            st.info(f"🎯 Confidence: **{conf:.1f}%**")
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame({
                'Star Rating': [f"{i} ⭐" for i in range(1, 6)],
                'Probability': [round(p, 4) for p in proba]
            })
            st.bar_chart(proba_df.set_index('Star Rating'))
            with st.expander("Show raw probabilities"):
                st.dataframe(proba_df)
        except FileNotFoundError:
            st.error("Model artifacts not found. Run the notebook first to generate model_artifacts/.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("ML Semester Project — Book Rating Predictor | books.toscrape.com + quotes.toscrape.com")
