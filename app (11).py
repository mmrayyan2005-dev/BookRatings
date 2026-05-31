import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import requests
from bs4 import BeautifulSoup
import time

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="Book Rating Predictor", page_icon="📚", layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
.star-row { font-size: 2rem; letter-spacing: 4px; }
.star-filled { color: #f5a623; }
.star-empty  { color: #d0d0d0; }
.prob-bar-wrap { margin: 4px 0; }
.prob-label { font-size: 0.85rem; color: #555; width: 60px; display: inline-block; }
.prob-track { display: inline-block; background: #eee; border-radius: 6px;
              width: 60%; height: 18px; vertical-align: middle; position: relative; }
.prob-fill  { background: linear-gradient(90deg, #f5a623, #e07b00);
              height: 100%; border-radius: 6px; transition: none !important; }
.prob-pct   { font-size: 0.82rem; color: #333; margin-left: 8px; }
.result-box { background: #fffbf0; border: 2px solid #f5a623; border-radius: 12px;
              padding: 1.2rem 1.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("best_model.joblib"),
        joblib.load("robust_scaler.joblib"),
        joblib.load("onehot_encoder.joblib"),
        joblib.load("X_train_columns.joblib"),
        joblib.load("category_stats.joblib"),
        joblib.load("genre_enrichment.joblib"),
    )

# ── Scrape book list for autocomplete ───────────────────────────
@st.cache_data(show_spinner="Loading book catalogue…")
def load_book_catalogue():
    """Scrape all book titles, prices and categories from books.toscrape.com."""
    BASE = "https://books.toscrape.com/"
    RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    books = []
    try:
        resp = requests.get(BASE, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        cat_links = soup.select("ul.nav-list ul li a")
        for cat_tag in cat_links[:50]:
            cat_name = cat_tag.text.strip()
            cat_url  = BASE + cat_tag["href"]
            page_url = cat_url
            while page_url:
                try:
                    r = requests.get(page_url, timeout=8)
                    ps = BeautifulSoup(r.text, "html.parser")
                    for art in ps.select("article.product_pod"):
                        try:
                            title = art.h3.a["title"]
                            price = float(re.sub(r"[^0-9.]", "", art.select_one(".price_color").text.strip()))
                            avail = art.select_one(".availability").text.strip()
                            rating = RATING_MAP.get(art.p["class"][1], 0)
                            books.append({"title": title, "price": price,
                                          "category": cat_name, "availability": avail,
                                          "actual_rating": rating})
                        except Exception:
                            pass
                    nxt = ps.select_one("li.next a")
                    if nxt:
                        page_url = cat_url.rsplit("/", 1)[0] + "/" + nxt["href"]
                    else:
                        break
                    time.sleep(0.2)
                except Exception:
                    break
    except Exception:
        pass
    return pd.DataFrame(books)

# ── Genre helpers ───────────────────────────────────────────────
GENRE_TO_CATEGORIES = {
    "mystery":    ["Mystery", "Crime", "Thriller"],
    "romance":    ["Romance", "Adult Fiction", "Womens Fiction", "New Adult"],
    "self-help":  ["Self Help", "Health", "Parenting", "Psychology"],
    "humor":      ["Humor"],
    "philosophy": ["Philosophy", "Religion", "Spirituality"],
    "fiction":    ["Fiction", "Historical Fiction", "Science Fiction",
                   "Fantasy", "Young Adult", "Classics", "Horror", "Poetry"],
}
CAT_TO_GENRE = {cat: g for g, cats in GENRE_TO_CATEGORIES.items() for cat in cats}

def extract_stock(s):
    if "In stock" in str(s):
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else 1
    return 0

def preprocess_input(title, price, category, availability,
                     scaler, encoder, X_cols, cat_stats_df, genre_enrich_df):
    genre_richness = dict(zip(genre_enrich_df["genre"],
                               genre_enrich_df["genre_tag_richness_score"]))
    genre_tagcount = dict(zip(genre_enrich_df["genre"],
                               genre_enrich_df["avg_tag_count"]))

    row       = cat_stats_df[cat_stats_df["category"] == category]
    cat_avg   = float(row["category_avg_price"].values[0])  if len(row) else 35.0
    cat_count = float(row["category_book_count"].values[0]) if len(row) else 20.0

    title_len  = len(title)
    word_count = len(title.split())
    has_num    = int(bool(re.search(r"\d", title)))
    stock      = extract_stock(availability)
    log_p      = np.log1p(price)
    ppc        = price / max(title_len, 1)
    genre      = CAT_TO_GENRE.get(category, "general")
    richness   = genre_richness.get(genre, genre_richness.get("general", 3.0))
    tagcount   = genre_tagcount.get(genre,  genre_tagcount.get("general", 3.0))

    num_df = pd.DataFrame(
        [[price, title_len, stock, word_count, ppc,
          cat_avg, cat_count, log_p, richness, tagcount]],
        columns=["price", "title_length", "stock_count", "word_count_title",
                 "price_per_title_char", "category_avg_price",
                 "category_book_count", "log_price",
                 "genre_tag_richness_score", "avg_tag_count_for_genre"]
    )
    num_scaled = pd.DataFrame(scaler.transform(num_df), columns=num_df.columns)

    encoder_cols = list(encoder.feature_names_in_)
    cat_row = {}
    for col in encoder_cols:
        if col == "category":      cat_row[col] = category
        elif col == "availability": cat_row[col] = availability
        else:                       cat_row[col] = ""
    cat_df  = pd.DataFrame([cat_row], columns=encoder_cols)
    cat_enc = encoder.transform(cat_df)
    cat_encoded_df = pd.DataFrame(
        cat_enc, columns=encoder.get_feature_names_out(encoder_cols))

    row_assembled = pd.concat(
        [num_scaled,
         pd.DataFrame([[has_num]], columns=["has_number_in_title"]),
         cat_encoded_df], axis=1)

    final = pd.DataFrame(0.0, index=[0], columns=X_cols)
    for c in row_assembled.columns:
        if c in final.columns:
            final[c] = row_assembled[c].values
    return final

# ── Render fixed star rating ─────────────────────────────────────
def render_stars(rating):
    filled = "★" * int(rating)
    empty  = "☆" * (5 - int(rating))
    return (f'<span class="star-row">'
            f'<span class="star-filled">{filled}</span>'
            f'<span class="star-empty">{empty}</span>'
            f'</span>')

# ── Render fixed probability bars (no hover shift) ───────────────
def render_prob_bars(proba):
    html = "<div style='margin-top:12px'>"
    for i, p in enumerate(proba, 1):
        pct = p * 100
        html += f"""
        <div class="prob-bar-wrap">
          <span class="prob-label">{i} ⭐</span>
          <span class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%"></div>
          </span>
          <span class="prob-pct">{pct:.1f}%</span>
        </div>"""
    html += "</div>"
    return html

# ── UI ──────────────────────────────────────────────────────────
st.title("📚 Book Rating Predictor")
st.markdown("Search for a book or enter details manually to predict its **star rating (1–5)**.")

CATEGORIES = sorted([
    "Travel","Mystery","Historical Fiction","Sequential Art","Classics",
    "Philosophy","Romance","Womens Fiction","Fiction","Childrens",
    "Religion","Nonfiction","Music","Science Fiction","Fantasy",
    "New Adult","Young Adult","Science","Poetry","Horror","Art",
    "Psychology","Autobiography","Parenting","Adult Fiction","Humor",
    "Spirituality","Christian Fiction","Business","Historical",
    "Contemporary","Self Help","Politics","Health","Thriller","Crime",
])

AVAILABILITY_OPTIONS = [
    "In stock (20 available)", "In stock (5 available)",
    "In stock (1 available)", "In stock", "Out of stock"
]

# ── Load book catalogue (with spinner, cached) ───────────────────
catalogue_df = load_book_catalogue()
has_catalogue = len(catalogue_df) > 0

# ── Book search / autocomplete ───────────────────────────────────
st.subheader("🔍 Search Book")

search_query = st.text_input("Type a book title to search", placeholder="e.g. Sharp Objects, Tipping…")

selected_book = None
if search_query and has_catalogue:
    matches = catalogue_df[
        catalogue_df["title"].str.contains(search_query, case=False, na=False)
    ].head(10)
    if not matches.empty:
        options = ["— select a book —"] + matches["title"].tolist()
        choice = st.selectbox("Matching books:", options)
        if choice != "— select a book —":
            selected_book = matches[matches["title"] == choice].iloc[0]
            st.success(f"✅ Selected: **{selected_book['title']}**")
    else:
        st.info("No matching books found. Fill in the details below manually.")
elif search_query and not has_catalogue:
    st.warning("Book catalogue could not be loaded (network unavailable). Please fill in details manually.")

st.markdown("---")
st.subheader("📋 Book Details")

# Pre-fill from selected book, else use defaults
default_title    = selected_book["title"]    if selected_book is not None else "The Hitchhiker's Guide to the Galaxy"
default_price    = float(selected_book["price"]) if selected_book is not None else 15.99
default_category = selected_book["category"] if selected_book is not None else "Fiction"
default_avail    = selected_book["availability"] if selected_book is not None else "In stock (20 available)"

# Normalise availability to match dropdown options
if selected_book is not None:
    raw_avail = str(selected_book["availability"])
    if "In stock" in raw_avail:
        m = re.search(r"\d+", raw_avail)
        if m:
            n = int(m.group())
            if n >= 20:
                default_avail = "In stock (20 available)"
            elif n >= 5:
                default_avail = "In stock (5 available)"
            else:
                default_avail = "In stock (1 available)"
        else:
            default_avail = "In stock"
    else:
        default_avail = "Out of stock"

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Book Title", value=default_title)
    # No max_value set — price is unlimited
    price = st.number_input("Price (GBP)", min_value=0.01, value=default_price, step=0.01,
                            format="%.2f")
with col2:
    cat_idx = CATEGORIES.index(default_category) if default_category in CATEGORIES else 0
    category = st.selectbox("Category", CATEGORIES, index=cat_idx)
    av_idx = AVAILABILITY_OPTIONS.index(default_avail) if default_avail in AVAILABILITY_OPTIONS else 0
    availability = st.selectbox("Availability", AVAILABILITY_OPTIONS, index=av_idx)

predict_btn = st.button("🔮 Predict Rating", use_container_width=True, type="primary")

if predict_btn:
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
            best_model, scaler, encoder, X_cols, cat_stats_df, genre_enrich_df = load_artifacts()
            with st.spinner("Predicting…"):
                processed = preprocess_input(
                    title.strip(), price, category, availability,
                    scaler, encoder, X_cols, cat_stats_df, genre_enrich_df
                )
                pred  = best_model.predict(processed)[0]
                proba = best_model.predict_proba(processed)[0]
                conf  = float(np.max(proba)) * 100

            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            st.markdown(
                f'<div class="result-box">'
                f'<b style="font-size:1.1rem">Predicted Rating: {pred} / 5 Stars</b><br>'
                f'{render_stars(pred)}'
                f'<br><span style="color:#888;font-size:0.9rem">Confidence: <b>{conf:.1f}%</b></span>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.subheader("📊 Probability by Star Rating")
            st.markdown(render_prob_bars(proba), unsafe_allow_html=True)

            with st.expander("Show raw probabilities table"):
                proba_df = pd.DataFrame({
                    "Star Rating": [f"{i} ⭐" for i in range(1, 6)],
                    "Probability": [f"{p:.4f}" for p in proba],
                    "Percentage":  [f"{p*100:.1f}%" for p in proba],
                })
                st.dataframe(proba_df, hide_index=True)

        except FileNotFoundError:
            st.error("⚠️ Model artifacts not found. Make sure all .joblib files are in the root of your repo.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("ML Semester Project — Book Rating Predictor | books.toscrape.com + quotes.toscrape.com")
