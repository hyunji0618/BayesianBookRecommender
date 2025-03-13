import streamlit as st
import xarray as xr
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from bs4 import BeautifulSoup

st.title("📚Book Recommendation System")

# Load datasets
@st.cache_data
def load_data():
    file_path = "book_recommendations2.nc"
    ds2 = xr.open_dataset(file_path, engine='netcdf4')
    df2 = ds2.to_dataframe().reset_index()

    raw_df = pd.read_csv(r"book_ratings_cleaned.csv").drop(columns=['Unnamed: 0'])
    book_data = pd.read_csv(r"books_updated.csv")

    return df2, raw_df, book_data

df2, raw_df, book_data = load_data()


# Filter User-IDs that have at least one valid book recommendation
valid_users = df2.dropna(subset=['Rec_1', 'Rec_2', 'Rec_3'])['User-ID'].unique()

# User selection with default ID if available
default_user_id = 278843  # Change as needed
if default_user_id in valid_users:
    default_index = list(valid_users).index(default_user_id)
else:
    default_index = 0  # Fallback to first valid user

random_user = st.selectbox("Select User ID", valid_users, index=default_index)


allow_image_search = st.checkbox("Allow Image Search 🔎 (Title + Author)", value=True)

user_books = raw_df[raw_df['User-ID'] == random_user][['ISBN', 'Book-Rating']].sort_values(by='Book-Rating', ascending=False)
user_books = user_books.merge(book_data, how='left', on='ISBN')


top_3_books = user_books.head(3)

# Recommended Books (Reshape: Convert Rec_1, Rec_2, Rec_3 into a single column 'ISBN')
reccd_books = df2[df2['User-ID'] == random_user][['User-ID', 'Rec_1', 'Rec_2', 'Rec_3']]
reccd_books = reccd_books.melt(id_vars=['User-ID'], value_name='ISBN').drop(columns=['variable'])
reccd_books = reccd_books.merge(book_data, how='left', on='ISBN')

# Image Fetching Functions
def is_valid_image(url, max_size=200):
    """Fetches and validates an image from a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=5)
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) < 2000:
            return None
        img = Image.open(BytesIO(response.content))
        if img.getbbox() is None:
            return None
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img
    except (requests.exceptions.RequestException, UnidentifiedImageError):
        return None

def get_bing_image(book_title, author_name):
    """Searches Bing Images for a book cover using title + author."""
    query = f"{book_title} {author_name} book cover".replace(" ", "+")
    search_url = f"https://www.bing.com/images/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all("img", class_="mimg")
        for img in img_tags:
            img_src = img.get("src")
            if img_src and img_src.startswith("https://tse"):
                return img_src
        return None
    except requests.exceptions.RequestException:
        return None

def fetch_image(row):
    """Fetches the best available book image, using Bing if enabled."""
    # First, try provided image URLs
    if "Image-URL-L" in row and isinstance(row["Image-URL-L"], str):
        img = is_valid_image(row["Image-URL-L"])
        if img:
            return img
    if "Image-URL-M" in row and isinstance(row["Image-URL-M"], str):
        img = is_valid_image(row["Image-URL-M"])
        if img:
            return img

    # If enabled, search Bing using Book Title + Author
    if allow_image_search and "Book-Title" in row and "Book-Author" in row:
        bing_img_url = get_bing_image(row["Book-Title"], row["Book-Author"])
        if bing_img_url:
            img = is_valid_image(bing_img_url)
            if img:
                return img

    return None


DEFAULT_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png"

# Display Top 3 Books
st.subheader("📖 Top 3 Rated Books")
cols = st.columns(3)
for i, row in top_3_books.iterrows():
    img = fetch_image(row)
    with cols[i]:
        st.image(
            img if img else DEFAULT_IMAGE,
            caption=f" {row['Book-Title']} \n  {row.get('Book-Author', 'Unknown')}",
            use_container_width=True
        )
# Display Recommended Books (Adjust columns dynamically)
st.subheader("📚 Recommended Books")

if not reccd_books.empty:
    num_books = len(reccd_books)
    cols = st.columns(num_books)  # Ensure we don't pass 0

    for i, row in reccd_books.iterrows():
        img = fetch_image(row)
        with cols[i]:
            st.image(
                img if img else DEFAULT_IMAGE,
                caption=f"{row['Book-Title']} \n {row.get('Book-Author', 'Unknown')}",
                use_container_width=True
            )
else:
    st.write("No recommendations available.")

