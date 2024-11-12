import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, font

# Data
data = {
    'title': ['The Matrix', 'Avengers', 'The Dark Knight', 'Shutter Island', 'Interstellar'],
    'genre': ['Action Sci-Fi', 'Action Adventure', 'Action Crime', 'Mystery Thriller', 'Adventure Drama']
}
movie_df = pd.DataFrame(data)

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_df['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movie_df[movie_df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:5]  # Show top 4 recommendations
        movie_indices = [i[0] for i in sim_scores]
        return movie_df[['title', 'genre']].iloc[movie_indices]
    except IndexError:
        return None

# Display recommendations
def show_recommendations():
    movie_title = movie_entry.get()

    if movie_title:
        recommendations = get_recommendations(movie_title)
        if recommendations is not None:
            messagebox.showinfo(
                "Recommendations",
                f"Movies similar to '{movie_title}':\n" +
                "\n".join([f"{row['title']} - {row['genre']}" for _, row in recommendations.iterrows()])
            )
        else:
            messagebox.showerror("Error", f"Movie '{movie_title}' not found in the dataset!\nTry another title.")
    else:
        messagebox.showwarning("Input Error", "Please enter a movie title.")

# Tkinter GUI
root = tk.Tk()
root.title("Enhanced Movie Recommendation System")
root.geometry("500x300")
root.configure(bg="#2c3e50")

# Fonts and Styles
title_font = font.Font(family="Arial", size=18, weight="bold")
label_font = font.Font(family="Arial", size=12)
button_font = font.Font(family="Arial", size=10, weight="bold")

# Title Label
title_label = tk.Label(root, text=" Movie Recommendation System", font=title_font,
                       bg="#2c3e50", fg="#ecf0f1")
title_label.pack(pady=20)

# Entry Label and Box
label = tk.Label(root, text="Enter a Movie Title:", font=label_font, bg="#2c3e50", fg="#ecf0f1")
label.pack(pady=5)

# Entry box for movie title
movie_entry = tk.Entry(root, font=label_font, width=30)
movie_entry.pack(pady=5)

# Recommend Button
recommend_button = tk.Button(root, text="Show Recommendations", font=button_font,
                             bg="#2980b9", fg="#ecf0f1", command=show_recommendations)
recommend_button.pack(pady=20)

# Start the main Tkinter loop
root.mainloop()
