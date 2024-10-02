import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
import random  # Import random for color generation

# Function to map similar texts with normalized percent similarity and provide top N options
def map_similar_texts_with_options(source_list, target_list, top_n=3):
    # Create a TF-IDF Vectorizer and compute the TF-IDF matrix
    vectorizer = TfidfVectorizer().fit_transform(source_list + target_list)
    
    # Split the TF-IDF matrix for source and target lists
    source_matrix = vectorizer[:len(source_list)]
    target_matrix = vectorizer[len(source_list):]
    
    # Compute cosine similarity between source and target lists
    similarity_matrix = cosine_similarity(source_matrix, target_matrix)
    
    # Normalize the similarity scores
    normalized_similarity_matrix = np.interp(similarity_matrix, (similarity_matrix.min(), similarity_matrix.max()), (50, 100))
    
    # Map each source text to the top N most similar target texts with their normalized similarity scores
    mapping = {}
    for i, source_text in enumerate(source_list):
        # Get the indices of the top N most similar target texts
        top_indices = np.argsort(normalized_similarity_matrix[i])[::-1][:top_n]
        top_similarities = [(target_list[idx], normalized_similarity_matrix[i][idx]) for idx in top_indices]
        
        # Store the source text and its top N similar target texts with normalized percentage similarities
        mapping[source_text] = top_similarities
    
    return mapping

# Function to run on button click
def display_results():
    source_input = source_text.get("1.0", tk.END).strip().split('\n')
    target_input = target_text.get("1.0", tk.END).strip().split('\n')
    
    mapped_texts = map_similar_texts_with_options(source_input, target_input)
    
    result_area.delete("1.0", tk.END)  # Clear previous results
    for source, top_matches in mapped_texts.items():
        result_area.insert(tk.END, f"'{source}' is mapped to:\n")
        for target, similarity in top_matches:
            result_area.insert(tk.END, f"   - '{target}' with {similarity:.2f}% similarity\n")
        result_area.insert(tk.END, "\n")
    
    plot_similarities(mapped_texts)

# Function to plot similarities
def plot_similarities(mapped_texts):
    num_sources = len(mapped_texts)
    fig, axes = plt.subplots(num_sources, 1, figsize=(10, 4 * num_sources))  # Create subplots

    if num_sources == 1:
        axes = [axes]  # Ensure axes is a list for a single subplot

    for ax, (source, top_matches) in zip(axes, mapped_texts.items()):
        targets = [match[0] for match in top_matches]
        similarities = [match[1] for match in top_matches]
        
        # Generate random colors for each bar
        colors = [random_color() for _ in range(len(similarities))]
        
        ax.barh(targets, similarities, color=colors)
        ax.set_xlim(0, 100)  # Set limit for x-axis
        ax.set_xlabel('Similarity Percentage')
        ax.set_title(f'Top Similarity Matches for: {source}')

    plt.tight_layout()
    plt.show()

# Function to generate a random color
def random_color():
    return (random.random(), random.random(), random.random())  # RGB values between 0 and 1

# Create the main application window
app = tk.Tk()
app.title("Text Similarity Mapper")

# Create and place the input fields and button
tk.Label(app, text="Source Texts (one per line):").pack()
source_text = scrolledtext.ScrolledText(app, width=40, height=10)
source_text.pack()

tk.Label(app, text="Target Texts (one per line):").pack()
target_text = scrolledtext.ScrolledText(app, width=40, height=10)
target_text.pack()

submit_button = tk.Button(app, text="Map Similar Texts", command=display_results)
submit_button.pack()

tk.Label(app, text="Results:").pack()
result_area = scrolledtext.ScrolledText(app, width=40, height=10)
result_area.pack()

# Start the Tkinter event loop
app.mainloop()
