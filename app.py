import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import faiss
import pandas as pd
import numpy as np
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Recipe Recommender App", layout="centered")

# Load Sentence Transformer Model (For FAISS Search)
@st.cache_resource()
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Load FAISS index and recipes metadata
@st.cache_resource()
def load_faiss_index():
    try:
        index = faiss.read_index("faiss_index.idx")

        # Ensure 'recipes_metadata.json' exists
        json_file = "recipes_metadata.json"
        if not os.path.exists(json_file):
            st.warning(f"{json_file} not found. Generating from CSV...")

            # Load CSV and check data
            csv_file = "JPNmaindishes_cleaned.csv"
            if not os.path.exists(csv_file):
                st.error(f"Error: {csv_file} not found. Please upload it.")
                return None, None

            df = pd.read_csv(csv_file)

            # Check if required columns exist
            required_columns = {"title", "ingredients", "instructions", "url"}
            if not required_columns.issubset(df.columns):
                st.error(f"Error: CSV must contain columns: {', '.join(required_columns)}")
                return None, None

            # Convert DataFrame to JSON
            recipes = df.to_dict(orient="records")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(recipes, f, indent=4)

        # Load recipes JSON
        with open(json_file, "r", encoding="utf-8") as f:
            recipes = json.load(f)

        return index, recipes

    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None, None

# Load a smaller LLM model
@st.cache_resource()
def load_llm():
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Enables lower precision for reduced memory usage
        device_map="auto",
        offload_folder="./offload"  # Add this line to specify an offload folder
    )

    return model, tokenizer


# Load FAISS and LLM globally to avoid repeated loading
index, recipes = load_faiss_index()
model, tokenizer = load_llm()

# Function to generate embeddings
def get_embedding(query):
    return embedding_model.encode([query], convert_to_numpy=True).reshape(1, -1)

# Function to retrieve recipes
def retrieve_recipes(query, k=5):
    if index is None or recipes is None:
        return []

    query_embedding = get_embedding(query)
    _, indices = index.search(query_embedding, k)

    # Retrieve matching recipes
    results = [recipes[i] for i in indices[0] if i < len(recipes)]

    return results

# Summary Generator
def generate_summary(recipe):
    title = recipe.get("name", "Unknown Recipe")
    summary_raw = recipe.get("summary", "").strip()
    if not summary_raw:
        summary_raw = "No summary available."

    ingredients_raw = recipe.get("ingredients", "No ingredients available.")

    instructions_raw = recipe.get("process", "")
    if not isinstance(instructions_raw, str):
        instructions_raw = str(instructions_raw)
    instructions_raw = instructions_raw.strip()

    if not instructions_raw or instructions_raw.lower() in ["nan", "none", "null"]:
        instructions_raw = "No instructions provided in the dataset."

    if isinstance(ingredients_raw, str):
        ingredients_list = ingredients_raw.split(" | ")
    elif isinstance(ingredients_raw, list) and all(isinstance(i, str) and len(i) > 1 for i in ingredients_raw):
        ingredients_list = ingredients_raw
    else:
        ingredients_list = ["No ingredients available."]

    # Format ingredients properly
    ingredients_str = "
".join([f"- {ing.strip()}" for ing in ingredients_list if len(ing.strip()) > 1])

    # Format instructions step-by-step
    instructions_str = "
".join([f"{i+1}. {step.strip()}" for i, step in enumerate(instructions_raw.split(" | ")) if step.strip()])

    # Return formatted content
    return f"
{summary_raw}

**Ingredients:**
{ingredients_str}

**Instructions:**
{instructions_str}"

# Streamlit UI
st.title("🍜 Recipe Recommender (Japanese Cuisine Only)")
st.write("Enter an ingredient or dish to get **Japanese recipe** recommendations.")

query = st.text_input("Enter an ingredient or dish:")
if query:
    recipes = retrieve_recipes(query)

    if recipes:
        for recipe in recipes:
            # Check if ingredients are available
            ingredients_raw = recipe.get("ingredients", "No ingredients available.")
            if isinstance(ingredients_raw, str):
                ingredients_list = ingredients_raw.split(" | ")  # Properly split string
            elif isinstance(ingredients_raw, list):
                ingredients_list = [i.strip() for i in ingredients_raw if len(i.strip()) > 1]  # Ensure clean formatting
            else:
                ingredients_list = ["No ingredients available."]

            # If no ingredients are found, display a message
            if not ingredients_list or all(ing.strip() == "" for ing in ingredients_list):
                st.warning("No ingredients found for this dish.")
            else:
                summary = generate_summary(recipe)
                st.subheader(recipe.get("name", "Unknown Recipe"))  # Safely get title
                st.write("**Summary:**", summary)
                st.write(f"[View Full Recipe]({recipe.get('url', '#')})")
    else:
        st.warning("No recipes found. Try a different ingredient or dish.")

# Footer
st.markdown("---")
st.markdown("Recipes sourced from AllRecipes")

