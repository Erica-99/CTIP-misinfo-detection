# This script can be used to run inference on our saved best classification model (Linear SVM with TF-IDF feature extraction)

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import re
import unidecode
import warnings
import argparse
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ------- CONFIG -------

INPUT_TEXT = """Federal appeals court judges on Friday asked about President Donald Trump’s sharing of anti-Muslim videos on his Twitter account, as they grilled a U.S. government lawyer about the legality of the president’s latest travel ban. 
Judge Pamela Harris asked government lawyer Hashim Mooppan about Trump’s Nov. 29 online sharing of three anti-Muslim videos posted on Twitter by a far-right British party leader. Lawyers say that and earlier statements by Trump prove the policy is 
aimed at blocking the entry of Muslims, rather than the president’s stated goal of preserving national security.""" # Put your text for inference here.

DATASET_PATH = "data/processed_dataset.csv"

MODEL_PATH = "saved_classifier_model.joblib"

# ------- FUNCTIONS -------

def preprocess(processing_text):
    in_text = processing_text
    
    # Remove newlines and tabs
    def remove_newlines_tabs(text):
        formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com') # Also join the . and com together in links for consistency.
        return formatted_text

    # Convert typographical apostrophes/quotemarks to regular apostrophes
    def normalise_apostrophes(text):
        normalised_text = text.replace("’", "'").replace("‘", "'")
        return normalised_text

    # Strip any html tags
    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    # Remove accented characters
    def accented_characters_removal(text):
        decoded_text = unidecode.unidecode(text)
        return decoded_text

    # Strip links from the text
    def remove_links(text):
        remove_https = re.sub(r'http\S+', '', text)
        remove_com = re.sub(r'\ [A-Za-z]*\.com', ' ', remove_https)
        return remove_com

    # Make everything lowercase for consistency between characters and semantics
    def set_lowercase(text):
        lower_text = text.lower()
        return lower_text

    # Reduce repeated characters and punctuation
    # This will cut down long strings of the same letter repeated to just two instances.
    # e.g. Amaaaaaaaaaaazzzzzinggg!!!!!! -> Amaazzingg!!!!!!
    def reduce_character_repetition(text):
        pattern_alpha = re.compile(r'([A-Za-z])\1{2,}', re.DOTALL) # Find all instances of repetition
        formatted_text = pattern_alpha.sub(r'\1\1', text) # Limit repetition to two characters for each instance
        return formatted_text

    CONTRACTIONS_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    }

    # Expand contractions into multiple words
    def expand_contractions(text, contraction_map=CONTRACTIONS_MAP):
        word_list = text.split(' ')
        for word in contraction_map.keys():
            if word in word_list:
                word_list = [item.replace(word, contraction_map[word]) for item in word_list]
                
        new_string = ' '.join(str(expanded) for expanded in word_list)
        return new_string

    # Remove the first period from the word "U.S." specifically without adding spaces
    def fix_usa(text):
        fixed_text = text.replace("u.s", "us")
        return fixed_text

    # Remove special characters
    def remove_special_characters(text):
        processed_text = re.sub(r'\W', ' ', text)
        processed_text = processed_text.replace('_', ' ') # Underscores are counted as a word character, but we remove them as well.
        return processed_text

    # Remove isolated letters 
    def removed_isolated_letters(text):
        stripped_text = re.sub(r'(?<=\s)[a-zA-Z](?=\s)', ' ', text) # Middle
        stripped_text = re.sub(r'^[a-zA-Z](?=\s)', '', stripped_text) # Start of string
        stripped_text = re.sub(r'(?<=\s)[a-zA-Z]$', '', stripped_text) # End of string
        return stripped_text

    # Condense whitespace
    def condense_whitespace(text):
        without_whitespace = re.sub(r'\s+', ' ', text)
        without_whitespace = without_whitespace.strip() # Remove from start and end of string.
        return without_whitespace

    in_text = in_text.map(remove_newlines_tabs) # Removes newlines and tabs
    in_text = in_text.map(normalise_apostrophes) # Standardises typographical quotes and apostrophes into ' and "
    in_text = in_text.map(strip_html_tags) # Removes HTML tags
    in_text = in_text.map(accented_characters_removal) # Simplifies accented characters into their english equivalents
    in_text = in_text.map(remove_links) # Finds and removes links.
    in_text = in_text.map(set_lowercase) # Makes all text lowercase
    in_text = in_text.map(reduce_character_repetition) # Reduces repetition of the same letter in a row to max of 2.
    in_text = in_text.map(lambda x: expand_contractions(x)) # Expands contractions. you're -> you are
    in_text = in_text.map(fix_usa) # Removes the period in U.S. to just be US
    in_text = in_text.map(remove_special_characters) # Removes all punctuation and special characters.
    in_text = in_text.map(removed_isolated_letters) # Removes single letters sitting on their own.
    in_text = in_text.map(condense_whitespace) # Condenses multiple spaces into one space.
    
    return in_text

if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-dataset", action="store_true", help="Whether to use a random post from the dataset for inference.")
    ap.add_argument("--use-text", default="", help="The string of text to run inference on.")
    args = ap.parse_args()
    
    if args.use_dataset:
        dataset = pd.read_csv(DATASET_PATH, encoding='utf-8', index_col=0)
        chosen_text = dataset.sample(1)
        print(f"Predicting on random text from dataset:\n {chosen_text}")
        preprocessed_text = preprocess(chosen_text['text'])
        prediction = model.predict(preprocessed_text)
        print(f"It is predicted that this post contains {prediction[0]} information.")
    else:
        if args.use_text:
            print(f"Predicting on entered text:\n {args.use_text}")
            preprocessed_text = preprocess(pd.DataFrame([args.use_text]))
            prediction = model.predict(preprocessed_text[0])
            print(f"It is predicted that this post contains {prediction[0]} information.")
        else:
            print(f"Predicting on saved example text (excerpt from a true post in the dataset):\n {INPUT_TEXT}")
            preprocessed_text = preprocess(pd.DataFrame([INPUT_TEXT]))
            prediction = model.predict(preprocessed_text[0])
            print(f"It is predicted that this post contains {prediction[0]} information.")