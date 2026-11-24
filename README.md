# Automated Genre Tagging from Song Lyrics (NLP)

Can a song's genre be predicted from its lyrics alone? This project builds an NLP pipeline that does exactly that, framed for a music streaming platform's catalog team that receives large volumes of tracks with missing or unreliable genre tags.

**[View the full notebook with all outputs](https://GH_USER.github.io/lyrics-genre-classification-nlp/)**

## Approach

- **Data:** [Lyrics Genre Dataset (Large)](https://huggingface.co/datasets/Yegor25/lyrics_genre_dataset_large) from Hugging Face, filtered to English, single-genre tracks and balanced across six genres (Country, Hip-Hop, Pop, R&B, Rap, Rock)
- **Preprocessing:** tokenisation, stop-word removal and lemmatisation with NLTK
- **Information extraction:** named-entity recognition with spaCy, used as a separate descriptive signal
- **Representations:** TF-IDF (8,000 features) and MiniLM sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Models:** Multinomial Naive Bayes and Logistic Regression on TF-IDF, Logistic Regression on embeddings

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes + TF-IDF | 0.432 | 0.396 |
| Logistic Regression + TF-IDF | 0.410 | 0.403 |
| Logistic Regression + MiniLM | 0.421 | **0.407** |

Six-class chance level is about 0.17.

- Rap and Country separate cleanly; Pop, Hip-Hop and Rap are frequently confused, which matches real-world genre crossover.
- Rap and Hip-Hop lyrics mention roughly 2 to 3 times more people per song than Rock or Country (spaCy NER).
- **Recommendation:** not accurate enough to auto-tag a catalog, but useful as a low-cost disagreement detector next to an audio-based tagger, sending conflicting tracks to human review.

## Tech stack

Python, pandas, NLTK, spaCy, scikit-learn, PyTorch, Hugging Face Transformers, Matplotlib

## Notes

The dataset is not included in this repo; download it from the Hugging Face link above.
Completed as an individual project for M508 Big Data Analytics, MSc Data Science, AI and Digital Business, Gisma University of Applied Sciences.
