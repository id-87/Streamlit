import re
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

SAMPLE_TEXTS = {
    "Custom / Starter": "Natural Language Processing is amazing! It helps machines understand human language.",
    "Cleaning Demo": "Hello!!! NLP, in 2026, is #Amazing... Right??",
    "Normalization Demo": "The cats are running faster than the dogs and the birds are flying.",
    "Context Demo": "He sat by the bank of the river. She visited the bank for a loan.",
    "Mixed Paragraph": "Text preprocessing is the first step in NLP. It includes cleaning, tokenization, stopword removal, stemming, lemmatization, vectorization, and embeddings.",
}

_STOPWORDS = None
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def get_stopwords_set():
    global _STOPWORDS
    if _STOPWORDS is None:
        ensure_nltk_resources()
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_special_chars: bool = True,
    remove_digits: bool = False,
    normalize_spaces: bool = True,
) -> str:
    cleaned = text

    if lowercase:
        cleaned = cleaned.lower()

    if remove_punctuation:
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    if remove_special_chars:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)

    if remove_digits:
        cleaned = re.sub(r"\d+", " ", cleaned)

    if normalize_spaces:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def get_removed_elements(original: str, cleaned: str) -> List[str]:
    cleaned_chars = set(cleaned)
    removed = []
    for ch in original:
        if ch.strip() and ch not in cleaned_chars and ch not in removed:
            removed.append(ch)
    return removed


def simple_subword_tokenize(text: str) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    tokens = []
    for word in words:
        if len(word) <= 4:
            tokens.append(word)
        else:
            tokens.append(word[:3])
            rest = word[3:]
            chunk_size = 3
            for i in range(0, len(rest), chunk_size):
                tokens.append("##" + rest[i:i + chunk_size])
    return tokens


def tokenize_text(text: str, tokenization_type: str = "Word") -> List[str]:
    if not text.strip():
        return []

    if tokenization_type == "Word":
        return re.findall(r"\b\w+\b", text)

    if tokenization_type == "Character":
        return [ch for ch in text if not ch.isspace()]

    if tokenization_type == "Subword":
        return simple_subword_tokenize(text)

    return re.findall(r"\b\w+\b", text)


def nltk_pos_to_wordnet(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def remove_stopwords_and_normalize(
    tokens: List[str],
    tokenization_type: str = "Word",
    remove_stopwords: bool = True,
    normalization_method: str = "None",
) -> Tuple[List[str], pd.DataFrame]:
    stop_words = get_stopwords_set()
    rows = []
    processed = []

    if tokenization_type != "Word":
        for tok in tokens:
            keep = not (remove_stopwords and tok.lower() in stop_words)
            transformed = tok

            if keep:
                processed.append(transformed)

            rows.append(
                {
                    "Original Token": tok,
                    "Stopword Removed?": "Yes" if not keep else "No",
                    "Transformed Token": transformed if keep else "—",
                }
            )
        return processed, pd.DataFrame(rows)

    try:
        pos_tags = nltk.pos_tag(tokens) if tokens else []
    except LookupError:
        ensure_nltk_resources()
        try:
            pos_tags = nltk.pos_tag(tokens) if tokens else []
        except Exception:
            pos_tags = [(tok, "NN") for tok in tokens]

    for tok, pos in pos_tags:
        keep = not (remove_stopwords and tok.lower() in stop_words)
        transformed = tok

        if keep:
            if normalization_method == "Stemming":
                transformed = _STEMMER.stem(tok)
            elif normalization_method == "Lemmatization":
                transformed = _LEMMATIZER.lemmatize(tok, nltk_pos_to_wordnet(pos))

            processed.append(transformed)

        rows.append(
            {
                "Original Token": tok,
                "POS": pos,
                "Stopword Removed?": "Yes" if not keep else "No",
                "Transformed Token": transformed if keep else "—",
            }
        )

    return processed, pd.DataFrame(rows)


def build_vocabulary(tokens: List[str]) -> pd.DataFrame:
    if not tokens:
        return pd.DataFrame(columns=["Index", "Token", "Frequency"])

    counts = Counter(tokens)
    sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    rows = []
    for idx, (token, freq) in enumerate(sorted_tokens):
        rows.append(
            {
                "Index": idx,
                "Token": token,
                "Frequency": freq,
            }
        )

    return pd.DataFrame(rows)


def vectorize_corpus(corpus: List[str], method: str = "TF-IDF"):
    docs = [doc for doc in corpus if isinstance(doc, str) and doc.strip()]
    if not docs:
        return pd.DataFrame(), [], method

    if method == "Bag of Words":
        vectorizer = CountVectorizer()
        label = "Bag of Words"
    else:
        vectorizer = TfidfVectorizer()
        label = "TF-IDF"

    matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out().tolist()
    df = pd.DataFrame(matrix.toarray(), columns=feature_names)

    return df, feature_names, label


def get_sentence_embeddings(sentences: List[str], model_name: str = "tfidf-demo") -> np.ndarray:
    docs = [s for s in sentences if isinstance(s, str) and s.strip()]
    if not docs:
        return np.zeros((1, 1))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs)
    return matrix.toarray()


def reduce_embeddings_2d(embeddings: np.ndarray, labels: List[str]):
    if embeddings is None or len(embeddings) == 0:
        return pd.DataFrame(columns=["x", "y", "label", "label_short"])

    if embeddings.shape[0] == 1:
        return pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "label": [labels[0]],
                "label_short": [shorten_text(labels[0])],
            }
        )

    n_components = 2 if embeddings.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    if n_components == 1:
        x_vals = reduced[:, 0]
        y_vals = np.zeros(len(x_vals))
    else:
        x_vals = reduced[:, 0]
        y_vals = reduced[:, 1]

    return pd.DataFrame(
        {
            "x": x_vals,
            "y": y_vals,
            "label": labels,
            "label_short": [shorten_text(label) for label in labels],
        }
    )


def cosine_similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    sim = cosine_similarity([vec_a], [vec_b])[0][0]
    return float(sim)


def extract_contextual_embeddings(sentence_a: str, sentence_b: str, target_word: str):
    token_a = target_word.lower()
    token_b = target_word.lower()

    tokens_a = re.findall(r"\b\w+\b", sentence_a.lower())
    tokens_b = re.findall(r"\b\w+\b", sentence_b.lower())

    if token_a not in tokens_a or token_b not in tokens_b:
        return {
            "found": False,
            "message": f"Could not find '{target_word}' in both sentences.",
        }

    index_a = tokens_a.index(token_a)
    index_b = tokens_b.index(token_b)

    context_vec_a = build_context_vector(tokens_a, index_a)
    context_vec_b = build_context_vector(tokens_b, index_b)

    similarity = cosine_similarity([context_vec_a], [context_vec_b])[0][0]

    return {
        "found": True,
        "index_a": index_a,
        "index_b": index_b,
        "token_a": target_word,
        "token_b": target_word,
        "similarity": float(similarity),
    }


def build_context_vector(tokens: List[str], index: int, window: int = 2) -> np.ndarray:
    left = tokens[max(0, index - window):index]
    right = tokens[index + 1:index + 1 + window]
    context = left + right

    if not context:
        return np.zeros(5)

    features = np.array(
        [
            len(context),
            sum(len(tok) for tok in context) / len(context),
            sum(1 for tok in context if tok in get_stopwords_set()),
            sum(1 for tok in context if tok.isalpha()),
            sum(ord(tok[0]) for tok in context) / len(context),
        ],
        dtype=float,
    )
    return features


def tokens_to_display_html(tokens: List[str]) -> str:
    if not tokens:
        return "<p><i>No tokens generated.</i></p>"

    chips = []
    for tok in tokens:
        chips.append(
            f"<span style='display:inline-block;padding:6px 10px;margin:4px;border-radius:999px;background:#E8EEF9;color:#1F2937;border:1px solid #CBD5E1;font-size:14px;'>{tok}</span>"
        )
    return "<div style='line-height:2.2;'>" + "".join(chips) + "</div>"


def split_into_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def shorten_text(text: str, max_len: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."