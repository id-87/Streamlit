import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    SAMPLE_TEXTS,
    clean_text,
    tokenize_text,
    remove_stopwords_and_normalize,
    build_vocabulary,
    vectorize_corpus,
    get_sentence_embeddings,
    reduce_embeddings_2d,
    cosine_similarity_score,
    extract_contextual_embeddings,
    tokens_to_display_html,
    get_removed_elements,
    split_into_sentences,
    ensure_nltk_resources,
)

st.set_page_config(
    page_title="Text Preprocessing to Embeddings",
    page_icon="🧠",
    layout="wide",
)

ensure_nltk_resources()

st.title("🧠 Text Preprocessing to Embeddings")
st.markdown(
    "Explore how raw text is cleaned, tokenized, normalized, vectorized, and finally converted into embeddings."
)

with st.sidebar:
    st.header("Controls")
    sample_choice = st.selectbox("Choose sample text", list(SAMPLE_TEXTS.keys()))
    raw_text = st.text_area(
        "Input text",
        value=SAMPLE_TEXTS[sample_choice],
        height=180,
    )

    st.subheader("Cleaning options")
    do_lowercase = st.checkbox("Lowercase", value=True)
    remove_punct = st.checkbox("Remove punctuation", value=True)
    remove_special = st.checkbox("Remove special characters", value=True)
    remove_digits = st.checkbox("Remove digits", value=False)
    normalize_spaces = st.checkbox("Normalize extra spaces", value=True)

    st.subheader("Tokenization")
    tokenization_type = st.selectbox(
        "Tokenization type",
        ["Word", "Character", "Subword"],
        index=0,
    )

    st.subheader("Normalization")
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    normalize_method = st.selectbox(
        "Normalization method",
        ["None", "Stemming", "Lemmatization"],
        index=2,
    )

    st.subheader("Vectorization")
    vectorization_method = st.selectbox(
        "Vectorization method",
        ["Bag of Words", "TF-IDF"],
        index=1,
    )

    st.subheader("Embeddings")
    embedding_model = st.selectbox(
        "Sentence embedding model",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"],
        index=0,
    )

if not raw_text.strip():
    st.warning("Please enter some text to begin.")
    st.stop()

cleaned_text = clean_text(
    raw_text,
    lowercase=do_lowercase,
    remove_punctuation=remove_punct,
    remove_special_chars=remove_special,
    remove_digits=remove_digits,
    normalize_spaces=normalize_spaces,
)

removed_items = get_removed_elements(raw_text, cleaned_text)
raw_tokens = tokenize_text(cleaned_text, tokenization_type)
processed_tokens, token_transform_df = remove_stopwords_and_normalize(
    raw_tokens,
    tokenization_type=tokenization_type,
    remove_stopwords=remove_stopwords,
    normalization_method=normalize_method,
)

vocab_df = build_vocabulary(processed_tokens)

sentences = split_into_sentences(raw_text)
if len(sentences) == 0:
    sentences = [raw_text]

active_sentence = cleaned_text if cleaned_text.strip() else raw_text
corpus_for_vectors = [active_sentence]
if len(sentences) > 1:
    corpus_for_vectors = sentences

vector_df, feature_names, vectorizer_label = vectorize_corpus(
    corpus_for_vectors,
    method=vectorization_method,
)

sentence_embeddings = get_sentence_embeddings(sentences, model_name=embedding_model)
embedding_2d_df = reduce_embeddings_2d(sentence_embeddings, labels=sentences)

st.markdown("---")

(tab1, tab2, tab3, tab4, tab5, tab6, tab7) = st.tabs([
    "1. Text Cleaning",
    "2. Tokenization",
    "3. Normalization",
    "4. Vocabulary",
    "5. Vectorization",
    "6. Word Embeddings",
    "7. Contextual Embeddings",
], on_change="rerun")

with tab1:
    st.subheader("Text Cleaning & Normalization")
    st.info(
        "Raw text often contains noise such as punctuation, symbols, mixed casing, or extra spaces. Cleaning standardizes text so later NLP steps become more consistent and easier to analyze."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original text")
        st.code(raw_text, language="text")
    with col2:
        st.markdown("### Cleaned text")
        st.code(cleaned_text, language="text")

    st.markdown("### Removed / changed elements")
    if removed_items:
        removed_df = pd.DataFrame({"Removed or changed": removed_items})
        st.dataframe(removed_df, use_container_width=True, hide_index=True)
    else:
        st.success("No visible elements were removed by the selected cleaning steps.")

    st.markdown("### What to observe")
    st.write(
        "Notice how the cleaned text becomes more regular. Small variations like punctuation and casing can create unnecessary differences for machine learning models."
    )

with tab2:
    st.subheader("Tokenization")
    st.info(
        "Tokenization splits text into smaller units called tokens. Depending on the strategy, tokens may be words, characters, or subword fragments."
    )

    st.markdown(f"### {tokenization_type} tokens")
    st.metric("Number of tokens", len(raw_tokens))
    st.markdown(tokens_to_display_html(raw_tokens), unsafe_allow_html=True)

    token_df = pd.DataFrame({"Token #": list(range(1, len(raw_tokens) + 1)), "Token": raw_tokens})
    st.dataframe(token_df, use_container_width=True, hide_index=True)

    st.markdown("### What to observe")
    st.write(
        "Different tokenization strategies change how the model sees the text. Word tokens preserve words, character tokens are much finer, and subwords help handle rare or unknown words."
    )

with tab3:
    st.subheader("Stopword Removal, Stemming & Lemmatization")
    st.info(
        "This stage removes very common words if needed and reduces tokens to simpler base forms. It helps focus on meaningful content and can reduce feature space size."
    )

    st.markdown("### Token transformation table")
    st.dataframe(token_transform_df, use_container_width=True, hide_index=True)

    st.markdown("### Final processed tokens")
    st.metric("Processed token count", len(processed_tokens))
    st.markdown(tokens_to_display_html(processed_tokens), unsafe_allow_html=True)

    st.markdown("### What to observe")
    st.write(
        "Compare the original tokens with the transformed ones. Stemming is more aggressive, while lemmatization usually produces cleaner dictionary-like base forms."
    )

with tab4:
    st.subheader("Vocabulary Building")
    st.info(
        "A vocabulary maps tokens to numerical indices. This creates the feature space that traditional machine learning models use for text input."
    )

    c1, c2 = st.columns(2)
    c1.metric("Vocabulary size", int(vocab_df.shape[0]))
    c2.metric("Unique processed tokens", int(vocab_df.shape[0]))

    st.markdown("### Token to index mapping")
    st.dataframe(vocab_df, use_container_width=True, hide_index=True)

    if not vocab_df.empty:
        fig_vocab = px.bar(
            vocab_df.sort_values("Frequency", ascending=False).head(20),
            x="Token",
            y="Frequency",
            color="Frequency",
            title="Top vocabulary tokens by frequency",
        )
        fig_vocab.update_layout(height=420)
        st.plotly_chart(fig_vocab, use_container_width=True)

    st.markdown("### What to observe")
    st.write(
        "Every token now has a numeric identity. The more repeated a token is, the more frequently it appears in the vocabulary statistics."
    )

with tab5:
    st.subheader("Text Vectorization")
    st.info(
        "Vectorization converts text into numbers. Bag-of-Words counts token occurrences, while TF-IDF weights tokens based on how informative they are across documents."
    )

    st.markdown(f"### {vectorizer_label} matrix")
    st.dataframe(vector_df, use_container_width=True)

    if not vector_df.empty:
        heatmap_df = vector_df.copy()
        heatmap_df.insert(0, "Document", [f"Doc {i+1}" for i in range(len(heatmap_df))])
        heatmap_long = heatmap_df.melt(id_vars="Document", var_name="Feature", value_name="Value")
        fig_heat = px.density_heatmap(
            heatmap_long,
            x="Feature",
            y="Document",
            z="Value",
            color_continuous_scale="Blues",
            title=f"{vectorizer_label} feature heatmap",
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### What to observe")
    st.write(
        "Sparse text vectors usually contain many zeros and only a few active features. TF-IDF gives less importance to common tokens and more importance to distinctive ones."
    )

with tab6:
    st.subheader("Word / Sentence Embeddings")
    st.info(
        "Dense embeddings map text to compact vectors that capture semantic meaning. Similar sentences tend to appear closer together in embedding space."
    )

    st.metric("Embedding dimension", int(sentence_embeddings.shape[1]))
    st.markdown("### First few embedding values")
    embed_preview = pd.DataFrame(sentence_embeddings[: min(3, len(sentence_embeddings)), :12])
    embed_preview.index = [f"Sentence {i+1}" for i in range(embed_preview.shape[0])]
    st.dataframe(embed_preview, use_container_width=True)

    if len(sentences) >= 2:
        similarity = cosine_similarity_score(sentence_embeddings[0], sentence_embeddings[1])
        st.metric("Cosine similarity of first two sentences", f"{similarity:.4f}")

    if embedding_2d_df is not None and not embedding_2d_df.empty:
        fig_scatter = px.scatter(
            embedding_2d_df,
            x="x",
            y="y",
            text="label_short",
            hover_data=["label"],
            title="2D projection of sentence embeddings",
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=12, color="#4F46E5"))
        fig_scatter.update_layout(height=520)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### What to observe")
    st.write(
        "Unlike Bag-of-Words, dense embeddings do not just count words. They place semantically related text closer together in a continuous vector space."
    )

with tab7:
    st.subheader("Contextual Embeddings")
    st.info(
        "Contextual embeddings change depending on the surrounding words. The same word can have different vector representations in different sentences because its meaning changes with context."
    )

    default_a = "He sat on the bank of the river."
    default_b = "She visited the bank to deposit money."

    col_a, col_b = st.columns(2)
    with col_a:
        context_a = st.text_input("Sentence A", value=default_a)
    with col_b:
        context_b = st.text_input("Sentence B", value=default_b)

    target_word = st.text_input("Target word to compare", value="bank")

    if target_word.strip():
        contextual_result = extract_contextual_embeddings(context_a, context_b, target_word.strip())

        if contextual_result["found"]:
            st.success(f"Target word '{target_word}' found in both sentences.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Sentence A token index", contextual_result["index_a"])
            m2.metric("Sentence B token index", contextual_result["index_b"])
            m3.metric("Contextual cosine similarity", f"{contextual_result['similarity']:.4f}")

            compare_df = pd.DataFrame({
                "Sentence": ["Sentence A", "Sentence B"],
                "Context": [context_a, context_b],
                "Matched token": [contextual_result["token_a"], contextual_result["token_b"]],
            })
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            st.markdown("### Interpretation")
            st.write(
                "If the similarity is lower, the model is treating the word as meaning something different in each sentence. This is what makes contextual embeddings powerful for modern NLP systems."
            )
        else:
            st.warning(contextual_result["message"])

    st.markdown("### What to observe")
    st.write(
        "Static embeddings give one vector per word, but contextual embeddings depend on surrounding words. That is why ambiguous words can be represented differently in different sentences."
    )