import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from collections import Counter
import plotly.graph_objects as go

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to check plagiarism
def check_plagiarism(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Function to read text from uploaded files
def read_document(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

# Function to create a dynamic gauge meter
def create_gauge_meter(similarity_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=similarity_score * 100,
        title={"text": "Plagiarism Similarity (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red" if similarity_score > 0.8 else "green"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": similarity_score * 100
            }
        }
    ))
    st.plotly_chart(fig)

# Function to find and visualize common words with frequency
def visualize_common_words(text1, text2):
    text1_words = preprocess_text(text1).split()
    text2_words = preprocess_text(text2).split()
    common_words = set(text1_words).intersection(set(text2_words))
    if not common_words:
        return None, None

    # Count the frequency of common words in both texts
    text1_counter = Counter(text1_words)
    text2_counter = Counter(text2_words)

    frequencies = {word: (text1_counter[word], text2_counter[word]) for word in common_words}

    # Create a bar chart for visualization
    fig = go.Figure()
    words = list(frequencies.keys())
    text1_freq = [frequencies[word][0] for word in words]
    text2_freq = [frequencies[word][1] for word in words]

    fig.add_trace(go.Bar(name="Text 1", x=words, y=text1_freq, marker_color="blue"))
    fig.add_trace(go.Bar(name="Text 2", x=words, y=text2_freq, marker_color="orange"))

    fig.update_layout(
        title="Common Words Frequency Comparison",
        xaxis_title="Words",
        yaxis_title="Frequency",
        barmode="group",
        template="plotly_white",
        title_x=0.5,
    )
    return common_words, fig

# Streamlit app
def main():
    # Centering the title using custom HTML and CSS
    st.markdown(
    """
    <style>
    .center-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    <div class="center-title"><b>Plagiarism Checker</b></div> 
    """,
    unsafe_allow_html=True
)

    page = st.radio("Select a section", ("Document Upload", "Paragraph Input"))

    if page == "Document Upload":
        st.subheader("Document Upload for Plagiarism Check")
        col1, col2 = st.columns(2)
        with col1:
            doc1 = st.file_uploader("Choose the first document", type=["txt", "pdf", "docx"])
        with col2:
            doc2 = st.file_uploader("Choose the second document", type=["txt", "pdf", "docx"])

        if doc1 and doc2:
            if st.button("Check Plagiarism"):
                with st.spinner("Analyzing documents..."):
                    text1 = read_document(doc1)
                    text2 = read_document(doc2)
                    similarity_score = check_plagiarism(text1, text2)
                    common_words, fig = visualize_common_words(text1, text2)

                st.subheader("Plagiarism Check Result")
                st.write(f"Similarity Score: {similarity_score:.2f}")
                create_gauge_meter(similarity_score)

                if similarity_score > 0.8:
                    st.warning("The documents are highly similar (possible plagiarism).")
                else:
                    st.success("The documents are not very similar.")
                
                # Display common words
                if common_words:
                    st.markdown("### Common Words Frequency:")
                    st.plotly_chart(fig)
                else:
                    st.write("No common words or phrases found.")
        else:
            st.info("Please upload two documents to check for plagiarism.")

    elif page == "Paragraph Input":
        st.subheader("Paragraph Input for Plagiarism Check")
        col1, col2 = st.columns(2)
        with col1:
            user_text = st.text_area("Enter your paragraph here:")
        with col2:
            reference_text = st.text_area("Enter the reference text to compare against:")

        if user_text and reference_text:
            if st.button("Check Paragraph Plagiarism"):
                similarity_score = check_plagiarism(user_text, reference_text)
                common_words, fig = visualize_common_words(user_text, reference_text)

                st.subheader("Plagiarism Check Result")
                st.write(f"Similarity Score: {similarity_score:.2f}")
                create_gauge_meter(similarity_score)

                if similarity_score > 0.8:
                    st.warning("The paragraph is highly similar to the reference text (possible plagiarism).")
                else:
                    st.success("The paragraph is not very similar.")
                
                # Display common words
                if common_words:
                    st.markdown("### Common Words Frequency:")
                    st.plotly_chart(fig)
                else:
                    st.write("No common words or phrases found.")
        else:
            st.info("Please enter both the paragraph and the reference text.")

if __name__ == "__main__":
    main()
