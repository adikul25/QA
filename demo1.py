import streamlit as st
from utils import QuestionAnswering4
import os

# Create an instance of the QuestionAnswering4 class
qa = QuestionAnswering4()

# Define the Streamlit app
def main():
    st.title("Question Answering Demo")

    # Add a file uploader for the documents directory
    documents_file = st.text_input("Please enter documents directory path")

    # Handle empty documents directory input
    if documents_file is None:
        st.warning("Please upload a documents directory.")
        return

    # # Get the path to the documents directory
    # documents_path = documents_file.name

    # Prepare the passages by reading in the documents from the directory
    try:
        qa.prepare_passages(documents_file                                                                                                                                                                                                                                                                                                                          )
        st.success("Passages prepared successfully!")
    except:
        st.warning("Please enter the directory path.")

    # Add a text input for the question
    question = st.text_input("Enter a question")

    # Handle empty question input
    if question == "":
        st.warning("Please enter a question.")
        return

    # Fetch the answers to the question
    try:
        answers = qa.fetch_answers(question)
    except:
        st.error("Error fetching answers. Please try again.")

    # Display the answers in a text area
    st.subheader("Answers")
    st.text_area("", value=answers, height=500, max_chars=None, key=None)

if __name__ == "__main__":
     main()
