# Install necessary packages
!pip install langchain
!pip install huggingface_hub
!pip install sentence_transformers
!pip install -U langchain-community
!pip install faiss-cpu
!pip install transformers

import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import textwrap
from transformers import pipeline

# Download the story and save it to a file
url = "https://raw.githubusercontent.com/Anshoomjain/langchain-proj1/main/A%20story%20to%20read.txt"
res = requests.get(url)
with open("Story.txt", "w") as f:
    f.write(res.text)

# Load the text from the file
loader = TextLoader('./Story.txt')
thestory = loader.load()

# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Print the loaded story (wrapped for readability)
print(wrap_text_preserve_newlines(str(thestory[0])))

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(thestory)

# Use embeddings from HuggingFace to create a vector store with FAISS
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Load the Hugging Face question-answering model
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Define a function to perform QA using the model
def answer_question(query):
    # Find relevant documents using similarity search
    relevant_docs = db.similarity_search(query)
    # Extract text from the first relevant document
    context = relevant_docs[0].page_content
    # Perform QA
    result = qa_pipeline(question=query, context=context)
    return result['answer']

# Example query
query = "who betrayed who?"
answer = answer_question(query)
print(f"Query: {query}")
print(f"Answer: {answer}")
