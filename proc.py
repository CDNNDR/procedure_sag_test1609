import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import ssl
import certifi

# --------------------------------------------------------------------------------------------------------------------------------------------
# Use certifi's certificates for SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set the OpenAI API key
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

# Set the screen to full width by default across devices
def wide_space_default():
    st.set_page_config(layout="wide")

# Call the function to apply the setting
wide_space_default()

# -----------------------------------------
# Function to load and split documents into manageable chunks
def load_and_split_documents():
    folder_path = "/Users/andrea/Documenti/PycharmProjects/streamapp/data"  # Your actual data folder path
    loader = DirectoryLoader(folder_path, glob="**/*.txt")
    documents = loader.load()

    # Ensure 'istruzioni' is included
    istruzioni_path = "/Users/andrea/Documenti/PycharmProjects/streamapp/data/istruzioni.txt"  # Your actual istruzioni.txt path
    with open(istruzioni_path, 'r') as file:
        istruzioni_content = file.read()
    documents.append(Document(page_content=istruzioni_content, metadata={"source": "istruzioni"}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs, istruzioni_content

# Load documents and 'istruzioni_content'
docs, istruzioni_content = load_and_split_documents()

# Initialize embeddings and FAISS vectorstore for document retrieval
@st.cache_resource
def setup_retrieval():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = setup_retrieval()

# Initialize memory
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = st.session_state['memory']

# Define a more conversational prompt including 'istruzioni_content' directly in the template
conversational_prompt = PromptTemplate(
    template=f"""
        {istruzioni_content}

        The following is a friendly conversation between a user and an assistant.
        The assistant is knowledgeable and provides detailed answers based on the provided documents below.
        If the assistant does not know the answer, it should politely admit it.

        Relevant documents:
        {{context}}

        {{chat_history}}
        User: {{question}}
        Assistant:
    """,
    input_variables=["context", "chat_history", "question"]
)

# Set up the LLM without the 'openai_api_key' argument
llm = OpenAI(
    temperature=0,
    model_name="gpt-4o-mini"  # Ensure you have access to this model, or replace with "gpt-3.5-turbo"
)

# Setup the Conversational Retrieval Chain using the LLM and the corrected prompt
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={'prompt': conversational_prompt}
)

# -------------------------------------------------------------------------------------------
# Streamlit UI setup
st.sidebar.title("🧭 Navigator")

# Add a button in the sidebar to clear the chat
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Ciao! Come posso aiutarti?"}]
    st.experimental_rerun()  # Refresh the page to clear the chat

# Add a button to re-run the page
if st.sidebar.button("Restart Lucy!"):
    st.experimental_rerun()  # This will force the Streamlit app to refresh

st.title("👩🏻‍💻 ‍🍀 Hi! I'm Lucy!")
st.caption("🚀 Your personal assistant powered by OpenAI")

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ciao! Come posso aiutarti?"}]

# Display conversation history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    assistant_response = None  # Initialize assistant_response

    # RAG mode with document retrieval
    with st.spinner("Recupero e generazione della risposta..."):
        # The ConversationalRetrievalChain handles retrieval internally
        response = qa_chain({"question": prompt})
        assistant_response = response['answer']

    # Store and display the assistant's response
    if assistant_response:
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})
        st.chat_message("assistant").write(assistant_response)