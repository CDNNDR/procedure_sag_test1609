import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
from langchain.prompts import PromptTemplate
import os
import ssl
import certifi
from git import Repo

# --------------------------------------------------------------------------------------------------------------------------------------------
# Use certifi's certificates for SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set the OpenAI API key
openai_api_key = st.secrets["openai_api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Set the screen to full width by default across devices
def wide_space_default():
    st.set_page_config(layout="wide")

# Call the function to apply the setting
wide_space_default()

# -------------------------------------------------

# Function to clone a GitHub repository
def clone_github_repo(github_url, local_dir):
    if not os.path.exists(local_dir):
        Repo.clone_from(github_url, local_dir)

# Correct GitHub URL
github_url = "https://github.com/CDNNDR/procedure_sag_test1609"
local_dir = "/tmp/github_repo"  # Temporary local directory

# Clone the repository
clone_github_repo(github_url, local_dir)

# -----------------------------------------
# Function to load and split documents into manageable chunks
def load_and_split_documents():
    folder_path = os.path.join(local_dir, "data")  # Adjust path as needed
    loader = DirectoryLoader(folder_path, glob="**/*.txt")
    documents = loader.load()

    # Ensure 'istruzioni' is included
    istruzioni_path = os.path.join(folder_path, "istruzioni.txt")
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
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass the API key explicitly
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

# Set up the LLM using ChatOpenAI for chat models
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-4o-mini"  # Replace with "gpt-3.5-turbo" if using that model
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
    st.experimental_set_query_params()  # Refresh the page

# Add a button to re-run the page
if st.sidebar.button("Restart Lucy!"):
    st.experimental_set_query_params()  # This will force the Streamlit app to refresh


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
