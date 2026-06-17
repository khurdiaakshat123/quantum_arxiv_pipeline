import os
# Force Hugging Face cache to be inside the project folder to bypass Windows PermissionError
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(SCRIPT_DIR, ".hf_cache")

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# Import LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. Environment & Setup
# ---------------------------------------------------------
env_path = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(env_path, override=True)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Quantum Network RAG Engine",
    page_icon="⚛️",
    layout="wide"
)

# Custom Styling (Glassmorphic touches in Markdown/HTML)
st.markdown("""
    <style>
    .reportview-container {
        background: #0D1117;
    }
    div.stButton > button:first-child {
        background-color: #00F2FE;
        color: #0D1117;
        border-radius: 8px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #4FACFE;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar Configuration (Settings & API Keys)
# ---------------------------------------------------------
st.sidebar.title("⚛️ RAG Configuration")
st.sidebar.markdown("Configure your AI providers and database connection settings below.")

# Provider Selector
provider = st.sidebar.selectbox(
    "LLM Provider (Chat)",
    options=["OpenAI (Paid Cloud)", "Ollama (Free Local)", "Groq (Free Cloud API)"],
    index=0
)

# Provider-Specific Parameters
openai_api_key = ""
groq_api_key = ""
ollama_url = "http://localhost:11434"
chat_model_name = ""
embedding_model_name = ""
embedding_provider = ""

if provider == "OpenAI (Paid Cloud)":
    st.sidebar.subheader("OpenAI Settings")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Provide your OpenAI API key. Required for OpenAI Embeddings and Chat Generation."
        )
    else:
        st.sidebar.success("Loaded OpenAI API Key from environment.")
        
    chat_model_name = st.sidebar.selectbox(
        "LLM Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    embedding_provider = "OpenAI (Paid Cloud)"
    embedding_model_name = st.sidebar.selectbox(
        "Embedding Model",
        options=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        index=0
    )

elif provider == "Ollama (Free Local)":
    st.sidebar.subheader("Ollama Local Settings")
    ollama_url = st.sidebar.text_input(
        "Ollama Server URL",
        value="http://localhost:11434",
        help="Local URL where Ollama is running."
    )
    chat_model_name = st.sidebar.selectbox(
        "Local LLM Model",
        options=["llama3", "phi3", "gemma2:2b"],
        index=0,
        help="Make sure you pulled the model first (e.g. 'ollama pull llama3')"
    )
    embedding_provider = "Ollama Local (Free)"
    embedding_model_name = st.sidebar.selectbox(
        "Local Embedding Model",
        options=["nomic-embed-text"],
        index=0,
        help="Make sure you pulled the embedding model first (e.g. 'ollama pull nomic-embed-text')"
    )
    st.sidebar.info("💡 Make sure Ollama application is running in the background.")

else:  # Groq (Free Cloud API)
    st.sidebar.subheader("Groq Settings")
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk-...",
            help="Provide your Groq API key from console.groq.com."
        )
    else:
        st.sidebar.success("Loaded Groq API Key from environment.")
        
    chat_model_name = st.sidebar.selectbox(
        "Groq LLM Model",
        options=["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0
    )
    
    # Select embedding source for Groq
    st.sidebar.subheader("Embeddings Source for Groq")
    embedding_provider = st.sidebar.selectbox(
        "Search Embedding Source",
        options=["Hugging Face Local (Free)", "Ollama Local (Free)", "OpenAI (Paid Cloud)"],
        index=0,
        help="Which embedding service to use to retrieve documents from your database."
    )
    
    if embedding_provider == "OpenAI (Paid Cloud)":
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            openai_api_key = st.sidebar.text_input(
                "OpenAI API Key (for search)",
                type="password",
                placeholder="sk-...",
                help="Required for OpenAI Embeddings."
            )
        embedding_model_name = st.sidebar.selectbox(
            "Embedding Model",
            options=["text-embedding-ada-002", "text-embedding-3-small"],
            index=0
        )
    elif embedding_provider == "Ollama Local (Free)":
        ollama_url = st.sidebar.text_input(
            "Ollama Server URL (for search)",
            value="http://localhost:11434",
            help="Required for local embedding generation."
        )
        embedding_model_name = st.sidebar.selectbox(
            "Embedding Model",
            options=["nomic-embed-text"],
            index=0
        )
    else:  # Hugging Face Local (Free)
        embedding_model_name = st.sidebar.selectbox(
            "Local HF Model",
            options=["all-MiniLM-L6-v2"],
            index=0
        )

# Supabase Settings
st.sidebar.subheader("Supabase Settings")
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")

# Determine default RPC dynamically to prevent dimension mismatches
if embedding_provider == "Hugging Face Local (Free)":
    default_rpc = "match_hf_documents"
elif embedding_provider in ["Ollama Local (Free)", "Ollama"]:
    default_rpc = "match_local_documents"
else:
    default_rpc = "match_documents"

rpc_function_name = st.sidebar.text_input(
    "RPC Function Name",
    value=default_rpc,
    help="The database function (RPC) in Supabase used for vector similarity search."
)

st.sidebar.subheader("Retrieval Hyperparameters")
match_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    help="Minimum similarity score required to include a text chunk."
)
match_count = st.sidebar.slider(
    "Top K Chunks",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Number of most relevant text chunks to retrieve."
)

# ---------------------------------------------------------
# 3. Connection Initializations
# ---------------------------------------------------------
@st.cache_resource
def get_supabase_client(url: str, key: str) -> Client:
    """
    Returns a cached Supabase client with URL sanitization.
    """
    if not url or not key:
        st.error("Missing SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY. Please verify your .env file.")
        st.stop()
    # Sanitize URL if it ends with /rest/v1/ or /rest/v1
    if url.endswith("/rest/v1/"):
        url = url[:-9]
    elif url.endswith("/rest/v1"):
        url = url[:-8]
    return create_client(url, key)

# Initialize Supabase client
supabase_client = get_supabase_client(supabase_url, supabase_key)

# Check for API credentials depending on selection
if provider == "OpenAI (Paid Cloud)" and not openai_api_key:
    st.warning("⚠️ OpenAI API Key is missing. Please enter your API key in the sidebar to start chatting.")
    st.stop()

if provider == "Groq (Free Cloud API)" and not groq_api_key:
    st.warning("⚠️ Groq API Key is missing. Please enter your API key in the sidebar to start chatting.")
    st.stop()

if provider == "Groq (Free Cloud API)" and embedding_provider == "OpenAI (Paid Cloud)" and not openai_api_key:
    st.warning("⚠️ OpenAI API Key is required for generating search embeddings. Please enter it in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# 4. RAG Retrieval Function
# ---------------------------------------------------------
def retrieve_context(query: str) -> list:
    """
    Embeds the query using the selected provider and performs a vector similarity search in Supabase.
    """
    try:
        if embedding_provider in ["OpenAI", "OpenAI (Paid Cloud)"]:
            embeddings = OpenAIEmbeddings(
                model=embedding_model_name,
                openai_api_key=openai_api_key
            )
        elif embedding_provider in ["Ollama", "Ollama Local (Free)"]:
            embeddings = OllamaEmbeddings(
                model=embedding_model_name,
                base_url=ollama_url
            )
        else:  # Hugging Face Local (Free)
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            
        # Convert text query to embedding vector
        query_embedding = embeddings.embed_query(query)
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        if "insufficient_quota" in str(e):
            st.info(
                "💡 **OpenAI Quota Exceeded**: Your account has run out of credits. "
                "Please add a credit balance on [OpenAI Billing](https://platform.openai.com/settings/organization/billing) to enable search."
            )
        elif "connection" in str(e).lower() or "refused" in str(e).lower():
            st.info(
                "💡 **Ollama Connection Failed**: Could not connect to the local Ollama server.\n"
                "- Verify Ollama is running in your taskbar / system tray.\n"
                "- Make sure you pulled the embedding model using `ollama pull nomic-embed-text`."
            )
        return []

    # Call Supabase vector similarity search function via RPC
    try:
        print(f"[DEBUG] Calling RPC '{rpc_function_name}' with threshold={match_threshold}, count={match_count}", flush=True)
        response = supabase_client.rpc(
            rpc_function_name,
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }
        ).execute()
        num_results = len(response.data) if response.data else 0
        print(f"[DEBUG] RPC '{rpc_function_name}' returned {num_results} results", flush=True)
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Supabase RPC search failed: {e}")
        st.info(
            "💡 **Dimension Mismatch Troubleshooting**:\n"
            "- If your Supabase vectors are in 1536 dimensions (OpenAI) or 768 dimensions (Ollama), but you search with 384 dimensions (HuggingFace), it will fail.\n"
            "- Ensure your selected RPC and database table match your selected embedding model."
        )
        return []

# ---------------------------------------------------------
# 5. UI Layout & Chat Stream Loop
# ---------------------------------------------------------
st.title("⚛️ Quantum Network RAG Engine")
st.caption("A specialized RAG-powered chatbot answering literature questions using ingested scientific papers.")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("View Retrieved Context Chunks"):
                for idx, chunk in enumerate(msg["context"], 1):
                    source = chunk.get("metadata", {}).get("title", "Unknown Source")
                    st.markdown(f"**Chunk {idx} (Source: {source}):**")
                    st.write(chunk.get("content") or chunk.get("chunk_content") or str(chunk))

# User Input
if user_query := st.chat_input("Ask a question about Quantum Networks..."):
    print(f"[DEBUG] Received User Query: '{user_query}'", flush=True)
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Retrieve Relevant Context
    with st.spinner("Retrieving relevant literature..."):
        context_chunks = retrieve_context(user_query)
    
    # Format context text for the prompt
    context_texts = []
    for chunk in context_chunks:
        content = chunk.get("content") or chunk.get("chunk_content") or chunk.get("text") or str(chunk)
        context_texts.append(content)
    
    formatted_context = "\n\n---\n\n".join(context_texts) if context_texts else "No relevant context found."
    print(f"[DEBUG] Formatted context length: {len(formatted_context)} characters across {len(context_texts)} chunks.", flush=True)

    # 3. Create LangChain Generation pipeline
    system_prompt = (
        "You are an expert quantum networking researcher. Answer the user's question using ONLY the provided context.\n"
        "If the answer is not in the context, say 'I cannot answer this based on the current literature'.\n\n"
        "Context:\n{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # Instantiate selected Chat LLM
    if provider == "OpenAI (Paid Cloud)":
        print(f"[DEBUG] Initializing OpenAI Chat LLM with model: {chat_model_name}", flush=True)
        llm = ChatOpenAI(
            model=chat_model_name,
            openai_api_key=openai_api_key,
            temperature=0.0,
            streaming=True
        )
    elif provider == "Ollama (Free Local)":
        print(f"[DEBUG] Initializing Ollama Chat LLM with model: {chat_model_name}", flush=True)
        llm = ChatOllama(
            model=chat_model_name,
            base_url=ollama_url,
            temperature=0.0
        )
    else:  # Groq (Free Cloud API)
        print(f"[DEBUG] Initializing Groq Chat LLM with model: {chat_model_name}", flush=True)
        llm = ChatGroq(
            model=chat_model_name,
            api_key=groq_api_key,
            temperature=0.0
        )

    rag_chain = prompt_template | llm | StrOutputParser()

    # 4. Generate & Stream Response in Streamlit Chat Window
    with st.chat_message("assistant"):
        # Display context inside expander for transparency
        if context_chunks:
            with st.expander("View Retrieved Context Chunks"):
                for idx, chunk in enumerate(context_chunks, 1):
                    meta = chunk.get("metadata") or {}
                    title = meta.get("title") or "Unknown Paper"
                    st.markdown(f"**Chunk {idx} (Source: {title}):**")
                    content = chunk.get("content") or chunk.get("chunk_content") or chunk.get("text") or str(chunk)
                    st.write(content)
        else:
            st.info("No matching papers found in vector search. The assistant will answer based on empty context.")

        # Stream response
        try:
            print(f"[DEBUG] Invoking stream for RAG chain...", flush=True)
            stream = rag_chain.stream({
                "context": formatted_context,
                "question": user_query
            })
            ai_response = st.write_stream(stream)
            print(f"[DEBUG] Stream finished successfully. Response length: {len(ai_response)} chars.", flush=True)
        except Exception as e:
            ai_response = "I cannot answer this due to an API error."
            print(f"[DEBUG] Exception during AI response generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            st.error(f"Error during AI response generation: {e}")
            if "insufficient_quota" in str(e):
                st.info(
                    "💡 **OpenAI Quota Exceeded**: Your API key works, but your OpenAI account has run out of credits.\n\n"
                    "To resolve this:\n"
                    "1. Go to [platform.openai.com/settings/organization/billing](https://platform.openai.com/settings/organization/billing).\n"
                    "2. Fund your account with a minimum credit balance (e.g., $5).\n"
                    "3. Wait 5-10 minutes for OpenAI to update your balance, then retry."
                )
            elif "connection" in str(e).lower() or "refused" in str(e).lower():
                st.info(
                    "💡 **Ollama Connection Failed**: Could not connect to the local Ollama server.\n"
                    "- Verify Ollama is running in your taskbar / system tray.\n"
                    "- Make sure you pulled the LLM model using `ollama pull llama3`."
                )
            elif "groq" in str(e).lower() or "gsk" in str(e).lower():
                st.info(
                    "💡 **Groq API Error**: Please verify your Groq API key is correct and you have not hit Groq rate limits."
                )

    # 5. Append Assistant Message to History
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
        "context": context_chunks
    })
