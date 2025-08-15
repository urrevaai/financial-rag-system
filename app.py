__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time
from dotenv import load_dotenv

@st.cache_resource
def setup_application():
    """
    Sets up the application by downloading data and building the vector store
    if it doesn't already exist.
    """
    if not os.path.exists("./chroma_db"):
        st.header("First-time Setup: Building Vector Database...")
        st.info("This is a one-time process and may take a few minutes.")
        
        with st.container():
            st.write("Step 1/2: Downloading SEC filings and extracting data...")
            from ingest_data import fetch_sec_filing, extract_risk_factors
            import json
            
            full_text, filing_date = fetch_sec_filing()
            if full_text:
                extract_risk_factors(full_text)
                with open("filing_metadata.json", "w") as f:
                    json.dump({"filing_date": filing_date}, f)
                st.write("✅ Data ingestion complete.")
            else:
                st.error("Failed to ingest data. Please check the logs.")
                return False

            st.write("Step 2/2: Building vector store... (This is the slow part)")
            from vector_store import build_vector_store
            build_vector_store()
            st.write("✅ Vector store built successfully.")
            
        st.success("Setup complete! The application is ready.")
        time.sleep(3)
        st.rerun()
    return True

setup_application()

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from tools import analyze_stock_trend, summarize_risk_factors, assess_news_sentiment

st.set_page_config(page_title="Financial Insights RAG System", page_icon="🚀", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #FFFFFF; text-align: center; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()
VECTOR_STORE_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    return vector_store

@st.cache_resource
def load_llm_chain():
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst AI..."),
        ("human", "CONTEXT:\n{context}\n\nADDITIONAL ANALYSIS:\n{analysis}\n\nQUESTION:\n{question}"),
    ])
    return prompt_template | llm

def financial_qa_tab(vector_store, llm_chain):
    st.header("💬 Financial Query (RAG)")
    st.write("Ask questions about the company's financial reports (10-K). The AI will retrieve relevant context to form an answer.")
    
    col1, col2 = st.columns(2)
    with col1:
        tool_question = st.selectbox(
            "Select an analysis type:",
            ("None", "Analyze Key Risks", "Analyze News Sentiment")
        )

    with col2:
        user_question = st.text_input("Or ask your own question:", placeholder="e.g., What is the company's strategy?")

    if st.button("🚀 Generate Insights", type="primary"):
        if not tool_question and not user_question:
            st.warning("Please select an analysis or enter a question.")
            return

        final_question = user_question if user_question else tool_question
        
        with st.spinner("Analyzing..."):
            additional_analysis = "No specific analysis tool was triggered."
            if tool_question == "Analyze Key Risks":
                additional_analysis = summarize_risk_factors()
            elif tool_question == "Analyze News Sentiment":
                additional_analysis = assess_news_sentiment()

            retrieved_docs = vector_store.similarity_search(final_question, k=3)
            context = ""
            for doc in retrieved_docs:
                filing_date = doc.metadata.get('filing_date', 'Unknown Date')
                context += f"From document filed on {filing_date}:\n{doc.page_content}\n\n"
            
            response = llm_chain.invoke({
                "context": context.strip(), "analysis": additional_analysis, "question": final_question
            })
            answer = response.content

            st.subheader("🤖 AI-Generated Insight")
            st.markdown(answer)
            with st.expander("Show Context & Tool Output"):
                st.subheader("Analysis Tool Output")
                st.markdown(additional_analysis)
                st.subheader("Retrieved Context from 10-K Report")
                st.text(context)

def market_analysis_tab():
    st.header(f"📈 Live Market Analysis for {analyze_stock_trend.__globals__['TICKER']}")
    
    if st.button("📊 Run Real-time Analysis", type="primary"):
        with st.spinner("Fetching live market data..."):
            analysis_result = analyze_stock_trend()
        
        if analysis_result:
            metrics = analysis_result["metrics"]
            chart_data = analysis_result["chart_data"]

            cols = st.columns(4)
            cols[0].metric(
                label="Latest Price",
                value=f"${float(metrics['latest_price']):.2f}",
                delta=f"${float(metrics['price_change']):.2f} ({float(metrics['price_change_pct']):.2%})"
            )
            cols[1].metric(label="Volume", value=f"{int(metrics['volume']) / 1_000_000:.2f}M")
            cols[2].metric(label="50-Day MA", value=f"${float(metrics['50_day_ma']):.2f}")
            cols[3].metric(label="200-Day MA", value=f"${float(metrics['200_day_ma']):.2f}")
            
            st.subheader("Data Preview (Last 10 Days)")
            st.dataframe(chart_data.tail(10))
            
        else:
            st.error("Failed to fetch market data. The API may be temporarily unavailable.")

def main():
    st.markdown('<h1 class="main-header">🚀 Apple based Financial Insights RAG System</h1>', unsafe_allow_html=True)
    
    vector_store = load_models()
    llm_chain = load_llm_chain()
    
    tab1, tab2 = st.tabs(["💬 Financial Q&A", "📈 Live Market Analysis"])
    
    with tab1:
        financial_qa_tab(vector_store, llm_chain)
        
    with tab2:
        market_analysis_tab()

if __name__ == "__main__":
    main()
