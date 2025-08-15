# 🚀 Apple based Financial Analysis RAG System

## https://financial-rag-system.streamlit.app/
`![Financial RAG System Preview](preview.png)`

This project is a sophisticated Retrieval-Augmented Generation (RAG) system designed to provide financial insights by integrating multiple data sources. It features a user-friendly Streamlit application that can answer questions about a company's financial reports, analyze real-time market data, and assess risks.

A key feature of this system is its **on-demand setup process**. The application automatically builds its own vector database the first time it's deployed, making it portable and easy to launch.

---

## ✨ Key Features

* **Automated First-Time Setup**: The app automatically ingests SEC filings and builds its own vector database on the initial run, eliminating the need to upload large database files.
* **Multi-Source Data Integration**: Combines SEC 10-K filings, real-time stock prices from `yfinance`, and recent news headlines.
* **Advanced RAG Q&A**: Uses a Groq-powered Llama 3 model to answer natural language questions based on context retrieved from financial reports.
* **Temporal Context**: Enriches the vector database with the filing date of documents, allowing the AI to understand the age of the information it's analyzing.
* **Real-time Market Analysis**: Features a dedicated dashboard to fetch and display live stock data, including price, volume, and key technical indicators.
* **In-depth Risk Assessment**: Automatically extracts and summarizes the "Risk Factors" section from 10-K reports for a focused and relevant risk analysis.

---

## 🏗️ System Architecture

The system is designed for easy deployment and robust performance.

### 1. On-Demand Setup (First Run Only)

When the Streamlit application starts for the first time on a new server, it checks if the `chroma_db` folder exists. If not, it automatically triggers a one-time setup process:

* **`ingest_data.py`**: Connects to the SEC EDGAR database to download the latest 10-K report, extracts the "Risk Factors" section, and saves the filing date.
* **`vector_store.py`**: The downloaded report is chunked, converted into vector embeddings, and stored in a new, persistent `chroma_db` folder on the server.
* The application then reloads to its main interface.

### 2. Online Query Processing (Normal Operation)

* **User Interface**: The Streamlit app presents two main tabs: "Financial Q&A" and "Live Market Analysis".
* **Market Analysis Tab**: Fetches live market data via the `yfinance` API and displays key metrics and a data table.
* **Financial Q&A Tab**:
    1.  **Tool Routing**: The app checks if the user's query triggers a specialized tool (e.g., asking about "risks").
    2.  **Retrieval**: The user's question is used to query the local `chroma_db` to find the most relevant text chunks from the 10-K report.
    3.  **Augmentation & Generation**: The question, retrieved context, and any tool output are sent to the Groq API to generate a final, context-aware answer using the Llama 3 model.

---

## 🛠️ Tech Stack

* **Application Framework**: Streamlit
* **LLM & RAG Orchestration**: LangChain
* **Large Language Model**: Llama 3 (via Groq API)
* **Embedding Model**: HuggingFace `all-MiniLM-L6-v2`
* **Vector Database**: ChromaDB (`0.4.15` for stability)
* **Data Sources**: SEC EDGAR API, `yfinance`

---

## ⚙️ Setup and Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/urrevaai/financial-rag-system/tree/main
cd financial-rag-project
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Up API Keys

You need a free API key from Groq to power the language model.

1.  Go to [GroqCloud](https://console.groq.com/keys) to get your API key.
2.  Create a file named `.env` in the root of your project folder.
3.  Add your key to the file like this:
    ```
    GROQ_API_KEY="gsk_YourKeyHere"
    ```

---

## 🚀 How to Run the Application Locally

Simply launch the Streamlit app. The setup process will run automatically if needed.

```bash
streamlit run app.py
```

---

## 🌐 Deployment Guide (Streamlit Community Cloud)

This project is optimized for easy deployment.

#### 1. Upload to GitHub

Push your project to a **public** GitHub repository. Make sure your `.gitignore` file is configured to **ignore** the `chroma_db` folder and your `.env` file. You only need to upload the script files.

#### 2. Deploy on Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
2.  Click "**New app**" and select your repository.
3.  Ensure the "Main file path" is set to `app.py`.
4.  Click on "**Advanced settings**" and go to the "**Secrets**" section.
5.  Paste your Groq API key in this format:
    ```toml
    GROQ_API_KEY = "gsk_YourActualKeyHere"
    ```
6.  Click "**Deploy!**".

The app will start, display the "First-time Setup" screen while it builds the database (this may take a few minutes), and then automatically reload into the main application, ready to use.

---

## 🔍 Troubleshooting

This repository includes built-in fixes for common deployment issues on Streamlit Community Cloud.

* **`sqlite3` Version Error**: Streamlit's environment has an outdated version of `sqlite3`. This project solves this by including `pysqlite3-binary` in `requirements.txt` and a patch at the top of `app.py` to force the use of the newer version.
* **ChromaDB Telemetry Warnings**: Harmless `capture() takes 1 positional argument but 3 were given` warnings are prevented by disabling ChromaDB's anonymous telemetry feature in the `vector_store.py` script.
* **Library Compatibility**: The `requirements.txt` file pins specific versions of `chromadb` and `langchain-community` to ensure they work together without `AttributeError` issues.

---

