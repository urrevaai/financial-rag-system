## ⚙️ Setup and Installation
First, you need to set up your environment and install the necessary packages.

Clone the Repository: Get the code from GitHub.

git clone <your-repo-url>
cd financial-rag-project
Create and Activate a Virtual Environment: This keeps your project's dependencies separate.

For Windows:

python -m venv venv
.\venv\Scripts\activate
For Mac/Linux:

python3 -m venv venv
source venv/bin/activate
Install Dependencies: This command installs all the required libraries.

pip install -r requirements.txt
Set Up API Key: The application needs a Groq API key to work.

Get a free key from GroqCloud.

Create a file named .env in your project folder.

Add your key to the file like this: GROQ_API_KEY="gsk_YourKeyHere"

## 🚀 How to Run the Application
Once the setup is complete, you can launch the application.

Run the Streamlit command from your terminal:

streamlit run app.py
Open your web browser and go to the local URL provided (usually http://localhost:8501).

The first time you run it, the app will automatically download the necessary financial data and build the database. This might take a few minutes, but it only happens once. After that, the app will load directly to the main interface.