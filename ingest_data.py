import yfinance as yf
import requests
import re
import json
import os
from bs4 import BeautifulSoup

# --- Configuration ---
TICKER = "AAPL"
COMPANY_CIK = "0000320193"
YOUR_EMAIL = "your_email@example.com"
HEADERS = {'User-Agent': f'YourName {YOUR_EMAIL}'}
OUTPUT_REPORT_PATH = "sec_10k_report.txt"
OUTPUT_RISK_FACTORS_PATH = "risk_factors.txt"

def fetch_news_headlines():
    """Scrapes Yahoo Finance for recent news headlines for the ticker."""
    print("Fetching news headlines...")
    news_headlines = []
    try:
        url = f"https://finance.yahoo.com/quote/{TICKER}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This selector targets the headlines in the main news feed of Yahoo Finance
        headlines = soup.find_all('h3', class_='Mb(5px)')
        for headline in headlines:
            if headline.a:
                news_headlines.append(headline.a.text)
        
        print(f"Found {len(news_headlines)} headlines.")
        return news_headlines
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news headlines: {e}")
        return []

def fetch_sec_filing():
    """Fetches the latest 10-K filing text and its filing date."""
    print("Fetching SEC 10-K filing...")
    try:
        submissions_url = f"https://data.sec.gov/submissions/CIK{COMPANY_CIK.zfill(10)}.json"
        response = requests.get(submissions_url, headers=HEADERS)
        response.raise_for_status()
        submissions = response.json()

        latest_10k_accession = None
        filing_date = None
        for i in range(len(submissions['filings']['recent']['filingDate'])):
            if submissions['filings']['recent']['form'][i] == '10-K':
                latest_10k_accession = submissions['filings']['recent']['accessionNumber'][i]
                filing_date = submissions['filings']['recent']['filingDate'][i]
                break

        if not latest_10k_accession:
            print("Could not find a 10-K filing.")
            return None, None

        filing_url = f"https://www.sec.gov/Archives/edgar/data/{COMPANY_CIK}/{latest_10k_accession.replace('-', '')}/{latest_10k_accession}.txt"
        response = requests.get(filing_url, headers=HEADERS)
        response.raise_for_status()
        full_text = response.text

        with open(OUTPUT_REPORT_PATH, "w", encoding='utf-8') as f:
            f.write(full_text)
        print(f"Successfully saved 10-K report to {OUTPUT_REPORT_PATH}")
        return full_text, filing_date

    except Exception as e:
        print(f"Error fetching SEC filing: {e}")
        return None, None

def extract_risk_factors(full_text):
    """Extracts the 'Risk Factors' section (Item 1A) from the 10-K filing."""
    print("Extracting 'Risk Factors' section (Item 1A)...")
    try:
        # Use regex to find Item 1A. This is a simplified pattern and might need adjustment.
        # It looks for "Item 1A." and captures text until it finds "Item 1B."
        pattern = re.compile(r'Item\s+1A\.\s+Risk\s+Factors\s*\.?(.*?)Item\s+1B\.', re.DOTALL | re.IGNORECASE)
        match = pattern.search(full_text)
        
        if match:
            risk_text = match.group(1).strip()
            with open(OUTPUT_RISK_FACTORS_PATH, 'w', encoding='utf-8') as f:
                f.write(risk_text)
            print(f"Successfully extracted and saved risk factors to {OUTPUT_RISK_FACTORS_PATH}")
        else:
            print("Could not find the 'Risk Factors' section using the primary pattern.")
            # Fallback pattern for documents that might not have the exact same structure
            pattern_fallback = re.compile(r'RISK\s+FACTORS(.*?)(?=\n\s*\n\s*Item|\Z)', re.DOTALL | re.IGNORECASE)
            match_fallback = pattern_fallback.search(full_text)
            if match_fallback:
                 risk_text = match_fallback.group(1).strip()
                 with open(OUTPUT_RISK_FACTORS_PATH, 'w', encoding='utf-8') as f:
                    f.write(risk_text)
                 print(f"Successfully extracted and saved risk factors using fallback pattern.")
            else:
                print("Fallback pattern also failed. Risk factors not extracted.")

    except Exception as e:
        print(f"An error occurred during risk factor extraction: {e}")

if __name__ == "__main__":
    print("--- Starting Upgraded Data Ingestion ---")
    full_text, filing_date = fetch_sec_filing()
    if full_text:
        extract_risk_factors(full_text)
        # Store filing_date for the vector store script to use
        with open("filing_metadata.json", "w") as f:
            json.dump({"filing_date": filing_date}, f)
        print(f"Filing date ({filing_date}) saved for metadata.")
    print("--- Data Ingestion Complete ---")