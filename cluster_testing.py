# Step 1: Import necessary libraries
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# Step 2: Load your Excel dataset (make sure the file is in the same directory)
df = pd.read_excel('nadil_dataset_final.xlsx')  # Change the path if needed

# Step 3: Data Refinement - Text Preprocessing

# Create a dictionary of abbreviations
abbreviations = {
    'PYT': 'Payment',
    'TRF': 'Transfer',
    'DEP': 'Deposit',
    'WDL': 'Withdrawal',
    'WD': 'Withdrawal',
    'POS': 'Point of Sale',
    'ATM': 'ATM Withdrawal',
    'CHQ': 'Cheque',
    'DD': 'Demand Draft',
    'BT': 'Bank Transfer',
    'ACH': 'Automated Clearing House',
    'NEFT': 'National Electronic Funds Transfer',
    'RTGS': 'Real-Time Gross Settlement',
    'IMPS': 'Immediate Payment Service',
    'UPI': 'Unified Payments Interface',
    'INT': 'Interest',
    'CHG': 'Charge',
    'FEE': 'Fee',
    'TXN': 'Transaction',
    'REV': 'Reversal',
    'EMI': 'Equated Monthly Installment',
    'CC': 'Credit Card',
    'POS REF': 'Point of Sale Refund',
    'BIL': 'Bill Payment',
    'BILP': 'Bill Payment',
    'INV': 'Investment',
    'REF': 'Refund',
    'SAL': 'Salary Credit',
    'SL': 'Salary Credit',
    'TFR': 'Transfer'
}


# Step 3.1: Normalize Capitalization and Expand Abbreviations
def clean_text(text, abbr_dict):
    # Convert text to lowercase
    text = text.lower()

    # Expand abbreviations
    for abbr, full_form in abbr_dict.items():
        text = re.sub(rf'\b{abbr.lower()}\b', full_form.lower(), text)

    return text


# Apply text cleaning to 'Particulars' column
df['cleaned_particulars'] = df['Discription'].apply(lambda x: clean_text(str(x), abbreviations))


# Step 4: Use Sentence Transformers to Create Embeddings

# Initialize the sentence transformer model (you can choose any model you prefer)
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings for the cleaned text
embeddings = model.encode(df['cleaned_particulars'].tolist())

# Step 5: Clustering using HDBSCAN

# Step 5.1: Normalize the embeddings (this is optional but can improve HDBSCAN performance)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Step 5.2: Apply HDBSCAN Clustering
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)  # Adjust min_cluster_size as needed
labels = hdbscan_model.fit_predict(embeddings_scaled)

# Step 6: Add the cluster labels to the dataframe
df['Cluster'] = labels

# Step 7: Print all clusters with their data

# Loop through all unique clusters (ignoring noise points labeled as -1)
unique_clusters = set(labels)

for cluster in unique_clusters:
    print(f"\nCluster {cluster} Transactions:")

    # Filter rows for the current cluster
    cluster_data = df[df['Cluster'] == cluster]

    # Print the details of each transaction in the cluster
    print(cluster_data[['Discription', 'Payments', 'Receipts', 'Balance']])

