import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to prevent GUI errors
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_data_from_database(db_name='transactions_data.db', table_name='transactions'):
    conn = sqlite3.connect(db_name)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except pd.io.sql.DatabaseError:
        df = pd.DataFrame() # Return empty DataFrame if table doesn't exist
    finally:
        conn.close()
    return df

def preprocess_data(df):
    df.dropna(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    return df

def analyze_and_train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    
    return rf_pred, lr_pred

def classify_transactions(df):
    large_threshold_usd = 10000  # USD
    small_threshold_usd = 100     # USD
    large_threshold_size = 100    # Arbitrary threshold for size
    large_threshold_weight = 100  # Arbitrary threshold for weight
    
    df['transaction_type'] = 'Unknown'
    df.loc[(df['input_total_usd'] > large_threshold_usd) | 
           (df['output_total_usd'] > large_threshold_usd) |
           (df['size'] > large_threshold_size) |
           (df['weight'] > large_threshold_weight), 'transaction_type'] = 'Illicit'
    df.loc[(df['input_total_usd'] < small_threshold_usd) &
           (df['output_total_usd'] < small_threshold_usd), 'transaction_type'] = 'Licit'
    return df

# Store results in a new SQLite database
def store_results_in_database(df, db_name='classified_transactions.db', table_name='transactions'):
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def network_analysis(df):
    # Grouping by hash and counting occurrences
    hash_counts = df['hash'].value_counts()
    return hash_counts

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message='No file selected.')
        if file:
            # Create uploads folder if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath, sep='\t')
            df = preprocess_data(df)
            
            # Classify transactions
            df = classify_transactions(df)

            # Store classified data in the database
            store_results_in_database(df) # Uses default 'classified_transactions.db'

            # Perform network analysis
            hash_counts = network_analysis(df)
            
            # Plotting the network analysis results
            plt.figure(figsize=(10, 6))
            hash_counts.plot(kind='bar', color='skyblue')
            plt.xlabel('Hash')
            plt.ylabel('Frequency')
            plt.title('Network Analysis: Hash Frequencies')
            plt.tight_layout()
            plt.savefig('static/network_analysis_plot.png')  # Save the plot to a file
            plt.close()  # Close the plot to free up resources
            
            success_message = f"File '{filename}' processed successfully. Network analysis plot generated and transactions classified and stored."
            return render_template('index.html', message=success_message)
    return render_template('upload.html', message='')

@app.route('/database', methods=['GET'])
def view_database():
    df = load_data_from_database(db_name='classified_transactions.db')
    if df.empty:
        return render_template('database.html', message="No data found in the database. Please upload a file first.")
    return render_template('database.html', tables=[df.to_html(classes='data', escape=False, index=False)], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
