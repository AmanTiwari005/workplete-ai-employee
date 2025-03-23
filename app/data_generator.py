# app/data_generator.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import datetime as dt

def generate_synthetic_lead_data(n_samples=1000, seed=42):
    """Generate synthetic lead data for the logistics industry."""
    np.random.seed(seed)
    
    # Generate basic features and target
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        random_state=seed
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[
        'inquiry_frequency', 
        'time_spent_on_site', 
        'pages_visited',
        'downloads', 
        'email_opens', 
        'company_size',
        'previous_orders',
        'quote_requests'
    ])
    
    # Normalize features to meaningful ranges
    df['inquiry_frequency'] = (df['inquiry_frequency'] * 10).round().clip(0, 20)
    df['time_spent_on_site'] = ((df['time_spent_on_site'] + 3) * 5).round().clip(1, 30)  # minutes
    df['pages_visited'] = ((df['pages_visited'] + 3) * 3).round().clip(1, 20)
    df['downloads'] = ((df['downloads'] + 3) * 1).round().clip(0, 10)
    df['email_opens'] = ((df['email_opens'] + 3) * 2).round().clip(0, 15)
    df['company_size'] = pd.cut(
        ((df['company_size'] + 3) * 100).round().clip(5, 1000),
        bins=[0, 20, 100, 500, 1000],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    )
    df['previous_orders'] = ((df['previous_orders'] + 3) * 0.5).round().clip(0, 5)
    df['quote_requests'] = ((df['quote_requests'] + 3) * 0.8).round().clip(0, 8)
    
    # Add more domain-specific features
    df['industry'] = np.random.choice(
        ['Manufacturing', 'Retail', 'Healthcare', 'Technology', 'Services'],
        size=n_samples
    )
    df['region'] = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central'],
        size=n_samples
    )
    
    # Last contact date (within last 90 days)
    today = dt.datetime.now()
    df['last_contact'] = [
        (today - dt.timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d')
        for _ in range(n_samples)
    ]
    
    # Initial contact date (within last 360 days)
    df['initial_contact'] = [
        (today - dt.timedelta(days=np.random.randint(90, 360))).strftime('%Y-%m-%d')
        for _ in range(n_samples)
    ]
    
    # Add lead source
    df['lead_source'] = np.random.choice(
        ['Website', 'Referral', 'Trade Show', 'Cold Call', 'Social Media'],
        size=n_samples
    )
    
    # Add target variable (converted: 1=yes, 0=no)
    df['converted'] = y
    
    # Add customer ID
    df['customer_id'] = [f'CUST-{i+10000}' for i in range(n_samples)]
    
    # If converted, add contract value
    avg_values = {
        'Small': 5000, 
        'Medium': 15000, 
        'Large': 50000, 
        'Enterprise': 120000
    }
    
    df['contract_value'] = 0
    for idx, row in df.iterrows():
        if row['converted'] == 1:
            base_value = avg_values[row['company_size']]
            variation = np.random.uniform(0.7, 1.3)
            df.at[idx, 'contract_value'] = int(base_value * variation)
    
    return df

if __name__ == "__main__":
    # Generate and save data
    df = generate_synthetic_lead_data(1500)
    df.to_csv("data/sample_leads.csv", index=False)
    print(f"Generated {len(df)} synthetic lead records")