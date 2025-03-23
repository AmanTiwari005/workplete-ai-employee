import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime as dt
import shap
import time
import base64
from io import BytesIO
from app.data_generator import generate_synthetic_lead_data
from app.data_processor import DataProcessor
from app.lead_scorer import LeadScorer

# Import our modules
from app.data_generator import generate_synthetic_lead_data
from app.data_processor import DataProcessor
from app.lead_scorer import LeadScorer

# Session state initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.model_trained = False
    st.session_state.feedback_records = []
    st.session_state.voice_enabled = False
    st.session_state.last_prediction = None

# Set page config
st.set_page_config(
    page_title="Workplete Lead Scoring AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    .metric-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
    }
    .hot-lead {
        color: #EF4444;
        font-weight: bold;
    }
    .warm-lead {
        color: #F59E0B;
        font-weight: bold;
    }
    .cool-lead {
        color: #10B981;
        font-weight: bold;
    }
    .cold-lead {
        color: #6B7280;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for voice synthesis (mockup)
def synthesize_voice(text):
    if not st.session_state.voice_enabled:
        return f"Voice synthesis disabled: {text}"
    
    # Inject JavaScript to list voices and use the first English one
    js_code = f"""
    <script>
    var voices = window.speechSynthesis.getVoices();
    var utterance = new SpeechSynthesisUtterance("{text}");
    utterance.lang = "en-US";
    // Use the first available English voice
    for (var i = 0; i < voices.length; i++) {{
        if (voices[i].lang === "en-US") {{
            utterance.voice = voices[i];
            break;
        }}
    }}
    utterance.volume = 1.0;
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
    </script>
    """
    html(js_code, height=0)
    return None

# Helper function to generate a download link for a dataframe
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Helper function to get lead quality color class
def get_lead_quality_class(probability):
    if probability >= 0.75:
        return "hot-lead"
    elif probability >= 0.5:
        return "warm-lead"
    elif probability >= 0.25:
        return "cool-lead"
    else:
        return "cold-lead"

# Helper function to get lead quality
def get_lead_quality(probability):
    if probability >= 0.75:
        return "Hot 🔥"
    elif probability >= 0.5:
        return "Warm 🔆"
    elif probability >= 0.25:
        return "Cool 🧊"
    else:
        return "Cold ❄️"

# Initialize data and models
def initialize_app():
    if st.session_state.initialized:
        return
    
    with st.spinner("Setting up AI Employee..."):
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        if not os.path.exists("data/sample_leads.csv"):
            df = generate_synthetic_lead_data(1500)
            df.to_csv("data/sample_leads.csv", index=False)
            st.session_state.df = df
        else:
            st.session_state.df = pd.read_csv("data/sample_leads.csv")
        
        st.session_state.initialized = True

# Train/load model
def setup_model():
    if st.session_state.model_trained:
        return
    
    with st.spinner("Training lead scoring model..."):
        data_processor = DataProcessor()
        X_train, X_test, y_train, y_test, _ = data_processor.process_data(st.session_state.df)  # Note: feature_names not needed here
        
        data_processor.save_pipeline()
        
        lead_scorer = LeadScorer(model_type='gradient_boosting')
        lead_scorer.train(X_train, y_train, preprocessor=data_processor.pipeline)  # Pass the pipeline
        
        metrics = lead_scorer.evaluate(X_test, y_test)
        
        lead_scorer.save_model()
        
        st.session_state.data_processor = data_processor
        st.session_state.lead_scorer = lead_scorer
        st.session_state.metrics = metrics
        st.session_state.test_data = (X_test, y_test)
        st.session_state.model_trained = True

# Sidebar navigation
def sidebar():
    st.sidebar.markdown("<div class='main-header'>Workplete AI</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sub-header'>Lead Scoring System</div>", unsafe_allow_html=True)
    
    navigation = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Lead Analyzer", "Model Performance", "Feedback Loop", "Settings"]
    )
    
    if st.sidebar.checkbox("Enable Voice Feedback", value=st.session_state.voice_enabled):
        st.session_state.voice_enabled = True
    else:
        st.session_state.voice_enabled = False
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This AI employee helps your sales team prioritize leads based on their conversion potential. "
        "It analyzes customer data and provides actionable insights to maximize efficiency."
    )
    
    return navigation

# Dashboard page (already complete)
def show_dashboard():
    st.markdown("<div class='main-header'>Lead Scoring Dashboard</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
        st.metric("Total Leads", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
        conversion_rate = df['converted'].mean() * 100
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
        avg_contract = df[df['converted'] == 1]['contract_value'].mean()
        st.metric("Avg. Contract Value", f"${avg_contract:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
        total_revenue = df[df['converted'] == 1]['contract_value'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='sub-header'>Conversion by Lead Source</div>", unsafe_allow_html=True)
        source_conversion = df.groupby('lead_source')['converted'].agg(['count', 'mean'])
        source_conversion.columns = ['Count', 'Conversion Rate']
        source_conversion['Conversion Rate'] = source_conversion['Conversion Rate'] * 100
        
        fig = px.bar(
            source_conversion.reset_index(),
            x='lead_source',
            y='Conversion Rate',
            color='Count',
            labels={'lead_source': 'Lead Source', 'Conversion Rate': 'Conversion Rate (%)'},
            title='Conversion Rate by Lead Source',
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='sub-header'>Revenue by Lead Source</div>", unsafe_allow_html=True)
        source_revenue = df[df['converted'] == 1].groupby('lead_source')['contract_value'].sum().reset_index()
        
        fig = px.pie(
            source_revenue,
            values='contract_value',
            names='lead_source',
            title='Revenue Distribution by Lead Source',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<div class='sub-header'>Company Size Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        size_conversion = df.groupby('company_size')['converted'].agg(['count', 'mean'])
        size_conversion.columns = ['Count', 'Conversion Rate']
        size_conversion['Conversion Rate'] = size_conversion['Conversion Rate'] * 100
        size_conversion = size_conversion.reindex(['Small', 'Medium', 'Large', 'Enterprise'])
        
        fig = px.bar(
            size_conversion.reset_index(),
            x='company_size',
            y='Conversion Rate',
            color='Count',
            labels={'company_size': 'Company Size', 'Conversion Rate': 'Conversion Rate (%)'},
            title='Conversion Rate by Company Size',
            color_continuous_scale=px.colors.sequential.Blues,
            category_orders={"company_size": ["Small", "Medium", "Large", "Enterprise"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        size_revenue = df[df['converted'] == 1].groupby('company_size')['contract_value'].agg(['sum', 'mean'])
        size_revenue.columns = ['Total Revenue', 'Average Revenue']
        size_revenue = size_revenue.reindex(['Small', 'Medium', 'Large', 'Enterprise'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=size_revenue.index,
            y=size_revenue['Average Revenue'],
            name='Average Revenue',
            marker_color='rgb(158,202,225)'
        ))
        
        fig.add_trace(go.Scatter(
            x=size_revenue.index,
            y=size_revenue['Total Revenue']/max(size_revenue['Total Revenue'])*max(size_revenue['Average Revenue']),
            name='Total Revenue (Scaled)',
            yaxis='y2',
            line=dict(color='rgb(8,48,107)', width=2)
        ))
        
        fig.update_layout(
            title='Revenue Metrics by Company Size',
            yaxis=dict(
                title='Average Revenue ($)'
            ),
            yaxis2=dict(
                title='Total Revenue (Scaled)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<div class='sub-header'>Recent Leads</div>", unsafe_allow_html=True)
    
    recent_leads = df.sort_values('last_contact', ascending=False).head(10)
    recent_leads['last_contact'] = pd.to_datetime(recent_leads['last_contact']).dt.strftime('%Y-%m-%d')
    recent_leads['converted'] = recent_leads['converted'].map({1: '✅ Yes', 0: '❌ No'})
    recent_leads['contract_value'] = recent_leads['contract_value'].apply(lambda x: f"${x:,.2f}" if x > 0 else "-")
    
    st.dataframe(
        recent_leads[['customer_id', 'company_size', 'industry', 'lead_source', 'last_contact', 'converted', 'contract_value']],
        use_container_width=True
    )
    
    st.markdown(
        get_csv_download_link(df, "all_leads.csv", "Download Complete Lead Data"),
        unsafe_allow_html=True
    )

# Lead Analyzer page
def show_lead_analyzer():
    st.markdown("<div class='main-header'>Lead Analyzer</div>", unsafe_allow_html=True)
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Analyze Existing Lead", "Score New Lead"]
    )
    
    if analysis_type == "Analyze Existing Lead":
        analyze_existing_lead()
    else:
        score_new_lead()

def analyze_existing_lead():
    df = st.session_state.df
    
    st.markdown("<div class='sub-header'>Select Lead to Analyze</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lead_id = st.selectbox("Select Customer ID", df['customer_id'].unique())
    
    with col2:
        industries = st.multiselect("Filter by Industry", df['industry'].unique(), default=df['industry'].unique())
    
    with col3:
        company_sizes = st.multiselect("Filter by Company Size", df['company_size'].unique(), default=df['company_size'].unique())
    
    filtered_df = df[(df['customer_id'] == lead_id) & 
                    (df['industry'].isin(industries)) & 
                    (df['company_size'].isin(company_sizes))]
    
    if not filtered_df.empty:
        lead_data = filtered_df.iloc[0]
        processed_lead = st.session_state.data_processor.process_new_lead(lead_data)
        probability = st.session_state.lead_scorer.predict_proba(processed_lead)[0, 1]
        
        st.markdown(f"### Lead: {lead_id}")
        st.markdown(f"**Conversion Probability**: {probability:.2%}")
        st.markdown(f"**Lead Quality**: <span class='{get_lead_quality_class(probability)}'>{get_lead_quality(probability)}</span>", unsafe_allow_html=True)
        
        shap_values = st.session_state.lead_scorer.explain_prediction(processed_lead)
        st.markdown("#### Key Factors Influencing Prediction")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, processed_lead, feature_names=st.session_state.lead_scorer.feature_names, plot_type="bar")
        st.pyplot(fig)
        
        if st.session_state.voice_enabled:
            st.write(synthesize_voice(f"This lead has a {probability:.2%} chance of converting and is classified as {get_lead_quality(probability)}."))
    else:
        st.error("No matching lead found.")

def score_new_lead():
    st.markdown("<div class='sub-header'>Enter New Lead Details</div>", unsafe_allow_html=True)
    
    # Form for lead scoring
    with st.form("new_lead_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_size = st.selectbox("Company Size", ["Small", "Medium", "Large", "Enterprise"], key="new_company_size")
            industry = st.selectbox("Industry", ["Manufacturing", "Retail", "Healthcare", "Technology", "Services"], key="new_industry")
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"], key="new_region")
            lead_source = st.selectbox("Lead Source", ["Website", "Referral", "Trade Show", "Cold Call", "Social Media"], key="new_lead_source")
        
        with col2:
            inquiry_frequency = st.number_input("Inquiry Frequency", min_value=0, max_value=20, value=5, key="new_inquiry_frequency")
            time_spent_on_site = st.number_input("Time Spent on Site (minutes)", min_value=0, max_value=30, value=10, key="new_time_spent")
            pages_visited = st.number_input("Pages Visited", min_value=0, max_value=20, value=5, key="new_pages_visited")
            downloads = st.number_input("Downloads", min_value=0, max_value=10, value=2, key="new_downloads")
            email_opens = st.number_input("Email Opens", min_value=0, max_value=15, value=3, key="new_email_opens")
            previous_orders = st.number_input("Previous Orders", min_value=0, max_value=5, value=0, key="new_previous_orders")
            quote_requests = st.number_input("Quote Requests", min_value=0, max_value=8, value=1, key="new_quote_requests")
        
        last_contact = st.date_input("Last Contact Date", value=dt.datetime.now(), key="new_last_contact")
        initial_contact = st.date_input("Initial Contact Date", value=dt.datetime.now() - dt.timedelta(days=30), key="new_initial_contact")
        
        submitted = st.form_submit_button("Score Lead")
    
    # Process lead scoring
    if submitted:
        new_lead = {
            'company_size': st.session_state.new_company_size,
            'industry': st.session_state.new_industry,
            'region': st.session_state.new_region,
            'lead_source': st.session_state.new_lead_source,
            'inquiry_frequency': st.session_state.new_inquiry_frequency,
            'time_spent_on_site': st.session_state.new_time_spent,
            'pages_visited': st.session_state.new_pages_visited,
            'downloads': st.session_state.new_downloads,
            'email_opens': st.session_state.new_email_opens,
            'previous_orders': st.session_state.new_previous_orders,
            'quote_requests': st.session_state.new_quote_requests,
            'last_contact': st.session_state.new_last_contact.strftime('%Y-%m-%d'),
            'initial_contact': st.session_state.new_initial_contact.strftime('%Y-%m-%d')
        }
        
        processed_lead = st.session_state.data_processor.process_new_lead(new_lead)
        probability = st.session_state.lead_scorer.predict_proba(processed_lead)[0, 1]
        st.session_state.last_prediction = (new_lead, probability)
        
        # Display score immediately
        st.markdown(f"### New Lead Score")
        st.markdown(f"**Conversion Probability**: {probability:.2%}")
        st.markdown(f"**Lead Quality**: <span class='{get_lead_quality_class(probability)}'>{get_lead_quality(probability)}</span>", unsafe_allow_html=True)
        
        if st.session_state.voice_enabled:
            st.write(synthesize_voice(f"This new lead has a {probability:.2%} chance of converting and is classified as {get_lead_quality(probability)}."))
    
    # Feedback section (always shown if there's a last prediction)
    if 'last_prediction' in st.session_state and st.session_state.last_prediction:
        new_lead, probability = st.session_state.last_prediction
        
        st.markdown("### Feedback on This Prediction")
        data_ok = st.radio("Is this lead data accurate?", ["Yes", "No", "Unsure"], key="data_feedback_new_lead")
        if data_ok == "No":
            true_outcome = st.selectbox("What was the actual outcome?", ["Converted", "Not Converted", "Unknown"], key="true_outcome_new_lead")
            comments = st.text_area("Why was the prediction inaccurate?", key="data_comments_new_lead")
        else:
            comments = st.text_area("Additional Comments (optional)", key="data_comments_optional_new_lead")
        
        if st.button("Submit Feedback", key="submit_feedback_new_lead"):
            feedback_record = {
                'lead_data': new_lead,
                'probability': probability,
                'feedback': data_ok,
                'comments': comments,
                'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            if data_ok == "No" and true_outcome != "Unknown":
                feedback_record['true_outcome'] = 1 if true_outcome == "Converted" else 0
            st.session_state.feedback_records.append(feedback_record)
            st.success("Feedback submitted successfully!")
            if st.session_state.voice_enabled:
                st.write(synthesize_voice("Thank you for your feedback!"))
            st.write(f"Debug: Total feedback records: {len(st.session_state.feedback_records)}")
            
# Model Performance page
def show_model_performance():
    st.markdown("<div class='main-header'>Model Performance</div>", unsafe_allow_html=True)
    
    metrics = st.session_state.metrics
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col3:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
        st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
    
    st.markdown("### Confusion Matrix")
    cm_fig = go.Figure(data=go.Heatmap(
        z=[[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        colorscale='Blues',
        text=[[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    st.plotly_chart(cm_fig, use_container_width=True)
    
    st.markdown("### Feature Importance")
    feature_importance = st.session_state.lead_scorer.get_feature_importance(top_n=10)
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title='Top 10 Important Features',
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Feedback Loop page
def show_feedback_loop():
    st.markdown("<div class='main-header'>Feedback History</div>", unsafe_allow_html=True)
    
    st.markdown("### Submitted Feedback")
    if not st.session_state.feedback_records:
        st.info("No feedback submitted yet.")
    else:
        feedback_df = pd.DataFrame(st.session_state.feedback_records)
        # Ensure all expected columns are present, fill missing with 'N/A'
        expected_columns = ['timestamp', 'probability', 'feedback', 'comments', 'true_outcome']
        for col in expected_columns:
            if col not in feedback_df.columns:
                feedback_df[col] = 'N/A'
        
        # Display the DataFrame with selected columns
        display_df = feedback_df[['timestamp', 'probability', 'feedback', 'comments', 'true_outcome']]
        st.dataframe(display_df, use_container_width=True)
        
        # Add download link
        st.markdown(get_csv_download_link(display_df, "feedback_history.csv", "Download Feedback History"), unsafe_allow_html=True)
        
        # Debug: Show raw data
        with st.expander("View Raw Feedback Data"):
            st.json(st.session_state.feedback_records)

# Settings page
def show_settings():
    st.markdown("<div class='main-header'>Settings</div>", unsafe_allow_html=True)
    
    st.markdown("### Model Configuration")
    model_type = st.selectbox("Select Model Type", ["gradient_boosting", "random_forest", "logistic_regression"], index=0)
    
    if st.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            st.session_state.model_trained = False
            st.session_state.lead_scorer = LeadScorer(model_type=model_type)
            X_train, X_test, y_train, y_test, _ = st.session_state.data_processor.process_data(st.session_state.df)
            st.session_state.lead_scorer.train(X_train, y_train, preprocessor=st.session_state.data_processor.pipeline)  # Pass the pipeline
            st.session_state.metrics = st.session_state.lead_scorer.evaluate(X_test, y_test)
            st.session_state.test_data = (X_test, y_test)
            st.session_state.model_trained = True
            st.success("Model retrained successfully!")
    
    st.markdown("### Data Management")
    if st.button("Regenerate Synthetic Data"):
        with st.spinner("Generating new data..."):
            st.session_state.df = generate_synthetic_lead_data(1500)
            st.session_state.df.to_csv("data/sample_leads.csv", index=False)
            st.session_state.model_trained = False  # Reset model training
            st.success("New synthetic data generated!")

# Main app logic
def main():
    initialize_app()
    setup_model()
    
    navigation = sidebar()
    
    if navigation == "Dashboard":
        show_dashboard()
    elif navigation == "Lead Analyzer":
        show_lead_analyzer()
    elif navigation == "Model Performance":
        show_model_performance()
    elif navigation == "Feedback Loop":
        show_feedback_loop()
    elif navigation == "Settings":
        show_settings()

if __name__ == "__main__":
    main()