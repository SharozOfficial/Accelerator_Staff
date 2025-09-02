import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from math import ceil
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

# Import modified functions from model_serving_utils
try:
    from model_serving_utils import (
        get_spark_session,
        run_databricks_healthcare_pipeline,
        HealthcareAIQueryProcessor,
        analyze_capacity_impact_optimized,
        identify_high_risk_patients_optimized,
        optimize_schedule_optimized,
        analyze_capacity_impact_pandas  
    )
    print("✅ Successfully imported functions from model_serving_utils")
except ImportError as e:
    st.error(f"❌ Error importing functions: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .staffing-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'spark_session' not in st.session_state:
        st.session_state.spark_session = None
    if 'ai_processor' not in st.session_state:
        st.session_state.ai_processor = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def validate_dataframe_columns(df, required_columns=None):
    """Validate that DataFrame has required columns"""
    if df is None or len(df) == 0:
        return False, "DataFrame is empty or None"
    
    if required_columns is None:
        # Core columns that should always be present
        core_required = ['patient_id', 'age', 'cancel_risk', 'los', 'cost_estimate', 
                        'priority_score', 'readmit_risk', 'wait_days']
    else:
        core_required = required_columns
    
    missing_columns = [col for col in core_required if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    return True, "All required columns present"

# def reconstruct_categorical_columns(df):
#     """Reconstruct original categorical columns from one-hot encoded columns"""
#     df_reconstructed = df.copy()
    
#     # Reconstruct gender
#     if 'gender_M' in df.columns:
#         df_reconstructed['gender'] = df['gender_M'].map({1: 'M', 0: 'F'})
    
#     # Reconstruct specialty
#     specialty_cols = [col for col in df.columns if col.startswith('specialty_')]
#     if specialty_cols:
#         def get_specialty(row):
#             for col in specialty_cols:
#                 if row[col] == 1:
#                     return col.replace('specialty_', '')
#             return 'Cardiology'  # Default fallback
        
#         df_reconstructed['specialty'] = df[specialty_cols].apply(get_specialty, axis=1)
    
#     # Reconstruct season
#     season_cols = [col for col in df.columns if col.startswith('season_')]
#     if season_cols:
#         def get_season(row):
#             for col in season_cols:
#                 if row[col] == 1:
#                     return col.replace('season_', '')
#             return 'Autumn'  # Default fallback (since it's not in dummy variables)
        
#         df_reconstructed['season'] = df[season_cols].apply(get_season, axis=1)
    
#     return df_reconstructed

def reconstruct_categorical_columns(df):
    """Reconstruct original categorical columns from one-hot encoded columns OR preserve existing ones"""
    df_reconstructed = df.copy()
    
    # Check if specialty column already exists with actual values
    if 'specialty' in df.columns and not df['specialty'].isna().all():
        # Specialty column already exists with real values - keep it as is
        print(f"✅ Using existing specialty column with {df['specialty'].nunique()} unique specialties")
    else:
        # Try to reconstruct from one-hot encoded columns (fallback)
        specialty_cols = [col for col in df.columns if col.startswith('specialty_')]
        if specialty_cols:
            def get_specialty(row):
                for col in specialty_cols:
                    if row[col] == 1:
                        return col.replace('specialty_', '')
                return 'General Surgery'  # Better default
            
            df_reconstructed['specialty'] = df[specialty_cols].apply(get_specialty, axis=1)
        else:
            # If no specialty data available at all, create a default
            df_reconstructed['specialty'] = 'General Surgery'
    
    # Check if gender column already exists with actual values
    if 'gender' in df.columns and not df['gender'].isna().all():
        # Gender column already exists - keep it
        print(f"✅ Using existing gender column")
    else:
        # Try to reconstruct from one-hot encoding
        if 'gender_M' in df.columns:
            df_reconstructed['gender'] = df['gender_M'].map({1: 'M', 0: 'F'})
        else:
            # Default gender distribution if not available
            df_reconstructed['gender'] = np.random.choice(['M', 'F'], len(df))
    
    # Check if season column already exists with actual values
    if 'season' in df.columns and not df['season'].isna().all():
        # Season column already exists - keep it
        print(f"✅ Using existing season column")
    else:
        # Try to reconstruct from one-hot encoding
        season_cols = [col for col in df.columns if col.startswith('season_')]
        if season_cols:
            def get_season(row):
                for col in season_cols:
                    if row[col] == 1:
                        return col.replace('season_', '')
                return 'Autumn'  # Default fallback
            
            df_reconstructed['season'] = df[season_cols].apply(get_season, axis=1)
        else:
            # Default season distribution if not available
            df_reconstructed['season'] = np.random.choice(['Winter', 'Spring', 'Summer', 'Autumn'], len(df))
    
    return df_reconstructed

def load_data_and_pipeline():
    """Load data and initialize pipeline - now uses pandas backend"""
    if not st.session_state.data_loaded:
        with st.spinner("Initializing Healthcare Analytics Pipeline..."):
            try:
                # Try to get Spark session (might not work in Streamlit)
                try:
                    st.session_state.spark_session = get_spark_session()
                except:
                    st.session_state.spark_session = None
                    st.info("Running in pandas mode")
                
                # Run pipeline (now uses pandas backend)
                st.session_state.pipeline_results = run_databricks_healthcare_pipeline()
                
                # Initialize AI Query Processor (works with or without Spark)
                st.session_state.ai_processor = HealthcareAIQueryProcessor(
                    spark_session=st.session_state.spark_session
                )
                
                st.session_state.data_loaded = True
                #st.success("✅ Pipeline initialized successfully!")
                
                # Validate the data
                df_test = get_pandas_dataframe()
                if df_test is not None:
                    is_valid, message = validate_dataframe_columns(df_test)
                    if not is_valid:
                        st.warning(f" Data validation warning: {message}")
                        st.info(f"Available columns: {list(df_test.columns)}")
                    else:
                        #st.success("Data structure validated successfully!")
                        # Check if we successfully reconstructed categorical columns
                        if all(col in df_test.columns for col in ['gender', 'specialty', 'season']):
                            pass
                            #st.info("Categorical columns reconstructed from one-hot encoding")
                
            except Exception as e:
                st.error(f"Error initializing pipeline: {str(e)}")
                st.exception(e)  # Show full traceback for debugging
                return False
    return True

def create_overview_metrics(df_pandas):
    """Create overview metrics cards with column validation"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Validate required columns exist
    required_cols = ['wait_days', 'readmit_risk', 'cost_estimate']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"Missing columns for metrics: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    with col1:
        total_patients = len(df_pandas)
        st.metric(
            label="📊 Total Patients",
            value=f"{total_patients:,}",
            #delta=f"+{total_patients//10} this month"
        )
    
    with col2:
        avg_wait = df_pandas['wait_days'].mean()
        st.metric(
            label="⏱️ Avg Wait Time",
            value=f"{avg_wait:.1f} days",
            #delta=f"-{avg_wait*0.1:.1f} from last month"
        )
    
    with col3:
        high_risk = len(df_pandas[df_pandas['readmit_risk'] > 0.7])
        st.metric(
            label="⚠️ High Risk Patients",
            value=f"{high_risk:,}",
            #delta=f"+{high_risk//20} this week"
        )
    
    with col4:
        total_cost = df_pandas['cost_estimate'].sum()
        st.metric(
            label="💰 Total Cost Estimate",
            value=f"£{total_cost/1000000:.1f}M",
            delta=f"+£{total_cost/10000000:.1f}M this quarter"
        )

def create_specialty_analysis(df_pandas):
    """Create specialty analysis visualizations with column validation"""
    st.subheader("🏥 Specialty Analysis")
    
    # Check for required columns
    required_cols = ['specialty', 'wait_days', 'cost_estimate', 'cancel_risk', 'readmit_risk', 'patient_id']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for specialty analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    # Specialty distribution
    col1, col2 = st.columns(2)
    
    with col1:
        specialty_counts = df_pandas['specialty'].value_counts()
        fig_pie = px.pie(
            values=specialty_counts.values,
            names=specialty_counts.index,
            title="Patient Distribution by Specialty"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Average wait time by specialty
        specialty_wait = df_pandas.groupby('specialty')['wait_days'].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            x=specialty_wait.values,
            y=specialty_wait.index,
            orientation='h',
            title="Average Wait Time by Specialty",
            labels={'x': 'Average Wait Days', 'y': 'Specialty'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Specialty performance table
    st.subheader("📈 Specialty Performance Summary")
    specialty_summary = df_pandas.groupby('specialty').agg({
        'patient_id': 'count',
        'wait_days': 'mean',
        'cost_estimate': 'mean',
        'cancel_risk': 'mean',
        'readmit_risk': 'mean'
    }).round(2)
    specialty_summary.columns = ['Patient Count', 'Avg Wait Days', 'Avg Cost', 'Avg Cancel Risk', 'Avg Readmit Risk']
    st.dataframe(specialty_summary, use_container_width=True)

def create_risk_analysis(df_pandas):
    """Create risk analysis visualizations with column validation"""
    st.subheader("⚠️ Risk Analysis")
    
    # Check for required columns
    required_cols = ['readmit_risk', 'specialty', 'cancel_risk']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for risk analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission risk distribution
        fig_hist = px.histogram(
            df_pandas,
            x='readmit_risk',
            nbins=20,
            title="Readmission Risk Distribution",
            labels={'x': 'Readmission Risk', 'y': 'Count'}
        )
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                          annotation_text="High Risk Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Cancellation risk by specialty
        cancel_risk = df_pandas.groupby('specialty')['cancel_risk'].mean().sort_values(ascending=False)
        fig_cancel = px.bar(
            x=cancel_risk.index,
            y=cancel_risk.values,
            title="Average Cancellation Risk by Specialty",
            labels={'x': 'Specialty', 'y': 'Average Cancellation Risk'}
        )
        fig_cancel.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cancel, use_container_width=True)

# def create_capacity_planning(df_pandas):
#     """Create improved capacity planning interface with staffing analysis"""
#     st.subheader("📊 Capacity Planning & Staffing Analysis")
    
#     # Calculate current staffing ratios from the data
#     if all(col in df_pandas.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
#         # Calculate average staffing ratios
#         total_planned = df_pandas['planned_count'].sum()
#         total_doctors = df_pandas['doctors_fte'].mean()  # Average per facility
#         total_nurses = df_pandas['nurses_fte'].mean()    # Average per facility
        
#         # Calculate patients per staff ratios
#         patients_per_doctor = total_planned / total_doctors if total_doctors > 0 else 50
#         patients_per_nurse = total_planned / total_nurses if total_nurses > 0 else 25
#     else:
#         # Use industry standard ratios if staffing data not available
#         patients_per_doctor = 50  # Conservative estimate
#         patients_per_nurse = 25   # Conservative estimate
#         total_doctors = 100       # Default for calculations
#         total_nurses = 200        # Default for calculations
    
#     total_patients = len(df_pandas)
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.markdown("### Capacity Settings")
        
#         # Current capacity settings
#         baseline_capacity = st.number_input(
#             "Current Weekly Capacity", 
#             min_value=50, 
#             max_value=500, 
#             value=100,
#             help="Number of patients that can be treated per week currently"
#         )
        
#         # Capacity uplift slider for real-time analysis
#         uplift_percentage = st.slider(
#             "Capacity Uplift (%)", 
#             min_value=0, 
#             max_value=100, 
#             value=20,
#             help="Percentage increase in weekly capacity"
#         )
        
#         # Calculate new capacity
#         new_capacity = int(baseline_capacity * (1 + uplift_percentage/100))
#         additional_capacity = new_capacity - baseline_capacity
        
#         # Display capacity comparison
#         st.info(f"**Current Capacity:** {baseline_capacity} patients/week")
#         st.success(f"**New Capacity:** {new_capacity} patients/week")
#         st.metric("Additional Weekly Capacity", f"+{additional_capacity}", f"{uplift_percentage}% increase")
        
#         # Calculate basic metrics
#         current_weeks_needed = np.ceil(total_patients / baseline_capacity)
#         new_weeks_needed = np.ceil(total_patients / new_capacity)
#         weeks_saved = current_weeks_needed - new_weeks_needed
        
#         st.markdown("### Impact Summary")
#         st.metric("Weeks to Clear Backlog", f"{int(new_weeks_needed)}", f"-{int(weeks_saved)} weeks")
        
#         # Calculate staffing requirements
#         st.markdown("### 👨‍⚕️ Staffing Requirements")
        
#         # Calculate additional staff needed for capacity increase
#         additional_doctors_needed = additional_capacity / patients_per_doctor
#         additional_nurses_needed = additional_capacity / patients_per_nurse
        
#         # Display current staffing info
#         col_staff1, col_staff2 = st.columns(2)
        
#         with col_staff1:
#             st.markdown("**Current Staffing:**")
#             st.write(f"👨‍⚕️ Doctors: {total_doctors:.0f}")
#             st.write(f"👩‍⚕️ Nurses: {total_nurses:.0f}")
#             st.write(f"📊 Patients/Doctor: {patients_per_doctor:.0f}")
#             st.write(f"📊 Patients/Nurse: {patients_per_nurse:.0f}")
        
#         with col_staff2:
#             st.markdown("**Additional Staff Needed:**")
#             st.write(f"👨‍⚕️ +{additional_doctors_needed:.1f} Doctors")
#             st.write(f"👩‍⚕️ +{additional_nurses_needed:.1f} Nurses")
    
#     with col2:
#         st.markdown("### Capacity & Staffing Scenarios")
        
#         # Create staffing scenarios
#         scenarios = []
        
#         # Scenario 1: Doctors only
#         doctors_only = np.ceil(additional_capacity / patients_per_doctor)
#         cost_doctors = doctors_only * 80000  # Average doctor salary
        
#         # Scenario 2: Nurses only  
#         nurses_only = np.ceil(additional_capacity / patients_per_nurse)
#         cost_nurses = nurses_only * 35000  # Average nurse salary
        
#         # Scenario 3: Mixed approach (60% doctors, 40% nurses)
#         mixed_doctors = np.ceil((additional_capacity * 0.6) / patients_per_doctor)
#         mixed_nurses = np.ceil((additional_capacity * 0.4) / patients_per_nurse)
#         cost_mixed = (mixed_doctors * 80000) + (mixed_nurses * 35000)
        
#         scenarios_data = {
#             'Scenario': ['Doctors Only', 'Nurses Only', 'Mixed Approach', 'Current Baseline'],
#             'Additional_Doctors': [doctors_only, 0, mixed_doctors, 0],
#             'Additional_Nurses': [0, nurses_only, mixed_nurses, 0],
#             'Annual_Cost': [cost_doctors, cost_nurses, cost_mixed, 0],
#             'Weekly_Capacity': [new_capacity, new_capacity, new_capacity, baseline_capacity],
#             'Feasibility': ['High cost, high expertise', 'Cost effective, good coverage', 'Balanced approach', 'Current state']
#         }
        
#         scenarios_df = pd.DataFrame(scenarios_data)
        
#         # Display scenarios as cards
#         st.markdown("#### 🎯 Staffing Scenarios to Achieve Target Capacity")
        
#         for idx, row in scenarios_df.iterrows():
#             if row['Scenario'] != 'Current Baseline':
#                 with st.container():
#                     scenario_col1, scenario_col2, scenario_col3 = st.columns([2, 2, 2])
                    
#                     with scenario_col1:
#                         st.markdown(f"**{row['Scenario']}**")
#                         if row['Additional_Doctors'] > 0:
#                             st.write(f"👨‍⚕️ +{row['Additional_Doctors']:.0f} Doctors")
#                         if row['Additional_Nurses'] > 0:
#                             st.write(f"👩‍⚕️ +{row['Additional_Nurses']:.0f} Nurses")
                    
#                     with scenario_col2:
#                         st.metric("Annual Cost", f"£{row['Annual_Cost']:,.0f}")
#                         st.write(f"📈 Weekly Capacity: {row['Weekly_Capacity']:.0f}")
                    
#                     with scenario_col3:
#                         st.write(f"💡 {row['Feasibility']}")
                    
#                     st.markdown("---")
        
#         # Timeline comparison with staffing context
#         st.markdown("### 📅 Implementation Timeline")
        
#         max_weeks = min(int(current_weeks_needed) + 5, 52)
#         weeks = list(range(1, max_weeks + 1))
        
#         current_backlog = []
#         new_backlog = []
        
#         for week in weeks:
#             current_treated = week * baseline_capacity
#             new_treated = week * new_capacity
            
#             current_remaining = max(0, total_patients - current_treated)
#             new_remaining = max(0, total_patients - new_treated)
            
#             current_backlog.append(current_remaining)
#             new_backlog.append(new_remaining)
        
#         fig_timeline = go.Figure()
        
#         # Current capacity line
#         fig_timeline.add_trace(go.Scatter(
#             x=weeks,
#             y=current_backlog,
#             mode='lines+markers',
#             name=f'Current Capacity ({baseline_capacity}/week)',
#             line=dict(color='#ff6b6b', width=3),
#         ))
        
#         # New capacity line
#         fig_timeline.add_trace(go.Scatter(
#             x=weeks,
#             y=new_backlog,
#             mode='lines+markers',
#             name=f'Enhanced Capacity ({new_capacity}/week)',
#             line=dict(color='#4ecdc4', width=3),
#         ))
        
#         fig_timeline.update_layout(
#             title=f"Backlog Reduction Timeline (Total: {total_patients:,} patients)",
#             xaxis_title="Week",
#             yaxis_title="Remaining Patients in Backlog",
#             hovermode='x unified',
#             height=400,
#         )
        
#         st.plotly_chart(fig_timeline, use_container_width=True)
        
#         # Financial analysis
#         st.markdown("### 💰 Return on Investment Analysis")
        
#         if 'cost_estimate' in df_pandas.columns:
#             avg_cost_per_patient = df_pandas['cost_estimate'].mean()
            
#             # Revenue from additional capacity
#             additional_monthly_patients = additional_capacity * 4
#             additional_monthly_revenue = additional_monthly_patients * avg_cost_per_patient
            
#             roi_metrics_col1, roi_metrics_col2 = st.columns(2)
            
#             with roi_metrics_col1:
#                 st.metric("Additional Monthly Revenue", f"£{additional_monthly_revenue:,.0f}")
#                 st.metric("Revenue per Additional Patient", f"£{avg_cost_per_patient:,.0f}")
                
#             with roi_metrics_col2:
#                 # Calculate payback period for mixed scenario
#                 if cost_mixed > 0:
#                     monthly_profit = additional_monthly_revenue - (cost_mixed/12)  # Monthly cost
#                     payback_months = cost_mixed / monthly_profit if monthly_profit > 0 else float('inf')
#                     st.metric("Payback Period (Mixed Scenario)", f"{payback_months:.1f} months")
#                     st.metric("Monthly Net Benefit", f"£{monthly_profit:,.0f}")

def create_capacity_planning(df_pandas):
    """Create improved capacity planning interface with FIXED staffing analysis"""
    st.subheader("📊 Capacity Planning & Staffing Analysis")
    
    total_patients = len(df_pandas)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Capacity Settings")
        
        # Current capacity settings
        baseline_capacity = st.number_input(
            "Current Weekly Capacity", 
            min_value=50, 
            max_value=500, 
            value=100,
            help="Number of patients that can be treated per week currently"
        )
        
        # Capacity uplift slider for real-time analysis
        uplift_percentage = st.slider(
            "Capacity Uplift (%)", 
            min_value=0, 
            max_value=100, 
            value=20,
            help="Percentage increase in weekly capacity"
        )
        
        # Calculate new capacity
        new_capacity = int(baseline_capacity * (1 + uplift_percentage/100))
        additional_capacity = new_capacity - baseline_capacity
        
        # Display capacity comparison
        st.info(f"**Current Capacity:** {baseline_capacity} patients/week")
        st.success(f"**New Capacity:** {new_capacity} patients/week")
        st.metric("Additional Weekly Capacity", f"+{additional_capacity}", f"{uplift_percentage}% increase")
        
        # Calculate basic metrics
        current_weeks_needed = np.ceil(total_patients / baseline_capacity)
        new_weeks_needed = np.ceil(total_patients / new_capacity)
        weeks_saved = current_weeks_needed - new_weeks_needed
        
        st.markdown("### Impact Summary")
        st.metric("Weeks to Clear Backlog", f"{int(new_weeks_needed)}", f"-{int(weeks_saved)} weeks")
        
        # FIXED STAFFING CALCULATIONS - using realistic industry ratios
        st.markdown("### 👨‍⚕️ Staffing Requirements")
        
        # Use realistic industry standard ratios
        PATIENTS_PER_DOCTOR = 25   # Industry standard: 1 doctor per 25 patients/week
        PATIENTS_PER_NURSE = 12    # Industry standard: 1 nurse per 12 patients/week
        
        # Calculate current staffing needs
        current_doctors_needed = baseline_capacity / PATIENTS_PER_DOCTOR
        current_nurses_needed = baseline_capacity / PATIENTS_PER_NURSE
        
        # Calculate additional staff needed for the uplift
        additional_doctors_needed = ceil(additional_capacity / PATIENTS_PER_DOCTOR)
        additional_nurses_needed = ceil(additional_capacity / PATIENTS_PER_NURSE)

        # Calculate current staffing ratios from the data
        if all(col in df_pandas.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
            # Calculate average staffing ratios
            total_planned = df_pandas['planned_count'].sum()
            total_doctors = df_pandas['doctors_fte'].mean()  # Average per facility
            total_nurses = df_pandas['nurses_fte'].mean()    # Average per facility
        
            # Calculate patients per staff ratios
            total_patient = df_pandas.shape[0]
            patients_per_doctor_curr = total_patient / total_doctors if total_doctors > 0 else 25
            patients_per_nurse_curr = total_patient / total_nurses if total_nurses > 0 else 12

            curr_doctors_needed = ceil(baseline_capacity / patients_per_doctor_curr)
            curr_nurses_needed = ceil(baseline_capacity / patients_per_nurse_curr) 
        else:
            # Use industry standard ratios if staffing data not available
            patients_per_doctor_curr = 25  # Conservative estimate
            patients_per_nurse = 12   # Conservative estimate
            # total_doctors = 100       # Default for calculations
            # total_nurses = 200        # Default for calculations
        
        # Total staff needed for new capacity
        # total_doctors_needed = new_capacity / PATIENTS_PER_DOCTOR
        # total_nurses_needed = new_capacity / PATIENTS_PER_NURSE

        # Calculate total staff needed for new capacity
        total_doctors_needed = curr_doctors_needed + additional_doctors_needed
        total_nurses_needed = curr_nurses_needed + additional_nurses_needed
        
        # Display current and required staffing
        col_staff1, col_staff2 = st.columns(2)
        
        with col_staff1:
            st.markdown("**Current Staffing:**")
            st.write(f"👨‍⚕️ Total Doctors: {round(total_doctors)}")
            st.write(f"👩‍⚕️ Total Nurses: {round(total_nurses)}")

            st.markdown("**Current Staffing needed as per baseline weekly capacity:**")
            st.write(f"👨‍⚕️ {curr_doctors_needed} Doctors")
            st.write(f"👨‍⚕️ {curr_nurses_needed} Nurses")
        
        with col_staff2:
            st.markdown("**Additional Staff Needed after capacity uplift:**")
            st.write(f"👨‍⚕️ +{additional_doctors_needed} Doctors")
            st.write(f"👩‍⚕️ +{additional_nurses_needed} Nurses")
            
            # Show total required
            st.markdown("**Total Required as per new weekly capcity:**")
            st.write(f"👨‍⚕️ {total_doctors_needed} Doctors")
            st.write(f"👩‍⚕️ {total_nurses_needed} Nurses")
    
    with col2:
        st.markdown("### Capacity & Staffing Scenarios")
        
        # FIXED: Dynamic staffing scenarios based on actual uplift
        # Annual salaries
        DOCTOR_SALARY = 80000
        NURSE_SALARY = 35000
        
        # Scenario 1: Doctors only
        doctors_only_additional = ceil(additional_capacity / PATIENTS_PER_DOCTOR)
        cost_doctors_only = doctors_only_additional * DOCTOR_SALARY
        
        # Scenario 2: Nurses only  
        nurses_only_additional = ceil(additional_capacity / PATIENTS_PER_NURSE)
        cost_nurses_only = nurses_only_additional * NURSE_SALARY
        
        # Scenario 3: Mixed approach (optimal ratio: 1 doctor per 2.5 nurses)
        # For additional capacity, use balanced approach
        mixed_doctors_additional = ceil(additional_capacity / (PATIENTS_PER_DOCTOR * 1.2))  # Slightly fewer doctors
        mixed_nurses_additional = ceil(additional_capacity / (PATIENTS_PER_NURSE * 0.8))   # More nurses
        cost_mixed = (mixed_doctors_additional * DOCTOR_SALARY) + (mixed_nurses_additional * NURSE_SALARY)
        
        # Create dynamic scenarios table
        scenarios_data = {
            'Scenario': ['Doctors Only', 'Nurses Only', 'Mixed Approach (Recommended)'],
            'Additional_Doctors': [doctors_only_additional, 0, mixed_doctors_additional],
            'Additional_Nurses': [0, nurses_only_additional, mixed_nurses_additional],
            'Annual_Cost': [cost_doctors_only, cost_nurses_only, cost_mixed],
            'Monthly_Cost': [cost_doctors_only/12, cost_nurses_only/12, cost_mixed/12],
            'Capacity_Added': [additional_capacity, additional_capacity, additional_capacity]
        }
        
        scenarios_df = pd.DataFrame(scenarios_data)
        
        # Display scenarios
        st.markdown(f"#### 🎯 Staffing Options for +{additional_capacity} Weekly Capacity")
        
        for idx, row in scenarios_df.iterrows():
            with st.container():
                scenario_col1, scenario_col2, scenario_col3 = st.columns([2, 2, 2])
                
                with scenario_col1:
                    st.markdown(f"**{row['Scenario']}**")
                    if row['Additional_Doctors'] > 0:
                        st.write(f"👨‍⚕️ +{row['Additional_Doctors']:.1f} Doctors")
                    if row['Additional_Nurses'] > 0:
                        st.write(f"👩‍⚕️ +{row['Additional_Nurses']:.1f} Nurses")
                
                with scenario_col2:
                    st.metric("Annual Cost", f"£{row['Annual_Cost']:,.0f}")
                    st.metric("Monthly Cost", f"£{row['Monthly_Cost']:,.0f}")
                
                with scenario_col3:
                    if idx == 0:
                        st.write("💡 High expertise, high cost")
                    elif idx == 1:
                        st.write("💡 Cost effective, good coverage")
                    else:
                        st.write("💡 Balanced approach")
                
                st.markdown("---")
        
        # Timeline comparison
        st.markdown("### 📅 Implementation Timeline")
        
        max_weeks = min(int(current_weeks_needed) + 5, 52)
        weeks = list(range(1, max_weeks + 1))
        
        current_backlog = []
        new_backlog = []
        
        for week in weeks:
            current_treated = week * baseline_capacity
            new_treated = week * new_capacity
            
            current_remaining = max(0, total_patients - current_treated)
            new_remaining = max(0, total_patients - new_treated)
            
            current_backlog.append(current_remaining)
            new_backlog.append(new_remaining)
        
        fig_timeline = go.Figure()
        
        # Current capacity line
        fig_timeline.add_trace(go.Scatter(
            x=weeks,
            y=current_backlog,
            mode='lines+markers',
            name=f'Current Capacity ({baseline_capacity}/week)',
            line=dict(color='#ff6b6b', width=3),
        ))
        
        # New capacity line
        fig_timeline.add_trace(go.Scatter(
            x=weeks,
            y=new_backlog,
            mode='lines+markers',
            name=f'Enhanced Capacity ({new_capacity}/week)',
            line=dict(color='#4ecdc4', width=3),
        ))
        
        fig_timeline.update_layout(
            title=f"Backlog Reduction Timeline (Total: {total_patients:,} patients)",
            xaxis_title="Week",
            yaxis_title="Remaining Patients in Backlog",
            hovermode='x unified',
            height=400,
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # FIXED ROI ANALYSIS with clear calculation explanation
        st.markdown("### 💰 Return on Investment Analysis")
        
        if 'cost_estimate' in df_pandas.columns:
            avg_cost_per_patient = df_pandas['cost_estimate'].mean()
            
            # Calculate additional revenue from increased capacity
            additional_weekly_patients = additional_capacity
            additional_monthly_patients = additional_weekly_patients * 4  # 4 weeks per month
            additional_annual_patients = additional_weekly_patients * 52  # 52 weeks per year
            
            # Revenue calculations
            additional_monthly_revenue = additional_monthly_patients * avg_cost_per_patient
            additional_annual_revenue = additional_annual_patients * avg_cost_per_patient
            
            # Use mixed scenario cost for ROI calculation
            annual_staff_cost = cost_mixed
            monthly_staff_cost = annual_staff_cost / 12
            
            # Net benefit calculations
            monthly_net_benefit = additional_monthly_revenue - monthly_staff_cost
            annual_net_benefit = additional_annual_revenue - annual_staff_cost
            
            # Payback period
            payback_months = annual_staff_cost / monthly_net_benefit if monthly_net_benefit > 0 else float('inf')
            
            # Display ROI metrics
            roi_col1, roi_col2 = st.columns(2)
            
            with roi_col1:
                st.metric("Additional Monthly Revenue", f"£{additional_monthly_revenue:,.0f}")
                st.metric("Additional Annual Revenue", f"£{additional_annual_revenue:,.0f}")
                st.metric("Revenue per Additional Patient", f"£{avg_cost_per_patient:,.0f}")
                
            with roi_col2:
                st.metric("Annual Staff Investment", f"£{annual_staff_cost:,.0f}")
                st.metric("Monthly Net Benefit", f"£{monthly_net_benefit:,.0f}")
                st.metric("Payback Period", f"{payback_months:.1f} months")
            
            # ROI Calculation Explanation
            with st.expander("📝 How ROI is Calculated - Example"):
                st.markdown(f"""
                **Example with {uplift_percentage}% capacity uplift:**
                
                **Step 1: Additional Capacity**
                - Current weekly capacity: {baseline_capacity} patients
                - Uplift: {uplift_percentage}% = +{additional_capacity} patients/week
                
                **Step 2: Revenue Impact**
                - Additional patients per month: {additional_capacity} × 4 = {additional_monthly_patients}
                - Average revenue per patient: £{avg_cost_per_patient:,.0f}
                - Additional monthly revenue: {additional_monthly_patients} × £{avg_cost_per_patient:,.0f} = £{additional_monthly_revenue:,.0f}
                
                **Step 3: Staffing Cost (Mixed Scenario)**
                - Additional doctors needed: {mixed_doctors_additional:.1f} × £{DOCTOR_SALARY:,.0f} = £{mixed_doctors_additional * DOCTOR_SALARY:,.0f}/year
                - Additional nurses needed: {mixed_nurses_additional:.1f} × £{NURSE_SALARY:,.0f} = £{mixed_nurses_additional * NURSE_SALARY:,.0f}/year
                - Total annual staffing cost: £{annual_staff_cost:,.0f}
                - Monthly staffing cost: £{monthly_staff_cost:,.0f}
                
                **Step 4: Net Benefit**
                - Monthly net benefit: £{additional_monthly_revenue:,.0f} - £{monthly_staff_cost:,.0f} = £{monthly_net_benefit:,.0f}
                - Payback period: £{annual_staff_cost:,.0f} ÷ £{monthly_net_benefit:,.0f} = {payback_months:.1f} months
                """)
        
        # Show the calculation method
        st.markdown("### 🔍 Staffing Calculation Method")
        st.info(f"""
        **Industry Standard Ratios Used:**
        - 1 Doctor per {PATIENTS_PER_DOCTOR} patients/week
        - 1 Nurse per {PATIENTS_PER_NURSE} patients/week
        """)

def search_medical_term(term):
    """Search for medical term using Wikipedia API"""
    try:
        # First try direct Wikipedia page lookup
        direct_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.replace(' ', '_')}"
        direct_response = requests.get(direct_url, timeout=10)
        
        if direct_response.status_code == 200:
            direct_data = direct_response.json()
            extract = direct_data.get('extract', '')
            
            # Check if it's a valid medical/relevant page
            if extract and len(extract) > 50:
                return {
                    'title': direct_data.get('title', term),
                    'summary': extract,
                    'url': direct_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }
        
        # If direct lookup fails, try search
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': f'"{term}"',  # Use exact term in quotes
            'srlimit': 3
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            search_results = search_data.get('query', {}).get('search', [])
            
            # Try to find the best match
            for result in search_results:
                page_title = result['title']
                
                # Skip disambiguation pages and unrelated results
                if '(disambiguation)' in page_title.lower():
                    continue
                    
                # Check if title contains our search term
                if term.lower() in page_title.lower():
                    # Get page summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
                    summary_response = requests.get(summary_url, timeout=10)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        extract = summary_data.get('extract', '')
                        
                        if extract and len(extract) > 50:
                            return {
                                'title': summary_data.get('title', term),
                                'summary': extract,
                                'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                            }
        
        return None
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def create_ai_query_interface():
    """Create AI-powered query interface - now works with pandas backend"""
    st.subheader("🤖 AI-Powered Query Interface")

    ai_tab1, ai_tab2 = st.tabs(["🤖 Data Query", "🔍 Medical Terms"])
    with ai_tab1:
        st.markdown("Ask questions about your healthcare data in natural language!")
        
        # Sample queries
        with st.expander("🔍 Sample Queries"):
            sample_queries = [
                "What is the average wait time for Cardiology and Orthopedics?",
                "Show me high risk patients with readmission risk above 70%",
                "Reduce cancellations by 15%",
                "What is the cost impact for Orthopedics?",
                "What is the cost impact by specialty?",
                "Show seasonal analysis for Winter",
                "What is the average cost for young adults?",
                "Impact of Baseline capacity 200 and Capacity Increase by 20%"
            ]
            for i, query in enumerate(sample_queries, 1):
                st.write(f"{i}. {query}")
        
        # Query input
        user_query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the average wait time for Orthopedics?",
            help="Ask questions about patient data, wait times, risks, costs, etc."
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            query_button = st.button("🔍 Process Query", type="primary")
        
        if query_button and user_query:
            with st.spinner("🧠 Processing your query..."):
                try:
                    if st.session_state.ai_processor:
                        df_reconstructed = get_pandas_dataframe()
                        if df_reconstructed is not None:
                            class DataFrameWrapper:
                                def __init__(self, df):
                                    self.df = df
                                    self.raw_pandas_df = df
                            
                            df_data = DataFrameWrapper(df_reconstructed)
                            result = st.session_state.ai_processor.process_natural_language_query(user_query, df_data)
                            
                            st.info(f"🏷️ **Detected Query Type:** {result.query_type}")
                            
                            is_function_query = (hasattr(result, 'query_type') and 
                                               result.query_type.lower() in ['cancellation_reduction', 'function', 'capacity_analysis'])
                            
                            if is_function_query:
                                response = st.session_state.ai_processor.format_response(result)
                                st.markdown("### 📊 Query Results")
                                st.markdown(response)
                            else:
                                if hasattr(result, 'result_data') and isinstance(result.result_data, pd.DataFrame):
                                    if len(result.result_data) > 0:
                                        st.markdown("### 📋 Resultant Data")
                                        st.dataframe(result.result_data, use_container_width=True)
                        else:
                            st.error("❌ Could not get data for query processing")
                    else:
                        st.error("❌ AI Query Processor not initialized")
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.exception(e)
    
    
    with ai_tab2:
        st.markdown("### Search Medical Terms")
        
        # Simple search box and button
        search_term = st.text_input("Enter medical term:", placeholder="e.g., Hypertension, Diabetes, Cardiology")
        
        if st.button("🔍 Search Medical Term", type="primary"):
            if search_term:
                with st.spinner(f"Searching for '{search_term}'..."):
                    result = search_medical_term(search_term)
                    
                    if result:
                        st.success(f"✅ Found information for: **{result['title']}**")
                        st.write(result['summary'])
                        
                        if result['url']:
                            st.markdown(f"🔗 [Read more]({result['url']})")
                    else:
                        st.warning(f"❌ No results found for '{search_term}'. Try checking spelling or using different terms.")
            else:
                st.warning("Please enter a medical term to search.")

def create_seasonal_analysis(df_pandas):
    """Create seasonal analysis with column validation"""
    st.subheader("🌤️ Seasonal Analysis")
    
    # Check for required columns
    required_cols = ['season', 'cancel_risk', 'readmit_risk']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for seasonal analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal distribution
        seasonal_counts = df_pandas['season'].value_counts()
        fig_seasonal = px.bar(
            x=seasonal_counts.index,
            y=seasonal_counts.values,
            title="Patient Count by Season",
            labels={'x': 'Season', 'y': 'Patient Count'}
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        # Seasonal risk analysis
        seasonal_risk = df_pandas.groupby('season').agg({
            'cancel_risk': 'mean',
            'readmit_risk': 'mean'
        }).round(3)
        
        fig_risk_seasonal = go.Figure()
        fig_risk_seasonal.add_trace(go.Bar(
            name='Cancel Risk',
            x=seasonal_risk.index,
            y=seasonal_risk['cancel_risk']
        ))
        fig_risk_seasonal.add_trace(go.Bar(
            name='Readmit Risk',
            x=seasonal_risk.index,
            y=seasonal_risk['readmit_risk']
        ))
        
        fig_risk_seasonal.update_layout(
            title='Average Risk by Season',
            xaxis_title='Season',
            yaxis_title='Risk Score',
            barmode='group'
        )
        st.plotly_chart(fig_risk_seasonal, use_container_width=True)

def get_pandas_dataframe():
    """Get pandas DataFrame from pipeline results with better error handling"""
    if st.session_state.pipeline_results:
        try:
            # Try different ways to get the pandas DataFrame
            df = None
            if 'raw_pandas_df' in st.session_state.pipeline_results:
                df = st.session_state.pipeline_results['raw_pandas_df']
            elif hasattr(st.session_state.pipeline_results.get('df_final'), 'df'):
                df = st.session_state.pipeline_results['df_final'].df
            elif hasattr(st.session_state.pipeline_results.get('df_final'), 'toPandas'):
                df = st.session_state.pipeline_results['df_final'].toPandas()
            else:
                # If all else fails, return the first 10k records for display
                df_final = st.session_state.pipeline_results.get('df_final')
                if df_final and hasattr(df_final, 'limit'):
                    df = df_final.limit(10000).toPandas()
                else:
                    st.error("❌ Unable to extract DataFrame from pipeline results")
                    return None
            
            # If we got a DataFrame, reconstruct categorical columns from one-hot encoding
            if df is not None:
                df_reconstructed = reconstruct_categorical_columns(df)
                return df_reconstructed
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error extracting DataFrame: {str(e)}")
            return None
    return None

def create_debug_info():
    """Create debug information panel"""
    st.subheader("🔍 Debug Information")
    
    if st.session_state.pipeline_results:
        st.write("**Pipeline Results Keys:**", list(st.session_state.pipeline_results.keys()))
        
        for key, value in st.session_state.pipeline_results.items():
            st.write(f"**{key}:** {type(value)}")
            
            # If it's a DataFrame wrapper, try to get more info
            if hasattr(value, 'df'):
                try:
                    st.write(f"  - DataFrame shape: {value.df.shape}")
                    st.write(f"  - DataFrame columns: {list(value.df.columns)}")
                except:
                    st.write(f"  - Could not access DataFrame info")
            elif hasattr(value, 'toPandas'):
                try:
                    sample_df = value.limit(5).toPandas()
                    st.write(f"  - DataFrame columns: {list(sample_df.columns)}")
                    st.write(f"  - Sample data:")
                    st.dataframe(sample_df.head())
                except:
                    st.write(f"  - Could not convert to pandas")
    else:
        st.write("No pipeline results available")
    
    # Test DataFrame extraction
    df_test = get_pandas_dataframe()
    if df_test is not None:
        st.write("**Successfully extracted DataFrame:**")
        st.write(f"Shape: {df_test.shape}")
        st.write(f"Columns: {list(df_test.columns)}")
        st.dataframe(df_test.head())
    else:
        st.write("**Failed to extract DataFrame**")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🕹️ Dashboard Controls")
    
    # Load data
    if st.sidebar.button("🔄 Initialize/Refresh Data", type="primary"):
        st.session_state.data_loaded = False
        st.session_state.pipeline_results = None
        load_data_and_pipeline()
    
    # Load data if not already loaded
    if not load_data_and_pipeline():
        st.stop()
    
    # Navigation
    page = st.sidebar.selectbox(
        "📊 Select Analysis",
        [
            "📈 Overview & Metrics",
            "📊 Capacity Planning",
            "🏥 Specialty Analysis", 
            "⚠️ Risk Analysis",
            "🌤️ Seasonal Analysis",
            "🤖 AI Query Interface"
        ]
    )
    
    # Get DataFrame for analysis
    try:
        df_pandas = get_pandas_dataframe()
        
        if df_pandas is not None and len(df_pandas) > 0:
            # Validate DataFrame structure
            is_valid, validation_message = validate_dataframe_columns(df_pandas)
            if not is_valid:
                st.warning(f"⚠️ Data validation warning: {validation_message}")
                st.info(f"Available columns: {list(df_pandas.columns)}")
                # Continue anyway since we now reconstruct the missing columns
            else:
                pass
            
            # Sample data for performance if too large
            if len(df_pandas) > 100000:
                df_display = df_pandas.sample(n=min(100000, len(df_pandas)), random_state=42)
            else:
                df_display = df_pandas
            
            # Page routing
            if page == "📈 Overview & Metrics":
                create_overview_metrics(df_display)
                st.markdown("---")
                
                # Quick insights with column validation
                st.subheader("🔍 Quick Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'specialty' in df_display.columns:
                        most_common_specialty = df_display['specialty'].mode().iloc[0]
                        st.info(f"📊 **Most Common Specialty:** {most_common_specialty}")
                        
                        highest_risk_specialty = df_display.groupby('specialty')['readmit_risk'].mean().idxmax()
                        st.warning(f"⚠️ **Highest Risk Specialty:** {highest_risk_specialty}")
                    else:
                        st.warning("⚠️ Specialty information not available")
                
                with col2:
                    if 'age' in df_display.columns:
                        avg_age = df_display['age'].mean()
                        st.info(f"👥 **Average Patient Age:** {avg_age:.1f} years")
                    else:
                        st.warning("⚠️ Age information not available")
                    
                    if 'season' in df_display.columns:
                        winter_patients = len(df_display[df_display['season'] == 'Winter'])
                        st.info(f"❄️ **Winter Patients:** {winter_patients:,}")
                    else:
                        st.warning("⚠️ Seasonal information not available")
            
            elif page == "🏥 Specialty Analysis":
                create_specialty_analysis(df_display)
            
            elif page == "⚠️ Risk Analysis":
                create_risk_analysis(df_display)
            
            elif page == "📊 Capacity Planning":
                create_capacity_planning(df_pandas)  # Use full dataset for analysis
            
            elif page == "🌤️ Seasonal Analysis":
                create_seasonal_analysis(df_display)
            
            elif page == "🤖 AI Query Interface":
                create_ai_query_interface()
        
        else:
            st.warning("⚠️ No data available. Please initialize the pipeline first.")
    
    except Exception as e:
        st.error(f"❌ Error loading data for display: {str(e)}")
        st.exception(e)
        
        # Show debug info automatically on error
        st.markdown("---")
        create_debug_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            🏩 Healthcare Analytics Dashboard | Built with Streamlit & Pandas | 
            Powered by Advanced ML & AI | Compatible with Databricks
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()