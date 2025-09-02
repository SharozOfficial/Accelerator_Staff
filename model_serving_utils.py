from mlflow.deployments import get_deploy_client
from faker import Faker
import random
import pandas as pd
import numpy as np
import time
import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import math

# For Databricks environment
try:
    from databricks import sql as databricks_sql
    from databricks.connect import DatabricksSession
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# Fallback imports for local development
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    import pyspark.sql.functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

# Simplified ML imports - using basic models instead of complex XGBoost
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import openai
from dataclasses import dataclass

# Global flag to determine execution mode
USE_DATABRICKS_SQL = True

def get_spark_session():
    """Get Spark session - compatible with both environments"""
    if DATABRICKS_AVAILABLE and USE_DATABRICKS_SQL:
        try:
            # Use Databricks Connect for Streamlit apps
            spark = DatabricksSession.builder.getOrCreate()
            return spark
        except Exception as e:
            print(f"Warning: Could not create Databricks session: {e}")
            # Fall back to reading from Delta tables via SQL
            return None
    
    if PYSPARK_AVAILABLE:
        try:
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder \
                    .appName("DatabricksHealthcareAnalytics") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                    .getOrCreate()
            return spark
        except Exception as e:
            print(f"Warning: Could not configure Spark: {e}")
            return None
    
    return None

def get_databricks_connection():
    """Get connection to Databricks SQL warehouse"""
    try:
        import os
        # These should be set as environment variables or in secrets
        connection = databricks_sql.connect(
            server_hostname=os.getenv('DATABRICKS_SERVER_HOSTNAME'),
            http_path=os.getenv('DATABRICKS_HTTP_PATH'),
            access_token=os.getenv('DATABRICKS_TOKEN')
        )
        return connection
    except Exception as e:
        print(f"Could not connect to Databricks SQL: {e}")
        return None

def execute_sql_query(query: str):
    """Execute SQL query using Databricks SQL connector"""
    if USE_DATABRICKS_SQL:
        connection = get_databricks_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=columns)
            except Exception as e:
                print(f"Error executing SQL query: {e}")
                return pd.DataFrame()
            finally:
                connection.close()
    
    # Fallback to Spark if available
    spark = get_spark_session()
    if spark:
        try:
            return spark.sql(query).toPandas()
        except Exception as e:
            print(f"Error executing Spark SQL: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

def load_healthcare_reference_data():
    """Load and process the healthcare data from CSV - FIXED to use actual CSV size"""
    try:
        # Try to load the CSV file
        df = pd.read_csv('elective_recovery.csv')
        actual_records = len(df)
        print(f"✅ Loaded CSV with {actual_records:,} records and columns: {list(df.columns)}")
        
        # Check if this is the new format with direct age and patient_id columns
        if 'age' in df.columns and 'patient_id' in df.columns:
            print("✅ Detected new dataset format with direct age and patient_id columns")
            
            # Clean specialty names for consistency
            specialty_mapping = {
                'Trauma & Orthopaedics': 'Orthopedics',
                'Hepatobiliary & Pancreatic Surgery': 'Hepatobiliary Surgery'
            }
            df['specialty_clean'] = df['specialty'].map(specialty_mapping).fillna(df['specialty'])
            
            # For compatibility with existing code, set age_min and age_max
            df['age_min'] = df['age']
            df['age_max'] = df['age']
            
            # If planned_count doesn't exist, create a default value
            if 'planned_count' not in df.columns:
                df['planned_count'] = 100  # Default weight for all records
            
            print(f"✅ Dataset ready with {actual_records:,} records")
            return df
            
        else:
            print("✅ Detected old dataset format with age_band - processing age bands")
            # Old format processing (keeping existing logic)
            specialty_mapping = {
                'Trauma & Orthopaedics': 'Orthopedics',
                'Hepatobiliary & Pancreatic Surgery': 'Hepatobiliary Surgery'
            }
            
            df['specialty_clean'] = df['specialty'].map(specialty_mapping).fillna(df['specialty'])
            
            # Parse age bands for old format
            df['age_min'] = df['age_band'].str.replace('+', '').str.split('-').str[0].astype(int)
            df['age_max'] = df['age_band'].str.replace('+', '').str.split('-').str[-1].astype(int)
            df.loc[df['age_band'].str.contains(r'\+'), 'age_max'] = 95
            
            return df
            
    except FileNotFoundError:
        print("⚠️ Warning: elective_recovery.csv not found. Using fallback data generation.")
        return create_fallback_reference_data()
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}. Using fallback data generation.")
        return create_fallback_reference_data()

def create_realistic_patient_data_from_reference():
    """
    FIXED: Use actual CSV data directly instead of generating synthetic data
    This preserves the exact number of records from your CSV file
    """
    print("📄 Processing actual healthcare data from CSV (no synthetic generation)...")
    
    # Load reference data - this now contains the ACTUAL data
    ref_data = load_healthcare_reference_data()
    actual_records = len(ref_data)
    
    print(f"📊 Using {actual_records:,} actual patient records from CSV")
    
    # Use the data directly instead of generating synthetic data
    df = ref_data.copy()
    
    # Ensure we have the required columns by mapping from existing ones
    if 'patient_id' not in df.columns:
        df['patient_id'] = range(1, len(df) + 1)
    
    # Use existing age or generate from age_min/age_max if needed
    if 'age' not in df.columns:
        df['age'] = df.apply(lambda row: np.random.randint(row['age_min'], row['age_max'] + 1) 
                           if 'age_min' in df.columns and 'age_max' in df.columns 
                           else 50, axis=1)
    
    # Use specialty_clean if available, otherwise specialty
    if 'specialty_clean' in df.columns:
        df['specialty'] = df['specialty_clean']
    
    # Ensure we have required columns with realistic defaults if missing
    required_columns = {
        'cancel_risk': lambda x: x.get('cancel_risk', 0.03),
        'los': lambda x: x.get('los', 3.0),
        'cost_estimate': lambda x: x.get('cost_estimate', 3000),
        'season': lambda x: x.get('season', 'Summer')
    }
    
    for col, default_func in required_columns.items():
        if col not in df.columns:
            if col == 'cancel_risk':
                df[col] = 0.03  # Default 3% cancellation risk
            elif col == 'los':
                df[col] = 3.0   # Default 3 days length of stay
            elif col == 'cost_estimate':
                df[col] = 3000  # Default £3000 cost
            elif col == 'season':
                df[col] = np.random.choice(['Winter', 'Spring', 'Summer', 'Autumn'], len(df))
    
    print(f"✅ Processed {len(df):,} patient records with required columns")
    return df

def create_fallback_reference_data():
    """Create fallback reference data if CSV file is not available - now supports direct age"""
    specialties = ['Orthopedics', 'ENT', 'General Surgery', 'Gynaecology', 
                  'Urology', 'Ophthalmology', 'Cardiology']
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    genders = ['M', 'F']
    
    fallback_data = []
    record_id = 1
    
    # Generate records across age ranges for each combination
    for specialty in specialties:
        for season in seasons:
            for gender in genders:
                # Generate multiple age records for each combination
                for age in range(18, 91, 5):  # Ages 18, 23, 28, ..., 88
                    # Simple realistic values by specialty
                    if specialty == 'Orthopedics':
                        avg_los, cost_estimate, cancel_risk = 3.8, 3800, 0.06
                    elif specialty == 'ENT':
                        avg_los, cost_estimate, cancel_risk = 1.7, 2200, 0.033
                    elif specialty == 'General Surgery':
                        avg_los, cost_estimate, cancel_risk = 4.2, 4200, 0.038
                    elif specialty == 'Ophthalmology':
                        avg_los, cost_estimate, cancel_risk = 1.2, 1800, 0.012
                    elif specialty == 'Cardiology':
                        avg_los, cost_estimate, cancel_risk = 5.1, 5200, 0.018
                    else:
                        avg_los, cost_estimate, cancel_risk = 3.0, 3000, 0.03
                    
                    fallback_data.append({
                        'patient_id': record_id,
                        'age': age,
                        'specialty_clean': specialty,
                        'specialty': specialty,
                        'gender': gender,
                        'los': avg_los,
                        'season': season,
                        'cancel_risk': cancel_risk,
                        'cost_estimate': cost_estimate,
                        'planned_count': 100,
                        'age_min': age,  # For compatibility
                        'age_max': age,   # For compatibility
                        'doctors_fte': 50,  # Default staffing
                        'nurses_fte': 100,  # Default staffing
                        'total_fte': 150
                    })
                    record_id += 1
    
    print(f"🔧 Created fallback dataset with {record_id-1} records")
    return pd.DataFrame(fallback_data)

def create_synthetic_patient_data_pandas(num_records=None):
    """
    FIXED: Create patient data using actual CSV records
    num_records parameter is ignored - we use actual CSV size
    """
    return create_realistic_patient_data_from_reference()

def calculate_priority_score_pandas(df):
    """Calculate priority score using pandas operations"""
    print("🔢 Computing priority scores...")
    
    # Initialize score
    df['priority_score'] = 0.0
    
    # Age factor
    df.loc[df['age'] > 70, 'priority_score'] += 0.15
    
    # LOS factor
    df.loc[df['los'] > 3, 'priority_score'] += 0.15
    
    # Cost factor
    df.loc[df['cost_estimate'] > 5000, 'priority_score'] += 0.15
    
    # Specialty weights
    specialty_weights = {
        "Cardiology": 0.15, "General Surgery": 0.13, "Urology": 0.11,
        "Gynaecology": 0.09, "Ophthalmology": 0.07, "Orthopedics": 0.05, "ENT": 0.03
    }
    
    for specialty, weight in specialty_weights.items():
        df.loc[df['specialty'] == specialty, 'priority_score'] += weight
    
    # Season factor
    df.loc[df['season'] == 'Winter', 'priority_score'] += 0.07
    df.loc[df['season'] == 'Autumn', 'priority_score'] += 0.03
    
    # Cap at 1.0
    df['priority_score'] = df['priority_score'].clip(upper=1.0)
    
    return df

def calculate_readmission_risk_pandas(df):
    """Calculate readmission risk using pandas operations - enhanced with staffing ratios"""
    print("⚕️ Computing readmission risks...")
    
    # Initialize readmission risk
    df['readmit_risk'] = 0.0
    
    # Age factor
    df.loc[df['age'] > 70, 'readmit_risk'] += 0.2
    
    # LOS factor  
    df.loc[df['los'] > 5, 'readmit_risk'] += 0.2
    
    # Cancel risk factor
    df['readmit_risk'] += 0.2 * df['cancel_risk']
    
    # Cost factor
    df.loc[df['cost_estimate'] > 6000, 'readmit_risk'] += 0.2
    
    # Season factor
    df.loc[df['season'] == 'Winter', 'readmit_risk'] += 0.2
    
    # NEW: Staffing ratio impact on readmission risk
    if all(col in df.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
        # Calculate staffing ratios
        df['patients_per_doctor'] = df['planned_count'] / (df['doctors_fte'] + 0.1)  # Avoid division by zero
        df['patients_per_nurse'] = df['planned_count'] / (df['nurses_fte'] + 0.1)
        
        # Higher patient-to-staff ratios increase readmission risk
        df.loc[df['patients_per_doctor'] > 60, 'readmit_risk'] += 0.1  # High doctor workload
        df.loc[df['patients_per_nurse'] > 30, 'readmit_risk'] += 0.1   # High nurse workload
    
    # Cap at 1.0
    df['readmit_risk'] = df['readmit_risk'].clip(upper=1.0)
    
    return df

def train_staffing_efficiency_model_pandas(df):
    """Train ML model focused on staffing efficiency instead of just readmission risk - FIXED"""
    print("🤖 Training Staffing Efficiency model with pandas data...")
    
    # Create a copy to avoid modifying original DataFrame - PRESERVE ORIGINAL CATEGORICAL COLUMNS
    df_model = df.copy()
    
    # IMPORTANT: Store original categorical columns before any processing
    original_specialty = df_model['specialty'].copy() if 'specialty' in df_model.columns else None
    original_gender = df_model['gender'].copy() if 'gender' in df_model.columns else None
    original_season = df_model['season'].copy() if 'season' in df_model.columns else None
    
    # Create efficiency labels based on staffing ratios and outcomes
    if all(col in df_model.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
        # Calculate efficiency metrics
        df_model['patients_per_doctor'] = df_model['planned_count'] / (df_model['doctors_fte'] + 0.1)
        df_model['patients_per_nurse'] = df_model['planned_count'] / (df_model['nurses_fte'] + 0.1)
        df_model['total_staff_efficiency'] = df_model['planned_count'] / (df_model['total_fte'] + 0.1) if 'total_fte' in df_model.columns else df_model['patients_per_doctor']
        
        # Define efficient operations (low cancellation risk + reasonable staffing ratios)
        efficient_conditions = (
            (df_model['cancel_risk'] < 0.05) & 
            (df_model['patients_per_doctor'] < 70) & 
            (df_model['patients_per_nurse'] < 35)
        )
        df_model['efficiency_label'] = efficient_conditions.astype(int)
        
    else:
        # Fallback to readmission-based model
        print("⚠️ Staffing data not available, using readmission risk model")
        df_model['efficiency_label'] = (df_model['readmit_risk'] < 0.3).astype(int)  # Inverse of high risk
        # Add default staffing metrics for consistency
        df_model['patients_per_doctor'] = 50.0
        df_model['patients_per_nurse'] = 25.0
        df_model['total_staff_efficiency'] = 0.3
    
    # Define numeric features explicitly to avoid categorical column issues
    numeric_features = ['age', 'los', 'cost_estimate', 'cancel_risk', 'priority_score']
    
    # Add staffing features if they exist and are numeric
    staffing_features = ['patients_per_doctor', 'patients_per_nurse', 'total_staff_efficiency']
    for feature in staffing_features:
        if feature in df_model.columns:
            # Ensure the feature is numeric
            df_model[feature] = pd.to_numeric(df_model[feature], errors='coerce')
            if not df_model[feature].isna().all():  # Only add if not all NaN
                numeric_features.append(feature)
    
    # Handle categorical variables with proper preservation
    df_encoded = df_model.copy()
    categorical_features = []
    
    # Process categorical variables while preserving originals
    if 'gender' in df_model.columns:
        categorical_features.append('gender')
        dummies = pd.get_dummies(df_encoded['gender'], prefix='gender', drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        numeric_features.extend(dummies.columns.tolist())
    
    if 'specialty' in df_model.columns:
        categorical_features.append('specialty')
        dummies = pd.get_dummies(df_encoded['specialty'], prefix='specialty', drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        numeric_features.extend(dummies.columns.tolist())
    
    if 'season' in df_model.columns:
        categorical_features.append('season')
        dummies = pd.get_dummies(df_encoded['season'], prefix='season', drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        numeric_features.extend(dummies.columns.tolist())
    
    # Prepare feature matrix with only numeric features
    feature_cols = [col for col in numeric_features if col in df_encoded.columns]
    X = df_encoded[feature_cols].fillna(0).select_dtypes(include=[np.number])  # Ensure only numeric columns
    y = df_encoded['efficiency_label']
    
    print(f"🎯 Training on {len(X):,} samples with {X.shape[1]} features")
    print(f"📊 Efficiency rate: {y.mean():.2%}")
    print(f"🔍 Features used: {list(X.columns)}")
    
    # Use RandomForest instead of XGBoost for simplicity and interpretability
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Fallback to basic logistic regression
        model = SklearnLogisticRegression(random_state=42, max_iter=1000)
    
    # Train model
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    
    # Generate predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
    
    # CRITICAL: Add predictions to ORIGINAL DataFrame, not the encoded one
    # This preserves all original categorical columns including specialty diversity
    df['prediction'] = predictions
    df['score'] = probabilities
    df['percentage'] = np.round(probabilities * 100, 2)
    
    # RESTORE original categorical columns if they were modified
    if original_specialty is not None:
        df['specialty'] = original_specialty
    if original_gender is not None:
        df['gender'] = original_gender  
    if original_season is not None:
        df['season'] = original_season
    
    # Calculate metrics
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y, probabilities)
    else:
        auc = 0.5  # Default for models without probability predictions
    accuracy = accuracy_score(y, predictions)
    
    print(f"✅ Model completed in {training_time:.2f}s")
    print(f"📈 Performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
    
    # Feature importance if available - use X.columns instead of undefined feature_cols
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("🔍 Top 5 Important Features:")
        print(importance_df.head())
    
    return df, model

def generate_wait_days_pandas(df):
    """Generate wait days using pandas"""
    print("⏰ Generating wait days...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate wait days based on priority score
    df['wait_days'] = (100 - (df['priority_score'] * 100)).astype(int) + \
                      np.random.randint(0, 31, len(df))
    
    return df

def save_to_delta_table(df, table_name="patient_data"):
    """Save DataFrame to Delta table"""
    try:
        # If we have Spark available, convert pandas to Spark and save
        spark = get_spark_session()
        if spark:
            spark_df = spark.createDataFrame(df)
            spark_df.write.format("delta") \
                  .mode("overwrite") \
                  .option("overwriteSchema", "true") \
                  .saveAsTable(table_name)
            print(f"💾 Data saved to Delta table: {table_name}")
            return True
        else:
            # Fallback: save as parquet or use Databricks SQL
            print("💾 Spark not available, saving to parquet file...")
            df.to_parquet(f"/tmp/{table_name}.parquet")
            return True
    except Exception as e:
        print(f"❌ Error saving to Delta table: {e}")
        return False

def analyze_capacity_impact_pandas(df, baseline_capacity=100, uplift_percentage=20):
    """Analyze capacity impact using pandas operations with enhanced staffing analysis"""
    print(f"📊 Analyzing capacity impact: {uplift_percentage}% uplift...")
    
    # Calculate new capacity
    new_capacity = int(baseline_capacity * (1 + uplift_percentage/100))
    additional_capacity = new_capacity - baseline_capacity
    
    # Sort by priority score and wait days
    df_sorted = df.sort_values(['priority_score', 'wait_days'], ascending=[False, False]).reset_index(drop=True)
    
    # Calculate queue positions and scheduling weeks
    df_sorted['queue_position'] = range(1, len(df_sorted) + 1)
    df_sorted['baseline_week'] = np.ceil(df_sorted['queue_position'] / baseline_capacity)
    df_sorted['new_week'] = np.ceil(df_sorted['queue_position'] / new_capacity)
    
    # Calculate summary metrics
    baseline_weeks_needed = int(df_sorted['baseline_week'].max())
    new_weeks_needed = int(df_sorted['new_week'].max())
    weeks_saved = baseline_weeks_needed - new_weeks_needed
    patients_moved_earlier = len(df_sorted[df_sorted['new_week'] < df_sorted['baseline_week']])
    avg_weeks_saved = (df_sorted['baseline_week'] - df_sorted['new_week']).mean()
    
    # Calculate staffing requirements if staffing data is available
    staffing_analysis = {}
    if all(col in df.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
        # Current staffing ratios
        avg_doctors = df['doctors_fte'].mean()
        avg_nurses = df['nurses_fte'].mean()
        current_planned = df['planned_count'].sum()
        
        # Calculate current ratios
        patients_per_doctor = current_planned / avg_doctors if avg_doctors > 0 else 50
        patients_per_nurse = current_planned / avg_nurses if avg_nurses > 0 else 25
        
        # Calculate additional staff needed
        additional_doctors_needed = additional_capacity / patients_per_doctor
        additional_nurses_needed = additional_capacity / patients_per_nurse
        
        # Mixed scenario (60% doctor capacity, 40% nurse capacity)
        mixed_doctors = (additional_capacity * 0.6) / patients_per_doctor
        mixed_nurses = (additional_capacity * 0.4) / patients_per_nurse
        
        # Cost calculations (approximate annual salaries)
        doctor_annual_cost = 80000
        nurse_annual_cost = 35000
        
        staffing_analysis = {
            'current_doctors': avg_doctors,
            'current_nurses': avg_nurses,
            'patients_per_doctor': patients_per_doctor,
            'patients_per_nurse': patients_per_nurse,
            'additional_doctors_needed': additional_doctors_needed,
            'additional_nurses_needed': additional_nurses_needed,
            'mixed_scenario_doctors': mixed_doctors,
            'mixed_scenario_nurses': mixed_nurses,
            'cost_doctors_only': additional_doctors_needed * doctor_annual_cost,
            'cost_nurses_only': additional_nurses_needed * nurse_annual_cost,
            'cost_mixed_scenario': (mixed_doctors * doctor_annual_cost) + (mixed_nurses * nurse_annual_cost)
        }
    
    # Create summary DataFrame
    summary_data = {
        'total_patients': len(df),
        'baseline_capacity': baseline_capacity,
        'new_capacity': new_capacity,
        'additional_capacity': additional_capacity,
        'baseline_weeks': baseline_weeks_needed,
        'new_weeks': new_weeks_needed,
        'weeks_saved': weeks_saved,
        'patients_moved_earlier': patients_moved_earlier,
        'avg_weeks_saved': avg_weeks_saved or 0.0,
        **staffing_analysis  # Add staffing analysis to summary
    }
    
    summary_df = pd.DataFrame([summary_data])
    
    # Create weekly backlog analysis
    weekly_data = []
    for week in range(1, min(baseline_weeks_needed + 1, 52)):
        baseline_backlog = len(df_sorted[df_sorted['baseline_week'] > week])
        new_backlog = len(df_sorted[df_sorted['new_week'] > week])
        weekly_data.append({
            'week': week,
            'baseline_backlog': baseline_backlog,
            'new_backlog': new_backlog
        })
    
    weekly_df = pd.DataFrame(weekly_data)
    
    return summary_df, weekly_df, df_sorted

def identify_high_risk_patients_pandas(df, risk_threshold=0.7):
    """Identify high-risk patients using pandas"""
    print(f"🚨 Identifying high-risk patients (threshold: {risk_threshold})...")
    
    high_risk_df = df[df['readmit_risk'] > risk_threshold].sort_values('readmit_risk', ascending=False)
    high_risk_count = len(high_risk_df)
    
    print(f"Found {high_risk_count:,} high-risk patients")
    return high_risk_df

def optimize_schedule_pandas(df, weekly_capacity=100, risk_weight=0.3):
    """Optimize schedule using pandas operations"""
    print("📅 Creating optimized schedule...")
    
    # Calculate combined score
    df['combined_score'] = (df['priority_score'] * (1 - risk_weight)) + (df['readmit_risk'] * risk_weight)
    
    # Sort by combined score and wait days
    df_scheduled = df.sort_values(['combined_score', 'wait_days'], ascending=[False, False]).reset_index(drop=True)
    
    # Calculate queue position and scheduled week
    df_scheduled['queue_position'] = range(1, len(df_scheduled) + 1)
    df_scheduled['scheduled_week'] = np.ceil(df_scheduled['queue_position'] / weekly_capacity)
    
    return df_scheduled

def run_databricks_healthcare_pipeline():
    """
    FIXED: Execute the complete healthcare analytics pipeline using actual CSV data
    """
    print("🏥 Starting Healthcare Analytics Pipeline (Using Actual CSV Data)")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Load actual data from CSV (no longer generates synthetic data)
        df = create_synthetic_patient_data_pandas()  # This now uses actual CSV data
        actual_records = len(df)
        print(f"📊 Loaded {actual_records:,} actual patient records")
        
        # Step 2: Calculate priority scores
        df = calculate_priority_score_pandas(df)
        
        # Step 3: Calculate readmission risks (now includes staffing factors)
        df = calculate_readmission_risk_pandas(df)
        
        # Step 4: Train staffing efficiency model (replaces XGBoost readmission model)
        df_with_predictions, ml_model = train_staffing_efficiency_model_pandas(df)
        
        # Step 5: Generate wait days
        df_final = generate_wait_days_pandas(df_with_predictions)
        
        # Step 6: Save to Delta table (if possible)
        save_to_delta_table(df_final)
        
        # Step 7: Run some analyses
        high_risk_patients = identify_high_risk_patients_pandas(df_final, risk_threshold=0.7)
        
        print(f"🚨 High-Risk Patients: {len(high_risk_patients):,}")
        print(f"⏱️  Total Pipeline Execution Time: {time.time() - start_time:.2f}s")
        print(f"✅ Processing complete for {actual_records:,} patients")
        print("=" * 70)
        
        # Return results in a format compatible with existing code
        class DataFrameWrapper:
            def __init__(self, df):
                self.df = df
            
            def limit(self, n):
                return DataFrameWrapper(self.df.head(n))
            
            def toPandas(self):
                return self.df
            
            def count(self):
                return len(self.df)
            
            def filter(self, condition):
                # This would need more complex logic for real Spark-like filtering
                return DataFrameWrapper(self.df)
        
        return {
            'df_final': DataFrameWrapper(df_final),
            'predictions': DataFrameWrapper(df_with_predictions),
            'ml_model': ml_model,
            'high_risk_patients': DataFrameWrapper(high_risk_patients),
            'raw_pandas_df': df_final  # For direct pandas access
        }
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        raise

# Rest of the code remains the same - compatibility functions and AI query processor...
# (Keeping all the existing functions for compatibility)

def analyze_capacity_impact_optimized(df, baseline_capacity=100, uplift_percentage=20):
    """Wrapper to maintain compatibility with existing code"""
    if hasattr(df, 'df'):  # DataFrameWrapper
        pandas_df = df.df
    else:
        pandas_df = df
    
    summary_df, weekly_df, scheduled_df = analyze_capacity_impact_pandas(
        pandas_df, baseline_capacity, uplift_percentage
    )
    
    # Return as DataFrameWrapper for compatibility
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(summary_df), DataFrameWrapper(weekly_df), DataFrameWrapper(scheduled_df)

def identify_high_risk_patients_optimized(df, risk_threshold=0.7):
    """Wrapper to maintain compatibility"""
    if hasattr(df, 'df'):
        pandas_df = df.df
    else:
        pandas_df = df
    
    high_risk_df = identify_high_risk_patients_pandas(pandas_df, risk_threshold)
    
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def count(self):
            return len(self.df)
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(high_risk_df)

def optimize_schedule_optimized(df, weekly_capacity=100, risk_weight=0.3):
    """Wrapper to maintain compatibility"""
    if hasattr(df, 'df'):
        pandas_df = df.df
    else:
        pandas_df = df
    
    scheduled_df = optimize_schedule_pandas(pandas_df, weekly_capacity, risk_weight)
    
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(scheduled_df)

# Keep the existing HealthcareAIQueryProcessor class but modify to use pandas
@dataclass
class QueryResult:
    """Structure for query results"""
    query_type: str
    sql_query: str = None
    function_name: str = None
    parameters: Dict = None
    result_data: Any = None
    visualization_type: str = None
    cost_impact: float = None
    bed_days_freed: int = None

class HealthcareAIQueryProcessor:
    def __init__(self, spark_session=None, openai_api_key: str = None):
        self.spark = spark_session
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

        # Define available functions and their descriptions
        self.available_functions = {
            'analyze_capacity_impact_optimized': {
                'description': 'Analyze impact of capacity increase on waiting list with staffing requirements',
                'parameters': ['baseline_capacity', 'uplift_percentage'],
                'returns': 'weeks_saved, patients_moved_earlier, staffing_requirements, cost_analysis'
            },
            'identify_high_risk_patients_optimized': {
                'description': 'Find patients with high readmission risk',
                'parameters': ['risk_threshold'],
                'returns': 'high_risk_patient_count, potential_cost_savings'
            },
            'optimize_schedule_optimized': {
                'description': 'Create optimized schedule balancing priority and risk',
                'parameters': ['weekly_capacity', 'risk_weight'],
                'returns': 'optimized_schedule, efficiency_gain'
            },
            'simulate_cancellation_reduction': {
                'description': 'Simulate impact of reducing cancellations',
                'parameters': ['reduction_percentage'],
                'returns': 'cost_savings, bed_days_freed, capacity_freed'
            }
        }
        
        # Define common query patterns (keeping existing patterns)
        self.query_patterns = {
            'average_wait_time': r'(average|avg|mean).*(wait|waiting).*(time|days|period)',
            'capacity_analysis': r'(capacity|increase|uplift|boost).*((\d+)%?|\d+)',
            'cancellation_reduction': r'(reduce|decrease|lower|reduction).*(cancel|cancellation).*((\d+)%?|\d+)',
            'high_risk_patients': r'(high.risk|risky|dangerous).*(patient|case)',
            'specialty_analysis': r'(specialty|department|ward|category).*(analysis|breakdown|summary)',
            'cost_impact': r'(cost|expense|budget|saving).*(impact|effect|analysis)',
            'seasonal_readmission': r'(season|seasonal|winter|summer|spring|autumn).*(readmission|readmit)',
            'seasonal_cost': r'(season|seasonal|winter|summer|spring|autumn).*(cost|expense|budget)',
            'seasonal_analysis': r'(season|seasonal|winter|summer|spring|autumn).*(analysis|impact|effect|comparison)',
            'seasonal_comparison': r'(which|what).*(season|seasonal).*(highest|most|best|worst)',
            'age_based_risk': r'(risk|readmit|readmission).*(age|elderly|young|adult|child|children|senior|group|segment)|(age|elderly|young|adult|child|children|senior|group|segment).*(risk|readmit|readmission)',
            'age_based_cancellation': r'(age|elderly|young|children|child|senior|middle-aged|group|segment).*(cancel|cancellation)',
            'age_based_cost': r'(age|elderly|young|child|children|senior|middle-aged|group|segment).*(cost|expense|budget)',
            'staffing_analysis': r'(staff|staffing|doctor|nurse|personnel).*(analysis|requirement|need)',
            'capacity_staffing': r'(capacity|increase|uplift).*(staff|staffing|doctor|nurse|personnel)'
        }
    
    def classify_query(self, user_query: str) -> str:
        """Classify the type of query based on patterns"""
        user_query_lower = user_query.lower()
        
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, user_query_lower):
                return query_type
        
        return 'general_analysis'
    
    def extract_parameters(self, user_query: str) -> Dict:
        """Extract numerical parameters from user query"""
        parameters = {}
        
        # Extract percentages first
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', user_query)
        if percent_match:
            parameters['percentage'] = float(percent_match.group(1))
    
        # Extract specific numbers
        number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_query)
        if number_matches:
            parameters['numbers'] = [float(n) for n in number_matches]
    
        # Extract baseline capacity - look for explicit baseline capacity mentions
        baseline_capacity_match = re.search(r'baseline\s+capacity.*?(\d+)', user_query.lower())
        if baseline_capacity_match:
            parameters['baseline_capacity'] = int(baseline_capacity_match.group(1))
            parameters['weekly_capacity'] = int(baseline_capacity_match.group(1))
    
        # If no baseline capacity mentioned but it's a capacity query, use default
        if ('capacity' in user_query.lower() or 'increase' in user_query.lower()) and 'baseline_capacity' not in parameters:
            parameters['baseline_capacity'] = 100
            parameters['weekly_capacity'] = 100
        
        # Extract risk weight
        risk_weight_match = re.search(r'risk.weight.*?(\d+(?:\.\d+)?)', user_query.lower())
        if risk_weight_match:
            parameters['risk_weight'] = float(risk_weight_match.group(1))
        
        # Extract threshold values
        threshold_match = re.search(r'threshold.*?(\d+(?:\.\d+)?)', user_query.lower())
        if threshold_match:
            parameters['threshold'] = float(threshold_match.group(1))
        
        # Extract specialties
        specialties = ["orthopedics", "ent", "general surgery", "gynaecology", 
                      "urology", "ophthalmology", "cardiology"]
        
        found_specialties = []
        for specialty in specialties:
            pattern = r'\b' + re.escape(specialty.lower()) + r'\b'
            if re.findall(pattern, user_query.lower()):
                found_specialties.append(specialty)
        
        if found_specialties:
            titled_specialties = [s.title() for s in found_specialties]
            parameters['specialties'] = titled_specialties

        # Extract age information (keeping existing logic)
        age_match = re.search(r'\bage(?:d)?\s*(\d{1,3})\b', user_query.lower())
        if age_match:
            parameters['age'] = int(age_match.group(1))

        age_range_match = re.search(r'between\s+(\d{1,3})\s+(?:and|to)\s+(\d{1,3})', user_query.lower())
        if age_range_match:
            parameters['age_min'] = int(age_range_match.group(1))
            parameters['age_max'] = int(age_range_match.group(2))

        age_groups = {
            "child": (0, 17),
            "young": (18, 35),
            "young adult": (18, 35),
            "adult": (36, 60),
            "elderly": (61, 120),
            "senior": (61, 120),
            "middle-aged": (40, 60),
        }

        for group in age_groups:
            if re.search(r'\b' + re.escape(group) + r'\b', user_query.lower()):
                parameters['age_group'] = group
                parameters['age_min'], parameters['age_max'] = age_groups[group]
        
        return parameters
    
    def execute_pandas_query(self, query_type: str, parameters: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """Execute queries using pandas operations instead of SQL"""
        
        if query_type == 'average_wait_time':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('specialty').agg({
                'wait_days': 'mean',
                'patient_id': 'count'
            }).rename(columns={'wait_days': 'avg_wait_days', 'patient_id': 'patient_count'})
            return result.reset_index()
        
        elif query_type == 'specialty_analysis':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('specialty').agg({
                'patient_id': 'count',
                'wait_days': 'mean',
                'cost_estimate': 'mean',
                'cancel_risk': 'mean',
                'readmit_risk': 'mean'
            }).rename(columns={
                'patient_id': 'patient_count',
                'wait_days': 'avg_wait_days',
                'cost_estimate': 'avg_cost',
                'cancel_risk': 'avg_cancel_risk',
                'readmit_risk': 'avg_readmit_risk'
            })
            return result.reset_index()
        
        elif query_type == 'high_risk_patients':
            threshold = parameters.get('percentage', 70) / 100
            high_risk_df = df[df['readmit_risk'] > threshold]
            
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                high_risk_df = high_risk_df[high_risk_df['specialty'].isin(parameters['specialties'])]
            
            result = high_risk_df.groupby('specialty').agg({
                'patient_id': 'count',
                'readmit_risk': 'mean',
                'cost_estimate': 'mean'
            }).rename(columns={
                'patient_id': 'high_risk_count',
                'readmit_risk': 'avg_risk',
                'cost_estimate': 'avg_cost'
            })
            return result.reset_index()
        
        elif query_type == 'cost_impact':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            filtered_df['potential_loss_from_cancellations'] = filtered_df['cost_estimate'] * filtered_df['cancel_risk']
            
            result = filtered_df.groupby('specialty').agg({
                'cost_estimate': ['sum', 'mean', 'count'],
                'potential_loss_from_cancellations': 'sum'
            })
            result.columns = ['total_cost', 'avg_cost', 'patient_count', 'potential_loss_from_cancellations']
            return result.reset_index()
        
        elif query_type == 'seasonal_analysis':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('season').agg({
                'cancel_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'cancel_risk': 'avg_cancel_risk',
                'patient_id': 'total_cases'
            })
            return result.reset_index().sort_values('avg_cancel_risk', ascending=False)
        
        elif query_type == 'age_based_risk':
            filtered_df = self.apply_age_filter(df, parameters)
            
            # Create age groups
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'readmit_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'readmit_risk': 'avg_readmit_risk',
                'patient_id': 'total_patients'
            })
            result = result[result['total_patients'] > 0]
            return result.reset_index()
        
        elif query_type == 'age_based_cancellation':
            filtered_df = self.apply_age_filter(df, parameters)
            
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'cancel_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'cancel_risk': 'avg_cancel_risk',
                'patient_id': 'total_patients'
            })
            result = result[result['total_patients'] > 0]
            return result.reset_index().sort_values('avg_cancel_risk', ascending=False)
        
        elif query_type == 'age_based_cost':
            filtered_df = self.apply_age_filter(df, parameters)
            
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'cost_estimate': ['mean', 'sum', 'count']
            })
            result.columns = ['avg_cost', 'total_cost', 'patient_count']
            result = result[result['patient_count'] > 0]
            return result.reset_index().sort_values('avg_cost', ascending=False)
        
        # Default query
        return df.head(10)
    
    def apply_age_filter(self, df, parameters):
        """Apply age filtering based on parameters"""
        filtered_df = df.copy()
        
        if 'age' in parameters:
            filtered_df = filtered_df[filtered_df['age'] == parameters['age']]
        elif 'age_min' in parameters and 'age_max' in parameters:
            filtered_df = filtered_df[(filtered_df['age'] >= parameters['age_min']) & 
                                    (filtered_df['age'] <= parameters['age_max'])]
        elif 'age_group' in parameters:
            group = parameters['age_group']
            if group == "child":
                filtered_df = filtered_df[(filtered_df['age'] >= 0) & (filtered_df['age'] <= 17)]
            elif group in ["young", "young adult"]:
                filtered_df = filtered_df[(filtered_df['age'] >= 18) & (filtered_df['age'] <= 35)]
            elif group == "adult":
                filtered_df = filtered_df[(filtered_df['age'] >= 36) & (filtered_df['age'] <= 60)]
            elif group in ["elderly", "senior"]:
                filtered_df = filtered_df[filtered_df['age'] > 60]
            elif group == "middle-aged":
                filtered_df = filtered_df[(filtered_df['age'] >= 40) & (filtered_df['age'] <= 60)]
        
        if 'specialties' in parameters and len(parameters['specialties']) > 0:
            filtered_df = filtered_df[filtered_df['specialty'].isin(parameters['specialties'])]
        
        return filtered_df
    
    def simulate_cancellation_reduction(self, df: pd.DataFrame, reduction_percentage: float) -> Dict:
        """Simulate the impact of reducing cancellations using pandas"""
        
        # Calculate current cancellation impact
        total_patients = len(df)
        current_cancellations = df['cancel_risk'].sum()
        avg_cost = df['cost_estimate'].mean()
        avg_los = df['los'].mean()
        
        # Calculate reduction impact
        cancellations_prevented = current_cancellations * (reduction_percentage / 100)
        cost_savings = cancellations_prevented * avg_cost
        bed_days_freed = cancellations_prevented * avg_los
        
        # Calculate capacity freed (assuming 5 days per week operation)
        weekly_capacity_freed = bed_days_freed / 7 * 5
        
        return {
            'current_expected_cancellations': round(current_cancellations, 2),
            'cancellations_prevented': round(cancellations_prevented, 2),
            'cost_savings': round(cost_savings, 2),
            'bed_days_freed': round(bed_days_freed, 2),
            'weekly_capacity_freed': round(weekly_capacity_freed, 2),
            'reduction_percentage': reduction_percentage
        }
    
    # def execute_function_call(self, function_name: str, parameters: Dict, df) -> Dict:
    #     """Execute specific function calls based on the query - enhanced with staffing analysis"""
        
    #     # Get pandas DataFrame from wrapper or direct DataFrame
    #     if hasattr(df, 'df'):
    #         pandas_df = df.df
    #     elif hasattr(df, 'raw_pandas_df'):
    #         pandas_df = df.raw_pandas_df  
    #     else:
    #         pandas_df = df
        
    #     if function_name == 'simulate_cancellation_reduction':
    #         reduction_pct = parameters.get('percentage', 20)
    #         return self.simulate_cancellation_reduction(pandas_df, reduction_pct)
        
    #     elif function_name == 'analyze_capacity_impact_optimized':
    #         baseline_cap = parameters.get('baseline_capacity', 100)
    #         uplift_pct = parameters.get('percentage', 20)
    #         total_patients = len(pandas_df)
    #         new_capacity = int(baseline_cap * (1 + uplift_pct/100))
    #         additional_capacity = new_capacity - baseline_cap
            
    #         # Calculate basic timing metrics
    #         current_weeks_needed = np.ceil(total_patients / baseline_cap)
    #         new_weeks_needed = np.ceil(total_patients / new_capacity)
    #         weeks_saved = current_weeks_needed - new_weeks_needed

    #         # Calculate patients moved earlier (same logic as app.py)
    #         patients_moved_earlier = 0
    #         for week in range(1, int(current_weeks_needed) + 1):
    #             current_week_patients = min(baseline_cap, total_patients - (week-1) * baseline_cap)
    #             new_week_patients = min(new_capacity, total_patients - (week-1) * new_capacity)
    #             if new_week_patients > current_week_patients:
    #                 patients_moved_earlier += (new_week_patients - current_week_patients)
            
    #         # Enhanced staffing analysis
    #         staffing_insights = {}
    #         if all(col in pandas_df.columns for col in ['doctors_fte', 'nurses_fte', 'planned_count']):
    #             # Calculate current staffing ratios
    #             avg_doctors = pandas_df['doctors_fte'].mean()
    #             avg_nurses = pandas_df['nurses_fte'].mean()
    #             current_planned = pandas_df['planned_count'].sum()
                
    #             patients_per_doctor = current_planned / avg_doctors if avg_doctors > 0 else 50
    #             patients_per_nurse = current_planned / avg_nurses if avg_nurses > 0 else 25
                
    #             # Calculate staffing requirements for additional capacity
    #             additional_doctors_needed = additional_capacity / patients_per_doctor
    #             additional_nurses_needed = additional_capacity / patients_per_nurse
                
    #             # Mixed scenario calculations
    #             mixed_doctors = (additional_capacity * 0.6) / patients_per_doctor
    #             mixed_nurses = (additional_capacity * 0.4) / patients_per_nurse
                
    #             # Cost calculations
    #             doctor_cost = 80000
    #             nurse_cost = 35000
                
    #             staffing_insights = {
    #                 'current_doctors': round(avg_doctors, 1),
    #                 'current_nurses': round(avg_nurses, 1),
    #                 'patients_per_doctor': round(patients_per_doctor, 1),
    #                 'patients_per_nurse': round(patients_per_nurse, 1),
    #                 'additional_doctors_needed': round(additional_doctors_needed, 1),
    #                 'additional_nurses_needed': round(additional_nurses_needed, 1),
    #                 'mixed_doctors': round(mixed_doctors, 1),
    #                 'mixed_nurses': round(mixed_nurses, 1),
    #                 'cost_doctors_only': round(additional_doctors_needed * doctor_cost, 0),
    #                 'cost_nurses_only': round(additional_nurses_needed * nurse_cost, 0),
    #                 'cost_mixed': round((mixed_doctors * doctor_cost) + (mixed_nurses * nurse_cost), 0)
    #             }
            
    #         # Calculate additional metrics for comprehensive analysis
    #         efficiency_gain = ((current_weeks_needed - new_weeks_needed) / current_weeks_needed) * 100 if current_weeks_needed > 0 else 0
            
    #         result = {
    #             'weeks_saved': int(weeks_saved),
    #             'patients_moved_earlier': int(patients_moved_earlier),
    #             'baseline_weeks': int(current_weeks_needed),
    #             'new_weeks': int(new_weeks_needed),
    #             'baseline_capacity': baseline_cap,
    #             'new_capacity': new_capacity,
    #             'additional_capacity': additional_capacity,
    #             'efficiency_gain': round(efficiency_gain, 1),
    #             'capacity_increase': f"{uplift_pct}%",
    #             'total_patients': total_patients,
    #             **staffing_insights
    #         }
            
    #         return result
        
    #     elif function_name == 'identify_high_risk_patients_optimized':
    #         threshold = parameters.get('percentage', 70) / 100
    #         high_risk_df = pandas_df[pandas_df['readmit_risk'] > threshold]
    #         high_risk_count = len(high_risk_df)
    #         avg_cost_high_risk = high_risk_df['cost_estimate'].mean() if high_risk_count > 0 else 0
            
    #         return {
    #             'high_risk_patient_count': high_risk_count,
    #             'risk_threshold': threshold,
    #             'avg_cost_high_risk': round(avg_cost_high_risk, 2),
    #             'potential_intervention_needed': high_risk_count > 0
    #         }
        
    #     return {}

    
    def execute_function_call(self, function_name: str, parameters: Dict, df) -> Dict:
        """Execute specific function calls based on the query - enhanced with staffing analysis"""
        # Get pandas DataFrame from wrapper or direct DataFrame
        if hasattr(df, 'df'):
            pandas_df = df.df
        elif hasattr(df, 'raw_pandas_df'):
            pandas_df = df.raw_pandas_df  
        else:
            pandas_df = df
    
        if function_name == 'simulate_cancellation_reduction':
            reduction_pct = parameters.get('percentage', 20)
            return self.simulate_cancellation_reduction(pandas_df, reduction_pct)
    
        elif function_name == 'analyze_capacity_impact_optimized':
            baseline_cap = parameters.get('baseline_capacity', 100)
            uplift_pct = parameters.get('percentage', 20)
            total_patients = len(pandas_df)
            new_capacity = int(baseline_cap * (1 + uplift_pct/100))
            additional_capacity = new_capacity - baseline_cap
        
            # Calculate basic timing metrics
            current_weeks_needed = np.ceil(total_patients / baseline_cap)
            new_weeks_needed = np.ceil(total_patients / new_capacity)
            weeks_saved = current_weeks_needed - new_weeks_needed

            # Calculate patients moved earlier
            patients_moved_earlier = 0
            for week in range(1, int(current_weeks_needed) + 1):
                current_week_patients = min(baseline_cap, total_patients - (week-1) * baseline_cap)
                new_week_patients = min(new_capacity, total_patients - (week-1) * new_capacity)
                if new_week_patients > current_week_patients:
                    patients_moved_earlier += (new_week_patients - current_week_patients)
        
            # FIXED: Use realistic industry standard staffing ratios
            PATIENTS_PER_DOCTOR = 25   # Industry standard
            PATIENTS_PER_NURSE = 12    # Industry standard
        
            # Calculate current staffing for baseline capacity
            current_doctors_needed = baseline_cap / PATIENTS_PER_DOCTOR
            current_nurses_needed = baseline_cap / PATIENTS_PER_NURSE
        
            # Calculate additional staffing needed for uplift
            additional_doctors_needed = additional_capacity / PATIENTS_PER_DOCTOR
            additional_nurses_needed = additional_capacity / PATIENTS_PER_NURSE
        
            # Mixed scenario calculations (optimal staffing mix)
            mixed_doctors = additional_capacity / (PATIENTS_PER_DOCTOR * 1.2)  # Slightly fewer doctors
            mixed_nurses = additional_capacity / (PATIENTS_PER_NURSE * 0.8)   # More nurses
        
            # Cost calculations
            doctor_cost = 80000
            nurse_cost = 35000
        
            staffing_insights = {
            'current_doctors': round(current_doctors_needed, 1),
            'current_nurses': round(current_nurses_needed, 1),
            'patients_per_doctor': PATIENTS_PER_DOCTOR,
            'patients_per_nurse': PATIENTS_PER_NURSE,
            'additional_doctors_needed': round(additional_doctors_needed, 1),
            'additional_nurses_needed': round(additional_nurses_needed, 1),
            'mixed_doctors': round(mixed_doctors, 1),
            'mixed_nurses': round(mixed_nurses, 1),
            'cost_doctors_only': round(additional_doctors_needed * doctor_cost, 0),
            'cost_nurses_only': round(additional_nurses_needed * nurse_cost, 0),
            'cost_mixed': round((mixed_doctors * doctor_cost) + (mixed_nurses * nurse_cost), 0)
            }
        
            # Calculate efficiency gain
            efficiency_gain = ((current_weeks_needed - new_weeks_needed) / current_weeks_needed) * 100 if current_weeks_needed > 0 else 0
        
            result = {
            'weeks_saved': int(weeks_saved),
            'patients_moved_earlier': int(patients_moved_earlier),
            'baseline_weeks': int(current_weeks_needed),
            'new_weeks': int(new_weeks_needed),
            'baseline_capacity': baseline_cap,
            'new_capacity': new_capacity,
            'additional_capacity': additional_capacity,
            'efficiency_gain': round(efficiency_gain, 1),
            'capacity_increase': f"{uplift_pct}%",
            'total_patients': total_patients,
            **staffing_insights
            }
        
            return result
    
        elif function_name == 'identify_high_risk_patients_optimized':
            threshold = parameters.get('percentage', 70) / 100
            high_risk_df = pandas_df[pandas_df['readmit_risk'] > threshold]
            high_risk_count = len(high_risk_df)
            avg_cost_high_risk = high_risk_df['cost_estimate'].mean() if high_risk_count > 0 else 0
        
            return {
            'high_risk_patient_count': high_risk_count,
            'risk_threshold': threshold,
            'avg_cost_high_risk': round(avg_cost_high_risk, 2),
            'potential_intervention_needed': high_risk_count > 0
            }
    
        return {}
    
    def process_natural_language_query(self, user_query: str, df) -> QueryResult:
        """Main method to process natural language queries"""
        
        # Step 1: Classify the query
        query_type = self.classify_query(user_query)
        
        # Step 2: Extract parameters
        parameters = self.extract_parameters(user_query)
        
        # Step 3: Determine if it's a SQL query or function call
        result = QueryResult(query_type=query_type)
        
        if query_type in ['cancellation_reduction', 'capacity_analysis', 'capacity_staffing']:
            # Function call approach
            if 'cancel' in user_query.lower():
                result.function_name = 'simulate_cancellation_reduction'
            elif 'capacity' in user_query.lower() or 'increase' in user_query.lower():
                result.function_name = 'analyze_capacity_impact_optimized'
            elif query_type == 'high_risk_analysis':
                result.function_name = 'identify_high_risk_patients_optimized'
            elif query_type == 'schedule_optimization':
                result.function_name = 'optimize_schedule_optimized'
            
        if result.function_name:
            result.parameters = parameters
            result.result_data = self.execute_function_call(result.function_name, parameters, df)
            
        else:
            # Pandas query approach instead of SQL
            pandas_df = df.df if hasattr(df, 'df') else df.raw_pandas_df if hasattr(df, 'raw_pandas_df') else df
            result.result_data = self.execute_pandas_query(query_type, parameters, pandas_df)
        
        return result
    
    def format_response(self, result: QueryResult) -> str:
        """Format the response in a user-friendly way - enhanced with staffing insights"""
        
        if result.function_name == 'simulate_cancellation_reduction':
            data = result.result_data
            return f"""
            **Cancellation Reduction Analysis**
            
            By reducing cancellations by {data['reduction_percentage']}%:
            
            • **Cost Savings**: £{data['cost_savings']:,.2f}
            • **Bed Days Freed**: {data['bed_days_freed']:.0f} days
            • **Weekly Capacity Freed**: {data['weekly_capacity_freed']:.1f} procedures
            • **Cancellations Prevented**: {data['cancellations_prevented']:.0f} cases
            
            This represents significant operational improvement and cost efficiency.
            """
        
        elif result.function_name == 'analyze_capacity_impact_optimized':
            data = result.result_data
            
            # Base capacity analysis
            response = f"""
            **Enhanced Capacity Impact Analysis with Staffing Requirements**
            
            **Timeline Impact:**
            • **Time Saved**: {data['weeks_saved']} weeks
            • **Patients Benefiting**: {data['patients_moved_earlier']} moved to earlier slots
            • **Current Timeline**: {data['baseline_weeks']} weeks to clear backlog
            • **New Timeline**: {data['new_weeks']} weeks to clear backlog
            • **Efficiency Gain**: {data['efficiency_gain']}%
            
            **Capacity Analysis:**
            • **Current Capacity**: {data['baseline_capacity']} patients/week
            • **Target Capacity**: {data['new_capacity']} patients/week
            • **Additional Weekly Capacity**: +{data['additional_capacity']} patients
            """
            
            # Add staffing analysis if available
            if 'current_doctors' in data:
                response += f"""
            
            **Current Staffing Analysis:**
            • **Current Doctors**: {data['current_doctors']} FTE
            • **Current Nurses**: {data['current_nurses']} FTE
            • **Current Ratios**: {data['patients_per_doctor']:.0f} patients/doctor, {data['patients_per_nurse']:.0f} patients/nurse
            
            **Staffing Options to Achieve Target Capacity:**
            
            **Option 1: Doctors Only**
            • Additional Doctors Needed: +{data['additional_doctors_needed']:.1f} FTE
            • Annual Cost: £{data['cost_doctors_only']:,.0f}
            • Benefits: High clinical expertise, complex case handling
            
            **Option 2: Nurses Only**
            • Additional Nurses Needed: +{data['additional_nurses_needed']:.1f} FTE
            • Annual Cost: £{data['cost_nurses_only']:,.0f}
            • Benefits: Cost-effective, excellent patient care coverage
            
            **Option 3: Mixed Approach (Recommended)**
            • Additional Doctors: +{data['mixed_doctors']:.1f} FTE
            • Additional Nurses: +{data['mixed_nurses']:.1f} FTE
            • Annual Cost: £{data['cost_mixed']:,.0f}
            • Benefits: Balanced approach, optimal skill mix, sustainable growth
            
            **Recommendation**: The mixed approach provides the best balance of clinical expertise, cost-effectiveness, and sustainable capacity growth.
            """
            
            return response
        
        elif isinstance(result.result_data, pd.DataFrame):
            # Format pandas DataFrame results
            if len(result.result_data) > 0:
                return f"**Query Results**: Found {len(result.result_data)} records\n\n" + \
                       result.result_data.to_string(index=False)
            else:
                return "No results found for your query."
        
        return "Query processed successfully."