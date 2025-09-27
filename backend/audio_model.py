# import xgboost as xgb  # Comment out if XGBoost not available
from sklearn.ensemble import GradientBoostingClassifier  # Alternative to XGBoost
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Any

from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key="AIzaSyB2kZcstDSK0aUBbKzyq-G9puCVB4nEnn4"  
)

class SharedContext:
    def __init__(self):
        self.features = None
        self.labels = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

# Instantiate the shared context
shared_context = SharedContext()

# ===========================================
# TOOLS FOR AGENTS TO USE (Using @tool decorator)
# ===========================================

@tool
def load_audio_features_csv(csv_path: str) -> str:
    """Load audio features from CSV file and prepare for training."""
    try:
        df = pd.read_csv(csv_path)
        
        # Auto-detect label column
        label_col = None
        for col in df.columns:
            if col.lower() in ['depression_label', 'label', 'target', 'class', 'depression']:
                label_col = col
                break
        
        if label_col is None:
            # Look for binary columns (0/1 values)
            for col in df.columns:
                if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
                    label_col = col
                    break
        
        if label_col is None:
            return "Error: Could not identify label column. Please ensure your CSV has a column named 'depression_label', 'label', or similar."
        
        print(f"Using label column: {label_col}")
        
        # Identify feature columns (exclude metadata and labels)
        metadata_cols = ['filename', 'folder_name', 'relative_path', 'full_path', 
                        'detected_emotion', 'depression_label_text', 'labeling_strategy', 
                        'file_index', label_col]
        
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Handle missing values
        df = df.dropna(subset=[label_col])
        
        # Clean tempo column if it's in list format
        if 'tempo' in feature_cols and df['tempo'].dtype == 'object':
            def extract_tempo(tempo_val):
                if isinstance(tempo_val, str) and '[' in tempo_val:
                    # Extract number from string like '[143.5546875]'
                    import re
                    match = re.findall(r'[\d.]+', tempo_val)
                    return float(match[0]) if match else 0
                return float(tempo_val) if tempo_val else 0
            df['tempo'] = df['tempo'].apply(extract_tempo)
        
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Store in shared context
        shared_context.features = df[feature_cols].values
        shared_context.labels = df[label_col].astype(int).values
        shared_context.feature_columns = feature_cols
        
        return f"Successfully loaded {len(df)} samples with {len(feature_cols)} audio features. Label column: {label_col}"
        
    except Exception as e:
        return f"Error loading CSV: {str(e)}"

@tool
def train_audio_depression_model() -> str:
    """Train XGBoost model on audio features for depression detection."""
    try:
        if shared_context.features is None or shared_context.labels is None:
            return "Missing features or labels. Run CSV loading first."
        
        features = shared_context.features
        labels = shared_context.labels
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Class distribution: {class_dist}")
        
        # Scale features
        features_scaled = shared_context.scaler.fit_transform(features)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Store test set in shared context for evaluation
        shared_context.X_test = X_test
        shared_context.y_test = y_test
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = counts[0] / counts[1] if len(counts) == 2 else 1
        
        # Train model with parameters optimized for audio features
        # Using GradientBoostingClassifier as alternative to XGBoost
        model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        # If XGBoost is available, use it instead
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                random_state=42,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8
            )
            print("Using XGBoost classifier")
        except ImportError:
            print("XGBoost not available, using GradientBoostingClassifier")
        
        model.fit(X_train, y_train)
        
        # Validate on test set
        val_pred_proba = model.predict_proba(X_test)[:, 1]
        val_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, val_pred_proba)
        
        # Store model
        shared_context.model = model
        
        # Save model and scaler to disk
        with open("audio_depression_model.pkl", "wb") as f:
            pickle.dump({
                'model': model,
                'scaler': shared_context.scaler,
                'feature_columns': shared_context.feature_columns
            }, f)
        
        # Generate classification report
        report = classification_report(y_test, val_pred, target_names=['Not Depressed', 'Depressed'])
        
        return f"Audio depression model trained successfully!\nTest AUC: {auc:.4f}\nClass distribution: {class_dist}\nTrain set size: {len(X_train)}, Test set size: {len(X_test)}\nModel and scaler saved to disk.\n\nTest Set Classification Report:\n{report}"
        
    except Exception as e:
        return f"Error training model: {str(e)}"

@tool
def predict_audio_depression_risk() -> str:
    """Predict depression risk on the test set from the train-test split."""
    try:
        if shared_context.model is None:
            return "No trained model available. Run training first."
        
        # Use the test set that was created during training
        if not hasattr(shared_context, 'X_test') or not hasattr(shared_context, 'y_test'):
            return "No test set available. The model needs to be retrained to create test predictions."
        
        # Make predictions on test set
        test_risks = shared_context.model.predict_proba(shared_context.X_test)[:, 1]
        test_predictions = shared_context.model.predict(shared_context.X_test)
        
        # Calculate test set performance
        test_auc = roc_auc_score(shared_context.y_test, test_risks)
        test_report = classification_report(shared_context.y_test, test_predictions, target_names=['Not Depressed', 'Depressed'])
        
        # Get feature importances for explanation
        feature_importance = shared_context.model.feature_importances_
        top_features = sorted(zip(shared_context.feature_columns, feature_importance), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        # Generate results for first 10 test samples
        results = []
        for i in range(min(10, len(shared_context.X_test))):
            risk = test_risks[i]
            pred = test_predictions[i]
            actual = shared_context.y_test[i]
            risk_level = "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.3 else "LOW"
            
            status = "✓ Correct" if pred == actual else "✗ Incorrect"
            results.append(f"Test Sample {i+1}: Risk: {risk:.3f} ({risk_level}) | Predicted: {'Depressed' if pred == 1 else 'Not Depressed'} | Actual: {'Depressed' if actual == 1 else 'Not Depressed'} | {status}")
        
        # Summary statistics
        high_risk_count = sum(test_risks > 0.7)
        medium_risk_count = sum((test_risks > 0.3) & (test_risks <= 0.7))
        low_risk_count = sum(test_risks <= 0.3)
        
        summary = f"TEST SET EVALUATION RESULTS\n"
        summary += "=" * 50 + "\n"
        summary += f"Test Set Size: {len(shared_context.X_test)} samples\n"
        summary += f"Test AUC: {test_auc:.4f}\n\n"
        summary += f"Risk Distribution:\n"
        summary += f"High risk (>0.7): {high_risk_count} samples\n"
        summary += f"Medium risk (0.3-0.7): {medium_risk_count} samples\n"
        summary += f"Low risk (<0.3): {low_risk_count} samples\n\n"
        summary += f"Most important features: {', '.join([f[0] for f in top_features])}\n\n"
        summary += "Sample Test Predictions:\n" + "\n".join(results) + "\n\n"
        summary += "Detailed Classification Report:\n" + test_report
        
        return summary
        
    except Exception as e:
        return f"Error making test predictions: {str(e)}"

@tool
def analyze_model_performance() -> str:
    """Analyze the trained model's feature importance and performance."""
    try:
        if shared_context.model is None or shared_context.feature_columns is None:
            return "No trained model available for analysis."
        
        # Feature importance analysis
        feature_importance = shared_context.model.feature_importances_
        feature_ranking = sorted(zip(shared_context.feature_columns, feature_importance), 
                               key=lambda x: x[1], reverse=True)
        
        analysis = "MODEL PERFORMANCE ANALYSIS\n"
        analysis += "=" * 50 + "\n\n"
        
        analysis += "Top 10 Most Important Features:\n"
        for i, (feature, importance) in enumerate(feature_ranking[:10]):
            analysis += f"{i+1:2d}. {feature:<25} {importance:.4f}\n"
        
        analysis += f"\nTotal features used: {len(shared_context.feature_columns)}\n"
        analysis += f"Model type: XGBoost Classifier\n"
        
        # Audio-specific feature categories
        audio_categories = {
            'pitch': ['pitch_mean', 'pitch_std'],
            'energy': ['rms_energy_mean', 'rms_energy_std'],
            'voice_quality': ['jitter', 'shimmer', 'hnr'],
            'temporal': ['duration_seconds', 'tempo', 'speaking_rate', 'silence_ratio'],
            'spectral': ['zcr_mean', 'zcr_std']
        }
        
        analysis += "\nFeature Category Analysis:\n"
        for category, features in audio_categories.items():
            category_features = [f for f in features if f in shared_context.feature_columns]
            if category_features:
                avg_importance = np.mean([dict(feature_ranking)[f] for f in category_features])
                analysis += f"{category.capitalize():<15} features: {avg_importance:.4f} avg importance\n"
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing model: {str(e)}"

# ===========================================
# AUTONOMOUS AGENTS
# ===========================================

data_loader = Agent(
    role="Audio Data Loading Specialist",
    goal="Load and prepare audio feature datasets for machine learning",
    backstory="""You are an expert in audio signal processing and feature engineering. 
    You load CSV files containing extracted audio features and prepare them for 
    depression detection model training.""",
    tools=[load_audio_features_csv],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

model_trainer = Agent(
    role="Audio ML Engineer",
    goal="Train robust depression detection models using audio features",
    backstory="""You are an experienced ML engineer specializing in audio-based 
    classification tasks. You train XGBoost models on audio features and optimize 
    them for depression detection with proper handling of class imbalance.""",
    tools=[train_audio_depression_model, analyze_model_performance],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

risk_predictor = Agent(
    role="Audio Depression Risk Assessment Specialist",
    goal="Analyze audio features and provide accurate depression risk predictions",
    backstory="""You are a specialist in audio-based mental health assessment using AI. 
    You analyze audio feature patterns to provide careful, accurate risk assessments 
    while being sensitive to the mental health domain.""",
    tools=[predict_audio_depression_risk],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ===========================================
# AUTONOMOUS TASKS
# ===========================================

data_loading_task = Task(
    description="""Load the audio features CSV file using the load_audio_features_csv tool.
    The CSV should contain extracted audio features like pitch, energy, voice quality, 
    and temporal features along with depression labels.
    
    Use csv_path parameter to specify the path to your audio features CSV file.""",
    expected_output="Confirmation that audio features and labels have been loaded successfully with feature count and sample count",
    agent=data_loader
)

model_training_task = Task(
    description="""Using the loaded audio features from the previous task, train an XGBoost 
    classifier for depression detection using the train_audio_depression_model tool.
    
    Then analyze the model performance using the analyze_model_performance tool to 
    understand which audio features are most important for depression detection.""",
    expected_output="Training completion with validation metrics, feature importance analysis, and model save confirmation",
    agent=model_trainer
)

prediction_task = Task(
    description="""Using the trained model from the previous task, evaluate the model performance 
    on the test set using the predict_audio_depression_risk tool.
    
    This will show how well the model performs on unseen data and provide detailed 
    predictions with risk scores and feature importance explanations.""",
    expected_output="Test set evaluation with performance metrics, risk analysis, and sample predictions with actual vs predicted comparisons",
    agent=risk_predictor
)

# ===========================================
# AUTONOMOUS CREW
# ===========================================

audio_depression_crew = Crew(
    agents=[data_loader, model_trainer, risk_predictor],
    tasks=[data_loading_task, model_training_task, prediction_task],
    verbose=True
)

# ===========================================
# MAIN EXECUTION FUNCTIONS
# ===========================================

def run_audio_depression_pipeline(csv_path: str):
    """Run the complete audio depression detection pipeline with train-test split"""
    print("Starting Audio-based Depression Detection System...")
    print("=" * 60)
    
    # Update the data loading task with the actual CSV path
    data_loading_task.description = f"""Load the audio features CSV file using the load_audio_features_csv tool 
    with csv_path="{csv_path}".
    
    The CSV should contain extracted audio features like pitch, energy, voice quality, 
    and temporal features along with depression labels. The system will automatically 
    perform train-test split on this data."""
    
    try:
        # Run the autonomous crew
        result = audio_depression_crew.kickoff()
        
        print("\n" + "=" * 60)
        print("AUDIO DEPRESSION ANALYSIS COMPLETE!")
        print("=" * 60)
        print(result)
        
        return result
        
    except Exception as e:
        print(f"Error in audio depression analysis: {str(e)}")
        return None

def predict_new_audio_samples(csv_path: str):
    """Predict depression risk for new audio samples"""
    print(f"\nAnalyzing new audio samples from: {csv_path}")
    
    # Create a quick prediction task
    prediction_task_new = Task(
        description=f"Analyze audio samples for depression risk using the predict_audio_depression_risk tool with csv_path='{csv_path}'",
        expected_output="Risk scores and explanations for the provided audio samples",
        agent=risk_predictor
    )
    
    # Create temporary crew for prediction
    prediction_crew = Crew(
        agents=[risk_predictor],
        tasks=[prediction_task_new],
        verbose=True
    )
    
    result = prediction_crew.kickoff()
    return result

# ===========================================
# EXAMPLE USAGE
# ===========================================

if __name__ == "__main__":
    # Configuration - Update this path to your actual CSV file
    CSV_PATH = r"C:\Users\Kruttika\OneDrive\Desktop\Kruttika\codessance\Code-Blooded_Codeissance\data\audio_features_cleaned.csv"  # Your CSV with features and labels
    
    print("Audio-based Depression Detection with CrewAI")
    print("=" * 60)
    
    # Run the complete pipeline with train-test split
    result = run_audio_depression_pipeline(CSV_PATH)
    
    if result:
        print("\nModel training and evaluation completed successfully!")
        print("The model has been trained on 80 of your data and evaluated on the remaining 20%.")
    else:
        print("Pipeline execution failed.")