import os
import librosa
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import re
from collections import defaultdict, Counter
import json
from typing import Dict, List, Optional, Tuple, Any

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available. Install with: pip install praat-parselmouth")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Install with: pip install noisereduce")

# Custom tool class since BaseTool is not available
class BaseTool(BaseModel):
    name: str
    description: str
    processor: Any = Field(...)
    
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)
    
    def _run(self, *args, **kwargs):
        pass

# Input/Output Models for CrewAI
class AudioProcessingInput(BaseModel):
    root_folder: str = Field(description="Path to the audio dataset")
    max_files: Optional[int] = Field(default=None, description="Maximum number of files to process")
    output_csv: str = Field(default="audio_features.csv", description="Output CSV filename")

class AgenticAudioProcessor:
    """
    Core audio processing system that extracts features
    """
    
    def __init__(self):
        self.supported_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.aac', '.ogg'}
        self.dataset_analysis = {}
        self.folder_structure = {}
        self.labeling_strategy = None
        self.emotion_mapping = {}
        
    def analyze_folder_structure(self, root_folder: str) -> Dict[str, Any]:
        """
        Intelligently analyze folder structure to understand dataset organization
        """
        print("Analyzing folder structure...")
        
        structure_info = {
            'total_audio_files': 0,
            'folder_levels': {},
            'file_patterns': [],
            'emotion_indicators': [],
            'speaker_indicators': [],
            'folder_names': [],
            'sample_files': []
        }
        
        # Walk through all folders
        for root, dirs, files in os.walk(root_folder):
            level = root.replace(root_folder, '').count(os.sep)
            folder_name = os.path.basename(root)
            
            if folder_name:  # Not root folder
                structure_info['folder_names'].append(folder_name.lower())
                
            if level not in structure_info['folder_levels']:
                structure_info['folder_levels'][level] = {'folders': [], 'file_count': 0}
            
            structure_info['folder_levels'][level]['folders'].append(folder_name)
            
            # Analyze audio files in current folder
            audio_files = [f for f in files if Path(f).suffix.lower() in self.supported_extensions]
            structure_info['total_audio_files'] += len(audio_files)
            structure_info['folder_levels'][level]['file_count'] += len(audio_files)
            
            # Sample files for pattern analysis
            if audio_files and len(structure_info['sample_files']) < 20:
                for file in audio_files[:5]:  # Sample up to 5 files per folder
                    structure_info['sample_files'].append({
                        'filename': file,
                        'folder_path': root,
                        'folder_name': folder_name,
                        'level': level
                    })
        
        self.folder_structure = structure_info
        print(f"Found {structure_info['total_audio_files']} audio files across {len(structure_info['folder_names'])} folders")
        return structure_info
    
    def detect_emotion_patterns(self, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently detect emotion patterns in filenames and folder names
        """
        print("\n🧠 DETECTING EMOTION PATTERNS...")
        
        # Common emotion indicators
        emotion_keywords = {
            'happiness': ['happy', 'joy', 'joyful','joyfully', 'euphoric', 'pleasant', 'surprise', 'neutral'],
            'sadness': ['sad', 'sadness', 'melancholy', 'sorrow'],
            'anger': ['angry', 'anger', 'mad', 'rage', 'furious'],
            'fear': ['fear', 'afraid', 'scared', 'terror'],
            'disgust': ['disgust', 'disgusted', 'revulsion']
        }
        
        detected_patterns = {
            'folder_emotions': defaultdict(list),
            'filename_emotions': defaultdict(list),
            'emotion_codes': defaultdict(list),
            'speaker_codes': defaultdict(list)
        }
        
        # Analyze folder names
        for folder_name in structure_info['folder_names']:
            for emotion_category, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in folder_name:
                        detected_patterns['folder_emotions'][emotion_category].append(folder_name)
        
        # Analyze filenames
        for file_info in structure_info['sample_files']:
            filename = file_info['filename'].lower()
            folder_name = file_info['folder_name'].lower()
            
            # Check for emotion keywords in filename
            for emotion_category, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in filename or keyword in folder_name:
                        detected_patterns['filename_emotions'][emotion_category].append(filename)
            
            # Look for pattern-based codes
            # Pattern 1: emotion letters (a, d, f, h, sa)
            if re.search(r'[_\-]([adfs]|sa|h)[_\-]', filename) or re.search(r'^([adfs]|sa|h)[_\-]', filename):
                detected_patterns['emotion_codes']['letter_based'].append(filename)
            
            # Pattern 2: numeric emotion codes (01-08)
            if re.search(r'\d{2}-\d{2}-(\d{2})-', filename):
                code_match = re.search(r'\d{2}-\d{2}-(\d{2})-', filename)
                if code_match:
                    emotion_code = code_match.group(1)
                    detected_patterns['emotion_codes'][f'numeric_{emotion_code}'].append(filename)
            
            # Pattern 3: speaker codes
            if re.search(r'(OAF|YAF|OAM|YAM)', filename):
                speaker_match = re.search(r'(OAF|YAF|OAM|YAM)', filename)
                detected_patterns['speaker_codes'][speaker_match.group(1)].append(filename)
        
        return detected_patterns
    
    def determine_labeling_strategy(self, structure_info: Dict[str, Any], emotion_patterns: Dict[str, Any]) -> Tuple[str, int, Dict[str, Any]]:
        """
        Intelligently determine the best labeling strategy based on detected patterns
        """
        print("\n🎯 DETERMINING LABELING STRATEGY...")
        
        strategies = []
        
        # Strategy 1: Folder-based emotions (TESS style)
        if emotion_patterns['folder_emotions']:
            score = sum(len(files) for files in emotion_patterns['folder_emotions'].values())
            strategies.append(('folder_emotions', score, emotion_patterns['folder_emotions']))
            print(f"   Found folder-based emotions: {dict(emotion_patterns['folder_emotions'])}")
        
        # Strategy 2: Filename letter codes (SAVEE style)
        if emotion_patterns['emotion_codes'].get('letter_based'):
            score = len(emotion_patterns['emotion_codes']['letter_based'])
            strategies.append(('letter_codes', score, emotion_patterns['emotion_codes']))
            print(f"   Found letter-based emotion codes in {score} files")
        
        # Strategy 3: Numeric emotion codes (RAVDESS style)
        numeric_codes = [k for k in emotion_patterns['emotion_codes'].keys() if k.startswith('numeric_')]
        if numeric_codes:
            score = sum(len(emotion_patterns['emotion_codes'][code]) for code in numeric_codes)
            strategies.append(('numeric_codes', score, emotion_patterns['emotion_codes']))
            print(f"   Found numeric emotion codes: {[code.replace('numeric_', '') for code in numeric_codes]}")
        
        # Strategy 4: Fixed filename patterns (custom dataset style)
        fixed_names = ['euphoric.wav', 'joyfully.wav', 'sad.wav']
        fixed_name_score = 0
        for file_info in structure_info['sample_files']:
            if any(fixed_name in file_info['filename'].lower() for fixed_name in ['euphoric', 'joyfully', 'surprised', 'sad']):
                fixed_name_score += 1
        if fixed_name_score > 0:
            strategies.append(('fixed_names', fixed_name_score, fixed_names))
            print(f"   Found fixed naming pattern in {fixed_name_score} files")
        
        # Choose best strategy
        if strategies:
            best_strategy = max(strategies, key=lambda x: x[1])
            self.labeling_strategy = best_strategy[0]
            print(f"\n✅ Selected strategy: {best_strategy[0]} (confidence: {best_strategy[1]} matches)")
            return best_strategy
        else:
            print("\n⚠️ No clear labeling strategy detected, using default heuristics")
            self.labeling_strategy = 'heuristic'
            return ('heuristic', 0, {})
    
    def assign_depression_label(self, filename: str, folder_name: str = "", file_path: str = "") -> Optional[Tuple[int, str, str]]:
        """
        Intelligently assign depression labels based on detected strategy
        """
        filename_lower = filename.lower()
        folder_lower = folder_name.lower() if folder_name else ""
        
        if self.labeling_strategy == 'folder_emotions':
            return self._label_by_folder_emotion(folder_lower)
        elif self.labeling_strategy == 'letter_codes':
            return self._label_by_letter_codes(filename_lower)
        elif self.labeling_strategy == 'numeric_codes':
            return self._label_by_numeric_codes(filename_lower)
        elif self.labeling_strategy == 'fixed_names':
            return self._label_by_fixed_names(filename_lower)
        else:
            return self._label_by_heuristics(filename_lower, folder_lower)
    
    def _label_by_folder_emotion(self, folder_name: str) -> Tuple[int, str, str]:
        """Label based on folder emotion detection"""
        # Not depressed emotions
        if any(keyword in folder_name for keyword in ['neutral', 'happy', 'joy', 'pleasant', 'surprise']):
            emotion = 'positive'
            for keyword in ['neutral', 'happy', 'joy', 'pleasant', 'surprise']:
                if keyword in folder_name:
                    emotion = keyword
                    break
            return 0, "not_depressed", emotion
        
        # Depressed emotions
        if any(keyword in folder_name for keyword in ['angry', 'sad', 'fear', 'disgust']):
            emotion = 'negative'
            for keyword in ['angry', 'sad', 'fear', 'disgust']:
                if keyword in folder_name:
                    emotion = keyword
                    break
            return 1, "depressed", emotion
        
        return 0, "not_depressed", "unknown"
    
    def _label_by_letter_codes(self, filename: str) -> Tuple[int, str, str]:
        """Label based on letter emotion codes (SAVEE style)"""
        # Check for happiness (not depressed)
        if re.search(r'[_\-]h[_\-]|^h[_\-]|[_\-]h\.|^h\.', filename):
            return 0, "not_depressed", "happiness"
        
        # Check for sadness (check 'sa' before 'a')
        if re.search(r'[_\-]sa[_\-]|^sa[_\-]|[_\-]sa\.|^sa\.', filename):
            return 1, "depressed", "sadness"
        
        # Check for anger
        if re.search(r'[_\-]a[_\-]|^a[_\-]|[_\-]a\.|^a\.', filename):
            return 1, "depressed", "anger"
        
        # Check for disgust
        if re.search(r'[_\-]d[_\-]|^d[_\-]|[_\-]d\.|^d\.', filename):
            return 1, "depressed", "disgust"
        
        # Check for fear
        if re.search(r'[_\-]f[_\-]|^f[_\-]|[_\-]f\.|^f\.', filename):
            return 1, "depressed", "fear"
        
        return 1, "depressed", "unknown"
    
    def _label_by_numeric_codes(self, filename):
        """Label based on numeric emotion codes (RAVDESS style)"""
        match = re.search(r'\d{2}-\d{2}-(\d{2})-(\d{2})-(\d{2})-(\d{2}-(\d{2}))', filename)
        if match:
            code = filename.split('-')[2]
            if code in ['01', '02', '03']:  # positive emotions
                return 0, "not_depressed", f"emotion_code_{code}"
            elif code in ['04', '05', '06', '07']:  # negative emotions
                return 1, "depressed", f"emotion_code_{code}"
            elif code == '08':  # skip
                return None
        
        return 1, "depressed", "unknown"
    
    def _label_by_fixed_names(self, filename):
        """Label based on fixed filename patterns"""
        base_name = os.path.splitext(filename)[0]
        
        if base_name == 'sad':
            return 1, "depressed", "sad"
        elif base_name in ['euphoric', 'joyfully', 'surprised']:
            return 0, "not_depressed", base_name
        
        return 0, "not_depressed", "unknown"
    
    def _label_by_heuristics(self, filename: str, folder_name: str) -> Tuple[int, str, str]:
        """Fallback heuristic labeling"""
        text_to_check = f"{filename} {folder_name}".lower()
        
        # Positive indicators
        if any(word in text_to_check for word in ['happy', 'joy', 'neutral', 'pleasant', 'surprise', 'euphoric']):
            return 0, "not_depressed", "positive_heuristic"
        
        # Negative indicators
        if any(word in text_to_check for word in ['sad', 'angry', 'fear', 'disgust', 'negative']):
            return 1, "depressed", "negative_heuristic"
        
        return 1, "depressed", "default_heuristic"
    
    def extract_audio_features(self, file_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=22050, duration=30)
            
            if len(y) == 0:
                return None
            
            # Apply noise reduction if available
            if NOISEREDUCE_AVAILABLE:
                y = nr.reduce_noise(y=y, sr=sr)
            
            duration = len(y) / sr
            features = {}
            
            # 1. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 2. Energy Features
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy_mean'] = np.mean(rms)
            features['rms_energy_std'] = np.std(rms)
            
            # 3. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # 4. Rhythm Features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 5. Voice Quality Features (if parselmouth available)
            if PARSELMOUTH_AVAILABLE:
                try:
                    sound = parselmouth.Sound(file_path)
                    pitch = sound.to_pitch(time_step=0.01, pitch_floor=50.0, pitch_ceiling=800.0)
                    pitch_values_praat = pitch.selected_array['frequency']
                    pitch_values_praat = pitch_values_praat[pitch_values_praat != 0]
                    
                    # Jitter, Shimmer, HNR calculations with fallbacks
                    try:
                        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 50, 800)
                        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                        
                        features['jitter'] = jitter if not (np.isnan(jitter) or np.isinf(jitter)) else 0
                        features['shimmer'] = shimmer if not (np.isnan(shimmer) or np.isinf(shimmer)) else 0
                    except:
                        features['jitter'] = 0
                        features['shimmer'] = 0
                    
                    try:
                        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 50, 0.1, 1.0)
                        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
                        features['hnr'] = hnr if not (np.isnan(hnr) or np.isinf(hnr)) else 0
                    except:
                        features['hnr'] = 0
                        
                except:
                    features['jitter'] = 0
                    features['shimmer'] = 0
                    features['hnr'] = 0
            else:
                features['jitter'] = 0
                features['shimmer'] = 0
                features['hnr'] = 0
            
            # 6. Silence and Pause Features
            energy_threshold = np.percentile(rms, 20)
            silence_frames = np.sum(rms < energy_threshold)
            total_frames = len(rms)
            features['silence_ratio'] = silence_frames / total_frames if total_frames > 0 else 0
            
            non_silent_duration = duration * (1 - features['silence_ratio'])
            features['speaking_rate'] = non_silent_duration / duration if duration > 0 else 0
           
            # Basic info
            features['duration_seconds'] = duration
            features['sample_rate'] = sr
            
            return features
            
        except Exception as e:
            print(f"    Error extracting features from {os.path.basename(file_path)}: {e}")
            return None
    
    def process_all_audio_files(self, root_folder, output_csv="agentic_audio_features.csv", max_files=None):
        """
        Main processing function that uses AI-like intelligence to process all audio files
        """
        print("🤖 AGENTIC AUDIO PROCESSING SYSTEM")
        print("=" * 60)
        print("Using AI-driven analysis to automatically detect and process your audio dataset")
        
        if not os.path.exists(root_folder):
            print(f"❌ Root folder not found: {root_folder}")
            return None
        
        # Step 1: Analyze folder structure
        structure_info = self.analyze_folder_structure(root_folder)
        
        # Step 2: Detect emotion patterns
        emotion_patterns = self.detect_emotion_patterns(structure_info)
        
        # Step 3: Determine labeling strategy
        strategy_info = self.determine_labeling_strategy(structure_info, emotion_patterns)
        
        print(f"\n📊 DATASET ANALYSIS COMPLETE")
        print(f"   Total audio files found: {structure_info['total_audio_files']}")
        print(f"   Folder levels: {len(structure_info['folder_levels'])}")
        print(f"   Selected labeling strategy: {self.labeling_strategy}")
        
        # Step 4: Process all files
        print(f"\n🎵 PROCESSING AUDIO FILES...")
        
        all_features = []
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        file_counter = 0
        
        # Walk through all folders and process audio files
        for root, dirs, files in os.walk(root_folder):
            folder_name = os.path.basename(root)
            
            audio_files = [f for f in files if Path(f).suffix.lower() in self.supported_extensions]
            
            for audio_file in audio_files:
                if max_files and file_counter >= max_files:
                    break
                
                file_counter += 1
                file_path = os.path.join(root, audio_file)
                relative_path = os.path.relpath(root, root_folder)
                
                print(f"\n[{file_counter}/{min(max_files or structure_info['total_audio_files'], structure_info['total_audio_files'])}] Processing: {audio_file}")
                print(f"   Path: {relative_path}")
                
                # Get label using intelligent strategy
                label_result = self.assign_depression_label(audio_file, folder_name, file_path)
                
                if label_result is None:
                    print(f"   ⏭️ Skipped (emotion code 08 or invalid)")
                    skipped_count += 1
                    continue
                
                depression_label, label_text, emotion = label_result
                print(f"   Detected: {emotion} → {label_text} ({depression_label})")
                
                # Extract features
                print(f"   Extracting features...")
                features = self.extract_audio_features(file_path)
                
                if features is not None:
                    # Add metadata
                    features['filename'] = audio_file
                    features['folder_name'] = folder_name
                    features['relative_path'] = relative_path
                    features['full_path'] = file_path
                    features['detected_emotion'] = emotion
                    features['depression_label'] = depression_label
                    features['depression_label_text'] = label_text
                    features['labeling_strategy'] = self.labeling_strategy
                    features['file_index'] = file_counter
                    
                    all_features.append(features)
                    processed_count += 1
                    print(f"   ✅ Success! Features: {len([k for k in features.keys() if k not in ['filename', 'folder_name', 'relative_path', 'full_path', 'detected_emotion', 'depression_label', 'depression_label_text', 'labeling_strategy', 'file_index']])}")
                else:
                    print(f"   ❌ Failed to extract features")
                    error_count += 1
                    
            if max_files and file_counter >= max_files:
                break
        
        # Step 5: Save results and generate report
        if all_features:
            df = pd.DataFrame(all_features)
            df.to_csv(output_csv, index=False)
            
            print(f"\n🎉 PROCESSING COMPLETE!")
            print(f"   Output file: {output_csv}")
            print(f"   Dataset shape: {df.shape}")
            print(f"   Files processed: {processed_count}")
            print(f"   Files skipped: {skipped_count}")
            print(f"   Files with errors: {error_count}")
            
            # Generate intelligent analysis report
            self._generate_analysis_report(df, strategy_info)
            
            return df
        else:
            print("❌ No audio files were processed successfully!")
            return None
    
    def _generate_analysis_report(self, df: pd.DataFrame, strategy_info: Tuple[str, int, Dict[str, Any]]):
        """Generate intelligent analysis report"""
        print(f"\nINTELLIGENT DATASET ANALYSIS REPORT")
        print("=" * 50)
        
        # Label distribution
        label_counts = df['depression_label'].value_counts().sort_index()
        print(f"Depression Label Distribution:")
        print(f"   Not Depressed (0): {label_counts.get(0, 0)} samples ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"   Depressed (1): {label_counts.get(1, 0)} samples ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        # Class balance analysis
        if len(label_counts) == 2:
            balance_ratio = min(label_counts) / max(label_counts)
            balance_status = "Well balanced" if balance_ratio > 0.8 else "Imbalanced" if balance_ratio > 0.5 else "Highly imbalanced"
            print(f"   Class Balance Ratio: {balance_ratio:.3f} {balance_status}")
        
        # Feature quality assessment
        metadata_cols = ['file_index', 'filename', 'folder_name', 'relative_path', 'full_path', 
                        'depression_label', 'depression_label_text', 'labeling_method']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"\nFeature Quality Assessment:")
        print(f"   Total features extracted: {len(feature_cols)}")
        
        # Check for missing values
        missing_features = df[feature_cols].isnull().sum()
        if missing_features.sum() > 0:
            print(f"   Features with missing values: {(missing_features > 0).sum()}")
        else:
            print(f"   No missing values in features")
        
        print(f"\nLABELING STRATEGY:")
        print(f"   Strategy used: {self.labeling_strategy}")
        print(f"   Confidence level: {strategy_info[1]} pattern matches")


# CrewAI Tools
class FolderAnalysisTool(BaseTool):
    name: str = "folder_analysis"
    description: str = "Analyze folder structure and detect audio files in the dataset"
    processor: AgenticAudioProcessor = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, root_folder: str) -> str:
        try:
            structure_info = self.processor.analyze_folder_structure(root_folder)
            return f"Found {structure_info['total_audio_files']} audio files in {len(structure_info['folder_names'])} folders"
        except Exception as e:
            return f"Error analyzing folder: {str(e)}"

class FeatureExtractionTool(BaseTool):
    name: str = "feature_extraction"
    description: str = "Extract audio features from files and create dataset"
    processor: AgenticAudioProcessor = Field(...)
    
    class Config:
        arbitrary_types_allowed = True

    def _run(self, root_folder: str, output_csv: str = "features.csv", max_files: Optional[int] = None) -> str:
        try:
            df = self.processor.process_all_audio_files(root_folder, output_csv, max_files)
            if df is not None:
                return f"Successfully extracted features from {len(df)} files and saved to {output_csv}"
            else:
                return "Failed to extract features"
        except Exception as e:
            return f"Error extracting features: {str(e)}"


# CrewAI Agents, Tasks, and Crew
def create_audio_processing_crew() -> Crew:
    """Create and configure the CrewAI crew for audio processing"""
    
    # Initialize processor
    processor = AgenticAudioProcessor()
    
    # Initialize tools
    folder_tool = FolderAnalysisTool(processor=processor)
    feature_tool = FeatureExtractionTool(processor=processor)
    
    # Create agents
    data_analyst = Agent(
        role='Audio Dataset Analyst',
        goal='Analyze and understand the structure of audio datasets for depression detection',
        backstory="""You are an expert in audio data analysis with deep knowledge of speech processing 
                     and emotion recognition. You specialize in understanding dataset structures and 
                     identifying optimal processing strategies.""",
        tools=[folder_tool],
        verbose=True
    )
    
    feature_engineer = Agent(
        role='Audio Feature Engineer',
        goal='Extract comprehensive audio features suitable for depression detection',
        backstory="""You are a specialist in audio signal processing and feature engineering. 
                     You know which acoustic features are most relevant for detecting depression 
                     in speech, including prosodic, spectral, and voice quality features.""",
        tools=[feature_tool],
        verbose=True
    )
    
    # Create tasks
    analysis_task = Task(
        description="""Analyze the audio dataset structure at {root_folder} and provide insights about:
        1. Number of audio files and their organization
        2. Folder structure and naming patterns
        3. Potential labeling strategies for depression detection
        4. Data quality assessment""",
        expected_output="A comprehensive analysis report of the dataset structure and organization",
        agent=data_analyst
    )
    
    feature_extraction_task = Task(
        description="""Extract comprehensive audio features from the dataset at {root_folder}:
        1. Process audio files and extract relevant acoustic features
        2. Apply intelligent labeling for depression detection
        3. Create a structured dataset with features and labels
        4. Generate quality assessment report
        
        Parameters:
        - root_folder: {root_folder}
        - max_files: {max_files}
        - output_csv: {output_csv}""",
        expected_output="A CSV file containing extracted audio features with depression labels",
        agent=feature_engineer,
        context=[analysis_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[data_analyst, feature_engineer],
        tasks=[analysis_task, feature_extraction_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew


# Main execution functions
def run_audio_processing_pipeline(
    root_folder: str,
    max_files: Optional[int] = None,
    output_csv: str = "audio_features.csv"
) -> Dict[str, Any]:
    """
    Run the complete audio processing pipeline using CrewAI
    """
    
    print("CREWAI AUDIO PROCESSING PIPELINE")
    print("=" * 50)
    
    # Create the crew
    crew = create_audio_processing_crew()
    
    # Define inputs
    inputs = {
        'root_folder': root_folder,
        'max_files': max_files,
        'output_csv': output_csv
    }
    
    try:
        # Execute the crew
        result = crew.kickoff(inputs=inputs)
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        return {
            'success': True,
            'result': result,
            'inputs': inputs
        }
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'inputs': inputs
        }


# Example usage
def run_example():
    """
    Example of how to use the CrewAI audio processing system
    """
    
    # Configuration
    ROOT_FOLDER = r"C:\Users\Kruttika\OneDrive\Desktop\Kruttika\codessance\dataset"
    MAX_FILES = 9000  # Limit for testing
    
    print("STARTING CREWAI AUDIO PROCESSING")
    print("=" * 60)
    
    # Run the complete pipeline
    result = run_audio_processing_pipeline(
        root_folder=ROOT_FOLDER,
        max_files=MAX_FILES,
        output_csv="audio_features.csv"
    )
    
    if result['success']:
        print(f"EXAMPLE COMPLETED SUCCESSFULLY!")
        print(f"Features saved to: audio_features.csv")
    else:
        print(f"Example failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("AUDIO PROCESSING SYSTEM - DIRECT MODE")
    print("=" * 60)
    
    # Direct processing without CrewAI
    processor = AgenticAudioProcessor()
    
    ROOT_FOLDER = r"C:\Users\Kruttika\OneDrive\Desktop\Kruttika\codessance\dataset"
    MAX_FILES = 8000
    
    # Process audio files
    print("Processing audio files...")
    df = processor.process_all_audio_files(ROOT_FOLDER, "audio_features.csv", MAX_FILES)
    
    if df is not None:
        print(f"SUCCESS! Features extracted and saved to: audio_features.csv")
        print(f"Dataset shape: {df.shape}")
    else:
        print("Audio processing failed")