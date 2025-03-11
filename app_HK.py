import sys
import os
from pathlib import Path
import datetime
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QLabel, QFileDialog, QTextEdit,
                           QProgressBar, QComboBox, QSpinBox, QGroupBox, QFormLayout,
                           QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)
    
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
            self.finished_signal.emit(result)
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False)

class TransSentiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TransSenti - Transit Sentiment Analysis Tool")
        self.setMinimumSize(800, 600)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Author information
        author_info = QLabel("by Dr. Lei Da, Prof. Sylvia He, Prof. Shuli Luo\nContact: greatradar@gmail.com\nThe Chinese University of Hong Kong")
        author_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author_info.setFont(QFont("Arial", 10))
        main_layout.addWidget(author_info)
        
        # Input/Output section
        io_group = QGroupBox("File Paths")
        io_layout = QFormLayout()
        
        # Input directory
        self.input_dir_layout = QHBoxLayout()
        self.input_dir_edit = QTextEdit()
        self.input_dir_edit.setMaximumHeight(28)
        self.input_dir_button = QPushButton("Browse...")
        self.input_dir_button.clicked.connect(self.select_input_dir)
        self.input_dir_layout.addWidget(self.input_dir_edit)
        self.input_dir_layout.addWidget(self.input_dir_button)
        io_layout.addRow("Original Data Directory:", self.input_dir_layout)
        
        # Prefiltered directory
        self.prefiltered_dir_layout = QHBoxLayout()
        self.prefiltered_dir_edit = QTextEdit()
        self.prefiltered_dir_edit.setMaximumHeight(28)
        self.prefiltered_dir_button = QPushButton("Browse...")
        self.prefiltered_dir_button.clicked.connect(self.select_prefiltered_dir)
        self.prefiltered_dir_layout.addWidget(self.prefiltered_dir_edit)
        self.prefiltered_dir_layout.addWidget(self.prefiltered_dir_button)
        io_layout.addRow("Prefiltered Results Directory:", self.prefiltered_dir_layout)
        
        # Cleaned directory
        self.cleaned_dir_layout = QHBoxLayout()
        self.cleaned_dir_edit = QTextEdit()
        self.cleaned_dir_edit.setMaximumHeight(28)
        self.cleaned_dir_button = QPushButton("Browse...")
        self.cleaned_dir_button.clicked.connect(self.select_cleaned_dir)
        self.cleaned_dir_layout.addWidget(self.cleaned_dir_edit)
        self.cleaned_dir_layout.addWidget(self.cleaned_dir_button)
        io_layout.addRow("Cleaned Results Directory:", self.cleaned_dir_layout)
        
        io_group.setLayout(io_layout)
        main_layout.addWidget(io_group)
        
        # API and Parameters section
        params_group = QGroupBox("API Settings & Processing Parameters")
        params_layout = QFormLayout()
        
        # API Key input
        self.api_key_edit = QLineEdit("sk-7c87ef2add054e439095db9b18c921e9")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setMinimumWidth(300)
        params_layout.addRow("API Key:", self.api_key_edit)
        
        # Base URL input
        self.base_url_edit = QLineEdit("https://api.deepseek.com")
        params_layout.addRow("Base URL:", self.base_url_edit)
        
        # Batch size
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 100)
        self.batch_size_spinner.setValue(50)
        params_layout.addRow("Batch Size:", self.batch_size_spinner)
        
        # API Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["deepseek-reasoner", "deepseek-chat", "deepseek-coder"])
        params_layout.addRow("Base LLM Model:", self.model_combo)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.prefilter_button = QPushButton("Run Keyword Prefiltering")
        self.prefilter_button.clicked.connect(self.run_prefiltering)
        buttons_layout.addWidget(self.prefilter_button)
        
        self.filter_button = QPushButton("Run LLM Filtering")
        self.filter_button.clicked.connect(self.run_filtering)
        buttons_layout.addWidget(self.filter_button)
        
        self.run_all_button = QPushButton("Run Complete Pipeline")
        self.run_all_button.clicked.connect(self.run_complete_pipeline)
        buttons_layout.addWidget(self.run_all_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)
        
        # Log display
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Set the main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Initialize default paths
        self.input_dir_edit.setText("Data/Twitter_Hong Kong")
        self.prefiltered_dir_edit.setText("prefiltered_results_HK")
        self.cleaned_dir_edit.setText("cleaned_results_HK")
        
        # Current worker thread
        self.worker = None
        
    def select_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir_edit.setText(directory)
            
    def select_prefiltered_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Prefiltered Directory")
        if directory:
            self.prefiltered_dir_edit.setText(directory)
            
    def select_cleaned_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Cleaned Directory")
        if directory:
            self.cleaned_dir_edit.setText(directory)
    
    def add_log_message(self, message):
        self.log_display.append(message)
        self.log_display.ensureCursorVisible()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def run_prefiltering(self):
        input_dir = self.input_dir_edit.toPlainText()
        prefiltered_dir = self.prefiltered_dir_edit.toPlainText()
        
        if not input_dir or not prefiltered_dir:
            self.add_log_message("Please select both input and prefiltered directories.")
            return
        
        self.add_log_message(f"Starting keyword prefiltering at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.progress_bar.setValue(0)
        
        # Disable buttons during processing
        self.prefilter_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        
        # Create worker thread for keyword prefiltering
        self.worker = WorkerThread(keyword_prefilter, input_dir, prefiltered_dir)
        self.worker.update_signal.connect(self.add_log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.prefiltering_finished)
        self.worker.start()
        
    def prefiltering_finished(self, has_results):
        self.add_log_message(f"Keyword prefiltering completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.progress_bar.setValue(100)
        
        # Re-enable buttons
        self.prefilter_button.setEnabled(True)
        self.filter_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        
        if has_results:
            self.add_log_message("Prefiltering found relevant posts that can now be processed with LLM filtering.")
        else:
            self.add_log_message("Warning: No posts matched the keyword prefiltering criteria.")
            
    def run_filtering(self):
        prefiltered_dir = self.prefiltered_dir_edit.toPlainText()
        cleaned_dir = self.cleaned_dir_edit.toPlainText()
        batch_size = self.batch_size_spinner.value()
        model = self.model_combo.currentText()
        api_key = self.api_key_edit.text()
        base_url = self.base_url_edit.text()
        
        if not prefiltered_dir or not cleaned_dir:
            self.add_log_message("Please select both prefiltered and cleaned directories.")
            return
        
        if not api_key:
            self.add_log_message("Please enter a valid API key.")
            return
        
        self.add_log_message(f"Starting LLM filtering at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_log_message(f"Using model: {model}, batch size: {batch_size}")
        self.progress_bar.setValue(0)
        
        # Disable buttons during processing
        self.prefilter_button.setEnabled(False)
        self.filter_button.setEnabled(False)
        self.run_all_button.setEnabled(False)
        
        # Create worker thread for LLM filtering
        self.worker = WorkerThread(datafilter, prefiltered_dir, cleaned_dir, batch_size, 
                                 api_key=api_key, base_url=base_url, model=model)
        self.worker.update_signal.connect(self.add_log_message)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.filtering_finished)
        self.worker.start()
        
    def filtering_finished(self, success):
        self.add_log_message(f"LLM filtering completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.progress_bar.setValue(100)
        
        # Re-enable buttons
        self.prefilter_button.setEnabled(True)
        self.filter_button.setEnabled(True)
        self.run_all_button.setEnabled(True)
        
    def run_complete_pipeline(self):
        input_dir = self.input_dir_edit.toPlainText()
        prefiltered_dir = self.prefiltered_dir_edit.toPlainText()
        cleaned_dir = self.cleaned_dir_edit.toPlainText()
        
        if not input_dir or not prefiltered_dir or not cleaned_dir:
            self.add_log_message("Please select all directories first.")
            return
        
        # Run prefiltering first
        self.run_prefiltering()
        
        # Connect the signal to trigger the next step
        self.worker.finished_signal.disconnect()
        self.worker.finished_signal.connect(self.prefilter_then_filter)
    
    def prefilter_then_filter(self, has_results):
        self.prefiltering_finished(has_results)
        
        if has_results:
            self.run_filtering()
        else:
            self.add_log_message("Skipping LLM filtering as no posts matched the prefiltering criteria.")
            # Re-enable buttons
            self.prefilter_button.setEnabled(True)
            self.filter_button.setEnabled(True)
            self.run_all_button.setEnabled(True)


# The following functions would be imported in actual implementation, but are included here for completeness

def keyword_prefilter(directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Define keywords for filtering in different languages
    keywords = [
        # Travel related
        "train", "travel", "trip", "旅行", "出行", "でんしゃ", "りょこう", "トリップ",
        "기차", "여행", "트립", "viaje", "excursión",

        # General transit terms
        "地铁", "地鐵", "metro", "subway", "métro", "地下鉄", "メトロ", "지하철", "전철", "지하철역",
        "港铁", "港鐵", "MTR", "轨道交通", "軌道交通", "rail", "transit",
        "transport ferroviaire", "transporte ferroviario",
        
        # Station names and other keywords would follow...
    ]
    
    total_posts = 0
    prefiltered_posts = 0
    
    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"prefiltered_{csv_file.name}"
        output_path = Path(output_directory) / output_filename
        
        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue
        
        print(f"\nPrefiltering file: {csv_file}")
        try:
            df = pd.read_csv(csv_file, index_col=0)
            df.reset_index(drop=False, inplace=True)
            df = df.rename(columns={df.columns[0]: "post_id"})
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
            
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_posts += len(df)
        
        # Filter posts containing any of the keywords
        filtered_posts = []
        
        for index, row in df.iterrows():
            text = row['text']
                
            # Case-insensitive keyword matching
            text_lower = text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                filtered_posts.append(row)
        
        prefiltered_posts += len(filtered_posts)
        
        if filtered_posts:
            output_df = pd.DataFrame(filtered_posts)
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved {len(filtered_posts)} prefiltered posts out of {len(df)} posts in this file.")
    
    print(f"\nSaved {prefiltered_posts} prefiltered posts out of {total_posts} total posts.")
    return prefiltered_posts > 0  # Return True if any posts were prefiltered

def datafilter(directory, output_directory, batch_size=50, api_key="sk-7c87ef2add054e439095db9b18c921e9", 
              base_url="https://api.deepseek.com", model="deepseek-reasoner"):
    from openai import OpenAI
    
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    total_post_counter = 0
    valid_post_counter = 0
    
    # Initialize the API client
    client = OpenAI(api_key=api_key, base_url=base_url)

    for csv_file in Path(directory).glob('*.csv'):
        output_filename = f"cleaned_{csv_file.name}"
        output_path = Path(output_directory) / output_filename

        if output_path.exists():
            print(f"File {output_filename} already exists. Skipping.")
            continue

        print(f"\nProcessing file: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Total rows in {csv_file.name}: {len(df)}")
        total_post_counter += len(df)

        valid_posts = []
        
        # Filter out posts that are not strings or too long
        valid_df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x) <= 512)].copy()
        
        # Process in batches
        for i in range(0, len(valid_df), batch_size):
            batch_df = valid_df.iloc[i:i+batch_size]
            
            # Format batch posts with original post_ids
            formatted_posts = "\n\n".join([f"Post {row['post_id']}: {row['text']}" for _, row in batch_df.iterrows()])
            
            print(f"Processing batch {i//batch_size + 1}/{(len(valid_df) + batch_size - 1)//batch_size}")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data filter"},
                    {"role": "user", "content": f"""Please evaluate the following {len(batch_df)} social media posts to determine if they are related to transit service quality, transit environment, or transit experience.
                    These may include aspects like reliability, crowdedness, comfort, safety and security, waiting conditions, service facilities, or travel experience.
                    Note that some posts might just mention transit keywords (metro, subway, etc.) without actually evaluating the transit system.

                    {formatted_posts}

                    Please answer in JSON format, with each post's original ID and whether it's relevant:
                    {{
                    "post[ID]": "yes/no",
                    "post[ID]": "yes/no",
                    ...
                    }}
                    Return only the JSON format."""},
                ],
                stream=False
            )

            try:
                response_text = response.choices[0].message.content.strip()
                print(f"Response received")
                
                # Extract JSON part if there's explanatory text
                if '{' in response_text and '}' in response_text:
                    json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                    results = eval(json_str)
                    
                    # Process results
                    for post_key, is_relevant in results.items():
                        post_id = post_key.replace("post", "")
                        if is_relevant.lower() == "yes":
                            matching_rows = batch_df[batch_df['post_id'] == int(post_id)]
                            if not matching_rows.empty:
                                valid_post_counter += 1
                                valid_posts.append(matching_rows.iloc[0].to_dict())
                else:
                    print(f"Invalid response format")
                    
            except Exception as e:
                print(f"Error processing batch response: {e}")

        if valid_posts:
            output_df = pd.DataFrame(valid_posts)
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nSaved {valid_post_counter} filtered posts out of {total_post_counter} total posts.")
    return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransSentiApp()
    window.show()
    sys.exit(app.exec())