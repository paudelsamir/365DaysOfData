# Implementing 69 percent of NLP with sentiment analysis

*Just Implementation*

## Table of Contents

- [🌐 Phase 1: Data Setup & Cleaning](#phase-1-data-setup--cleaning)
  - [Data Loading](#data-loading)
  - [Basic Cleaning Pipeline](#basic-cleaning-pipeline)
  - [Text Analysis](#text-analysis)
- [🌐 Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
  - [Basic Stats](#basic-stats)
  - [Word Analysis](#word-analysis)
  - [Advanced EDA](#advanced-eda)
- [🌐 Phase 3: Vectorization & Embeddings](#phase-3-vectorization--embeddings)
  - [Non-Semantic Vectors](#non-semantic-vectors)
  - [Word2Vec Implementation](#word2vec-implementation)
  - [Pre-trained Embeddings](#pre-trained-embeddings)
  - [BERT Embeddings](#bert-embeddings)
  - [FastText Implementation](#fasttext-implementation)
- [🌐 Phase 4: Traditional Neural Networks [Revision]](#phase-4-traditional-neural-networks)
  - [PyTorch Setup](#pytorch-setup)
  - [Simple MLP](#simple-mlp)
  - [LSTM Implementation](#lstm-implementation)
  - [BiLSTM Implementation](#bilstm-implementation)
  - [CNN for Text](#cnn-for-text)
  - [GRU Implementation](#gru-implementation)
  - [Training Pipeline](#training-pipeline)
- [🌐 Phase 5: Advanced Architectures](#phase-5-advanced-architectures)
  - [BERT Fine-tuning](#bert-fine-tuning)
  - [RoBERTa Implementation](#roberta-implementation)
  - [DistilBERT Implementation](#distilbert-implementation)
  - [Custom Transformer](#custom-transformer)
  - [Prompt Engineering](#prompt-engineering)
  - [RAG Implementation (giving more resources)](#rag-implementation)
- [🌐 Phase 6: Visualization & Monitoring](#phase-6-visualization--monitoring)
  - [TensorBoard Setup](#tensorboard-setup)
  - [Model Analysis](#model-analysis)
  - [Error Analysis](#error-analysis)
  - [Model Comparison](#model-comparison)
  - [Advanced Metrics](#advanced-metrics)
- [🌐 Phase 7: Deployment & Production](#phase-7-deployment--production)
  - [Model Optimization](#model-optimization)
  - [API Development](#api-development)
  - [Docker Setup](#docker-setup)
  - [Web Interface](#web-interface)
  - [Monitoring Setup](#monitoring-setup)
  - [Testing Pipeline](#testing-pipeline)
  - [Deployment](#deployment)


> [!NOTE]
> This roadmap reflects the practical steps I can implement based on my current theoretical understanding refined with claude. My goal is to cover around 70% of core NLP concepts through this sentiment analysis project. As I progress, I'll learn and incorporate additional topics beyond my present knowledge.

## Phase 1: Data Setup & Cleaning

### Data Loading 
- [x] Download IMDb 50K dataset from Kaggle/HuggingFace
- [x] Load dataset
- [x] Check data shape: `print(dataset.shape)`, count value
- [x] Sample 10 reviews and read them manually
- [x] Do more (get overview as much as you can) like missing value and duplicates reviews 

### Basic Cleaning Pipeline (as this is already cleaned data, we don't have to do much)
- [x] Install libraries:
- [x] Remove duplicate rows
- [x] Write function `remove_html_tags()` using BeautifulSoup
- [x] Write function `remove_punctuation()` using regex or string.punctuation
- [x] Write function `remove_urls()` using regex
- [x] Write function `remove_stopwords()` using NLTK or spaCy
- [x] Write function `remove_emojis()` using regex
- [x] Write function `expand_contractions()` (don't→do not, I'll→I will)
- [x] Chain all functions into `preprocess_text()` pipeline
- [x] Lemmatize words using spaCy or NLTK
- [x] Apply pipeline to entire dataset and save as `cleaned_imdb.csv`

### Text Analysis
- [x] Count words per review: `reviews['word_count'] = reviews['text'].apply(lambda x: len(x.split()))`
- [x] Plot word count distribution using matplotlib
- [x] Find top 50 most frequent words using `Counter`
- [x] Create separate word frequency lists for positive/negative reviews
- [x] Build vocabulary dictionary: `word_to_idx = {word: i for i, word in enumerate(vocab)}`

## Phase 2: Exploratory Data Analysis

> **We have explored most of these steps during yesterday's data cleaning and preprocessing. This phase is essentially merged with Phase 1, but is separated here for a more structured

### Dataset Overview
- [x] Check data shape and structure: `print(df.shape)`, `df.info()`
- [x] Check for missing values: `df.isnull().sum()`
- [x] Identify and remove duplicate data: `df.duplicated().sum()`, `df.drop_duplicates()`

### Target Variable Distribution
- [x] Plot countplot for sentiment classes: `sns.countplot(x='label', data=df)`
- [x] Calculate class balance statistics: `df['label'].value_counts(normalize=True)`

### Text Length Analysis
- [x] Plot distribution of character counts per review: `df['char_count'] = df['text'].apply(len)`
- [x] Plot distribution of word counts per review: `df['word_count'] = df['text'].apply(lambda x: len(x.split()))`
- [x] Plot average word length distribution: `df['avg_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))`

### Word Cloud Visualization
- [x] Generate word cloud for positive reviews
- [x] Generate word cloud for negative reviews

### Corpus Analysis
- [x] Extract entire corpus from reviews: `' '.join(df['text'])`
- [x] Compute frequency distribution of most common words

### N-gram Analysis
- [x] Calculate unigram frequency for positive and negative reviews
- [x] Calculate bigram frequency for positive and negative reviews
- [x] Calculate trigram frequency for positive and negative reviews

### Basic Stats
- [x] Calculate average review length by sentiment
- [x] Count unique words: `len(set(all_words))`
- [x] Find longest and shortest reviews
- [x] Create length histogram: `plt.hist(word_counts, bins=50)`
- [x] Calculate sentiment distribution: `sentiment_counts = df['label'].value_counts()`

### Word Analysis
- [x] Install textblob: `pip install textblob`
- [x] Calculate sentiment scores: `TextBlob(text).sentiment.polarity`
- [x] Extract most common bigrams using `nltk.bigrams()`
- [x] Create word clouds: `pip install wordcloud`
- [x] Generate separate wordclouds for positive/negative reviews
- [x] Save all visualizations as PNG files

### Advanced EDA
- [x] Install spacy model: `python -m spacy download en_core_web_sm`
- [x] Extract POS tags for sample reviews
- [x] Count adjectives, adverbs in positive vs negative reviews
- [x] Create correlation matrix between text features
- [x] Export EDA summary to `eda_report.html` (just for structured way)

---

## Phase 3: Vectorization & Embeddings 

### Non-Semantic Vectors
- [x] Implement Bag of Words using `CountVectorizer`
- [x] Set parameters: `max_features=10000, min_df=2, max_df=0.95`
- [x] Transform train/test data: `X_train_bow = vectorizer.fit_transform(train_texts)`
- [x] Implement TF-IDF using `TfidfVectorizer`
- [x] Compare BoW vs TF-IDF shapes and sparsity
- [x] Save vectors as `.pkl` files

### Pre-trained Embeddings
- [x] Download GloVe vectors: `wget http://nlp.stanford.edu/data/glove.6B.zip`
- [x] Load GloVe into dictionary: `glove_dict = {}`
- [x] Create embedding matrix for your vocabulary
- [x] Handle OOV words with zero vectors or random initialization
- [x] Save embedding matrix as numpy array

### BERT Embeddings
- [ ] Install transformers: `pip install transformers torch`
- [ ] Load BERT model: `from transformers import AutoTokenizer, AutoModel`
- [ ] Initialize: `tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')`
- [ ] Initialize: `model = AutoModel.from_pretrained('bert-base-uncased')`
- [ ] Write function `get_bert_embeddings(texts)` that returns [CLS] tokens
- [ ] Process first 1000 reviews and save embeddings
- [ ] Compare embedding sizes: GloVE vs BERT

---

## Phase 4: Advanced Architectures 

### BERT Fine-tuning
- [ ] Create BERT classifier:
  ```python
  class BERTClassifier(nn.Module):
      def __init__(self):
          self.bert = AutoModel.from_pretrained('bert-base-uncased')
          self.classifier = nn.Linear(768, 2)
  ```
- [ ] Tokenize with BERT tokenizer: `inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')`
- [ ] Use different learning rates: `1e-5` for BERT, `1e-3` for classifier
- [ ] Train for 3-5 epochs (BERT needs fewer epochs)
- [ ] Save fine-tuned model

### RoBERTa Implementation
- [ ] Replace BERT with RoBERTa: `roberta-base`
- [ ] Compare tokenization differences
- [ ] Fine-tune RoBERTa on IMDb data
- [ ] Compare RoBERTa vs BERT performance

### DistilBERT Implementation
- [ ] Implement DistilBERT: `distilbert-base-uncased`
- [ ] Compare model size vs BERT
- [ ] Measure inference time differences
- [ ] Compare accuracy vs efficiency trade-offs

### Custom Transformer
- [ ] Implement basic transformer encoder:
  ```python
  class TransformerBlock(nn.Module):
      def __init__(self, embed_dim, num_heads):
          self.attention = nn.MultiheadAttention(embed_dim, num_heads)
          self.norm1 = nn.LayerNorm(embed_dim)
          self.norm2 = nn.LayerNorm(embed_dim)
          self.ffn = nn.Sequential(...)
  ```
- [ ] Add positional encoding
- [ ] Stack multiple transformer blocks
- [ ] Train from scratch on IMDb

### Prompt Engineering
- [ ] Install OpenAI API: `pip install openai`
- [ ] Create prompt templates:
  - "Classify the sentiment: [REVIEW] \nSentiment:"
  - "This movie review is [positive/negative]: [REVIEW]"
- [ ] Test few-shot prompting with 5 examples
- [ ] Compare zero-shot vs few-shot performance
- [ ] Save prompting results to JSON

### RAG Implementation
- [ ] Install FAISS: `pip install faiss-cpu`
- [ ] Create knowledge base from training reviews
- [ ] Implement retrieval function using BERT embeddings
- [ ] Build RAG pipeline: retrieve → augment → classify
- [ ] Compare RAG vs fine-tuning performance

---

## Phase 5: Visualization & Monitoring 

### TensorBoard Setup
- [ ] Install tensorboard: `pip install tensorboard`
- [ ] Add logging to training loop:
  ```python
  from torch.utils.tensorboard import SummaryWriter
  writer = SummaryWriter('runs/experiment_1')
  writer.add_scalar('Loss/Train', loss, epoch)
  ```
- [ ] Log accuracy, F1-score, learning rate
- [ ] Launch TensorBoard: `tensorboard --logdir=runs`

### Model Analysis
- [ ] Log attention weights for transformer models
- [ ] Create attention visualization function
- [ ] Plot embedding projections using t-SNE
- [ ] Visualize learned word embeddings
- [ ] Save all visualizations as images

### Error Analysis
- [ ] Create confusion matrix using seaborn
- [ ] Find most misclassified examples
- [ ] Analyze failure patterns (short reviews, specific topics)
- [ ] Create error analysis report

### Model Comparison
- [ ] Create results comparison table:
  | Model | Accuracy | F1-Score | Training Time | Model Size |
- [ ] Plot ROC curves for all models
- [ ] Create performance vs efficiency scatter plot
- [ ] Generate model comparison report

### Advanced Metrics
- [ ] Implement BLEU score (for generated text)
- [ ] Calculate perplexity for language models
- [ ] Measure inference latency for each model
- [ ] Create comprehensive evaluation dashboard

---

## Phase 6: Deployment & Production 

### Model Optimization
- [ ] Convert best model to TorchScript: `torch.jit.script(model)`
- [ ] Quantize model: `torch.quantization.quantize_dynamic(model)`
- [ ] Measure size reduction and speed improvement
- [ ] Export to ONNX: `torch.onnx.export(model, dummy_input, 'model.onnx')`

### API Development
- [ ] Install FastAPI: `pip install fastapi uvicorn`
- [ ] Create `app.py`:
  ```python
  from fastapi import FastAPI
  app = FastAPI()
  
  @app.post("/predict")
  def predict_sentiment(text: str):
      # implement prediction
  ```
- [ ] Load model once at startup
- [ ] Add request validation using Pydantic
- [ ] Test API with curl commands

### Docker Setup
- [ ] Create `Dockerfile`:
  ```dockerfile
  FROM python:3.9-slim
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] Build image: `docker build -t sentiment-api .`
- [ ] Run container: `docker run -p 8000:8000 sentiment-api`
- [ ] Test containerized API

### Web Interface
- [ ] Install Streamlit: `pip install streamlit`
- [ ] Create `streamlit_app.py`:
  - Text input box
  - Predict button
  - Show confidence scores
  - Display prediction history
- [ ] Run app: `streamlit run streamlit_app.py`
- [ ] Add file upload for batch processing

### Monitoring Setup
- [ ] Add request logging to API
- [ ] Track prediction confidence distribution
- [ ] Monitor API response times
- [ ] Create health check endpoint: `/health`
- [ ] Setup basic alerts for API failures

### Testing Pipeline
- [ ] Create `test_api.py` with pytest
- [ ] Test API endpoints with various inputs
- [ ] Test model inference correctness
- [ ] Add integration tests for preprocessing pipeline
- [ ] Run tests: `pytest test_api.py -v`

### Deployment 
- [ ] Deploy to cloud platform (AWS, GCP, or Azure)
- [ ] Setup CI/CD with GitHub Actions
- [ ] Configure auto-scaling
- [ ] Setup domain and SSL certificate
- [ ] Monitor production metrics

---
