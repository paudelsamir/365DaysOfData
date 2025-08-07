# Implementing 76 percent of NLP

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
> This reflects the practical steps I can implement based on my current theoretical understanding (refined with claude with hints on the way). My goal is to cover approximately 76% of core NLP concepts through this sentiment analysis project. As I progress, I'll learn and incorporate additional topics beyond my present knowledge.

## Phase 1: Data Setup & Cleaning 

### Data Loading
- [ ] Download IMDb 50K dataset from Kaggle/HuggingFace
- [ ] Load dataset using `datasets` library: `load_dataset('imdb')`
- [ ] Create train/test splits (40K train, 10K test)
- [ ] Check data shape: `print(dataset.shape)`
- [ ] Sample 100 reviews and read them manually
- Do more (get overview as much as you can)

### Basic Cleaning Pipeline
- [ ] Install libraries: `pip install beautifulsoup4 nltk spacy pandas torch` 
- [ ] Write function `remove_html_tags()` using BeautifulSoup if needed
- [ ] Write function `remove_urls()` using regex
- [ ] Write function `expand_contractions()` (don't→do not, I'll→I will)
- [ ] Write function `remove_special_chars()` keeping only alphanumeric + basic punctuation
- [ ] Chain all functions into `preprocess_text()` pipeline
- [ ] Apply pipeline to entire dataset and save as `cleaned_imdb.csv`

### Text Analysis
- [ ] Count words per review: `reviews['word_count'] = reviews['text'].apply(lambda x: len(x.split()))`
- [ ] Plot word count distribution using matplotlib
- [ ] Find top 50 most frequent words using `Counter`
- [ ] Create separate word frequency lists for positive/negative reviews
- [ ] Build vocabulary dictionary: `word_to_idx = {word: i for i, word in enumerate(vocab)}`

---

## Phase 2: Exploratory Data Analysis 

### Basic Stats
- [ ] Calculate average review length by sentiment
- [ ] Count unique words: `len(set(all_words))`
- [ ] Find longest and shortest reviews
- [ ] Create length histogram: `plt.hist(word_counts, bins=50)`
- [ ] Calculate sentiment distribution: `sentiment_counts = df['label'].value_counts()`

### Word Analysis
- [ ] Install textblob: `pip install textblob`
- [ ] Calculate sentiment scores: `TextBlob(text).sentiment.polarity`
- [ ] Extract most common bigrams using `nltk.bigrams()`
- [ ] Create word clouds: `pip install wordcloud`
- [ ] Generate separate wordclouds for positive/negative reviews
- [ ] Save all visualizations as PNG files

### Advanced EDA
- [ ] Install spacy model: `python -m spacy download en_core_web_sm`
- [ ] Extract POS tags for sample reviews
- [ ] Count adjectives, adverbs in positive vs negative reviews
- [ ] Create correlation matrix between text features
- [ ] Export EDA summary to `eda_report.html`

---

## Phase 3: Vectorization & Embeddings 

### Non-Semantic Vectors
- [ ] Implement Bag of Words using `CountVectorizer`
- [ ] Set parameters: `max_features=10000, min_df=2, max_df=0.95`
- [ ] Transform train/test data: `X_train_bow = vectorizer.fit_transform(train_texts)`
- [ ] Implement TF-IDF using `TfidfVectorizer`
- [ ] Compare BoW vs TF-IDF shapes and sparsity
- [ ] Save vectors as `.pkl` files

### Word2Vec Implementation
- [ ] Install gensim: `pip install gensim`
- [ ] Tokenize all reviews: `tokenized_reviews = [text.split() for text in texts]`
- [ ] Train Word2Vec: `model = Word2Vec(tokenized_reviews, vector_size=300, min_count=5)`
- [ ] Save model: `model.save('imdb_word2vec.model')`
- [ ] Test word similarity: `model.wv.most_similar('good')`
- [ ] Create sentence vectors by averaging word vectors
- [ ] Implement document vectorization function

### Pre-trained Embeddings
- [ ] Download GloVe vectors: `wget http://nlp.stanford.edu/data/glove.6B.zip`
- [ ] Load GloVe into dictionary: `glove_dict = {}`
- [ ] Create embedding matrix for your vocabulary
- [ ] Handle OOV words with zero vectors or random initialization
- [ ] Save embedding matrix as numpy array

### BERT Embeddings
- [ ] Install transformers: `pip install transformers torch`
- [ ] Load BERT model: `from transformers import AutoTokenizer, AutoModel`
- [ ] Initialize: `tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')`
- [ ] Initialize: `model = AutoModel.from_pretrained('bert-base-uncased')`
- [ ] Write function `get_bert_embeddings(texts)` that returns [CLS] tokens
- [ ] Process first 1000 reviews and save embeddings
- [ ] Compare embedding sizes: Word2Vec vs BERT

### FastText Implementation
- [ ] Install fasttext: `pip install fasttext`
- [ ] Train FastText model on your data
- [ ] Compare with Word2Vec on OOV words
- [ ] Extract subword information
- [ ] Save FastText model and vectors

---

## Phase 4: Traditional Neural Networks 

### PyTorch Setup
- [ ] Create `dataset.py`: Custom Dataset class inheriting from `torch.utils.data.Dataset`
- [ ] Implement `__len__` and `__getitem__` methods
- [ ] Create `collate_fn` for padding sequences to same length
- [ ] Setup DataLoader: `train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)`
- [ ] Test data loading: print first batch shapes

### Simple MLP
- [ ] Create `models.py` file
- [ ] Implement MLP class:
  ```python
  class MLP(nn.Module):
      def __init__(self, input_size, hidden_size, num_classes):
          # implement layers
      def forward(self, x):
          # implement forward pass
  ```
- [ ] Use average word embeddings as input
- [ ] Train for 10 epochs, save best model
- [ ] Calculate accuracy on test set

### LSTM Implementation
- [ ] Implement LSTM classifier:
  ```python
  class LSTMClassifier(nn.Module):
      def __init__(self, vocab_size, embed_dim, hidden_dim):
          self.embedding = nn.Embedding(vocab_size, embed_dim)
          self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
          self.fc = nn.Linear(hidden_dim, 2)
  ```
- [ ] Handle variable length sequences with padding
- [ ] Use last LSTM output for classification
- [ ] Train model and save checkpoints every epoch

### BiLSTM Implementation
- [ ] Modify LSTM to bidirectional: `nn.LSTM(..., bidirectional=True)`
- [ ] Adjust linear layer input size: `hidden_dim * 2`
- [ ] Compare BiLSTM vs LSTM performance
- [ ] Plot training curves

### CNN for Text
- [ ] Implement CNN classifier:
  ```python
  class CNNClassifier(nn.Module):
      def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters):
          self.convs = nn.ModuleList([
              nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
              for fs in filter_sizes
          ])
  ```
- [ ] Use multiple filter sizes: [3, 4, 5]
- [ ] Apply max pooling after convolution
- [ ] Concatenate all conv outputs

### GRU Implementation
- [ ] Replace LSTM with GRU: `nn.GRU(embed_dim, hidden_dim)`
- [ ] Compare GRU vs LSTM training time
- [ ] Compare final accuracies
- [ ] Save all model checkpoints

### Training Pipeline
- [ ] Create `train.py` with training loop
- [ ] Implement early stopping based on validation loss
- [ ] Add learning rate scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau`
- [ ] Save training logs to CSV file
- [ ] Calculate precision, recall, F1-score for each model

---

## Phase 5: Advanced Architectures 

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

## Phase 6: Visualization & Monitoring 

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

## Phase 7: Deployment & Production 

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
