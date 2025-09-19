"""
Enhanced Lexical Feature Extraction for SALSA
Combines ASR transcripts with linguistic features and semantic embeddings.
"""
import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


class LexicalFeatureExtractor:
    """Extract comprehensive lexical features from transcripts."""
    
    def __init__(self, language='en'):
        self.language = language
        
        # Load spaCy model for POS tagging and linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Load sentence transformer for semantic embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sentence_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sentence_model.eval()
        
        # Freeze sentence model
        for param in self.sentence_model.parameters():
            param.requires_grad = False
        
        # Disfluency markers
        self.disfluencies = {
            'filled_pauses': ['uh', 'um', 'er', 'ah', 'eh'],
            'repetitions': [],  # Will be detected dynamically
            'false_starts': ['i mean', 'well', 'you know'],
            'repairs': []  # Will be detected dynamically
        }
        
        self.stop_words = set(stopwords.words('english'))
    
    def extract_pos_features(self, transcript):
        """Extract part-of-speech ratios and linguistic features."""
        if not self.nlp:
            return {}
        
        doc = self.nlp(transcript.lower())
        
        # Count POS tags
        pos_counts = Counter([token.pos_ for token in doc if not token.is_space])
        total_words = sum(pos_counts.values())
        
        if total_words == 0:
            return {
                'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0,
                'adv_ratio': 0.0, 'pronoun_ratio': 0.0, 'det_ratio': 0.0,
                'prep_ratio': 0.0, 'conj_ratio': 0.0
            }
        
        features = {
            'noun_ratio': pos_counts.get('NOUN', 0) / total_words,
            'verb_ratio': pos_counts.get('VERB', 0) / total_words,
            'adj_ratio': pos_counts.get('ADJ', 0) / total_words,
            'adv_ratio': pos_counts.get('ADV', 0) / total_words,
            'pronoun_ratio': pos_counts.get('PRON', 0) / total_words,
            'det_ratio': pos_counts.get('DET', 0) / total_words,
            'prep_ratio': pos_counts.get('ADP', 0) / total_words,
            'conj_ratio': pos_counts.get('CCONJ', 0) / total_words,
        }
        
        return features
    
    def extract_fluency_features(self, transcript):
        """Extract disfluency and fluency-related features."""
        transcript_lower = transcript.lower()
        words = word_tokenize(transcript_lower)
        
        # Count filled pauses
        filled_pause_count = sum(1 for word in words if word in self.disfluencies['filled_pauses'])
        
        # Count repetitions (simple heuristic: consecutive identical words)
        repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and words[i] not in self.stop_words:
                repetitions += 1
        
        # Count false starts
        false_start_count = sum(1 for phrase in self.disfluencies['false_starts'] 
                               if phrase in transcript_lower)
        
        # Calculate WPM approximation (assuming 150 WPM as baseline)
        # This is a rough estimate - in practice you'd use actual timing
        estimated_duration = len(words) / 2.5  # Rough estimate in seconds
        wpm_estimate = (len(words) / estimated_duration * 60) if estimated_duration > 0 else 0
        
        features = {
            'filled_pause_rate': filled_pause_count / len(words) if words else 0,
            'repetition_rate': repetitions / len(words) if words else 0,
            'false_start_rate': false_start_count / len(words) if words else 0,
            'total_disfluencies': filled_pause_count + repetitions + false_start_count,
            'estimated_wpm': wpm_estimate,
            'fluency_score': 1.0 - min(1.0, (filled_pause_count + repetitions) / len(words)) if words else 0
        }
        
        return features
    
    def extract_lexical_diversity(self, transcript):
        """Extract lexical diversity and complexity measures."""
        words = word_tokenize(transcript.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        if not words:
            return {
                'type_token_ratio': 0.0,
                'mean_word_length': 0.0,
                'vocabulary_richness': 0.0,
                'content_word_ratio': 0.0
            }
        
        # Type-Token Ratio
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # Mean word length
        mean_word_length = np.mean([len(word) for word in words])
        
        # Vocabulary richness (variation of TTR)
        vocab_richness = len(unique_words) / (len(words) ** 0.5)
        
        # Content word ratio
        all_words = word_tokenize(transcript.lower())
        content_words = [w for w in all_words if w.isalpha() and w not in self.stop_words]
        content_ratio = len(content_words) / len(all_words) if all_words else 0
        
        features = {
            'type_token_ratio': ttr,
            'mean_word_length': mean_word_length,
            'vocabulary_richness': vocab_richness,
            'content_word_ratio': content_ratio
        }
        
        return features
    
    def extract_syntactic_features(self, transcript):
        """Extract syntactic complexity features."""
        sentences = sent_tokenize(transcript)
        
        if not sentences:
            return {
                'mean_sentence_length': 0.0,
                'sentence_count': 0,
                'clause_density': 0.0,
                'subordination_ratio': 0.0
            }
        
        # Mean sentence length
        words_per_sentence = [len(word_tokenize(sent)) for sent in sentences]
        mean_sentence_length = np.mean(words_per_sentence) if words_per_sentence else 0
        
        # Count subordinate clauses (rough heuristic)
        subordination_markers = ['because', 'since', 'although', 'while', 'if', 'when', 'that', 'which', 'who']
        subordinate_clauses = sum(1 for sent in sentences 
                                 for marker in subordination_markers 
                                 if marker in sent.lower())
        
        subordination_ratio = subordinate_clauses / len(sentences) if sentences else 0
        
        # Clause density (approximation)
        total_words = sum(words_per_sentence)
        clause_density = subordinate_clauses / total_words if total_words > 0 else 0
        
        features = {
            'mean_sentence_length': mean_sentence_length,
            'sentence_count': len(sentences),
            'clause_density': clause_density,
            'subordination_ratio': subordination_ratio
        }
        
        return features
    
    def get_semantic_embeddings(self, transcript):
        """Get sentence embeddings from pre-trained model."""
        # Tokenize and encode
        inputs = self.tokenizer(
            transcript, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.sentence_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.squeeze(0)  # Shape: (384,) for all-MiniLM-L6-v2
    
    def extract_all_features(self, transcript):
        """Extract all lexical features from transcript."""
        if not transcript or not transcript.strip():
            # Return zero features for empty transcripts
            feature_dict = {
                # POS features
                'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0,
                'adv_ratio': 0.0, 'pronoun_ratio': 0.0, 'det_ratio': 0.0,
                'prep_ratio': 0.0, 'conj_ratio': 0.0,
                # Fluency features
                'filled_pause_rate': 0.0, 'repetition_rate': 0.0, 'false_start_rate': 0.0,
                'total_disfluencies': 0.0, 'estimated_wpm': 0.0, 'fluency_score': 0.0,
                # Diversity features
                'type_token_ratio': 0.0, 'mean_word_length': 0.0,
                'vocabulary_richness': 0.0, 'content_word_ratio': 0.0,
                # Syntactic features
                'mean_sentence_length': 0.0, 'sentence_count': 0,
                'clause_density': 0.0, 'subordination_ratio': 0.0
            }
            semantic_emb = torch.zeros(384)  # MiniLM embedding size
            return feature_dict, semantic_emb
        
        # Extract all feature types
        pos_features = self.extract_pos_features(transcript)
        fluency_features = self.extract_fluency_features(transcript)
        diversity_features = self.extract_lexical_diversity(transcript)
        syntactic_features = self.extract_syntactic_features(transcript)
        semantic_embeddings = self.get_semantic_embeddings(transcript)
        
        # Combine all features
        feature_dict = {
            **pos_features,
            **fluency_features, 
            **diversity_features,
            **syntactic_features
        }
        
        return feature_dict, semantic_embeddings


class LexicalBranch(nn.Module):
    """Complete lexical branch with feature extraction and MLP head."""
    
    def __init__(self, semantic_dim=384, linguistic_features=20, hidden_dim=256, output_dim=512):
        super().__init__()
        
        self.feature_extractor = LexicalFeatureExtractor()
        
        # MLP for combining linguistic features with semantic embeddings
        total_input_dim = semantic_dim + linguistic_features
        
        self.feature_head = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.feature_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, transcripts):
        """
        Forward pass for lexical branch.
        
        Args:
            transcripts: List of transcript strings or batch of transcripts
            
        Returns:
            Lexical feature embeddings: (B, output_dim)
        """
        if isinstance(transcripts, str):
            transcripts = [transcripts]
        
        batch_linguistic_features = []
        batch_semantic_embeddings = []
        
        for transcript in transcripts:
            # Extract features
            linguistic_dict, semantic_emb = self.feature_extractor.extract_all_features(transcript)
            
            # Convert linguistic features to tensor
            linguistic_features = torch.tensor(
                list(linguistic_dict.values()), 
                dtype=torch.float32
            )
            
            batch_linguistic_features.append(linguistic_features)
            batch_semantic_embeddings.append(semantic_emb)
        
        # Stack into batch tensors
        linguistic_features = torch.stack(batch_linguistic_features)
        semantic_embeddings = torch.stack(batch_semantic_embeddings)
        
        # Move to same device as model
        device = next(self.parameters()).device
        linguistic_features = linguistic_features.to(device)
        semantic_embeddings = semantic_embeddings.to(device)
        
        # Combine features
        combined_features = torch.cat([semantic_embeddings, linguistic_features], dim=-1)
        
        # Pass through MLP
        output = self.feature_head(combined_features)
        
        return output