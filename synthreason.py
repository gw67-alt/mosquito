import numpy as np
from collections import defaultdict, deque, Counter
from typing import Dict, List, Tuple, Optional, Callable
import random
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from snntorch import spikegen
from sklearn.preprocessing import StandardScaler
import re
KB_LIMIT = 13177

class SpikingFrequencyPredictor:
    def __init__(self):
        self.bigram_frequencies: Dict[Tuple[str, str], int] = {}
        self.frequency_features: List[List[float]] = []
        self.snn_model: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.sorted_bigrams: List[Tuple[str, str]] = []
        self.unigram_counts: Dict[str, int] = Counter()
        self.num_base_features: int = 6  # Updated for combinatorial features
        self.feature_operations: Optional[List[Optional[Callable[[np.ndarray], np.ndarray]]]] = None
        self.num_steps = 5
        self.beta = 0.5
        self.spike_grad = surrogate.fast_sigmoid()
        self.current_text = ""

    def preprocess_text(self, text: str) -> List[str]:
        words = text.split()
        return [word for word in words if word]

    def extract_bigram_frequencies(self, text: str) -> Dict[Tuple[str, str], int]:
        words = self.preprocess_text(text)
        self.unigram_counts = Counter(words)
        bigram_counts = Counter()
        for i in range(len(words) - 1):
            bigram = (words[i], words[i + 1])
            bigram_counts[bigram] += 1
        self.bigram_frequencies = dict(bigram_counts)
        self.sorted_bigrams = [
            item[0] for item in sorted(
                self.bigram_frequencies.items(),
                key=lambda x: (x[1], *x[0], x[0][1]),
                reverse=True
            )
        ]
        return self.bigram_frequencies
    def contains_integer(self, s):
        return bool(re.search(r'[-+]?\b\d+\b', s))
    # Lambda infinite combinatorial feature generator
    def create_bigram_frequency_features(self) -> List[List[float]]:
        if not self.bigram_frequencies:
            return []
        text_content = self.current_text
        words = self.preprocess_text(text_content)

        neural_features = []

        # Lambda calculus inspired feature generator
        def infinite_features(w1, w2, idx):
            # Lambda calculus fixed-point combinator
            Y = lambda f: (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))
            
            # Combinatorial feature calculation using fixed-point
            comb_feature = Y(lambda f: lambda n: 1 if n == 0 else n * f(n-1))(idx % 17)
            freq = self.bigram_frequencies.get(bigram, 0)

            return [
                idx if words[freq] == "the" else 0,
                idx if words[idx] == "is" else 0,
                idx if words[idx] == "and" else 0,
                idx if words[idx] == "or" else 0,

            
            ]

        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            w1, w2 = bigram
            features = infinite_features(w1, w2, self.bigram_frequencies.get(bigram, 0))
            neural_features.append(features)

        if neural_features:
            self.num_base_features = len(neural_features[0]) - 1

        self.frequency_features = neural_features
        return neural_features

    def _apply_feature_operations(self, X_data: np.ndarray) -> np.ndarray:
        if not self.feature_operations:
            return X_data
        if X_data.ndim != 2 or X_data.shape[1] != self.num_base_features:
            return X_data
        X_transformed = X_data.astype(float).copy()
        for i in range(self.num_base_features):
            if i < len(self.feature_operations):
                operation = self.feature_operations[i]
                if operation:
                    try:
                        X_transformed[:, i] = operation(X_data[:, i].astype(float))
                    except Exception:
                        X_transformed[:, i] = X_data[:, i].astype(float)
        return X_transformed

    def _encode_features_to_spikes(self, features: np.ndarray) -> torch.Tensor:
        features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        features_tensor = torch.FloatTensor(features_normalized)
        spike_data = spikegen.rate(features_tensor, num_steps=self.num_steps)
        return spike_data

    def _create_spiking_network(self, input_size: int) -> nn.Module:
        snn_model = nn.Sequential(
            nn.Linear(input_size, 512),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            nn.Linear(512, 512),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad),
            nn.Linear(512, 1),
            snn.Leaky(beta=self.beta, init_hidden=True, spike_grad=self.spike_grad, output=True)
        )
        return snn_model

    def _spiking_forward_pass(self, spike_data: torch.Tensor) -> torch.Tensor:
        spk_rec = []
        mem_rec = []
        utils.reset(self.snn_model)
        for step in range(self.num_steps):
            spk_out, mem_out = self.snn_model(spike_data[step])
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        spk_rec = torch.stack(spk_rec)
        mem_rec = torch.stack(mem_rec)
        output = torch.sum(spk_rec, dim=0)
        return output, spk_rec, mem_rec
    def load_text_file(self, file_path: str) -> str:
        print(f"VERBOSE: Attempting to load text from local file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                words = content.lower().split()[:KB_LIMIT]
                print(f"VERBOSE: Successfully loaded {len(words)} words from {file_path}.")
                return ' '.join(words)
        except FileNotFoundError:
            print(f"VERBOSE: File {file_path} not found. Using internal sample text.")
            return self.get_sample_text()
        except Exception as e:
            print(f"VERBOSE: Error loading file {file_path}: {e}. Using internal sample text.")
            return self.get_sample_text()
            
    def generate_spiking_predictions(self, num_variations: int = 1) -> List[Dict[Tuple[str, str], float]]:
        """Generate predictions using the trained spiking network."""
        print(f"VERBOSE: Generating {num_variations} predictions with SNN")
        
        if self.snn_model is None:
            print("VERBOSE: No trained SNN model available")
            return []
        
        new_frequency_sets = []
        base_X = np.array([f[1:] for f in self.frequency_features])
        
        for variation in range(num_variations):
            print(f"VERBOSE: Generating SNN variation {variation + 1}")
            
            # Add noise to features
            noise_factor = 0.1 + (variation * 0.02)
            X_noised = base_X.copy()
            
            for j in range(X_noised.shape[1]):
                noise = np.random.normal(0, noise_factor * np.abs(X_noised[:, j] + 0.01))
                X_noised[:, j] = np.maximum(0, X_noised[:, j] + noise)
            
            # Transform and scale
            X_transformed = self._apply_feature_operations(X_noised)
            X_scaled = self.scaler.transform(X_transformed)
            
            # Convert to spikes
            spike_data = self._encode_features_to_spikes(X_scaled)
            
            # Generate predictions
            with torch.no_grad():
                predictions, _, _ = self._spiking_forward_pass(spike_data)
            
            # Convert to frequency dictionary
            predictions_np = predictions.numpy().flatten()
            predictions_np = np.argsort(predictions_np)
            
            new_freq_dict = {
                bigram: float(predictions_np[i]) 
                for i, bigram in enumerate(self.sorted_bigrams) 
                if i < len(predictions_np)
            }
            
            new_frequency_sets.append(new_freq_dict)
        
        print(f"VERBOSE: Generated {len(new_frequency_sets)} SNN prediction sets")
        return new_frequency_sets
    def expand_text_from_bigrams(self,
                                 frequency_dict: Dict[Tuple[str, str], float],
                                 text_length: int = 100,
                                 seed_phrase: Optional[str] = None) -> str:
        print(f"VERBOSE: Starting text expansion. Target length: {text_length}. Seed: '{seed_phrase if seed_phrase else 'None'}'")
        if not frequency_dict:
            print("VERBOSE: Error: No frequency data provided for text expansion.")
            return "Error: No frequency data provided."

        transitions = defaultdict(list)
        for (w1, w2), count in frequency_dict.items():
            if count > 0: 
                transitions[w1].append((w2, count))
        
        if not transitions:
            print("VERBOSE: Error: Frequency data has no usable transitions.")
            return "Error: Frequency data has no usable transitions."

        generated_text_list = []
        current_word: Optional[str] = None
        num_words_to_generate = text_length
        start_word_selected_from_seed = False

        if seed_phrase:
            seed_words = self.preprocess_text(seed_phrase) 
            if seed_words:
                print(f"VERBOSE: Processed seed phrase: {seed_words}")
                potential_start_node = seed_words[-1]
                if potential_start_node in transitions and transitions[potential_start_node]:
                    generated_text_list.extend(seed_words)
                    current_word = potential_start_node
                    start_word_selected_from_seed = True
                    num_words_to_generate = text_length - len(generated_text_list)
                    print(f"VERBOSE: Started with seed. Current word: '{current_word}'. Words to generate: {num_words_to_generate}.")
                    if num_words_to_generate <= 0:
                        final_text = ' '.join(generated_text_list[:text_length])
                        print(f"VERBOSE: Seed phrase already meets/exceeds target length. Generated text: '{final_text[:50]}...'")
                        return final_text

        if not start_word_selected_from_seed:
            print("VERBOSE: Selecting a starting word (seed not used or invalid).")
            valid_starting_unigrams = {w:c for w,c in self.unigram_counts.items() if w in transitions and transitions[w]}
            if valid_starting_unigrams:
                sorted_starters = sorted(valid_starting_unigrams.items(), key=lambda item: item[1], reverse=True)
                starters = [item[0] for item in sorted_starters]
                weights = [item[1] for item in sorted_starters]
                current_word = random.choices(starters, weights=weights, k=1)[0]
                print(f"VERBOSE: Selected start word '{current_word}' based on weighted unigram counts.")
            elif any(transitions.values()):
                possible_starters = [w1 for w1, w2_list in transitions.items() if w2_list]
                if possible_starters:
                    current_word = random.choice(possible_starters)
                    print(f"VERBOSE: Selected start word '{current_word}' randomly from possible transition starters.")
                else:
                    print("VERBOSE: Error: Cannot determine any valid starting word from transitions.")
                    return "Error: Cannot determine any valid starting word."
            else:
                print("VERBOSE: Error: Cannot determine a starting word (no valid transitions).")
                return "Error: Cannot determine a starting word (no valid transitions)."
            
            if current_word:
                generated_text_list.append(current_word)
                num_words_to_generate = text_length - 1
            else:
                print("VERBOSE: Error: Failed to select a starting word.")
                return "Error: Failed to select a starting word."

        for i in range(max(0, num_words_to_generate)):
            if not current_word or current_word not in transitions or not transitions[current_word]:
                print(f"VERBOSE: Current word '{current_word}' has no further transitions. Attempting to restart.")
                valid_restart_candidates = [w for w, trans_list in transitions.items() if trans_list]
                if not valid_restart_candidates:
                    print("VERBOSE: No valid restart candidates found. Ending generation.")
                    break 
                
                restart_options = {w:c for w,c in self.unigram_counts.items() if w in valid_restart_candidates}
                if restart_options:
                    sorted_restart_options = sorted(restart_options.items(), key=lambda item: item[1], reverse=True)
                    starters = [item[0] for item in sorted_restart_options]
                    weights = [item[1] for item in sorted_restart_options]
                    current_word = random.choices(starters, weights=weights, k=1)[0]
                    print(f"VERBOSE: Restarted with word '{current_word}' (weighted choice).")
                else:
                    current_word = random.choice(valid_restart_candidates)
                    print(f"VERBOSE: Restarted with word '{current_word}' (random choice).")
                if not current_word:
                    print("VERBOSE: Failed to select a restart word. Ending generation.")
                    break 

            possible_next_words, weights = zip(*transitions[current_word])
            next_word = random.choices(possible_next_words, weights=weights, k=1)[0]
            generated_text_list.append(next_word)
            current_word = next_word

        final_text = ' '.join(generated_text_list)
        print(f"VERBOSE: Text expansion complete. Generated {len(generated_text_list)} words. Preview: '{final_text[:70]}...'")
        return final_text
    def train_spiking_predictor(self) -> None:
        if not self.frequency_features:
            print("No frequency features available for SNN training")
            return
        X_raw = np.array([f[1:] for f in self.frequency_features])
        y = np.array([f[0] for f in self.frequency_features])
        X_transformed = self._apply_feature_operations(X_raw)
        X_scaled = self.scaler.fit_transform(X_transformed)
        spike_data = self._encode_features_to_spikes(X_scaled)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        self.snn_model = self._create_spiking_network(X_scaled.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.snn_model.parameters(), lr=0.001)
        num_epochs = 5
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output, spk_rec, mem_rec = self._spiking_forward_pass(spike_data)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Example usage
if __name__ == "__main__":
    """Enhanced text generation using spiking neural networks."""
    print("VERBOSE: Starting spiking neural network text generation")
    
    # Initialize spiking components
    predictor = SpikingFrequencyPredictor()
    
    # Load and process text
    text_content = predictor.load_text_file("test.txt")
    predictor.current_text = text_content  # Store for access in feature creation
    predictor.extract_bigram_frequencies(text_content)
    predictor.create_bigram_frequency_features()
    
    # Train spiking network
    predictor.train_spiking_predictor()
    
    print("\n" + "="*60)
    print("SPIKING NEURAL NETWORK TEXT GENERATOR READY")
    print("="*60)
    print("Enter text prompts to generate responses. Type 'quit' to exit.")
    print("="*60)
    
    while True:
        user_input = input("\nUSER: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        # Perform multilinear linking with spiking networks
        
        # Generate with spiking network
        spiking_frequencies = predictor.generate_spiking_predictions(num_variations=1)
        
        if spiking_frequencies:
            # Enhance frequencies based on linking results
            enhanced_frequencies = spiking_frequencies[0].copy()
            words = predictor.preprocess_text(text_content)
          
            generated_text = predictor.expand_text_from_bigrams(
                enhanced_frequencies,
                text_length=200,
                seed_phrase=user_input
            )
            
            print("\n" + "="*50)
            print("SPIKING NEURAL NETWORK GENERATION")
            print("="*50)
            print(generated_text)
            print("="*50)
        else:
            print("VERBOSE: No spiking predictions generated")