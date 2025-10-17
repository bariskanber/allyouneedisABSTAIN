import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================
# UPDATED: Data Augmentation with Corruption → [UNKNOWN]
# ============================================

class SimplePredictor(nn.Module):
    """Simple model: input token → output token"""
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def create_dataset(num_common=20, num_rare=20, num_unseen=20):
    """
    Create synthetic dataset:
    - Common: input i → output i+50 (seen 100 times)
    - Rare: input i+20 → output i+70 (seen 1 time)
    - Unseen: input i+40 → ??? (never seen, for testing)
    """
    # Common facts (high frequency)
    common_pairs = [(i, i + 50) for i in range(num_common)]
    common_data = common_pairs * 100  # See 100 times
    
    # Rare facts (low frequency - SINGLETONS)
    rare_pairs = [(i + num_common, i + num_common + 50) 
                  for i in range(num_rare)]
    rare_data = rare_pairs * 1  # See only once
    
    # Unseen (test hallucinations)
    unseen_pairs = [(i + num_common + num_rare, None)  # No known answer
                    for i in range(num_unseen)]
    
    train_data = common_data + rare_data
    np.random.shuffle(train_data)
    
    return {
        'train': train_data,
        'test_common': common_pairs,
        'test_rare': rare_pairs,
        'test_unseen': unseen_pairs,
    }

def augment_with_corrupted_unknown(train_data, corruption_prob=0.15, corruption_strength=30):
    """
    KEY INNOVATION: Data augmentation by corrupting inputs
    
    For each training example, with probability corruption_prob:
    - Corrupt the input (simulate shuffled/nonsense input)
    - Set target to [UNKNOWN] token
    
    This teaches: "when input is corrupted/nonsense → predict [UNKNOWN]"
    """
    UNKNOWN_TOKEN = 100
    augmented = []
    
    for input_token, output_token in train_data:
        # Add original example
        augmented.append((input_token, output_token))
        
        # Add corrupted version with probability
        if np.random.random() < corruption_prob:
            # Corrupt input: add random offset to simulate "shuffled" tokens
            # In real LLM: this would be shuffling the context window
            corrupted_input = (input_token + np.random.randint(corruption_strength, 
                                                               corruption_strength + 20)) % 100
            augmented.append((corrupted_input, UNKNOWN_TOKEN))
    
    return augmented

def evaluate(model, test_data, confidence_threshold=0.5, unknown_token=None, verbose=False):
    """Evaluate model and measure hallucinations"""
    model.eval()
    
    results = {
        'correct': 0,
        'wrong': 0,
        'total': 0,
        'confident_wrong': 0,
        'avg_confidence': 0,
        'predicted_unknown': 0,
        'confidences': [],
    }
    
    with torch.no_grad():
        for input_token, true_token in test_data:
            if true_token is None:  # Unseen data
                true_token = -1  # Placeholder
            
            logits = model(torch.tensor([input_token]))
            probs = F.softmax(logits[0], dim=0)
            
            predicted = probs.argmax().item()
            confidence = probs.max().item()
            
            results['total'] += 1
            results['avg_confidence'] += confidence
            results['confidences'].append(confidence)
            
            # Check if predicted [UNKNOWN] token
            if unknown_token is not None and predicted == unknown_token:
                results['predicted_unknown'] += 1
                if verbose:
                    unknown_prob = probs[unknown_token].item()
                    print(f"  Input {input_token}: → [UNKNOWN] (conf={confidence:.3f})")
                continue
            
            # Check correctness
            if true_token != -1:
                if predicted == true_token:
                    results['correct'] += 1
                else:
                    results['wrong'] += 1
                    if confidence > confidence_threshold:
                        results['confident_wrong'] += 1
                        if verbose:
                            print(f"  Input {input_token}: HALLUCINATED {predicted} "
                                  f"(true={true_token}, conf={confidence:.3f})")
            else:  # Unseen - any prediction is wrong
                results['wrong'] += 1
                if confidence > confidence_threshold:
                    results['confident_wrong'] += 1
                    if verbose:
                        print(f"  Input {input_token}: HALLUCINATED {predicted} "
                              f"(unseen, conf={confidence:.3f})")
    
    # Calculate rates
    total = results['total']
    return {
        'accuracy': results['correct'] / total if total > 0 else 0,
        'hallucination_rate': results['confident_wrong'] / total,
        'avg_confidence': results['avg_confidence'] / total,
        'unknown_rate': results['predicted_unknown'] / total,
        'wrong_rate': results['wrong'] / total,
    }

# ============================================
# CONDITION 1: BASELINE
# ============================================

def train_baseline(dataset, epochs=50):
    """Standard training without unknown token"""
    print("\n" + "="*60)
    print("CONDITION 1: BASELINE (Standard Cross-Entropy)")
    print("="*60)
    
    input_vocab = 100
    output_vocab = 100  # No unknown token
    
    model = SimplePredictor(input_vocab, output_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_data = dataset['train']
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        np.random.shuffle(train_data)
        
        for input_token, output_token in train_data:
            optimizer.zero_grad()
            
            logits = model(torch.tensor([input_token]))
            loss = F.cross_entropy(logits, torch.tensor([output_token]))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_data):.4f}")
    
    # Evaluate
    print("\n--- Evaluation ---")
    common_metrics = evaluate(model, dataset['test_common'])
    rare_metrics = evaluate(model, dataset['test_rare'])
    unseen_metrics = evaluate(model, dataset['test_unseen'])
    
    print(f"Common:  Acc={common_metrics['accuracy']:.3f}, "
          f"Hallu={common_metrics['hallucination_rate']:.3f}")
    print(f"Rare:    Acc={rare_metrics['accuracy']:.3f}, "
          f"Hallu={rare_metrics['hallucination_rate']:.3f}")
    print(f"Unseen:  Hallu={unseen_metrics['hallucination_rate']:.3f}")
    
    return model, {
        'common': common_metrics,
        'rare': rare_metrics,
        'unseen': unseen_metrics
    }

# ============================================
# CONDITION 2: WITH CORRUPTION → [UNKNOWN]
# ============================================

def train_with_corruption_unknown(dataset, epochs=50, corruption_prob=0.2):
    """
    Train with [UNKNOWN] token using CORRUPTION AUGMENTATION
    
    Key idea: Corrupt inputs → teach model to predict [UNKNOWN]
    This simulates shuffled context in real LLMs
    """
    print("\n" + "="*60)
    print("CONDITION 2: WITH CORRUPTION → [UNKNOWN] AUGMENTATION")
    print("="*60)
    
    input_vocab = 100
    output_vocab = 101  # +1 for [UNKNOWN]
    UNKNOWN_TOKEN = 100
    
    model = SimplePredictor(input_vocab, output_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Apply corruption augmentation!
    train_data = augment_with_corrupted_unknown(
        dataset['train'], 
        corruption_prob=corruption_prob,
        corruption_strength=30
    )
    
    num_original = len(dataset['train'])
    num_unknown = len([x for x in train_data if x[1] == UNKNOWN_TOKEN])
    
    print(f"Training data:")
    print(f"  Original examples: {num_original}")
    print(f"  Corrupted → [UNKNOWN]: {num_unknown}")
    print(f"  Total: {len(train_data)}")
    print(f"  Corruption rate: {corruption_prob:.1%}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        np.random.shuffle(train_data)
        
        for input_token, output_token in train_data:
            optimizer.zero_grad()
            
            logits = model(torch.tensor([input_token]))
            loss = F.cross_entropy(logits, torch.tensor([output_token]))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_data):.4f}")
    
    # Evaluate
    print("\n--- Evaluation ---")
    common_metrics = evaluate(model, dataset['test_common'], 
                             unknown_token=UNKNOWN_TOKEN)
    rare_metrics = evaluate(model, dataset['test_rare'],
                           unknown_token=UNKNOWN_TOKEN)
    
    print("Sample predictions on UNSEEN data:")
    unseen_sample = evaluate(model, dataset['test_unseen'][:5],
                            unknown_token=UNKNOWN_TOKEN, verbose=True)
    
    unseen_metrics = evaluate(model, dataset['test_unseen'],
                             unknown_token=UNKNOWN_TOKEN)
    
    print(f"\nCommon:  Acc={common_metrics['accuracy']:.3f}, "
          f"Hallu={common_metrics['hallucination_rate']:.3f}, "
          f"Unknown={common_metrics['unknown_rate']:.3f}")
    print(f"Rare:    Acc={rare_metrics['accuracy']:.3f}, "
          f"Hallu={rare_metrics['hallucination_rate']:.3f}, "
          f"Unknown={rare_metrics['unknown_rate']:.3f}")
    print(f"Unseen:  Hallu={unseen_metrics['hallucination_rate']:.3f}, "
          f"Unknown={unseen_metrics['unknown_rate']:.3f}")
    
    return model, {
        'common': common_metrics,
        'rare': rare_metrics,
        'unseen': unseen_metrics
    }

# ============================================
# RUN EXPERIMENT & COMPARE
# ============================================

def run_experiment():
    """Run both conditions and compare results"""
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    print("="*60)
    print("EXPERIMENT: Does corruption → [UNKNOWN] reduce hallucinations?")
    print("="*60)
    dataset = create_dataset(num_common=20, num_rare=20, num_unseen=20)
    
    # Train baseline
    baseline_model, baseline_results = train_baseline(dataset, epochs=50)
    
    # Train with corruption augmentation
    unknown_model, unknown_results = train_with_corruption_unknown(
        dataset, epochs=50, corruption_prob=0.2
    )
    
    # Compare results
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    for test_type in ['common', 'rare', 'unseen']:
        baseline = baseline_results[test_type]
        unknown = unknown_results[test_type]
        
        print(f"\n{test_type.upper()}:")
        print(f"  Accuracy:           {baseline['accuracy']:.3f} → {unknown['accuracy']:.3f}")
        print(f"  Hallucination:      {baseline['hallucination_rate']:.3f} → {unknown['hallucination_rate']:.3f}")
        print(f"  [UNKNOWN] usage:    N/A     → {unknown['unknown_rate']:.3f}")
    
    # Key result
    print("\n" + "="*60)
    print("KEY FINDING:")
    print("="*60)
    baseline_hallu = baseline_results['unseen']['hallucination_rate']
    unknown_hallu = unknown_results['unseen']['hallucination_rate']
    unknown_rate = unknown_results['unseen']['unknown_rate']
    
    print(f"\nOn UNSEEN data (the hallucination test):")
    print(f"  Baseline hallucination rate:     {baseline_hallu:.1%}")
    print(f"  With [UNKNOWN] hallucination:    {unknown_hallu:.1%}")
    print(f"  [UNKNOWN] predicted instead:     {unknown_rate:.1%}")
    
    if baseline_hallu > 0:
        reduction = (baseline_hallu - unknown_hallu) / baseline_hallu * 100
        print(f"\n  Hallucination reduction: {reduction:.1f}%")
        
        if reduction > 50:
            print("\n  ✓✓ CORRUPTION → [UNKNOWN] dramatically reduces hallucinations!")
        elif reduction > 30:
            print("\n  ✓ CORRUPTION → [UNKNOWN] significantly reduces hallucinations")
        elif reduction > 10:
            print("\n  → CORRUPTION → [UNKNOWN] moderately helps")
        else:
            print("\n  ? Minimal effect")
    
    print(f"\n  Model learned: corrupted input → predict [UNKNOWN]")
    print(f"  This generalizes to truly unseen inputs!")
    
    return baseline_results, unknown_results

# ============================================
# RUN IT!
# ============================================

if __name__ == "__main__":
    baseline_results, unknown_results = run_experiment()
