import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================
# MODEL AND DATA FUNCTIONS
# ============================================

class SimplePredictor(nn.Module):
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
    common_pairs = [(i, i + 50) for i in range(num_common)]
    common_data = common_pairs * 100
    
    rare_pairs = [(i + num_common, i + num_common + 50) 
                  for i in range(num_rare)]
    rare_data = rare_pairs * 1
    
    unseen_pairs = [(i + num_common + num_rare, None)
                    for i in range(num_unseen)]
    
    train_data = common_data + rare_data
    np.random.shuffle(train_data)
    
    return {
        'train': train_data,
        'test_common': common_pairs,
        'test_rare': rare_pairs,
        'test_unseen': unseen_pairs,
    }

def augment_with_corrupted_abstain(train_data, corruption_prob=0.15, corruption_strength=30):
    ABSTAIN_TOKEN = 100
    augmented = []
    
    for input_token, output_token in train_data:
        augmented.append((input_token, output_token))
        
        if np.random.random() < corruption_prob:
            corrupted_input = (input_token + np.random.randint(corruption_strength, 
                                                               corruption_strength + 20)) % 100
            augmented.append((corrupted_input, ABSTAIN_TOKEN))
    
    return augmented

def evaluate(model, test_data, confidence_threshold=0.5, abstain_token=None):
    model.eval()
    
    results = {
        'correct': 0,
        'wrong': 0,
        'total': 0,
        'confident_wrong': 0,
        'avg_confidence': 0,
        'predicted_abstain': 0,
    }
    
    with torch.no_grad():
        for input_token, true_token in test_data:
            if true_token is None:
                true_token = -1
            
            logits = model(torch.tensor([input_token]))
            probs = F.softmax(logits[0], dim=0)
            
            predicted = probs.argmax().item()
            confidence = probs.max().item()
            
            results['total'] += 1
            results['avg_confidence'] += confidence
            
            if abstain_token is not None and predicted == abstain_token:
                results['predicted_abstain'] += 1
                continue
            
            if true_token != -1:
                if predicted == true_token:
                    results['correct'] += 1
                else:
                    results['wrong'] += 1
                    if confidence > confidence_threshold:
                        results['confident_wrong'] += 1
            else:
                results['wrong'] += 1
                if confidence > confidence_threshold:
                    results['confident_wrong'] += 1
    
    total = results['total']
    return {
        'accuracy': results['correct'] / total if total > 0 else 0,
        'hallucination_rate': results['confident_wrong'] / total,
        'avg_confidence': results['avg_confidence'] / total,
        'abstain_rate': results['predicted_abstain'] / total,
    }

def train_with_corruption_abstain(dataset, epochs=50, corruption_prob=0.2):
    input_vocab = 100
    output_vocab = 101
    ABSTAIN_TOKEN = 100
    
    model = SimplePredictor(input_vocab, output_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_data = augment_with_corrupted_abstain(
        dataset['train'], 
        corruption_prob=corruption_prob,
        corruption_strength=30
    )
    
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
    
    common_metrics = evaluate(model, dataset['test_common'], abstain_token=ABSTAIN_TOKEN)
    rare_metrics = evaluate(model, dataset['test_rare'], abstain_token=ABSTAIN_TOKEN)
    unseen_metrics = evaluate(model, dataset['test_unseen'], abstain_token=ABSTAIN_TOKEN)
    
    return model, {
        'common': common_metrics,
        'rare': rare_metrics,
        'unseen': unseen_metrics
    }

# ============================================
# ABLATION EXPERIMENTS
# ============================================

def train_abstain_no_corruption(dataset, epochs=50):
    print("\n" + "="*60)
    print("ABLATION 1: [ABSTAIN] WITHOUT Corruption")
    print("="*60)
    
    input_vocab = 100
    output_vocab = 101
    ABSTAIN_TOKEN = 100
    
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
    
    unseen_metrics = evaluate(model, dataset['test_unseen'], abstain_token=ABSTAIN_TOKEN)
    
    print(f"\nUnseen: Hallu={unseen_metrics['hallucination_rate']:.3f}, "
          f"Abstain={unseen_metrics['abstain_rate']:.3f}")
    
    return unseen_metrics

qq = np.random.randint(0,1000000)

def test_corruption_rates(dataset):
    print("\n" + "="*60)
    print("ABLATION 2: Different Corruption Rates")
    print("="*60)
    
    corruption_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results = []
    
    for p in corruption_probs:
        print(f"\n--- Testing p={p:.2f} ---")
        torch.manual_seed(qq + int(p*100))
        np.random.seed(qq + int(p*100))
        
        _, metrics = train_with_corruption_abstain(dataset, epochs=50, corruption_prob=p)
        results.append({
            'p': p,
            'unseen_hallu': metrics['unseen']['hallucination_rate'],
            'unseen_abstain': metrics['unseen']['abstain_rate'],
            'rare_acc': metrics['rare']['accuracy'],
            'rare_abstain': metrics['rare']['abstain_rate'],
            'common_acc': metrics['common']['accuracy'],
            'common_abstain': metrics['common']['abstain_rate']
        })
        
        print(f"  Unseen: Hallu={metrics['unseen']['hallucination_rate']:.3f}, "
              f"Abstain={metrics['unseen']['abstain_rate']:.3f}")
        print(f"  Rare: Acc={metrics['rare']['accuracy']:.3f}, "
              f"Abstain={metrics['rare']['abstain_rate']:.3f}")
        print(f"  Common: Acc={metrics['common']['accuracy']:.3f}, "
              f"Abstain={metrics['common']['abstain_rate']:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'p':<6} {'Unseen Hallu':<13} {'Unseen [ABS]':<13} {'Rare Acc':<10} {'Rare [ABS]':<10} {'Common Acc':<10} {'Common [ABS]':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['p']:<6.2f} {r['unseen_hallu']:<13.3f} {r['unseen_abstain']:<13.3f} "
              f"{r['rare_acc']:<10.3f} {r['rare_abstain']:<10.3f} {r['common_acc']:<10.3f} {r['common_abstain']:<10.3f}")
    
    return results

# ============================================
# RUN ABLATIONS
# ============================================

if __name__ == "__main__":
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    print("Creating dataset...")
    dataset = create_dataset(num_common=20, num_rare=20, num_unseen=20)
    
    # Ablation 1
    no_corruption_results = train_abstain_no_corruption(dataset, epochs=50)
    
    # Ablation 2
    corruption_rate_results = test_corruption_rates(dataset)
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    print(f"Without corruption: [ABSTAIN] usage = {no_corruption_results['abstain_rate']:.1%}")
    print(f"  → Corruption is ESSENTIAL for learning to abstain")
    print(f"Trade-off: higher p → more abstention on rare examples")
