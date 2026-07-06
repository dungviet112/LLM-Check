"""
Script to extract last layer contextual embeddings of the last generated token
from the fava_annot dataset and save them to disk.

Usage:
    python extract_fava_embeddings.py --model 'open-llama-7b' --n_samples 100
"""

import argparse
import torch
from six.moves import cPickle as pkl
from common_utils import get_full_model_name
from utils_fava_annotated import get_fava_data, extract_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="open-llama-7b", 
                   help="model name or path for the evaluator model")
parser.add_argument("--n_samples", type=int, default=100,
                   help="number of samples to extract embeddings for")
parser.add_argument("--output_file", type=str, default=None,
                   help="path to save embeddings (default: data/embeddings_fava_annot_<model>_<n_samples>.pkl)")

args = parser.parse_args()

if __name__ == "__main__":
    n_samples = args.n_samples
    model_name_or_path = get_full_model_name(args.model.lower())[1]
    
    print(f"Model: {args.model.lower()}, Samples: {n_samples}")
    print(f"Loading fava_annot dataset...")
    
    # Load dataset
    sample_data, _ = get_fava_data(n_samples=n_samples)
    
    print(f"Extracting last layer embeddings...")
    # Extract embeddings
    embeddings, labels = extract_embeddings(model_name_or_path, sample_data, args)
    
    # Convert embeddings list to tensor for easier manipulation
    embeddings_tensor = torch.stack(embeddings)  # Shape: (n_samples, hidden_dim)
    
    print(f"Embeddings shape: {embeddings_tensor.shape}")
    print(f"Labels distribution: {sum(labels)} hallucinated, {len(labels) - sum(labels)} non-hallucinated")
    
    # Determine output file path
    if args.output_file is None:
        output_file = f"data/embeddings_fava_annot_{args.model.lower()}_{n_samples}.pkl"
    else:
        output_file = args.output_file
    
    # Save embeddings and labels
    with open(output_file, "wb") as f:
        pkl.dump([embeddings_tensor, labels], f)
    
    print(f"✓ Embeddings saved to: {output_file}")
    print(f"  - Embeddings tensor shape: {embeddings_tensor.shape}")
    print(f"  - Labels: {len(labels)}")
