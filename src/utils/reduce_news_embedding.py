"""
Reduce news embedding dimension using PCA.

Usage:
    python src/utils/reduce_news_embedding.py --input src/data/TWII/feature5-sc47-2013-2025-finlab/cls_embeddings.npy --dim 64
    python src/utils/reduce_news_embedding.py --input src/data/TWII/feature5-sc47-2013-2025-finlab/cls_embeddings.npy --dim 32
"""

import argparse
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path


def reduce_embedding_dim(input_path: str, target_dim: int, output_path: str = None):
    print(f"Loading {input_path}...")
    embeddings = np.load(input_path)
    original_shape = embeddings.shape
    print(f"Original shape: {original_shape}")  # (num_assets, num_days, 768)

    num_assets, num_days, embed_dim = original_shape

    if target_dim >= embed_dim:
        print(f"Target dim {target_dim} >= original dim {embed_dim}, skipping.")
        return None

    # Reshape to 2D for PCA: (num_assets * num_days, embed_dim)
    embeddings_2d = embeddings.reshape(-1, embed_dim)
    print(f"Reshaped for PCA: {embeddings_2d.shape}")

    # Fit PCA
    print(f"Fitting PCA to reduce {embed_dim} -> {target_dim} dimensions...")
    pca = PCA(n_components=target_dim, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings_2d)

    # Report explained variance
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio: {explained_var:.4f} ({explained_var*100:.2f}%)")

    # Reshape back to 3D: (num_assets, num_days, target_dim)
    embeddings_reduced = embeddings_reduced.reshape(num_assets, num_days, target_dim)
    print(f"Reduced shape: {embeddings_reduced.shape}")

    # Generate output path if not specified
    if output_path is None:
        input_dir = Path(input_path).parent
        output_path = str(input_dir / f"cls_embeddings_dim{target_dim}.npy")

    # Save
    print(f"Saving to {output_path}...")
    np.save(output_path, embeddings_reduced.astype(np.float32))

    # Verify
    loaded = np.load(output_path)
    print(f"Verified: {loaded.shape}, dtype={loaded.dtype}")
    print(f"Stats: min={loaded.min():.4f}, max={loaded.max():.4f}, mean={loaded.mean():.4f}, std={loaded.std():.4f}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce news embedding dimension using PCA")
    parser.add_argument("--input", type=str, required=True, help="Input cls_embeddings.npy path")
    parser.add_argument("--dim", type=int, default=64, help="Target dimension (default: 64)")
    parser.add_argument("--output", type=str, default=None, help="Output path (optional)")

    args = parser.parse_args()
    reduce_embedding_dim(args.input, args.dim, args.output)
