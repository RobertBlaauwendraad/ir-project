import pyterrier as pt
import pyt_splade
import torch
import os
import warnings

warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


def main():
    torch.manual_seed(26)  # for reproducibility

    def detect_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    device = detect_device()
    print(f"Using device: {device}")

    # Load SPLADE model
    splade = pyt_splade.Splade(model="naver/splade-cocondenser-ensembledistil", device=device, max_length=256)

    # Load Robust04 dataset
    dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")

    # Set up indexer
    index_dir = os.path.abspath("./robust04_splade_index")
    indexer = pt.IterDictIndexer(index_dir, pretokenised=True, verbose=True, overwrite=True)

    # Combine title and body fields using vectorized operation
    combine = pt.apply.text(lambda doc: doc.get('title', '') + ' ' + doc.get('body', ''))

    # Create an indexing pipeline: encode documents with SPLADE, then index
    indexing_pipeline = combine >> splade.doc_encoder(text_field='text') >> indexer

    # Build the index
    index_ref = indexing_pipeline.index(dataset.get_corpus_iter(), batch_size=64)


if __name__ == "__main__":
    main()
