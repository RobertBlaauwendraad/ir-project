import ir_datasets
import pyterrier as pt

pt.init()
dataset = pt.get_dataset('irds:msmarco-passage')

indexer = pt.IterDictIndexer('./msmarco_index', overwrite=True)

index_ref = indexer.index(dataset.get_corpus_iter())