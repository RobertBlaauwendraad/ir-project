import pyterrier as pt
from pyterrier.measures import *
import pandas as pd

dataset = pt.get_dataset("msmarco_passage")

index_ref = dataset.get_index("terrier_stemmed")
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", verbose=True, properties={"terrier.max.threads": "16"})

topics = dataset.get_topics('dev.small')
topics['query'] = topics['query'].str.replace(r'[\?\:\!]', '', regex=True)
qrels = dataset.get_qrels('dev.small')

experiment = pt.Experiment(
    [bm25],
    topics,
    qrels,
    [nDCG@10, AP@100]
)

print(experiment)