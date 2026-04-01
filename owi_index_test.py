import pyterrier as pt
if not pt.started():
    pt.init()

index_path = "./data/owi_splade_index" 

try:
    owi_index = pt.IndexFactory.of(index_path)
    
    stats = owi_index.getCollectionStatistics()
    print(f"Index Loaded Successfully!")
    print(f"Number of Documents: {stats.getNumberOfDocuments()}")
    print(f"Number of Unique Terms: {stats.getNumberOfUniqueTerms()}")
    
except Exception as e:
    print(f"Could not load index: {e}")
    print("Double-check the path. It should be the folder containing 'data.properties'.")
