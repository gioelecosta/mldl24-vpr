from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

'''
Miners combined with losses: 
Mining functions take a batch of n embeddings and return k pairs/triplets to be used for calculating the loss:
Embeddings are vector that rapresent input data, in this case images, and archors are is a specific type of embedding that
serves as a reference point for calculating similarities or distances with other embeddings.
In the context of pair and triplet mining, anchors are typically the "center" or "query" embeddings that are used to find positive and negative examples.
In particular:
- Pair miners output a tuple of size 4: (anchors, positives), (anchors, negatives).
- Triplet miners output a tuple of size 3: (anchors, positives, negatives).
- Without a tuple miner, loss functions will by default use all possible pairs/triplets in the batch.
'''

def get_loss(loss_name):
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

def get_miner(miner_name, margin=0.1):
    if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
    if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    return None
