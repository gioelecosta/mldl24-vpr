import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)

        # build index
        else:
            # build an IndexFlatL2 index with dimensionality embed_size
            # is a flat index that uses the L2 Euclidean distance to compute the similarity between vectors
            faiss_index = faiss.IndexFlatL2(embed_size) 
        
        # add references
        faiss_index.add(r_list)

        # search  max(k_values) nearest to queries in the faiss_index where are the references
        # predictions is a vector containing the indices of the  nearest 
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        
        
        # start calculating recall_at_k in this way:
        # for each prediction we check if the ground truth index is present in the top-N predicted indices,
        # where N is each value in k_values.
        # If the ground truth index is found, we increments the corresponding correct_at_k value 
        # and stop the second loop.
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        # dictionary where keys are the values of k_values vector and
        # values are the correct prediction proportion from correct_at_k
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print('\n') 
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performance on {dataset_name}"))
        
        return d, predictions
