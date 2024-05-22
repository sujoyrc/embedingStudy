import os
import numpy as np
import pandas as pd
from IsoScore import IsoScore
np.random.seed(123)

# Functions for isotropy calculations
def first_order_computation_isotropy(input_array):
    _, n = input_array.shape
    ones_vec_transposed = np.ones((n, 1))
    norm_term = np.linalg.norm(input_array.dot(ones_vec_transposed))
    score = (n - norm_term) / (n + norm_term)
    return score

def second_order_computation_isotropy(input_array):
    num_vecs, n = input_array.shape
    ones_vec_transposed = np.ones((n, 1))
    U, S, V = np.linalg.svd(input_array)
    sigma_min = S[-1]
    sigma_max = S[0]
    norm_term = np.linalg.norm(input_array.dot(ones_vec_transposed))
    score = (n - norm_term + (0.5 * sigma_min * sigma_min)) / (
        n + norm_term + (0.5 * sigma_max * sigma_max)
    )
    return score

# Main function
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go one level from src
    question_embeddings_dir = os.path.join(base_dir, 'inputs', 'question_embeddings')
    answer_embeddings_dir = os.path.join(base_dir, 'inputs', 'answer_embeddings')
    plots_dir = os.path.join(base_dir, 'plots')
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    # DataFrame to store the results
    results_df = pd.DataFrame(columns=['Embedding Type', 'Model', 'Normal IsoScore', 'First Order IsoScore', 'Second Order IsoScore'])
    
    # Process question and answer embeddings and append to the same DataFrame
    results_list = []
    for embeddings_type, embeddings_dir in [('Question', question_embeddings_dir), ('Answer', answer_embeddings_dir)]:
        for file_name in sorted(os.listdir(embeddings_dir)):
            if file_name.endswith('.npy'):
                full_path = os.path.join(embeddings_dir, file_name)
                embeddings = np.load(full_path)

                # Check if the dtype is float16 and convert to float32 if necessary
                if embeddings.dtype == np.float16:
                    embeddings = embeddings.astype(np.float32)
                
                # Calculate isotropy scores
                normal_isoscore = IsoScore.IsoScore(embeddings) * 100
                first_order_isoscore = first_order_computation_isotropy(embeddings) * 100
                second_order_isoscore = second_order_computation_isotropy(embeddings) * 100

                # Append to results list
                results_list.append({
                    'Embedding Type': embeddings_type,
                    'Model': file_name, 
                    'Normal IsoScore': normal_isoscore, 
                    'First Order IsoScore': first_order_isoscore, 
                    'Second Order IsoScore': second_order_isoscore
                })

    # Convert list to DataFrame and save to Excel file
    results_df = pd.DataFrame(results_list)
    results_df.to_excel(os.path.join(outputs_dir, 'isotropy_scores.xlsx'), index=False)