import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import default_rng
import datetime
from tqdm import tqdm
np.random.seed(123)


# Load embeddings from .npy file
def load_embeddings(embedding_file_path):
    embedding_array = np.load(embedding_file_path)
    embedding_array = embedding_array.astype(np.float32) if embedding_array.dtype == np.float16 else embedding_array
    return embedding_array


def bootstrap_min_sim(question_array, answer_array, all_sentence_id_info, k=5, p=100, m=100):

    min_sims = []
    rng = default_rng()
    all_sentence_id_info = np.array(all_sentence_id_info)

    for _ in tqdm(range(m), total=m):
        sample_indices = rng.choice(len(question_array), size=p, replace=False)
        sampled_questions = question_array[sample_indices]
        sampled_answers = all_sentence_id_info[sample_indices]

        sim_matrix = sampled_questions.dot(answer_array.T)
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
        # Fetch top-k similarities
        top_k_sims = np.take_along_axis(sim_matrix, top_k_indices, axis=1)
        # Fetch similarities for correct answers
        correct_sims = sim_matrix[np.arange(p), sampled_answers]
        # Find minimum similarity in top-k and the correct answer's similarity
        min_sims_per_question = np.min(top_k_sims)
        # Find minimum of these minimums across all questions
        min_sims.append(np.min(min_sims_per_question)-0.01)

    return np.array(min_sims)


def bootstrap_accuracy_top_k_with_matrix(sim_matrix, all_sentence_id_info, k=5, p=100, m=100):

    accuracies = []
    rng = default_rng()
    all_sentence_id_info = np.array(all_sentence_id_info)

    for _ in tqdm(range(m), total=m):
        sample_indices = rng.choice(len(sim_matrix), size=p, replace=False)
        sampled_sims = sim_matrix[sample_indices, :]
        sampled_answers = all_sentence_id_info[sample_indices]

        top_k_indices = np.argsort(sampled_sims, axis=1)[:, -k:]
        correct = np.array([
            sampled_answers[i] in top_k_indices[i] and sim_matrix[i, all_sentence_id_info[i]] != 0 
            for i in range(p)
        ])
        accuracies.append(np.mean(correct))

    mean_accuracy = round(100 * np.mean(accuracies), 2)
    lower_bound = np.percentile(accuracies, 2.5)
    upper_bound = np.percentile(accuracies, 97.5)
    ci = (round(100 * lower_bound, 2), round(100 * upper_bound, 2))

    return mean_accuracy, ci


def find_optimal_threshold(sim_matrix, all_sentence_id_info, sim_thresholds_per_draw, j=100\
                           , percentile_range=(20, 50, 5)):

    # Compute the baseline accuracy
    baseline_accuracy, _ = bootstrap_accuracy_top_k_with_matrix(sim_matrix, all_sentence_id_info)

    # Initialize variables to find the optimal threshold and accuracy
    best_threshold = None
    best_accuracy = None
    best_percentile = None

    # Iterate over the specified percentile range
    for percentile in np.arange(*percentile_range):
        print("Running for ", percentile)
        threshold = np.percentile(sim_thresholds_per_draw, percentile)
        masked_sims = np.where(sim_matrix >= threshold, sim_matrix, 0)

        # Compute accuracy with the threshold applied
        threshold_accuracy, _ = bootstrap_accuracy_top_k_with_matrix(masked_sims, all_sentence_id_info)

        # Update best results if this accuracy is within the tolerance and is better than any found so far
        if abs(threshold_accuracy - baseline_accuracy) <= j:
            if best_accuracy is None or threshold_accuracy > best_accuracy:
                best_threshold = threshold
                best_accuracy = threshold_accuracy
                best_percentile = percentile

    return best_threshold, baseline_accuracy, best_accuracy, best_percentile


def get_thresholds_for_model(model_name,question_embeddings,answer_embeddings,all_sentence_id_info, m=1000):
    this_question_embedding=question_embeddings[model_name]
    answer_embedding=answer_embeddings[model_name]
    this_embedding_array=answer_embedding
    
    sim_thresholds_per_draw=bootstrap_min_sim(this_question_embedding,this_embedding_array,all_sentence_id_info,m=m)
    
    sim_matrix_full=this_question_embedding.dot(this_embedding_array.T)
    optimal_threshold, baseline_accuracy, accuracy_with_threshold,percentile = find_optimal_threshold(sim_matrix_full, all_sentence_id_info, sim_thresholds_per_draw)
    
    return optimal_threshold, baseline_accuracy,accuracy_with_threshold,percentile


# Main function
if __name__ == '__main__':
    # Set the base directory relative to the src directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go one level from src
    question_embeddings_dir = os.path.join(base_dir, 'inputs', 'question_embeddings')
    answer_embeddings_dir = os.path.join(base_dir, 'inputs', 'answer_embeddings')
    plots_dir = os.path.join(base_dir, 'plots')
    outputs_dir = os.path.join(base_dir, 'outputs')

    # Initialize dictionaries to store distributions
    question_embeddings = {}
    answer_embeddings = {}

    # Process each pair of question and answer embeddings
    for file_name in os.listdir(question_embeddings_dir):
        if file_name.endswith('.npy'):
            # Use the same filename for both question and answer embeddings
            model_name = os.path.splitext(file_name)[0]  # Remove '.npy' extension
            
            # Load question and answer embeddings
            question_embeddings_path = os.path.join(question_embeddings_dir, file_name)
            answer_embeddings_path = os.path.join(answer_embeddings_dir, file_name)
            
            question_array = load_embeddings(question_embeddings_path)
            answer_array = load_embeddings(answer_embeddings_path)

            question_embeddings[model_name] = question_array
            answer_embeddings[model_name] = answer_array

    results_thresholding={}
    for model in list(question_embeddings.keys()):
        t1=datetime.datetime.now()
        print ("Running for ",model," at ",t1)
        all_sentence_id_info = np.arange(len(question_embeddings[model]))
        this_optimal,this_baseline,this_modified,this_percentile=get_thresholds_for_model(model,question_embeddings,answer_embeddings,all_sentence_id_info,m=500)
        results_thresholding[model]=(this_optimal,this_baseline,this_modified,this_percentile)
        t2=datetime.datetime.now()
        runtime=(t2-t1).total_seconds()
        print (results_thresholding[model])
        print ("Runtime for model ",model," is ",runtime," seconds")

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(results_thresholding, orient='index', columns=['optimal_threshold', 'baseline_accuracy', 'accuracy_with_threshold', 'percentile'])
    
    # Save the DataFrame to an Excel file
    df.to_excel(os.path.join(outputs_dir, 'results_thresholding.xlsx'), index=True)