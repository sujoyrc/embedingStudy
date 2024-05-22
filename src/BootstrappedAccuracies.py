import numpy as np
import pandas as pd
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import default_rng
from tqdm import tqdm
np.random.seed(123)


# Load embeddings from .npy file
def load_embeddings(embedding_file_path):
    embedding_array = np.load(embedding_file_path)
    embedding_array = embedding_array.astype(np.float32) if embedding_array.dtype == np.float16 else embedding_array
    return embedding_array


def get_accuracy_top_k(question_array, answer_array, k=5):

    all_sentence_id_info = np.arange(len(question_array))
    embedding_array = answer_array

    # Calculate similarity matrix
    sim_matrix = question_array.dot(embedding_array.T)

    # Get indices of top-k embeddings with highest similarity for each question
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]

    # Compute accuracy by checking if the true index is within the top-k indices
    correct_answers = np.array([all_sentence_id_info[i] in top_k_indices[i] for i in range(len(question_array))])
    top_k_accuracy = np.mean(correct_answers)

    return top_k_accuracy


def bootstrap_accuracy_top_k(question_array, answer_array, k=5, p=100, m=100):

    all_sentence_id_info = np.arange(len(question_array))
    embedding_array = answer_array
    accuracies = []
    rng = default_rng()
    all_sentence_id_info=np.array(all_sentence_id_info)

    for _ in tqdm(range(m),total=m):
        sample_indices = rng.choice(len(question_array), size=p, replace=False)
        sampled_questions = question_array[sample_indices]
        sampled_answers = all_sentence_id_info[sample_indices]

        sim_matrix = sampled_questions.dot(embedding_array.T)
        top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
        correct = np.array([sampled_answers[i] in top_k_indices[i] for i in range(p)])
        accuracies.append(np.mean(correct))

    mean_accuracy = round(100*np.mean(accuracies),2)

    lower_bound = np.percentile(accuracies, 2.5)
    upper_bound = np.percentile(accuracies, 97.5)
    
    upper_bound=round(100*upper_bound,2)
    lower_bound=round(100*lower_bound,2)
    ci=(lower_bound,upper_bound)

    return mean_accuracy,ci


def evaluate_models_bootstrapped(question_embeddings, answer_embeddings\
                                 , outputs_dir, k_range=range(5, 105, 5)):
    results = pd.DataFrame(columns=[f"k={k}" for k in k_range], index=question_embeddings.keys())
    for model_name, embeddings in question_embeddings.items():
        print ("processing ",model_name)     
        output_dir = os.path.join(outputs_dir, 'bootstrapped_accuracies')
        os.makedirs(output_dir,exist_ok=True)
        output_file_name=model_name+'_bootstrapped_accuracies.xlsx'
        output_file_name=os.path.join(output_dir,output_file_name) 
        
        if os.path.isfile(output_file_name):
            print (model_name," already processed ... skipping")
            continue
            
        print ("Running for ",model_name)
        embedding_file = answer_embeddings[model_name]
        for k in k_range:
            mean_accuracy, ci = bootstrap_accuracy_top_k(embeddings, embedding_file, k=k)
            results.at[model_name, f"k={k}"] = f"{mean_accuracy} ({ci[0]}, {ci[1]})"

        results_model=results.T
        results_model.to_excel(output_file_name,index=True)
        
    

    return results


def compute_accuracies_for_full_data(question_embeddings, answer_embeddings, k_range=range(5, 101, 5)):
    # Prepare DataFrame
    df = pd.DataFrame(index=[f'k={k}' for k in k_range], columns=question_embeddings.keys())
    
    # Iterate over each model and compute accuracy for each k
    for model_name, embeddings in tqdm(question_embeddings.items(),total=len(answer_embeddings.keys())):
        print ("Processing ",model_name)
        embedding_file = answer_embeddings[model_name]
        for k in k_range:
            accuracy = get_accuracy_top_k(embeddings, embedding_file, k=k)
            df.loc[f'k={k}', model_name] = round(100*accuracy,2)
    
    return df


def plot_swimlanes(data, ks, plot_title, plots_dir):
    # Ensure data index is proper if string indices are used
    data_filtered = data.loc[ks]

    # Extracting model names from columns
    models = data_filtered.columns

    # Setup figure and axes
    fig, axes = plt.subplots(nrows=len(ks), figsize=(10, len(ks) * 4), sharex=False)
    if len(ks) == 1:
        axes = [axes]  # Ensure axes is iterable for a single row scenario

    # Plotting each swimlane
    for ax, k in zip(axes, ks):
        # Getting the specific row for current K
        row = data_filtered.loc[k]
        means = []
        cis = []
        for model in models:
            # Parsing mean and CI from the data
            value=row[model]
            #print ("VALUE:",value)
            mean,lower,upper=value.split(' ')[0],value.split(' ')[1],value.split(' ')[2]
            #print (lower,upper)
            mean = float(mean)
            lower = float(lower.strip('(').strip(','))
            upper = float(upper.strip(')'))

            means.append(mean)
            # Correctly formatting ci for matplotlib errorbar
            cis.append([mean - lower, upper - mean])

        # Horizontal line plot for CI and dot for mean
        y_positions = np.arange(len(models))
        for pos, mean, (ci_lower, ci_upper) in zip(y_positions, means, cis):
            ax.errorbar(mean, pos, xerr=[[ci_lower], [ci_upper]], fmt='o', color='black', capsize=2, markersize=5, alpha=0.7)


        ax.set_yticks(y_positions)
        ax.set_yticklabels(models)
        ax.invert_yaxis()  # Invert to match the order in the DataFrame
        ax.set_xlabel('Percentage (%)')
        ax.set_title(f' {k}')
        ax.set_xlim(0, 100)  # Assuming the data is in percentage

    fig.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top to accommodate the suptitle
    plt.savefig(os.path.join(plots_dir, plot_title + '.png'))  # Save the swimlane plot
    plt.close()  # Close the plot


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

    # Bootstrapped Accuracies
    bootstrapped_accuracies=evaluate_models_bootstrapped(question_embeddings,answer_embeddings,outputs_dir)
    bootstrapped_accuracies.T.to_excel(os.path.join(outputs_dir, 'bootstrapped_accuracies.xlsx'), index=True)
    bootstrapped_accuracies=pd.read_excel(os.path.join(outputs_dir, 'bootstrapped_accuracies.xlsx'),index_col=[0])

    # Full Data Accuracies
    accuracy_df = compute_accuracies_for_full_data(question_embeddings, answer_embeddings)
    accuracy_df.to_excel(os.path.join(outputs_dir, 'accuracy_df.xlsx'), index=True)
    accuracy_df=pd.read_excel(os.path.join(outputs_dir, 'accuracy_df.xlsx'),index_col=[0])

    plot_swimlanes(bootstrapped_accuracies,['k=5','k=25','k=50','k=100'],"bootstrapped_accuracies",plots_dir)