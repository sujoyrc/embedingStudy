import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from scipy.stats import sem, t
from scipy.stats import percentileofscore
import re
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
np.random.seed(123)


# Load embeddings from .npy file
def load_embeddings(embedding_file_path):
    embedding_array = np.load(embedding_file_path)
    embedding_array = embedding_array.astype(np.float32) if embedding_array.dtype == np.float16 else embedding_array
    return embedding_array


# Function to calculate similarity matrix
def get_orig_sims(question_array, answer_array):
    
    all_sentence_id_info = np.arange(len(question_array))
    embedding_array = answer_array
    
    # Calculate similarity matrix
    sim_matrix = question_array.dot(embedding_array.T)

    # Extract similarities for the correct answers
    correct_indices = np.arange(len(question_array))
    correct_sims = sim_matrix[correct_indices, all_sentence_id_info]

    return correct_sims


# Function to get random, best, and worst k similarities
def get_random_best_and_worst_k_sims(question_array,answer_array,k=5):
    embedding_array = answer_array
    sim_matrix=question_array.dot(embedding_array.T)
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    worst_k_indices = np.argsort(sim_matrix, axis=1)[:, :k]
    top_k_sims = np.take_along_axis(sim_matrix, top_k_indices, axis=1)
    worst_k_sims = np.take_along_axis(sim_matrix, worst_k_indices, axis=1)
    top_k_mean_sims=np.mean(top_k_sims,axis=1)
    worst_k_mean_sims=np.mean(worst_k_sims,axis=1)
    random_sims=[]
    for j in range(question_array.shape[0]):
        random_index=np.random.randint(0,embedding_array.shape[0])
        this_random_sim=sim_matrix[j,random_index]
        random_sims.append(this_random_sim)
    random_sims=np.array(random_sims)
    return random_sims,top_k_mean_sims, worst_k_mean_sims


# Function to get only top k similarities
def get_only_top_k_sims(question_array,answer_array,k):
    embedding_array = answer_array
    sim_matrix=question_array.dot(embedding_array.T)
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
    top_k_sims = np.take_along_axis(sim_matrix, top_k_indices, axis=1)
    top_k_sims_unfolded=top_k_sims.ravel()
    return top_k_sims_unfolded


# Function to plot and save similarity histograms
def plot_sims_histograms(top_k_sims, correct_sims, random_sims, bottom_k_sims,title, plots_dir, kde=True):
    plt.figure(figsize=(6, 6))
    sns.histplot(top_k_sims.flatten(), color='blue', stat='density', kde=kde, label='Top K Sims', alpha=0.3,bins=50)
    sns.histplot(correct_sims.flatten(), color='green', stat='density', kde=kde, label='Correct Sims', alpha=0.3,bins=50)
    sns.histplot(random_sims.flatten(), color='red', stat='density', kde=kde, label='Random Sims', alpha=0.3,bins=50)
    sns.histplot(bottom_k_sims.flatten(), color='orange', stat='density', kde=kde, label='Bottom K Sims', alpha=0.3,bins=50)
    plt.legend()
    plt.title(title)
    plt.xlim(-0.1,1.1)
    plt.savefig(os.path.join(plots_dir, title + '.png'))  # Save plot
    plt.close()  # Close the plot


# Main function to process embeddings and generate plots
def get_and_plot_hists(question_array,answer_array,title,plots_dir,k=5):
    correct_sims=get_orig_sims(question_array,answer_array)
    random_sims,top_k_sims,bottom_k_sims=get_random_best_and_worst_k_sims(question_array\
                                                                          ,answer_array,k)
    plot_sims_histograms(top_k_sims,correct_sims,random_sims,bottom_k_sims,title,plots_dir)
    return top_k_sims,correct_sims,random_sims,bottom_k_sims


# Function to plot and save ECDFs
def plot_cdfs(ecdf_random,ecdf_correct,ecdf_top_k,model,output_dir):
    ax = plt.subplot()
    ecdf_random.cdf.plot(ax,color='red',label='random')
    ecdf_correct.cdf.plot(ax,color='green',label='correct')
    ecdf_top_k.cdf.plot(ax,color='blue',label='top-5')
    ax.legend()
    ax.set_title('Comparing CDF for '+model+' model')
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xlim(-0.1,1.1)
    plt.savefig(os.path.join(output_dir, model + '_ecdf.png'))  # Save ECDF plot
    plt.close()  # Close the plot


# Bootstrap function to calculate probabilities with confidence intervals
def bootstrap_prob_with_ci(source, target, percentile_limit, sample_size=1, n_bootstrap=50000):
    threshold = np.percentile(source, percentile_limit)
    source_filtered = source[source <= threshold]
    ecdf = ECDF(target)
    
    estimates = []
    for _ in tqdm(range(n_bootstrap), total=n_bootstrap):
        sampled = np.random.choice(source_filtered, size=sample_size, replace=True)
        probabilities = [1 - ecdf(x) for x in sampled]
        avg_prob = np.mean(probabilities)
        estimates.append(avg_prob)
    
    mean_estimate = np.mean(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)
    
    return mean_estimate, ci_lower, ci_upper


# Function to compute probabilities and CIs for each model and k
def compute_probabilities_and_cis(model_dists, dist_type, percentile_limit, n_bootstrap=50000):
    results = pd.DataFrame(index=np.arange(5, 101, 5), columns=model_dists.keys())
    for model_name, model_vectors_list in model_dists.items():
        dist = dist_type[model_name]
        for i, model_vectors in enumerate(model_vectors_list):
            k_value = 5 * (i + 1)  # k=5,10,...,100
            prob, ci_lower, ci_upper = bootstrap_prob_with_ci(
                model_vectors, dist, percentile_limit, n_bootstrap=n_bootstrap)
            results.loc[k_value, model_name] = f"{prob * 100:.2f}% ({ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%)"
    return results


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
            value = row[model].strip('%')
            mean, ci = value.split(' (')[0], value.split(' (')[1].strip(')%')
            lower, upper = ci.split(',')
            mean = float(re.sub('%','',mean))
            lower = float(re.sub('%','',lower))
            upper = float(re.sub('%','',upper))

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
        ax.set_title(f'K = {k}')
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
    random_dists = {}
    model_dists = {}
    correct_dists = {}

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
            
            # Get similarities and plot histograms
            top_k_sims, correct_sims, random_sims, bottom_k_sims = get_and_plot_hists(question_array, answer_array, model_name, plots_dir)
            
            # Calculate ECDFs and plot them
            ecdf_random = stats.ecdf(random_sims)
            ecdf_correct = stats.ecdf(correct_sims)
            ecdf_top_k = stats.ecdf(top_k_sims)
            plot_cdfs(ecdf_random, ecdf_correct, ecdf_top_k, model_name, plots_dir)
            
            # Store distributions
            random_dists[model_name] = random_sims
            correct_dists[model_name] = correct_sims
            
            # Calculate varying top k similarities
            k_varying_distribution = []
            for top_k in tqdm(range(5,105,5),total=20):
                this_top_k_this_model = get_only_top_k_sims(question_array, answer_array, top_k)
                k_varying_distribution.append(this_top_k_this_model)
            
            model_dists[model_name] = k_varying_distribution
 
    # Compute probabilities and CIs for each model and k
    results_ci_10_with_random = compute_probabilities_and_cis(model_dists, random_dists, percentile_limit=10)
    results_ci_50_with_random = compute_probabilities_and_cis(model_dists, random_dists, percentile_limit=50)
    results_ci_10_with_correct = compute_probabilities_and_cis(model_dists, correct_dists, percentile_limit=10)
    results_ci_50_with_correct = compute_probabilities_and_cis(model_dists, correct_dists, percentile_limit=50)

    # Save the results to Excel files in the outputs directory
    results_ci_10_with_random.to_excel(os.path.join(outputs_dir, 'results_ci_10_with_random.xlsx'), index=True)
    results_ci_50_with_random.to_excel(os.path.join(outputs_dir, 'results_ci_50_with_random.xlsx'), index=True)
    results_ci_10_with_correct.to_excel(os.path.join(outputs_dir, 'results_ci_10_with_correct.xlsx'), index=True)
    results_ci_50_with_correct.to_excel(os.path.join(outputs_dir, 'results_ci_50_with_correct.xlsx'), index=True)

    # Calling the plot_swimlanes function in main after computing the results
    plot_swimlanes(results_ci_10_with_correct, [5, 10, 25, 100], "results_ci_10_with_correct", plots_dir)
    plot_swimlanes(results_ci_50_with_correct, [5, 10, 25, 100], "results_ci_50_with_correct", plots_dir)
    plot_swimlanes(results_ci_10_with_random, [5, 10, 25, 100], "results_ci_10_with_random", plots_dir)
    plot_swimlanes(results_ci_50_with_random, [5, 10, 25, 100], "results_ci_50_with_random", plots_dir)