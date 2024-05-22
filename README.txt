Directory Structure:
--------------------
There are 4 main folders in this directory:

1. inputs
2. outputs
3. plots
4. src

Inputs Folder:
--------------
- Contains 2 subdirectories: 
  - question_embeddings
  - answer_embeddings
- Each subdirectory holds the corresponding embeddings in numpy (.npy) format. 
- The file names in both subdirectories must be consistent to ensure the code runs correctly. Refer to the sample files provided for the correct naming convention.

Outputs Folder:
---------------
- Stores all the generated results in Excel sheet format.

Plots Folder:
-------------
- Stores all the generated plots and histograms.

Src Folder:
-----------
- Contains the code files. There are 4 main Python scripts which can be executed individually to get the desired results:
  1. EmbeddingComparison.py
  2. BootstrappedAccuracies.py
  3. BootstrappedSimilarities.py
  4. IsotropyScores.py

- The Python scripts are self-sufficient and automatically take input embeddings and save outputs to the required folders.

Python Scripts Description:
---------------------------
1. EmbeddingComparison.py:
   - Calculates correct, random, top k, and bottom k similarities for the embeddings.
   - Computes the CDF scores and saves the similarity and CDF plots in the plots folder.
   - Calculates P(Correct | Top-K) and P(Random | Top-K) at the 10th and 50th percentiles with confidence intervals for k = 5 to 100.
   - Stores the results in Excel files in the outputs folder.
   - Plots swimlanes which are saved in the plots folder.

2. BootstrappedAccuracies.py:
   - Calculates bootstrapped accuracy and full data accuracies (accuracy_df) for k = 5 to 100.
   - Stores the results in Excel files in the outputs folder.
   - The swimlane for bootstrapped accuracies is saved in the plots folder.

3. BootstrappedSimilarities.py:
   - Calculates Similarity Threshold and the corresponding accuracy.
   - Stores the results in Excel files in the outputs folder, including optimal threshold, baseline accuracy, accuracy with threshold, and percentile.

4. IsotropyScores.py:
   - Calculates Isotropy scores (Isoscores), first order isotropy scores, and second order isotropy scores for all embeddings (both questions and answers).
   - Stores the results in Excel format in the outputs directory.

Library Imports/Requirements:
-----------------------------
- The following libraries are required and should be imported in each of the Python scripts:

  import numpy as np
  import os
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from statsmodels.distributions.empirical_distribution import ECDF
  from scipy import stats
  from scipy.stats import sem, t
  from scipy.stats import percentileofscore
  from numpy.random import default_rng
  import re
  import warnings
  from tqdm import tqdm
  import datetime
  from IsoScore import IsoScore

