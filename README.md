##**K-IBL and SVM Performance Analysis Project**##

**Authors**: Zoë Finelli, Onat Bitirgen, Emre Karaoglu, Noel Torres Carretero  
**Date**: 02 November 2025


**Introduction:**

This project conducts a comprehensive performance analysis of the **k-Instance-Based Learning (k-IBL)** algorithm and **Support Vector Machines (SVM)**. It follows a rigorous multi-step experimental workflow to compare:

1.  A tuned baseline **k-IBL** algorithm.
    
2.  A tuned baseline **SVM** algorithm.
    
3.  k-IBL enhanced with **Feature Weighting (FW)** techniques.
    
4.  k-IBL enhanced with **Instance Reduction (IR)** techniques.
    
5.  SVM performance on IR-reduced datasets.

The analysis uses 10-fold cross-validation and rigorous statistical tests (Friedman, Nemenyi, Wilcoxon, t-tests) to identify champion algorithms and measure the impact of FW and IR methods on accuracy, efficiency, and storage.

**Dependencies**

Check the requirements.txt file for dependencies.

**Project Setup & Installation**

1.  **Open the Project**
    
    * In PyCharm, go to **File > Open**.
    
    * Navigate to and select the main project folder.

2.  **Create the Virtual Environment (venv)**
    
    * Go to **PyCharm > Settings** (macOS) or **File > Settings** (Windows/Linux).
    
    * Navigate to **Project: [Your Project Name] > Python Interpreter**.
    
    * Click the **gear icon > Add...**.
    
    * Select **Virtualenv Environment** on the left.
    
    * Ensure **New environment** is selected and the **Location** points to a `venv` folder inside your project.
    
    * Click **OK**. PyCharm will create the `venv` and set it as the project interpreter.

3.  **Install Dependencies**
    
    * Open the **Terminal** tab at the bottom of PyCharm.
    
    * Ensure the `(venv)` prefix is visible in your terminal prompt.
    
    * Run the following command to install all required packages:
        ```bash
        pip install -r requirements.txt
        ```
4.  **Run the Scripts**
    
    * You can now run any script by right-clicking it in the `analysis_runners` folder and selecting **Run '...'**.
    
    * PyCharm will use the correct interpreter and all installed packages.

**Project Structure**

*   /core\_modules/: Contains the core implementations of algorithms (k-IBL, SVM), distance functions, voting methods, and IR reducers.
    
*   /analysis\_runners/: Contains all experiment and statistical analysis scripts.
    
*   /data/: Contains folder DataCBR with adult and pen-based datasets (included in this directory).
    
*   /results/: Default output directory for all CSVs, JSON files, and plots. Sub-folders are created automatically.
    

**Experimental Workflow (How to Run)**

This project must be run in a specific order. Scripts in later steps **depend on the results from earlier steps**.

**Crucially, you must manually edit configuration dictionaries in the scripts to "pass" the winning algorithm from one step to the next.**

**Step 1: Run Baseline Experiments**

These scripts run the main experiments and generate the raw results.

1.  **get\_best\_kibl\_runner.py**
    
    *   **What it does:** Runs the full k-IBL grid search (K, Distance, Voting, Retention).
        
    *   **Generates:** kibl\_detailed\_fold\_results\_FINAL\_...csv
        
2.  **svm\_baseline\_runner.py** 
    
    *   **What it does:** Runs the full SVM grid search (Kernel, C, Gamma).
        
    *   **Generates:** svm\_params\_...\_results.json
        

**Step 2: Find Baseline Winners**

These scripts analyze the raw results to find the "champion" for each baseline.

1.  **get\_bestKIBL\_stats\_runner.py** 
    
    *   **What it does:** Analyzes the k-IBL CSV to find the best configuration (e.g., K=7, HEOM, ModPlurality, NR).
        
    *   **Review the output:** Note the winning config.
        
2.  **svm\_baseline\_stats\_runner.py** 
    
    *   **What it does:** Analyzes the SVM JSON to find the best configuration (e.g., k=rbf, C=10.0, g=1.0).
        
    *   **Review the output:** Note the winning config.
        

**Step 3: Manual Update (Part 1)**

You must now **manually edit** the "consumer" scripts with the winners you found in Step 2.

*   **feature\_weighting\_runner.py**
    
    *   Update the BEST\_CONFIGS dictionary with your k-IBL winner.
        
*   **ir\_baseline\_runner.py**
    
    *   Update the BEST\_CONFIGS dictionary with your k-IBL winner.
        
*   **svm\_ir\_runner.py**
    
    *   Update the SVM\_PARAMS dictionary with your SVM winner.
        
*   **bestKIBL\_fw\_stats\_runner.py**
    
    *   Update the BASELINE\_CONFIGS dictionary with your k-IBL winner.
        
*   **bestKIBL\_ir\_stats\_runner.py**
    
    *   Update the BASELINE\_CONFIGS dictionary with your k-IBL winner.
        
*   **bestKIBL\_svm\_stats\_runner.py**
    
    *   Update the BEST\_CONFIGS dictionary with **both** your k-IBL and SVM winners.
        
*   **svm\_irSVM\_stats\_runner.py**
    
    *   Update the BEST\_SVM\_CONFIGS dictionary with your SVM winner.
        

**See the "Configuration Guide" below for critical formatting rules!**

**Step 4: Run Secondary Experiments**

Now that the scripts are configured with the correct winners, run them.

1.  **feature\_weighting\_runner.py** 
    
    *   **What it does:** Runs the "best" k-IBL using FW.
        
    *   **Generates:** fw\_detailed\_fold\_results\_...csv
        
2.  **ir\_baseline\_runner.py**
    
    *   **What it does:** Runs the "best" k-IBL using IR.
        
    *   **Generates:** ir\_detailed\_fold\_results\_...csv
        
    *   **Also Generates:** .npy files of reduced datasets in /results/ir\_reduced\_datasets/.
        
3.  **svm\_ir\_runner.py** 
    
    *   **What it does:** Runs the "best" SVM on the .npy reduced datasets.
        
    *   **Generates:** svm\_ir\_fold\_results\_...json
        

**Step 5: Run Secondary Statistical Analysis**

These scripts compare the baselines to their new variations.

1.  **bestKIBL\_fw\_stats\_runner.py**
    
    *   **Finds:** The best FW method (e.g., mutual\_info) that is (or isn't) significantly better than the baseline.
        
2.  **bestKIBL\_ir\_stats\_runner.py**
    
    *   **Finds:** The best IR method (e.g., ICF) based on Accuracy, Time, and Storage.
        
3.  **bestKIBL\_svm\_stats\_runner.py**
    
    *   **Compares:** The k-IBL champion vs. the SVM champion.
        
4.  **svm\_irSVM\_stats\_runner.py**
    
    *   **Compares:** The baseline SVM vs. the SVM on IR-reduced datasets.
        

**Step 6: Manual Update (Part 2)**

You have one final script to update.

*   **final\_stats\_runner.py**
    
    *   Update the BEST\_ALGOS dictionary with the three grand champions:
        
        1.  The k-IBL Baseline ('Baseline') from Step 2.
            
        2.  The winning FW method ('FW') from Step 5.
            
        3.  The winning IR method ('IR') from Step 5.
            

**Step 7: Run Final Analysis**

1.  **final\_stats\_runner.py (Step 6.d)**
    
    *   **What it does:** Runs the final 3-way comparison (Baseline vs. FW vs. IR) and generates the final CD diagram.
        

**Important Configuration Guide**

The statistical scripts work by **filtering text** from CSV and JSON files. If your manual configurations don't **exactly match** the format of the output files, the scripts will fail.

**1\. k-IBL Configuration (All Stats Scripts)**

*   **K**: Must be an **integer** (e.g., 7, not 7.0).
    
*   **Strings**: Must be **exact case** (e.g., "HEOM", "ModPlurality", "NR").
    

Python

\# GOOD

'Baseline': {"K": 7, "Distance": "HEOM", "Voting": "ModPlurality", "Retention": "NR"}

\# BAD - will fail

'Baseline': {"K": 7.0, "distance": "heom", "Voting": "modplurality", "Retention": "nr"}

**2\. SVM Configuration (All Stats Scripts)**

*   **k (kernel)**: Must be **lowercase** (e.g., "rbf", "linear").
    
*   **C**: Must be an **integer** or **float** _exactly_ as it was defined in svm\_baseline\_runner.py's PARAM\_GRID (e.g., 10, 0.1).
    
*   **g (gamma)**: Must be a **float** or **string** _exactly_ as defined (e.g., 1.0, "scale").
    

Python

\# GOOD

'svm': {"k": "rbf", "C": 10, "g": 1.0}

'svm': {"k": "linear", "C": 0.1, "g": "scale"}

\# BAD - will fail

'svm': {"k": "RBF", "C": 10.0, "g": 1} # Kernel case, C type, and g type are all wrong

**3\. Skipping Champions (in final\_stats\_runner.py)**

If you found that no FW or IR method was a winner, you can set its value to 'none' (as a string) or None (as the type). The script has been modified to detect this and skip that algorithm in the final comparison.

Python

'pen-based': {

  'Baseline': {...},

  'FW': 'none',  # This will skip the FW comparison for pen-based

  'IR': 'ENN',

}
