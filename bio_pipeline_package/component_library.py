# Arjun Subedi
# 8/20/24
# Components which inherit from base_component and can be used in bio_pipeline.

# Imports
from bio_pipeline_package.base_component import Block
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import hdbscan
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import LocalOutlierFactor
from kneed import KneeLocator


class CSV_Pandas(Block):
    def extra_init(self):
        logging.info("Running CSV_Pandas")
        self.input_type = 'csv'
        self.working_type = 'pandas_df'
        self.output_type = 'pickle'

    def process(self, entry):
        logging.info("CSV Reader processing")
        logging.info(f"First few rows of the loaded data: \n{entry.head()}")  
        return entry

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureSelection(Block):
    def extra_init(self):
        logging.info("Running Feature Selection")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pickle'

    def process(self, entry, args):
        logging.info("Starting Feature Selection")
        target_column = args[0]  # The target variable column name
        n_features = args[1]  # Number of features to select

        # Split data into features and target
        X = entry.drop(columns=[target_column])
        y = entry[target_column]

        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Fill missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0])

        # Fit RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)

        # Select important features using SelectFromModel
        selector = SelectFromModel(clf, prefit=True, max_features=n_features)
        X_selected = selector.transform(X)

        # Get the names of the selected columns
        selected_columns = X.columns[selector.get_support()]
        selected_df = pd.DataFrame(X_selected, columns=selected_columns)
        selected_df[target_column] = y

        # Hardcoded output path for selected features
        selected_features_file = "outputs/selectedfeatures.csv"
        pd.DataFrame(selected_columns, columns=['Selected Features']).to_csv(selected_features_file, index=False)
        
        logging.info(f"Selected features saved to {selected_features_file}")
        
        # Return the dataframe with selected features
        logging.info("Feature Selection completed")
        return selected_df


class HDBSCANCluster(Block):
    def extra_init(self):
        logging.info("Running HDBSCAN clustering")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pandas_df'

    def process(self, entry, args):
        logging.info("Starting HDBSCAN clustering")
        min_cluster_size = args[0]

        # Select only numerical columns and fill missing values
        entry = entry.select_dtypes(include=[np.number])
        entry = entry.fillna(entry.mean())

        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(entry)

        # Add the cluster labels to the DataFrame
        entry['hdbscan_labels'] = cluster_labels

        # Reorder columns to move 'hdbscan_labels' to the first column
        cols = ['hdbscan_labels'] + [col for col in entry.columns if col != 'hdbscan_labels']
        entry = entry[cols]

        # Write the output to a CSV file in the outputs folder
        output_csv_path = os.path.join('outputs', 'HDBSCAN_Results.csv')
        entry.to_csv(output_csv_path, index=False)
        logging.info(f"HDBSCAN results saved to {output_csv_path}")

        # Write the output to a pickle file (for pipeline processing)
        return entry
    
class PCAnalysis(Block):

    def extra_init(self):
        logging.info("Running PCA analysis")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pickle'

    def process(self, entry, args):
        logging.info("Starting PCA analysis")

        comp = args[0]
        labels = args[1]

        if labels:
            lab = entry[labels].tolist()
            entry = entry.drop(labels, axis=1)

        entry = entry.select_dtypes(include=[np.number])
        entry = entry.fillna(entry.mean())

        pca = PCA(n_components=comp)
        principalComponents = pca.fit_transform(entry)
        principalDf = pd.DataFrame(principalComponents)
        PC_list = range(1, comp + 1)
        principalDf.columns = ['PC' + str(b) for b in PC_list]

        if labels:
            principalDf['labels'] = lab

        logging.info("PCA columns: %s", principalDf.columns)
        logging.info("PCA analysis completed")

        return principalDf

class tSNE(Block):
    def extra_init(self):
        logging.info("Running t-SNE analysis")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pickle'

    def process(self, entry, args):
        logging.info("Starting t-SNE processing")

        try:
            dim = args[0]
            perplexity = args[1]
            labels = args[2]

            if labels and labels in entry.columns:
                lab = entry[labels].tolist()
                entry = entry.drop(labels, axis=1)
            else:
                lab = None

            entry = entry.apply(pd.to_numeric, errors='coerce')
            entry = entry.dropna(axis=1, how='any')

            logging.info(f"t-SNE input shape: {entry.shape}")

            tsne = TSNE(n_components=dim, perplexity=perplexity, n_iter=1000, verbose=1)
            tsneComponents = tsne.fit_transform(entry)

            logging.info("t-SNE computation completed")

            tsneDf = pd.DataFrame(tsneComponents, columns=[f'tSNE{i+1}' for i in range(dim)])

            if lab:
                tsneDf['labels'] = lab

            return tsneDf

        except Exception as e:
            logging.error("Error during t-SNE processing", exc_info=True)
            raise e
        
class LOFAndFiltering(Block):
    def extra_init(self):
        logging.info("Running LOF and Filtering Module")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'csv'

    def process(self, entry, args):
        logging.info("Starting LOF detection and filtering")
        n_neighbors = args[0]
        contamination = args[1]
        cutoff = args[2]

        # Filter only numerical columns
        entry = entry.select_dtypes(include=[np.number])

        if entry.empty:
            logging.error("No numerical data available after filtering.")
            return entry

        entry = entry.fillna(entry.mean())

        try:
            # Apply LOF
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            lof.fit(entry)  # Fit the model to the data
            entry['LOF_Actual_Scores'] = -lof.negative_outlier_factor_

            # Move LOF_Actual_Scores to the first column
            lof_scores = entry.pop('LOF_Actual_Scores')
            entry.insert(0, 'LOF_Actual_Scores', lof_scores)

            # Save the entire dataset with LOF scores
            full_output_path = 'outputs/lof_scores_dataset.csv'
            logging.info(f"Writing LOF scores for entire dataset to {full_output_path}")
            entry.to_csv(full_output_path, index=False)
            logging.info(f"LOF scores dataset written to {full_output_path}")

            # Apply filtering based on cutoff
            filtered_entry = entry[entry['LOF_Actual_Scores'] <= cutoff]

            # Save the filtered dataset
            filtered_output_path = 'outputs/filtered_data.csv'
            logging.info(f"Writing filtered output to {filtered_output_path}")
            filtered_entry.to_csv(filtered_output_path, index=False)
            logging.info(f"Filtered data written to {filtered_output_path}")

            return entry

        except ValueError as e:
            logging.error(f"Error during LOF processing: {e}")
            return entry


        
class LOFDetector(Block):
    def extra_init(self):
        logging.info("Running LOF Detector")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pandas_df'

    def process(self, entry, args):
        logging.info("Starting LOF detection")
        n_neighbors = args[0]
        contamination = args[1]

        # Filter only numerical columns
        entry = entry.select_dtypes(include=[np.number])

        if entry.empty:
            logging.error("No numerical data available after filtering.")
            return entry

        entry = entry.fillna(entry.mean())

        try:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            lof.fit(entry)  # Fit the model to the data
            lof_actual_scores = pd.Series(-lof.negative_outlier_factor_, name="LOF_Actual_Scores")

            # Concatenate the actual LOF scores with the original data
            entry = pd.concat([lof_actual_scores, entry], axis=1)

            # Manually ensure writing the main output file
            full_output_path = 'comms_files/2_3.pkl'
            logging.info(f"Writing main output to {full_output_path}")
            entry.to_pickle(full_output_path)
            logging.info(f"Main output write completed.")

            return entry

        except ValueError as e:
            logging.error(f"Error during LOF processing: {e}")
            return entry


class FilteringModule(Block):
    def extra_init(self):
        logging.info("Running Filtering Module")
        self.input_type = 'pandas_df'  # This should match what LOFDetector outputs
        self.working_type = 'pandas_df'
        self.output_type = 'csv'

    def process(self, entry, args):
        logging.info("Starting filtering process")

        # Verify the input type and log its structure
        logging.info(f"Type of entry before processing: {type(entry)}")
        logging.info(f"Entry's first few rows:\n{entry.head() if isinstance(entry, pd.DataFrame) else 'Not a DataFrame'}")

        # Attempt to explicitly ensure it is a DataFrame
        if not isinstance(entry, pd.DataFrame):
            logging.warning("Entry is not a DataFrame, attempting to convert.")
            try:
                entry = pd.DataFrame(entry)
                logging.info("Conversion to DataFrame successful.")
            except Exception as e:
                logging.error(f"Failed to convert entry to DataFrame: {e}", exc_info=True)
                return entry

        # Log the DataFrame's columns
        logging.info(f"Columns in DataFrame: {entry.columns}")

        cutoff = args[0]

        # Ensure the LOF_Actual_Scores column exists
        if 'LOF_Actual_Scores' not in entry.columns:
            logging.error("LOF_Actual_Scores column not found in the input data.")
            return entry

        # Apply the filtering logic
        filtered_entry = entry[entry['LOF_Actual_Scores'] <= cutoff]
        logging.info(f"Filtered data has {filtered_entry.shape[0]} rows after applying cutoff.")

        # Define the output path for the CSV file
        output_directory = 'output/filtered_data'
        os.makedirs(output_directory, exist_ok=True)
        full_output_path = os.path.join(output_directory, 'filtered_data.csv')

        try:
            # Write the filtered DataFrame to a CSV file
            logging.info(f"Writing filtered output to {full_output_path}")
            filtered_entry.to_csv(full_output_path, index=False)
            logging.info(f"Filtered data written to {full_output_path}")
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}", exc_info=True)

        return filtered_entry


class plot(Block):

    def extra_init(self):
        logging.info("Running plot")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'png'

    def process(self, entry, args):
        logging.info("Starting plotting process")

        xlab = args[0]
        ylab = args[1]
        plotTitle = args[2]
        labels = args[3]
        plot_type = args[4]
        zlab = args[5] if len(args) > 5 else None

        plt.figure(figsize=(10, 6))

        if plot_type == '3D':
            ax = plt.axes(projection='3d')
            if labels in entry.columns:
                unique_labels = pd.unique(entry[labels])
                for label in unique_labels:
                    cluster_data = entry[entry[labels] == label]
                    ax.scatter(cluster_data.get(xlab, []), cluster_data.get(ylab, []), cluster_data.get(zlab, []), s=50, label=label)
                ax.set_zlabel(zlab)
            else:
                ax.scatter(entry.get(xlab, []), entry.get(ylab, []), entry.get(zlab, []), s=50)
        else:
            if labels in entry.columns:
                unique_labels = pd.unique(entry[labels])
                for label in unique_labels:
                    cluster_data = entry[entry[labels] == label]
                    plt.scatter(cluster_data.get(xlab, []), cluster_data.get(ylab, []), s=50, label=label)
                plt.legend(title='Labels')
            else:
                plt.scatter(entry.get(xlab, []), entry.get(ylab, []), s=50)

        plt.title(f'{plotTitle}')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.savefig(str(plotTitle))
        plt.show()

        logging.info(f"Plot saved as {plotTitle}")

class LabeledPlot(Block):

    def extra_init(self):
        logging.info("Running LabeledPlot")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'png'

    def process(self, entry, args):
        logging.info("Starting plotting process with custom labels")

        xlab = args[0]
        ylab = args[1]
        plotTitle = args[2]
        label_column = 'Ponds'  # Assuming the 'Ponds' column contains your labels
        plot_type = args[4]
        zlab = args[5] if len(args) > 5 else None

        plt.figure(figsize=(10, 6))

        if plot_type == '3D':
            ax = plt.axes(projection='3d')
            # Ensure that the label column exists in the DataFrame
            if label_column in entry.columns:
                labels = entry[label_column]
                ax.scatter(entry.get(xlab, []), entry.get(ylab, []), entry.get(zlab, []), s=50)
                for i in range(len(entry)):
                    ax.text(entry.get(xlab).values[i], 
                            entry.get(ylab).values[i], 
                            entry.get(zlab).values[i], 
                            labels.values[i])  # Add the 'Ponds' values as labels
                ax.set_zlabel(zlab)
            else:
                logging.error(f"Label column '{label_column}' not found in the dataset.")
        else:
            if label_column in entry.columns:
                labels = entry[label_column]
                plt.scatter(entry.get(xlab, []), entry.get(ylab, []), s=50)
                for i in range(len(entry)):
                    plt.text(entry.get(xlab).values[i], 
                             entry.get(ylab).values[i], 
                             labels.values[i])  # Add the 'Ponds' values as labels
            else:
                logging.error(f"Label column '{label_column}' not found in the dataset.")

        plt.title(f'{plotTitle}')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.savefig(str(plotTitle))
        plt.show()

        logging.info(f"Plot saved as {plotTitle}")

class LabeledPCAnalysis(Block):
    def extra_init(self):
        logging.info("Running Labeled PCA analysis")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pickle'

    def process(self, entry, args):
        logging.info("Starting Labeled PCA analysis")

        comp = args[0]
        labels = args[1]

        # Retain 'Ponds' column separately for labeling later
        if 'Ponds' in entry.columns:
            ponds_labels = entry['Ponds']
        else:
            logging.error("Label column 'Ponds' not found in the dataset.")
            ponds_labels = None

        # Drop non-numeric columns except 'Ponds'
        entry_numeric = entry.select_dtypes(include=[np.number])
        entry_numeric = entry_numeric.fillna(entry_numeric.mean())

        # Perform PCA
        pca = PCA(n_components=comp)
        principalComponents = pca.fit_transform(entry_numeric)
        principalDf = pd.DataFrame(principalComponents)
        PC_list = range(1, comp + 1)
        principalDf.columns = ['PC' + str(b) for b in PC_list]

        # Add 'Ponds' column back to the PCA result for labeling
        if ponds_labels is not None:
            principalDf['Ponds'] = ponds_labels

        logging.info("PCA columns: %s", principalDf.columns)
        logging.info("Labeled PCA analysis completed")

        return principalDf


class Logarithmic_Classifier(Block):
    def extra_init(self):
        logging.info("Running Logarithmic Classifier")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'json'

    def process(self, entry):
        logging.info("Starting Logarithmic Classifier processing")
        out_dict = {'test': 'hi'}
        logging.info("Logarithmic Classifier processing completed")
        return out_dict

class KMeansCluster(Block):
    def extra_init(self):
        logging.info("Running KMeans clustering with Elbow Method")
        self.input_type = 'pickle'
        self.working_type = 'pandas_df'
        self.output_type = 'pandas_df'

    def process(self, entry, args):
        logging.info("Starting KMeans clustering with Elbow Method")
        max_clusters = args[0]  # Maximum number of clusters to test
        plot_elbow = args[1]  # Boolean flag for whether to plot the elbow curve

        # Ensure that only numerical data is used
        entry = entry.select_dtypes(include=[np.number])
        entry = entry.fillna(entry.mean())  # Handle missing values

        # List to store the WCSS values for each number of clusters
        wcss = []

        # Run K-means for cluster numbers from 1 to max_clusters
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(entry)
            wcss.append(kmeans.inertia_)  # WCSS (inertia) for each cluster

        # Create the output directory if it doesn't exist
        output_directory = 'outputs'
        os.makedirs(output_directory, exist_ok=True)

        # Plot the Elbow curve and save it in the outputs folder
        if plot_elbow:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
            plt.grid(True)
            elbow_plot_path = os.path.join(output_directory, 'elbow_method.png')
            plt.savefig(elbow_plot_path)  # Save the elbow plot in outputs folder
            plt.show()
            logging.info(f"Elbow plot saved to {elbow_plot_path}")

        # Automatically detect the optimal number of clusters (using KneeLocator)
        kl = KneeLocator(range(1, max_clusters + 1), wcss, curve='convex', direction='decreasing')
        optimal_clusters = kl.elbow

        # Log the optimal number of clusters
        logging.info(f"The optimal number of clusters (K) determined by the Elbow Method is: {optimal_clusters}")

        # Run KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        kmeans.fit(entry)

        # Add the cluster labels as a new column in the DataFrame
        entry['kmeans_labels'] = kmeans.labels_

        # Reorder columns to move 'kmeans_labels' to the first column
        cols = ['kmeans_labels'] + [col for col in entry.columns if col != 'kmeans_labels']
        entry = entry[cols]

        # Save the final DataFrame with KMeans labels to CSV in the outputs folder
        output_csv_path = os.path.join(output_directory, 'KMeans.csv')
        entry.to_csv(output_csv_path, index=False)
        logging.info(f"KMeans results saved to {output_csv_path}")

        return entry
    
class ClusterComparison(Block):
    def extra_init(self):
        logging.info("Running Cluster Comparison")
        self.input_type = 'pandas_df'
        self.working_type = 'pandas_df'
        self.output_type = 'pandas_df'
    
    def process(self, entry, args):
        # The ground truth column (e.g., "Diagnosis")
        ground_truth_column = args[0]  

        # Get the ground truth labels
        ground_truth = entry[ground_truth_column]

        # KMeans and HDBSCAN labels (assuming these columns exist after clustering blocks)
        kmeans_labels = entry['kmeans_labels']
        hdbscan_labels = entry['hdbscan_labels']

        # Calculate Adjusted Rand Index (ARI)
        ari_kmeans_vs_ground = adjusted_rand_score(ground_truth, kmeans_labels)
        ari_hdbscan_vs_ground = adjusted_rand_score(ground_truth, hdbscan_labels)
        ari_kmeans_vs_hdbscan = adjusted_rand_score(kmeans_labels, hdbscan_labels)

        # Calculate Adjusted Mutual Information (AMI)
        ami_kmeans_vs_hdbscan = adjusted_mutual_info_score(kmeans_labels, hdbscan_labels)

        # Log the results
        logging.info(f"ARI (KMeans vs Ground Truth): {ari_kmeans_vs_ground}")
        logging.info(f"ARI (HDBSCAN vs Ground Truth): {ari_hdbscan_vs_ground}")
        logging.info(f"ARI (KMeans vs HDBSCAN): {ari_kmeans_vs_hdbscan}")
        logging.info(f"AMI (KMeans vs HDBSCAN): {ami_kmeans_vs_hdbscan}")

        # Save results to CSV file
        results = {
            "Metric": ["ARI (KMeans vs Ground Truth)", "ARI (HDBSCAN vs Ground Truth)", "ARI (KMeans vs HDBSCAN)", "AMI (KMeans vs HDBSCAN)"],
            "Score": [ari_kmeans_vs_ground, ari_hdbscan_vs_ground, ari_kmeans_vs_hdbscan, ami_kmeans_vs_hdbscan]
        }
        results_df = pd.DataFrame(results)

        # Define output directory and file path within the project
        output_directory = 'outputs'
        os.makedirs(output_directory, exist_ok=True)  # Create the folder if it doesn't exist
        output_file = os.path.join(output_directory, 'cluster_comparison_scores.csv')

        # Save the scores to a CSV file
        results_df.to_csv(output_file, index=False)
        logging.info(f"Cluster comparison scores saved to {output_file}")

        return entry  # Return the entry for further processing if needed
    
class Write_Outputs(Block):
    def extra_init(self):
        logging.info("Running Write Outputs")
        self.input_type = 'json'
        self.working_type = 'dict'
        self.output_type = 'json'

    def process(self, entry):
        logging.info("Starting Write Outputs processing")
        out_dict = {'test': 'hi_2'}
        logging.info("Write Outputs processing completed")
        return out_dict