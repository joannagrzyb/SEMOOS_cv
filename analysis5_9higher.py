import os
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import chi2

from methods.moo_ensemble_SW import MooEnsembleSVC
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from methods.moo_ensemble_bootstrap_pruned import MooEnsembleSVCbootstrapPruned
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from methods.feature_selection_clf import FeatueSelectionClf
from utils.load_dataset import find_datasets
from utils.plots import scatter_pareto_chart, diversity_bar_plot
from utils.wilcoxon_ranking_grid_all import pairs_metrics_multi_grid_all


base_estimator = {'SVM': SVC(probability=True)}
methods = {
    "MooEnsembleSVC_SW": MooEnsembleSVC(base_classifier=base_estimator, cross_val=True),
    "MooEnsembleSVCbootstrap_SW": MooEnsembleSVCbootstrap(base_classifier=base_estimator, cross_val=True),
    "MooEnsembleSVCbootstrapPruned_SW": MooEnsembleSVCbootstrapPruned(base_classifier=base_estimator, cross_val=True),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    "SVM": SVC(),
    "FS": FeatueSelectionClf(base_estimator, chi2),
    "FSIRSVM": 0
}

methods_alias = [
                "SEMOOS",
                "SEMOOSb",
                "SEMOOSbp",
                "RS",
                "SVM",
                "FS",
                "FSIRSVM"
                ]

metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]
diversity_measures = ["Entropy", "KW", "Disagreement", "Q statistic"]

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods_alias) * len(base_estimator)
n_metrics = len(metrics_alias)


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/9higher')
n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))

data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
diversity = np.zeros((n_datasets, 4, n_folds, len(diversity_measures)))

datasets = []
for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    datasets.append(dataset)
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                filename = "results/experiment_server/experiment5_cross_val_in_opt_9h/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
                if not os.path.isfile(filename):
                    # print("File not exist - %s" % filename)
                    continue
                scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                data_np[dataset_id, metric_id, clf_id] = scores
                mean_score = np.mean(scores)
                mean_scores[dataset_id, metric_id, clf_id] = mean_score
                std = np.std(scores)
                stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset, clf_name, metric)

            for div_measure_id, div_measure in enumerate(diversity_measures):
                try:
                    filename = "results/experiment_server/experiment5_cross_val_in_opt_9h/diversity_results/%s/%s.csv" % (dataset, clf_name)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    diversity_raw = np.genfromtxt(filename, delimiter=' ', dtype=np.float32)
                    if np.isnan(diversity_raw).all():
                        pass
                    else:
                        diversity_raw = np.nan_to_num(diversity_raw)
                        diversity[dataset_id, clf_id] = diversity_raw
                except:
                    print("Error loading diversity data!", dataset, clf_name, div_measure)

diversity_m = np.mean(diversity, axis=2)
diversity_mean = np.mean(diversity_m, axis=0)


# Plotting

# Wilcoxon ranking grid - statistic test for methods: SEMOOS, SEMOOSb, SEMOOSbp and all metrics
# UNCOMMENT ONLY methods SEMOOS, SEMOOSb, SEMOOSbp
# pairs_metrics_multi_grid(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment5_cross_val_in_opt_9h", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot_grid_variants", ref_methods=methods_alias[0:3], offset=-75)

# Wilcoxon ranking grid - statistic test for all methods vs: SEMOOS, SEMOOSb, SEMOOSbp and all metrics
pairs_metrics_multi_grid_all(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment5_cross_val_in_opt_9h", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot_grid_all", ref_methods=methods_alias[0:3], offset=-75)


# Diveersity bar Plotting
diversity_bar_plot(diversity_mean, diversity_measures, methods_alias[:4], experiment_name="experiment5_cross_val_in_opt_9h")
