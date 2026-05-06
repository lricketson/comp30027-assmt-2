import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from pipelines import test_k_vals, run_nn_grid_search, run_lgbm_grid_search


def plot_model_confusion_matrix(
    y_true, y_pred, classes, title="Confusion Matrix", save_path=None
):
    """
    A universal confusion matrix plotter that works for any model.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )

    plt.title(title, fontsize=16, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved matrix to {save_path}")

    plt.show()


def evaluate_knn_tier(feature_df, k_values, tier_name, save_path):
    """Runs scaled and unscaled KNN, finds the winner, and plots the matrix."""
    print(f"\n========== Evaluating {tier_name.upper()} ==========")

    print("\n--- Running Scaled Tests ---")
    k_s, acc_s, y_val_s, preds_s = test_k_vals(feature_df, k_values, scale=True)

    print("\n--- Running Unscaled Tests ---")
    k_u, acc_u, y_val_u, preds_u = test_k_vals(feature_df, k_values, scale=False)

    if acc_s >= acc_u:
        print(f"\n🏆 WINNER: Scaled data with k={k_s} (Accuracy: {acc_s:.4f})")
        best_y_val, best_preds, best_k, mode = y_val_s, preds_s, k_s, "Scaled"
    else:
        print(f"\n🏆 WINNER: Unscaled data with k={k_u} (Accuracy: {acc_u:.4f})")
        best_y_val, best_preds, best_k, mode = y_val_u, preds_u, k_u, "Unscaled"

    unique_classes = sorted(best_y_val.unique())
    plot_title = f"KNN - {tier_name} ({mode}, k={best_k})"

    plot_model_confusion_matrix(
        y_true=best_y_val,
        y_pred=best_preds,
        classes=unique_classes,
        title=plot_title,
        save_path=save_path,
    )


def evaluate_nn_tier(
    feature_df, learning_rates, epoch_counts, hidden_sizes, tier_name, save_path
):
    """Runs the NN grid search and automatically plots the matrix."""
    print(f"\n========== Evaluating {tier_name.upper()} ==========")

    best_hp, best_model, y_val, y_pred = run_nn_grid_search(
        feature_df,
        learning_rates,
        epoch_counts,
        hidden_sizes,
        ignore_scale_cols=["hog"],
    )

    unique_classes = sorted(y_val.unique())
    plot_title = f"Neural Network - {tier_name}"

    plot_model_confusion_matrix(
        y_true=y_val,
        y_pred=y_pred,
        classes=unique_classes,
        title=plot_title,
        save_path=save_path,
    )


def evaluate_lgbm_tier(
    feature_df, learning_rates, max_depths, n_estimators_list, tier_name, save_path
):
    """Runs the LGBM grid search and automatically plots the matrix."""
    print(f"\n========== Evaluating {tier_name.upper()} ==========")

    best_hp, best_model, y_val, y_pred = run_lgbm_grid_search(
        feature_df,
        learning_rates,
        max_depths,
        n_estimators_list,
        ignore_scale_cols=["hog"],
    )

    unique_classes = sorted(y_val.unique())
    plot_title = f"LGBM - {tier_name}"

    plot_model_confusion_matrix(
        y_true=y_val,
        y_pred=y_pred,
        classes=unique_classes,
        title=plot_title,
        save_path=save_path,
    )
