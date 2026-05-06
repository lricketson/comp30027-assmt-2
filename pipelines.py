from sklearn.model_selection import train_test_split
from constants import LEAKAGE_COLS, TARGET_COL
from preprocessing import scale_data, one_hot_encode
from sklearn.preprocessing import LabelEncoder
from nn_util import train_network, make_prediction, calculate_accuracy
from knn_util import do_knn
from lightgbm import LGBMClassifier
import pandas as pd


def knn_testing_pipeline(feature_df: pd.DataFrame, k, scale):
    X = feature_df.drop(columns=LEAKAGE_COLS)
    Y = feature_df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=2718
    )
    if scale:
        X_train, X_val = scale_data(
            X_train=X_train, X_test=X_val, ignore_keywords=["hog"], return_numpy=False
        )

    correct_preds = 0
    total_tests = len(X_val)

    predictions = []

    print(f"Testing {total_tests} unseen images...")
    for i in range(total_tests):
        test_image_features = X_val.iloc[i]
        true_label = y_val.iloc[i]

        prediction = do_knn(
            image=test_image_features,
            training_images=X_train,
            training_image_labels=y_train,
            k=k,
        )

        predictions.append[prediction]

        if prediction == true_label:
            correct_preds += 1

    accuracy = correct_preds / total_tests
    print(f"Final accuracy: {accuracy:.4f}")
    return accuracy, y_val, predictions


def test_k_vals(df, k_values, scale):
    best_accuracy = 0
    best_k = None
    best_y_val = None
    best_predictions = None

    for k in k_values:
        print(f"K-val: {k}")
        accuracy, y_val, predictions = knn_testing_pipeline(df, k, scale)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_y_val = y_val
            best_predictions = predictions
    print("Best hyperparameters found!")
    print(f"Best k: {best_k} | Accuracy: {best_accuracy:.3f}")
    return best_k, best_accuracy, best_y_val, best_predictions


def run_nn_grid_search(
    feature_df, learning_rates, epoch_counts, hidden_sizes, ignore_scale_cols=["hog"]
):
    """
    Takes a full feature DataFrame, prepares the data, runs a grid search
    for Neural Network hyperparameters, and returns the best results.
    """
    # 1. Data Splitting
    X = feature_df.drop(columns=LEAKAGE_COLS, errors="ignore")
    Y = feature_df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=2718,
        stratify=Y,
    )

    # 2. Scaling
    X_train_scaled, X_val_scaled = scale_data(
        X_train=X_train,
        X_test=X_val,
        ignore_keywords=ignore_scale_cols,
        return_numpy=True,
    )

    # 3. Label Encoding & OHE
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    unique_classes = label_encoder.classes_

    y_train_ohe = one_hot_encode(
        Y_ints=y_train_encoded, num_classes=len(unique_classes)
    )

    # 4. Grid Search Setup
    best_hyperparams = {
        "best_lr": None,
        "best_ec": None,
        "best_hs": None,
        "best_accuracy": 0,
    }

    best_model_params = None
    best_predictions = None

    print("\nStarting search for best hyperparameters...")
    total_num_tests = len(learning_rates) * len(epoch_counts) * len(hidden_sizes)
    tests_completed = 0

    # 5. The Grid Search Loop
    for learning_rate in learning_rates:
        for epoch_count in epoch_counts:
            for hidden_size in hidden_sizes:
                # The \r keeps the terminal clean by overwriting the same line
                proportion_completed = tests_completed / total_num_tests
                print(
                    f"Grid Search Progress: {100 * proportion_completed:.2f}% complete",
                    end="\r",
                )

                params_dict = train_network(
                    X=X_train_scaled,
                    hidden_size=hidden_size,
                    output_size=len(unique_classes),
                    epochs=epoch_count,
                    Y=y_train_ohe,
                    learning_rate=learning_rate,
                )

                predictions, probs = make_prediction(
                    X=X_val_scaled,
                    classes_array=unique_classes,
                    params_dict=params_dict,
                )

                accuracy = calculate_accuracy(preds=predictions, true_labels=y_val)

                if accuracy > best_hyperparams["best_accuracy"]:
                    best_hyperparams["best_accuracy"] = accuracy
                    best_hyperparams["best_lr"] = learning_rate
                    best_hyperparams["best_ec"] = epoch_count
                    best_hyperparams["best_hs"] = hidden_size

                    best_model_params = params_dict
                    best_predictions = predictions

                    # Print on a new line so it doesn't get overwritten by the progress bar
                    print(
                        f"\n---> New best! Accuracy: {accuracy * 100:.2f}% | LR: {learning_rate}, Epochs: {epoch_count}, HS: {hidden_size}"
                    )

                tests_completed += 1

    # 6. Final Report
    print("\nSearch over!")
    print("Best hyperparameters are as follows:")
    print(
        f"Accuracy: {best_hyperparams['best_accuracy'] * 100:.2f}% | "
        f"LR: {best_hyperparams['best_lr']}, "
        f"Epochs: {best_hyperparams['best_ec']}, "
        f"HS: {best_hyperparams['best_hs']}"
    )

    return best_hyperparams, best_model_params, y_val, best_predictions


def run_lgbm_grid_search(
    feature_df: pd.DataFrame,
    learning_rates: list,
    max_depths: list,
    n_estimators_list: list,
    ignore_scale_cols=["hog"],
):
    """
    Runs an exhaustive search for LGBM hyperparameters.
    """
    X = feature_df.drop(columns=LEAKAGE_COLS)
    Y = feature_df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=2718, stratify=Y
    )
    X_train_scaled, X_val_scaled = scale_data(
        X_train=X_train,
        X_test=X_val,
        ignore_keywords=ignore_scale_cols,
        return_numpy=False,
    )
    # LGBM requires encoded variables
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    best_hyperparams = {
        "best_lr": None,
        "best_depth": None,
        "best_trees": None,
        "best_accuracy": 0,
    }

    print("\nStarting search for best hyperparams...")

    total_tests = len(learning_rates) * len(max_depths) * len(n_estimators_list)
    tests_completed = 0

    for lr in learning_rates:
        for depth in max_depths:
            for n_estimators in n_estimators_list:
                proportion = tests_completed / total_tests
                print(
                    f"Grid Search Progress: {100 * proportion:.2f}% complete", end="\r"
                )
                model = LGBMClassifier(
                    learning_rate=lr,
                    max_depth=depth,
                    n_estimators=n_estimators,
                    objective="multiclass",  # means we have 3+ classes
                    random_state=2718,
                    n_jobs=-1,  # use all CPU cores
                    verbose=-1,  # don't flood the terminal
                )
                model.fit(X_train_scaled, y_train_encoded)
                accuracy = model.score(X_val_scaled, y_val_encoded)

                if accuracy > best_hyperparams["best_accuracy"]:
                    best_hyperparams["best_accuracy"] = accuracy
                    best_hyperparams["best_lr"] = lr
                    best_hyperparams["best_depth"] = depth
                    best_hyperparams["best_trees"] = n_estimators
                    best_model = model
                    print(
                        f"\n---> New best! Accuracy: {accuracy * 100:.2f}% | LR: {lr}, Depth: {depth}, Trees: {n_estimators}"
                    )
                tests_completed += 1
    print("\nSearch over!")
    print(
        f"Best Accuracy: {best_hyperparams['best_accuracy'] * 100:.2f}% | LR: {best_hyperparams['best_lr']}, Depth: {best_hyperparams['best_depth']}, Trees: {best_hyperparams['best_trees']}"
    )
    return best_hyperparams, best_model
