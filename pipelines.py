from sklearn.model_selection import train_test_split
from constants import LEAKAGE_COLS
from main import scale_data, one_hot_encode
from sklearn.preprocessing import LabelEncoder
from nn_util import train_network, make_prediction, calculate_accuracy
from knn_util import do_knn


def knn_testing_pipeline(training_features, k, scale):
    train_df, val_df = train_test_split(
        training_features, test_size=0.2, random_state=2718
    )
    X_train = train_df.drop(columns=LEAKAGE_COLS, errors="ignore")
    y_train = train_df["class_name"]

    X_val = val_df.drop(columns=LEAKAGE_COLS, errors="ignore")
    y_val = val_df["class_name"]

    if scale:
        X_train, X_val = scale_data(
            X_train=X_train, X_test=X_val, ignore_keywords=["hog"], return_numpy=False
        )

    correct_preds = 0
    total_tests = len(X_val)

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

        if prediction == true_label:
            correct_preds += 1

    accuracy = correct_preds / total_tests
    print(f"Final accuracy: {accuracy:.4f}")
    return accuracy


def test_k_vals(df, k_values, scale):
    best_accuracy = 0
    best_k = None
    for k in k_values:
        print(f"K-val: {k}")
        accuracy = knn_testing_pipeline(df, k, scale)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    print("Best hyperparameters found!")
    print(f"Best k: {best_k} | Accuracy: {best_accuracy:.3f}")
    return best_k, best_accuracy


def run_nn_grid_search(
    feature_df, learning_rates, epoch_counts, hidden_sizes, ignore_scale_cols=["hog"]
):
    """
    Takes a full feature DataFrame, prepares the data, runs a grid search
    for Neural Network hyperparameters, and returns the best results.
    """
    # 1. Data Splitting
    X = feature_df.drop(columns=LEAKAGE_COLS, errors="ignore")
    Y = feature_df["class_name"]

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2718, stratify=Y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.2,
        random_state=2718,
        stratify=y_train_temp,
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

    return best_hyperparams
