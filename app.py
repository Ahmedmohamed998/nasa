import os as _os
_os.environ["TRANSFORMERS_NO_TF"] = "1"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)
from sklearn.multiclass import OneVsRestClassifier
import pickle
import os
import json
from datetime import datetime
from streamlit_chatbot_app import main as chatbot_main, add_footer as chatbot_footer

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Exoplanet Classification Suite", page_icon="🚀", layout="wide"
)
import warnings

warnings.filterwarnings("ignore")

# Import additional libraries if available
try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Import additional libraries if available
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.neural_network import MLPClassifier


class ExoplanetClassifierSuite:
    """
    Comprehensive exoplanet classification system with multiple ML algorithms
    """

    def __init__(self, models_dir="exoplanet_models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.model_performance = {}
        self.training_history = []

        # Create models directory
        os.makedirs(models_dir, exist_ok=True)

        # Initialize available models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all available ML models"""
        self.available_models = {
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
            ),
            "svm": SVC(probability=True, random_state=42, class_weight="balanced"),
            "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "naive_bayes": GaussianNB(),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.available_models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
            )

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.available_models["lightgbm"] = LGBMClassifier(
                n_estimators=500,
                num_leaves=64,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                random_state=42,
                verbose=-1,
            )

    
    def prepare_exoplanet_data_for_classification(
        self, k2_df=None, tess_df=None, kepler_df=None
    ):
        """
        Merge and prepare exoplanet datasets for classification tasks
        """
        print("Step 1: Standardizing planet names...")

        # Clean and standardize planet names
        def clean_planet_name(name):
            if pd.isna(name):
                return None
            return str(name).strip().upper()

        features_list = []

        # Define common column mappings
        common_features = [
            "ra",
            "dec",
            "stellar_mag",
            "orbital_period",
            "planet_radius",
            "transit_depth",
            "insolation",
            "equilibrium_temp",
            "stellar_teff",
            "stellar_logg",
            "stellar_radius",
            "model_snr",
            "fpflag_nt",
            "fpflag_ss",
            "fpflag_co",
            "fpflag_ec",
            "impact",
            "duration",
        ]

        # ---------------------- K2 ----------------------
        if k2_df is not None and not k2_df.empty:
            print("Step 3: Processing K2...")
            k2_df["pl_name_clean"] = k2_df.get(
                "pl_name", pd.Series(dtype="object")
            ).apply(clean_planet_name)
            k2_df["mission"] = "K2"
            k2_mapping = {"ra": "ra", "dec": "dec"}
            k2_features = pd.DataFrame(
                {
                    "target_id": k2_df.get("epic_id", pd.Series(dtype="object")),
                    "object_name": k2_df.get("k2_name", pd.Series(dtype="object")),
                    "pl_name_clean": k2_df["pl_name_clean"],
                    "mission": k2_df["mission"],
                }
            )

            # Map available columns
            k2_df_cols_lower = {col.lower().strip(): col for col in k2_df.columns}
            for orig_col, new_col in k2_mapping.items():
                actual_col = k2_df_cols_lower.get(orig_col.lower().strip())
                if actual_col and new_col in common_features:
                    k2_features[new_col] = pd.to_numeric(
                        k2_df[actual_col], errors="coerce"
                    )

            # Add empty feature columns for K2
            for col in common_features:
                if col not in k2_features.columns:
                    k2_features[col] = np.nan

            # Classification for K2 (size-based if radius available)
            if "planet_radius" in k2_features.columns:
                try:
                    k2_features["classification"] = k2_features[
                        "planet_radius"
                    ].apply(size_classification)
                except Exception:
                    k2_features["classification"] = "Unknown"
            else:
                k2_features["classification"] = "Unknown"
            features_list.append(k2_features)

        # ---------------------- TESS ----------------------
        if tess_df is not None and not tess_df.empty:
            print("Step 4: Processing TESS...")
            tess_df["pl_name_clean"] = tess_df.get(
                "pl_name", pd.Series(dtype="object")
            ).apply(clean_planet_name)
            tess_df["mission"] = "TESS"
            tess_mapping = {
                "ra": "ra",
                "dec": "dec",
                "st_tmag": "stellar_mag",
                "pl_orbper": "orbital_period",
                "pl_rade": "planet_radius",
                "pl_trandep": "transit_depth",
                "pl_insol": "insolation",
                "pl_eqt": "equilibrium_temp",
                "st_teff": "stellar_teff",
                "st_logg": "stellar_logg",
                "st_rad": "stellar_radius",
            }

            tess_features = pd.DataFrame(
                {
                    "target_id": tess_df.get("tid", pd.Series(dtype="object")),
                    "object_name": tess_df.get("toi", pd.Series(dtype="object")),
                    "pl_name_clean": tess_df["pl_name_clean"],
                    "mission": tess_df["mission"],
                }
            )

            tess_df_cols_lower = {col.lower().strip(): col for col in tess_df.columns}
            for orig_col, new_col in tess_mapping.items():
                actual_col = tess_df_cols_lower.get(orig_col.lower().strip())
                if actual_col and new_col in common_features:
                    tess_features[new_col] = pd.to_numeric(
                        tess_df[actual_col], errors="coerce"
                    )

            # Add missing columns
            for col in common_features:
                if col not in tess_features.columns:
                    tess_features[col] = np.nan

            # TESS disposition → classification
            disp_map = {
                "PC": "Candidate",
                "FP": "False Positive",
                "CP": "Confirmed",
                "KP": "Confirmed",
                "APC": "Candidate",
                "FA": "False Positive",
            }
            if "tfopwg_disp" in tess_df.columns:
                tess_features["classification"] = (
                    tess_df["tfopwg_disp"].map(disp_map).fillna("Unknown")
                )
                # Synthetic FP flag (if FP in TESS disposition)
                tess_features["fpflag_ec"] = tess_df["tfopwg_disp"].apply(
                    lambda x: 1 if str(x).upper() == "FP" else 0
                )
            else:
                tess_features["classification"] = "Unknown"
                tess_features["fpflag_ec"] = np.nan
            features_list.append(tess_features)

        # ---------------------- Kepler ----------------------
        if kepler_df is not None and not kepler_df.empty:
            print("Step 5: Processing Kepler...")
            kepler_df["pl_name_clean"] = kepler_df.get(
                "kepler_name", pd.Series(dtype="object")
            ).apply(clean_planet_name)
            kepler_df["mission"] = "Kepler"
            kepler_mapping = {
                "ra": "ra",
                "dec": "dec",
                "koi_kepmag": "stellar_mag",
                "koi_period": "orbital_period",
                "koi_prad": "planet_radius",
                "koi_depth": "transit_depth",
                "koi_insol": "insolation",
                "koi_teq": "equilibrium_temp",
                "koi_steff": "stellar_teff",
                "koi_slogg": "stellar_logg",
                "koi_srad": "stellar_radius",
                "koi_model_snr": "model_snr",
                "koi_fpflag_nt": "fpflag_nt",
                "koi_fpflag_ss": "fpflag_ss",
                "koi_fpflag_co": "fpflag_co",
                "koi_fpflag_ec": "fpflag_ec",
                "koi_impact": "impact",
                "koi_duration": "duration",
            }

            kepler_features = pd.DataFrame(
                {
                    "target_id": kepler_df.get("kepid", pd.Series(dtype="object")),
                    "object_name": kepler_df.get(
                        "kepoi_name", pd.Series(dtype="object")
                    ),
                    "pl_name_clean": kepler_df["pl_name_clean"],
                    "mission": kepler_df["mission"],
                }
            )

            kepler_df_cols_lower = {
                col.lower().strip(): col for col in kepler_df.columns
            }
            for orig_col, new_col in kepler_mapping.items():
                actual_col = kepler_df_cols_lower.get(orig_col.lower().strip())
                if actual_col and new_col in common_features:
                    kepler_features[new_col] = pd.to_numeric(
                        kepler_df[actual_col], errors="coerce"
                    )

            # Add missing columns
            for col in common_features:
                if col not in kepler_features.columns:
                    kepler_features[col] = np.nan

            # Kepler classification
            if "koi_disposition" in kepler_df.columns:
                kepler_features["classification"] = (
                    kepler_df["koi_disposition"]
                    .map(
                        {
                            "CONFIRMED": "Confirmed",
                            "CANDIDATE": "Candidate",
                            "FALSE POSITIVE": "False Positive",
                        }
                    )
                    .fillna("Unknown")
                )
            else:
                kepler_features["classification"] = "Unknown"
            features_list.append(kepler_features)

        # ---------------------- Combine ----------------------
        if not features_list:
            return pd.DataFrame(), [], pd.DataFrame()

        print("Step 6: Combining datasets...")
        combined_df = pd.concat(features_list, ignore_index=True, sort=False)

        # Analyze missing data
        print(f"Combined dataset shape: {combined_df.shape}")
        missing_counts = combined_df[common_features].isnull().sum()
        missing_percentages = (missing_counts / len(combined_df)) * 100

        missing_summary = pd.DataFrame(
            {
                "Column": missing_counts.index,
                "Missing_Count": missing_counts.values,
                "Missing_Percentage": missing_percentages.values,
            }
        ).sort_values("Missing_Percentage")

        # Identify usable features (<80% missing)
        usable_features = missing_summary[
            missing_summary["Missing_Percentage"] < 80
        ]["Column"].tolist()
        self.feature_cols = usable_features
        print(f"Usable features: {usable_features}")

        return combined_df, usable_features, missing_summary



    def prepare_for_ml(
        self,
        combined_df,
        usable_features,
        target_col="classification",
        min_completeness=0.3,
    ):
        """
        Prepare the combined dataset for machine learning
        """
        print("Preparing data for machine learning...")

        # Remove rows with unknown classification for supervised learning
        ml_df = combined_df[combined_df[target_col] != "Unknown"].copy()
        print(f"Dataset shape after removing unknowns: {ml_df.shape}")

        if len(ml_df) == 0:
            print("No labeled data available for supervised learning!")
            return None, None, None

        # Calculate data completeness for each object
        ml_df["data_completeness"] = ml_df[usable_features].count(axis=1) / len(
            usable_features
        )

        # Filter objects with sufficient data
        complete_objects = ml_df["data_completeness"] >= min_completeness
        ml_df_filtered = ml_df[complete_objects].copy()

        print(
            f"Objects with >{min_completeness*100}% data completeness: {complete_objects.sum()}"
        )
        print(f"Final dataset shape: {ml_df_filtered.shape}")

        if len(ml_df_filtered) == 0:
            print(
                f"No objects meet the {min_completeness*100}% completeness threshold!"
            )
            return None, None, None

        # Advanced missing value handling
        print("Applying advanced missing value imputation...")

        # Group by mission for mission-specific imputation
        mission_medians = {}
        for mission in ml_df_filtered["mission"].unique():
            mission_data = ml_df_filtered[ml_df_filtered["mission"] == mission]
            mission_medians[mission] = mission_data[usable_features].median()

        # Fill missing values with mission-specific medians, then overall median
        for col in usable_features:
            if col in ml_df_filtered.columns:
                ml_df_filtered[col] = pd.to_numeric(
                    ml_df_filtered[col], errors="coerce"
                )

                # Fill with mission-specific median first
                for mission in ml_df_filtered["mission"].unique():
                    mask = (ml_df_filtered["mission"] == mission) & (
                        ml_df_filtered[col].isnull()
                    )
                    if mask.any() and not pd.isna(mission_medians[mission][col]):
                        ml_df_filtered.loc[mask, col] = mission_medians[mission][col]

                # Fill remaining with overall median
                overall_median = ml_df_filtered[col].median()
                ml_df_filtered[col] = ml_df_filtered[col].fillna(overall_median)

        # Prepare features and target
        X = ml_df_filtered[usable_features].copy()
        y = ml_df_filtered[target_col].copy()

        # Remove any objects with all missing features
        valid_rows = ~X.isnull().all(axis=1)
        X = X[valid_rows]
        y = y[valid_rows]

        print(f"Final ML dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        return X, y, ml_df_filtered

    def train_multiple_models(self, X, y, models_to_train=None, cv_folds=5):
        """
        Train multiple ML models and compare performance
        """
        if models_to_train is None:
            models_to_train = list(self.available_models.keys())

        print(f"Training {len(models_to_train)} models...")

        # Encode labels for compatibility with all models
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le

        # Save the label encoder
        le_path = os.path.join(self.models_dir, "label_encoder.pkl")
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        print(f"Label encoder saved to {le_path}")

        # Split data
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # We need the original string labels for reporting
        y_test = le.inverse_transform(y_test_encoded)

        results = {}

        for model_name in models_to_train:
            if model_name not in self.available_models:
                print(f"Model {model_name} not available, skipping...")
                continue

            print(f"\nTraining {model_name}...")

            # Create and train model
            model = self.available_models[model_name]
            scaler = StandardScaler()

            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                # Train model
                model.fit(X_train_scaled, y_train_encoded)

                # Make predictions
                y_pred_encoded = model.predict(X_test_scaled)
                y_pred = le.inverse_transform(y_pred_encoded)

                y_pred_proba = (
                    model.predict_proba(X_test_scaled)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                # Cross-validation
                cv_scores = cross_val_score(
                    model,
                    X_train_scaled,
                    y_train_encoded,
                    cv=cv_folds,
                    scoring="accuracy",
                )

                # Store results
                results[model_name] = {
                    "model": model,
                    "scaler": scaler,
                    "feature_columns": list(X.columns),
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "y_pred_proba": y_pred_proba,
                    "classification_report": classification_report(
                        y_test, y_pred, zero_division=0
                    ),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                }

                print(
                    f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}"
                )
            except Exception as e:
                print(f"Error training model {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        # Store results for successfully trained models
        successful_models = {
            name: res for name, res in results.items() if "error" not in res
        }
        if successful_models:
            self.models.update(
                {name: successful_models[name]["model"] for name in successful_models}
            )
            self.scalers.update(
                {name: successful_models[name]["scaler"] for name in successful_models}
            )
            self.model_performance.update(successful_models)

        if not self.model_performance:
            print("\nNo models were trained successfully.")
            return {}, None

        # Find best model
        best_model_name = max(
            self.model_performance, key=lambda x: self.model_performance[x]["cv_mean"]
        )
        print(
            f"\nBest model: {best_model_name} (CV Accuracy: {self.model_performance[best_model_name]['cv_mean']:.3f})"
        )

        self.feature_cols = list(X.columns)
        # Store training history
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "models_trained": list(results.keys()),
            "best_model": best_model_name,
            "best_cv_score": float(self.model_performance[best_model_name]["cv_mean"]),
            "dataset_shape": [int(d) for d in X.shape],
            "feature_columns": list(X.columns),
        }
        self.training_history.append(training_record)

        return results, best_model_name

    def visualize_model_performance(self, results=None, save_plots=True):
        """
        Create comprehensive visualizations of model performance
        """
        if results is None:
            results = self.model_performance

        if not results:
            print("No model results available for visualization")
            return

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Model Accuracy Comparison
        plt.subplot(3, 4, 1)
        model_names = list(results.keys())
        accuracies = [results[name]["accuracy"] for name in model_names]
        cv_means = [results[name]["cv_mean"] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.bar(x - width / 2, accuracies, width, label="Test Accuracy", alpha=0.8)
        plt.bar(x + width / 2, cv_means, width, label="CV Mean", alpha=0.8)
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. F1 Score Comparison
        plt.subplot(3, 4, 2)
        f1_scores = [results[name]["f1_score"] for name in model_names]
        plt.bar(model_names, f1_scores, alpha=0.8, color="orange")
        plt.xlabel("Models")
        plt.ylabel("F1 Score")
        plt.title("Model F1 Score Comparison")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 3. Cross-validation scores with error bars
        plt.subplot(3, 4, 3)
        cv_means = [results[name]["cv_mean"] for name in model_names]
        cv_stds = [results[name]["cv_std"] for name in model_names]
        plt.errorbar(
            range(len(model_names)),
            cv_means,
            yerr=cv_stds,
            marker="o",
            capsize=5,
            capthick=2,
        )
        plt.xlabel("Models")
        plt.ylabel("CV Accuracy")
        plt.title("Cross-Validation Scores")
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.grid(True, alpha=0.3)

        # 4. Best model confusion matrix
        best_model = max(results, key=lambda x: results[x]["cv_mean"])
        plt.subplot(3, 4, 4)
        cm = results[best_model]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{best_model} - Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 5-8. Individual model confusion matrices (top 4 models)
        top_4_models = sorted(
            results.keys(), key=lambda x: results[x]["cv_mean"], reverse=True
        )[:4]

        for i, model_name in enumerate(top_4_models):
            plt.subplot(3, 4, 5 + i)
            cm = results[model_name]["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False)
            plt.title(f'{model_name}\nAcc: {results[model_name]["accuracy"]:.3f}')
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

        # 9. Training time vs accuracy (if available)
        plt.subplot(3, 4, 9)
        # This would require timing data - placeholder for now
        plt.scatter(accuracies, cv_means, s=100, alpha=0.7)
        for i, model in enumerate(model_names):
            plt.annotate(
                model,
                (accuracies[i], cv_means[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        plt.xlabel("Test Accuracy")
        plt.ylabel("CV Accuracy")
        plt.title("Test vs CV Accuracy")
        plt.grid(True, alpha=0.3)

        # 10. Feature importance (if available for best model)
        plt.subplot(3, 4, 10)
        best_model_obj = results[best_model]["model"]
        if hasattr(best_model_obj, "feature_importances_"):
            feature_importance = (
                pd.DataFrame(
                    {
                        "feature": self.feature_cols,
                        "importance": best_model_obj.feature_importances_,
                    }
                )
                .sort_values("importance", ascending=True)
                .tail(10)
            )

            plt.barh(feature_importance["feature"], feature_importance["importance"])
            plt.xlabel("Importance")
            plt.title(f"{best_model} - Top Features")
        else:
            plt.text(
                0.5,
                0.5,
                "Feature importance\nnot available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        # 11. Model complexity vs performance
        plt.subplot(3, 4, 11)
        # Approximate complexity ranking
        complexity_score = {
            "naive_bayes": 1,
            "logistic_regression": 2,
            "knn": 3,
            "svm": 4,
            "random_forest": 5,
            "extra_trees": 5,
            "gradient_boosting": 6,
            "lightgbm": 7,
        }

        x_complexity = [complexity_score.get(model, 5) for model in model_names]
        plt.scatter(x_complexity, cv_means, s=100, alpha=0.7)
        for i, model in enumerate(model_names):
            plt.annotate(
                model,
                (x_complexity[i], cv_means[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        plt.xlabel("Model Complexity (Approximate)")
        plt.ylabel("CV Accuracy")
        plt.title("Complexity vs Performance")
        plt.grid(True, alpha=0.3)

        # 12. ROC Curve for best model (if multiclass, show macro average)
        plt.subplot(3, 4, 12)
        if results[best_model]["y_pred_proba"] is not None:
            y_test = results[best_model]["y_test"]
            y_pred_proba = results[best_model]["y_pred_proba"]

            # For multiclass, compute macro-average ROC
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc
            from itertools import cycle

            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)

            if len(classes) == 2:
                fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            else:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(len(classes)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute macro-average ROC curve and ROC area
                all_fpr = np.unique(
                    np.concatenate([fpr[i] for i in range(len(classes))])
                )
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(classes)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                mean_tpr /= len(classes)
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                plt.plot(
                    fpr["macro"],
                    tpr["macro"],
                    label=f'Macro-avg ROC (AUC = {roc_auc["macro"]:.2f})',
                )

            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{best_model} - ROC Curve")
            plt.legend()
        else:
            plt.text(
                0.5,
                0.5,
                "ROC curve\nnot available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        plt.tight_layout()

        if save_plots:
            plt.savefig(
                os.path.join(self.models_dir, "model_performance_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"Performance plots saved to {self.models_dir}/model_performance_analysis.png"
            )

        plt.show()

        return fig

    def save_models(self, models_to_save=None):
        """
        Save trained models, scalers, and metadata
        """
        if models_to_save is None:
            models_to_save = list(self.models.keys())

        print(f"Saving {len(models_to_save)} models...")

        for model_name in models_to_save:
            if model_name not in self.models:
                print(f"Model {model_name} not found, skipping...")
                continue

            # Create model-specific directory
            model_dir = os.path.join(self.models_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.models[model_name], f)

            # Save scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scalers[model_name], f)

            # Save metadata
            metadata = {
                "model_name": model_name,
                "feature_columns": self.feature_cols,
                "performance": {
                    "accuracy": float(self.model_performance[model_name]["accuracy"]),
                    "f1_score": float(self.model_performance[model_name]["f1_score"]),
                    "cv_mean": float(self.model_performance[model_name]["cv_mean"]),
                    "cv_std": float(self.model_performance[model_name]["cv_std"]),
                },
                "training_timestamp": datetime.now().isoformat(),
                "classes": (
                    [int(c) for c in self.models[model_name].classes_]
                    if hasattr(self.models[model_name], "classes_")
                    else []
                ),
            }

            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  {model_name} saved to {model_dir}")

        # Save training history
        history_path = os.path.join(self.models_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        # Save feature columns
        features_path = os.path.join(self.models_dir, "feature_columns.json")
        with open(features_path, "w") as f:
            json.dump(self.feature_cols, f, indent=2)

        print(f"All models and metadata saved to {self.models_dir}")

    def load_model(self, model_name):
        """
        Load a specific trained model with its scaler and metadata
        """
        model_dir = os.path.join(self.models_dir, model_name)

        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} not found!")
            return None

        try:
            # Load model
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            # Load metadata
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update instance variables
            self.models[model_name] = model
            self.scalers[model_name] = scaler

            # Load feature columns from feature_columns.json if it exists
            feature_cols_path = os.path.join(self.models_dir, "feature_columns.json")
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, "r") as f:
                    self.feature_cols = json.load(f)
            else:
                # Fallback to metadata if feature_columns.json doesn't exist
                self.feature_cols = metadata.get("feature_columns", [])

            print(f"Model {model_name} loaded successfully!")
            print(f"  Accuracy: {metadata['performance']['accuracy']:.3f}")
            print(f"  Features: {len(self.feature_cols)}")
            print(f"  Feature columns: {', '.join(self.feature_cols)}")
            print(f"  Training date: {metadata['training_timestamp']}")

            return model, scaler, metadata

        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return None

    def load_all_models(self):
        """
        Load all available models from the models directory
        """
        if not os.path.exists(self.models_dir):
            print(f"Models directory {self.models_dir} not found!")
            return

        model_dirs = [
            d
            for d in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, d))
        ]

        print(f"Loading {len(model_dirs)} models...")

        loaded_models = []
        for model_name in model_dirs:
            result = self.load_model(model_name)
            if result is not None:
                loaded_models.append(model_name)

        print(f"Successfully loaded {len(loaded_models)} models: {loaded_models}")
        return loaded_models

    def predict_single_object(self, model_name, **kwargs):
        """
        Make prediction for a single object using specified model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded!")
        if not hasattr(self, "label_encoder") or self.label_encoder is None:
            raise RuntimeError(
                "Label encoder not loaded. Please train or load a model first."
            )

        # Get feature columns for the specific model from its metadata
        model_dir = os.path.join(self.models_dir, model_name)
        metadata_path = os.path.join(model_dir, "metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model_feature_cols = metadata.get("feature_columns", self.feature_cols)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to global feature columns if metadata is not found or invalid
            model_feature_cols = self.feature_cols

        # Create DataFrame from input
        single_data = pd.DataFrame([kwargs])

        # Ensure all feature columns are present
        for col in model_feature_cols:
            if col not in single_data.columns:
                single_data[col] = np.nan

        # Select and order features
        X_single = single_data[model_feature_cols].copy()

        # Handle missing values
        for col in model_feature_cols:
            X_single[col] = pd.to_numeric(X_single[col], errors="coerce")

        # Fill with median or 0 if all NaN
        for col in X_single.columns:
            if X_single[col].isna().all():
                X_single[col] = 0
            else:
                X_single[col] = X_single[col].fillna(X_single[col].median())

        # Scale features
        X_single_scaled = self.scalers[model_name].transform(X_single)

        # Make prediction
        model = self.models[model_name]
        prediction_encoded = model.predict(X_single_scaled)
        prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]

        probabilities = (
            model.predict_proba(X_single_scaled)[0]
            if hasattr(model, "predict_proba")
            else None
        )

        # Create probability dictionary
        prob_dict = {}
        if probabilities is not None:
            classes = self.label_encoder.classes_
            prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}

        return prediction, prob_dict

    def predict_batch(self, model_name, data, save_results=True):
        """
        Make predictions on batch data using specified model
        """
        if model_name not in self.models:
            print(f"Model {model_name} not loaded!")
            return None

        print(f"Making batch predictions with {model_name}...")

        try:
            # Get feature columns for the specific model from its metadata
            model_dir = os.path.join(self.models_dir, model_name)
            metadata_path = os.path.join(model_dir, "metadata.json")
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                model_feature_cols = metadata.get("feature_columns", self.feature_cols)
            except (FileNotFoundError, json.JSONDecodeError):
                # Fallback to global feature columns if metadata is not found or invalid
                model_feature_cols = self.feature_cols

            # Create a case-insensitive mapping of column names
            column_mapping = {col.lower(): col for col in data.columns}

            # Check for missing features (case-insensitive)
            missing_features = []
            feature_mapping = {}
            for feature in model_feature_cols:
                feature_lower = feature.lower()
                if feature_lower in column_mapping:
                    feature_mapping[feature] = column_mapping[feature_lower]
                else:
                    missing_features.append(feature)

            if missing_features:
                print("Available columns:", data.columns.tolist())
                print("Required features:", model_feature_cols)
                raise ValueError(
                    f"Missing required features: {', '.join(missing_features)}"
                )

            # Prepare data with correct feature order using the mapping
            X_batch = pd.DataFrame()
            for feature in model_feature_cols:
                X_batch[feature] = data[feature_mapping[feature]]

            # Handle missing values and convert to numeric
            for col in model_feature_cols:
                X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")

            # Fill missing values with median of the column
            column_medians = X_batch.median()
            X_batch = X_batch.fillna(column_medians).fillna(0)

            # Scale features
            X_batch_scaled = self.scalers[model_name].transform(X_batch)

            # Make predictions
            model = self.models[model_name]
            predictions_encoded = model.predict(X_batch_scaled)
            predictions = self.label_encoder.inverse_transform(predictions_encoded)

            probabilities = (
                model.predict_proba(X_batch_scaled)
                if hasattr(model, "predict_proba")
                else None
            )

            # Create results DataFrame
            results_df = data.copy()
            results_df["predicted_class"] = predictions
            results_df["model_used"] = model_name

            # Add probability columns
            if probabilities is not None:
                classes = self.label_encoder.classes_
                for i, class_name in enumerate(classes):
                    col_name = f'prob_{str(class_name).replace(" ", "_").replace("-", "_").lower()}'
                    results_df[col_name] = probabilities[:, i]

                # Add confidence (max probability)
                results_df["confidence"] = np.max(probabilities, axis=1)

            print(f"Predictions completed for {len(data)} objects")
            print(f"Predicted classes distribution:")
            print(results_df["predicted_class"].value_counts())

            if save_results:
                output_path = os.path.join(
                    self.models_dir,
                    f'batch_predictions_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                )
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")

            return results_df

        except Exception as e:
            print(f"Error in batch prediction: {str(e)}")
            return None

    def ensemble_predict(self, data, models_to_use=None, method="voting"):
        """
        Make ensemble predictions using multiple models
        """
        if models_to_use is None:
            models_to_use = list(self.models.keys())

        if len(models_to_use) < 2:
            print("Need at least 2 models for ensemble prediction!")
            return None

        print(f"Making ensemble predictions with {len(models_to_use)} models...")

        # Get feature columns from the first model in the ensemble list
        first_model_name = models_to_use[0]
        model_dir = os.path.join(self.models_dir, first_model_name)
        metadata_path = os.path.join(model_dir, "metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model_feature_cols = metadata.get("feature_columns", self.feature_cols)
        except (FileNotFoundError, json.JSONDecodeError):
            model_feature_cols = self.feature_cols

        # Prepare data
        X_batch = data[model_feature_cols].copy()
        for col in model_feature_cols:
            X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")
        X_batch = X_batch.fillna(X_batch.median()).fillna(0)

        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []

        for model_name in models_to_use:
            if model_name not in self.models:
                print(f"Model {model_name} not available, skipping...")
                continue

            # Scale and predict
            X_scaled = self.scalers[model_name].transform(X_batch)
            model = self.models[model_name]

            predictions = model.predict(X_scaled)
            all_predictions.append(predictions)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_scaled)
                all_probabilities.append(probabilities)

        # Ensemble prediction
        if method == "voting":
            # Majority voting
            ensemble_predictions_encoded = []
            for i in range(len(data)):
                votes = [pred[i] for pred in all_predictions]
                ensemble_pred = max(set(votes), key=votes.count)
                ensemble_predictions_encoded.append(ensemble_pred)
            ensemble_predictions = self.label_encoder.inverse_transform(
                ensemble_predictions_encoded
            )

        elif method == "averaging" and all_probabilities:
            # Average probabilities
            avg_probabilities = np.mean(all_probabilities, axis=0)
            ensemble_predictions_encoded = np.argmax(avg_probabilities, axis=1)
            ensemble_predictions = self.label_encoder.inverse_transform(
                ensemble_predictions_encoded
            )
        else:
            ensemble_predictions = ["N/A"] * len(data)

        # Create results
        results_df = data.copy()
        results_df["ensemble_prediction"] = ensemble_predictions
        results_df["ensemble_method"] = method
        results_df["models_used"] = ",".join(models_to_use)

        # Add individual model predictions
        for i, model_name in enumerate(models_to_use):
            if i < len(all_predictions):
                results_df[f"pred_{model_name}"] = self.label_encoder.inverse_transform(
                    all_predictions[i]
                )

        print(f"Ensemble predictions completed!")
        print(f"Predicted classes distribution:")
        print(results_df["ensemble_prediction"].value_counts())

        return results_df

    def model_interpretability_analysis(self, model_name, X_sample=None):
        """
        Analyze model interpretability and feature importance
        """
        if model_name not in self.models:
            print(f"Model {model_name} not loaded!")
            return None

        model = self.models[model_name]

        # Get feature columns for the specific model from its metadata
        model_dir = os.path.join(self.models_dir, model_name)
        metadata_path = os.path.join(model_dir, "metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model_feature_cols = metadata.get("feature_columns", self.feature_cols)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to global feature columns if metadata is not found or invalid
            model_feature_cols = self.feature_cols

        print(f"\nModel Interpretability Analysis: {model_name}")
        print("=" * 50)

        results = {"model_name": model_name}

        # Feature importance (for tree-based models)
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": model_feature_cols,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            print("\nFeature Importance (Top 10):")
            print(feature_importance.head(10).to_string(index=False))
            results["feature_importance"] = feature_importance

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(top_features["feature"], top_features["importance"])
            plt.xlabel("Feature Importance")
            plt.title(f"{model_name} - Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        # Coefficients (for linear models)
        elif hasattr(model, "coef_"):
            if len(model.classes_) == 2:  # Binary classification
                coefficients = pd.DataFrame(
                    {"feature": model_feature_cols, "coefficient": model.coef_[0]}
                ).sort_values("coefficient", key=abs, ascending=False)
            else:  # Multiclass
                coefficients = pd.DataFrame(
                    {
                        "feature": model_feature_cols,
                        "avg_abs_coefficient": np.mean(np.abs(model.coef_), axis=0),
                    }
                ).sort_values("avg_abs_coefficient", ascending=False)

            print("\nModel Coefficients (Top 10):")
            print(coefficients.head(10).to_string(index=False))
            results["coefficients"] = coefficients

        # Model complexity metrics
        complexity_info = {}
        if hasattr(model, "n_estimators"):
            complexity_info["n_estimators"] = model.n_estimators
        if hasattr(model, "max_depth"):
            complexity_info["max_depth"] = model.max_depth
        if hasattr(model, "n_features_in_"):
            complexity_info["n_features"] = model.n_features_in_

        if complexity_info:
            print(f"\nModel Complexity:")
            for key, value in complexity_info.items():
                print(f"  {key}: {value}")
            results["complexity"] = complexity_info

        return results

    def hyperparameter_optimization(
        self, model_name, X, y, param_grid=None, cv_folds=5
    ):
        """
        Perform hyperparameter optimization for a specific model
        """
        if model_name not in self.available_models:
            print(f"Model {model_name} not available!")
            return None

        print(f"Performing hyperparameter optimization for {model_name}...")

        # Default parameter grids
        default_param_grids = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.15],
                "max_depth": [3, 5, 7],
            },
            "svm": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.001, 0.01],
                "kernel": ["rbf", "poly"],
            },
        }

        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})

        if not param_grid:
            print(f"No parameter grid available for {model_name}")
            return None

        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Grid search
        base_model = self.available_models[model_name]
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_scaled, y)

        # Results
        print(f"\nBest parameters for {model_name}:")
        print(grid_search.best_params_)
        print(f"Best CV score: {grid_search.best_score_:.3f}")

        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        self.scalers[model_name] = scaler

        return (
            grid_search.best_estimator_,
            grid_search.best_params_,
            grid_search.best_score_,
        )

    def generate_model_report(self, output_path=None):
        """
        Generate comprehensive model performance report
        """
        if not self.model_performance:
            print("No model performance data available!")
            return None

        report = []
        report.append("EXOPLANET CLASSIFICATION MODEL REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total models trained: {len(self.model_performance)}")
        report.append(f"Feature columns: {len(self.feature_cols)}")
        report.append("")

        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 30)

        # Sort models by CV performance
        sorted_models = sorted(
            self.model_performance.items(), key=lambda x: x[1]["cv_mean"], reverse=True
        )

        report.append(
            f"{'Model':<20} {'Accuracy':<10} {'F1 Score':<10} {'CV Mean':<10} {'CV Std':<10}"
        )
        report.append("-" * 60)

        for model_name, performance in sorted_models:
            report.append(
                f"{model_name:<20} {performance['accuracy']:<10.3f} "
                f"{performance['f1_score']:<10.3f} {performance['cv_mean']:<10.3f} "
                f"{performance['cv_std']:<10.3f}"
            )

        report.append("")

        # Best model details
        best_model = sorted_models[0]
        report.append(f"BEST MODEL: {best_model[0]}")
        report.append("-" * 30)
        report.append(
            f"CV Accuracy: {best_model[1]['cv_mean']:.3f} ± {best_model[1]['cv_std']:.3f}"
        )
        report.append(f"Test Accuracy: {best_model[1]['accuracy']:.3f}")
        report.append(f"F1 Score: {best_model[1]['f1_score']:.3f}")
        report.append("")

        # Classification report for best model
        report.append("DETAILED CLASSIFICATION REPORT (Best Model)")
        report.append("-" * 40)
        report.append(best_model[1]["classification_report"])
        report.append("")

        # Features used
        report.append("FEATURES USED")
        report.append("-" * 15)
        for i, feature in enumerate(self.feature_cols, 1):
            report.append(f"{i:2d}. {feature}")
        report.append("")

        # Training history
        if self.training_history:
            report.append("TRAINING HISTORY")
            report.append("-" * 20)
            for i, session in enumerate(self.training_history, 1):
                report.append(f"Session {i}: {session['timestamp']}")
                report.append(
                    f"  Best model: {session['best_model']} (CV: {session['best_cv_score']:.3f})"
                )
                report.append(f"  Dataset shape: {session['dataset_shape']}")
                report.append("")

        # Join report
        report_text = "\n".join(report)

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")

        print(report_text)
        return report_text


def initialize_session_state():
    """Initialize or reset the session state for the Streamlit app."""
    if "classifier" not in st.session_state:
        st.session_state.classifier = ExoplanetClassifierSuite()

    # Try to load the label encoder first
    le_path = os.path.join(st.session_state.classifier.models_dir, "label_encoder.pkl")
    if os.path.exists(le_path):
        try:
            with open(le_path, "rb") as f:
                st.session_state.classifier.label_encoder = pickle.load(f)
                print("Label encoder loaded.")
        except Exception as e:
            print(f"Error loading label encoder: {e}")

    # Try to load all available models
    loaded_models = st.session_state.classifier.load_all_models()

    if loaded_models:
        st.session_state.models_loaded = True
        st.session_state.available_models = loaded_models

        # Load feature columns from feature_columns.json
        feature_cols_path = os.path.join(
            st.session_state.classifier.models_dir, "feature_columns.json"
        )
        if os.path.exists(feature_cols_path):
            with open(feature_cols_path, "r") as f:
                st.session_state.classifier.feature_cols = json.load(f)
                print("Available features:", st.session_state.classifier.feature_cols)

        # Initialize model_results for loaded models
        if "model_results" not in st.session_state:
            st.session_state.model_results = {}

        # Add performance metrics for each loaded model
        for model_name in loaded_models:
            model_dir = os.path.join(st.session_state.classifier.models_dir, model_name)
            metadata_path = os.path.join(model_dir, "metadata.json")

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    performance_data = {
                        "accuracy": metadata["performance"]["accuracy"],
                        "f1_score": metadata["performance"]["f1_score"],
                        "cv_mean": metadata["performance"]["cv_mean"],
                        "cv_std": metadata["performance"]["cv_std"],
                        "y_pred": None,
                        "y_test": None,
                        "y_pred_proba": None,
                    }

                    # Add additional metrics if available
                    if "classification_report" in metadata:
                        performance_data["classification_report"] = metadata[
                            "classification_report"
                        ]
                    if "confusion_matrix" in metadata:
                        performance_data["confusion_matrix"] = metadata[
                            "confusion_matrix"
                        ]

                    st.session_state.model_results[model_name] = performance_data
            except Exception as e:
                print(f"Error loading metadata for {model_name}: {str(e)}")
                st.session_state.model_results[model_name] = {
                    "accuracy": 0.0,
                    "f1_score": 0.0,
                    "cv_mean": 0.0,
                    "cv_std": 0.0,
                    "y_pred": None,
                    "y_test": None,
                    "y_pred_proba": None,
                }
    else:
        st.session_state.models_loaded = False
        st.session_state.available_models = []


def process_and_display_data(classifier, k2_df=None, tess_df=None, kepler_df=None):
    """Helper function to process and display data summary."""
    try:
        # Store the dataframes in session state
        st.session_state["k2_df"] = k2_df
        st.session_state["tess_df"] = tess_df
        st.session_state["kepler_df"] = kepler_df

        # Prepare data
        with st.spinner("Preparing and classifying data..."):
            combined_df, usable_features, missing_summary = (
                classifier.prepare_exoplanet_data_for_classification(
                    k2_df=k2_df, tess_df=tess_df, kepler_df=kepler_df
                )
            )

        # Store prepared data in session state
        st.session_state["combined_df"] = combined_df
        st.session_state["usable_features"] = usable_features
        st.session_state["missing_summary"] = missing_summary

        st.success("Data loaded and prepared successfully!")

        # Display data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("K2 Samples", len(k2_df) if k2_df is not None else 0)
        with col2:
            st.metric("TESS Samples", len(tess_df) if tess_df is not None else 0)
        with col3:
            st.metric("Kepler Samples", len(kepler_df) if kepler_df is not None else 0)

        # Display feature information
        st.subheader("Usable Features")
        st.write(f"Number of usable features: {len(usable_features)}")
        st.write(usable_features)

        # Display missing data summary
        st.subheader("Missing Data Analysis")
        st.dataframe(missing_summary)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")


def create_streamlit_app():

    # Initialize session state
    initialize_session_state()

    # Use the classifier from session state
    classifier = st.session_state.classifier

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        [
            "Home",
            "Guided Discovery",
            "Data Upload",
            "Model Training",
            "Predictions",
            "Hyperparameter Tuning",
            "Analysis",
            "Challenge Info",
            "Chatbot Assistant",
        ],
    )

    if page == "Home":
        # Custom CSS for a more modern look
        st.markdown(
            """
            <style>
                .main .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }
                h1 {
                    font-size: 2.5rem !important;
                }
                .st-emotion-cache-1y4p8pa {
                    padding-top: 2rem;
                }
                .card {
                    background-color: #ffffff;
                    border-radius: 10px;
                    padding: 2rem;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    transition: all 0.3s ease;
                    height: 100%;
                }
                .card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                }
                .card .icon {
                    font-size: 3rem;
                    margin-bottom: 1rem;
                }
                .card h4, .card p {
                    color: #000;
                }
                .model-tags {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.75rem;
                    justify-content: center;
                    padding-top: 1rem;
                }
                .model-tag {
                    background-color: #e1e8f0;
                    color: #333;
                    padding: 0.5rem 1rem;
                    border-radius: 15px;
                    font-size: 0.9rem;
                    font-weight: 500;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # --- Hero Section ---
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.title("Exoplanet Classification Suite")
            st.markdown(
                """
                <p style='font-size: 1.2em; color: #555;'>
                Welcome to your personal dashboard for exploring and classifying distant worlds. 
                This tool leverages state-of-the-art machine learning to analyze data from NASA missions and help distinguish between confirmed exoplanets, candidates, and false positives.
                </p>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.image(
                "https://exoplanets.nasa.gov/rails/active_storage/disk/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaDdDVG9JYTJWNVNTSWhaalYzTW1ZM01ESmljblp6TVRkeU5IZHpOM2g0YUhOb2JXOXJZd1k2QmtWVU9oQmthWE53YjNOcGRHbHZia2tpUFdGMGRHRmphRzFsYm5RN0lHWnBiR1Z1WVcxbFBTSmxlRzh1Y0c1bklqc2dabWxzWlc1aGJXVXFQVlZVUmkwNEp5ZGxlRzh1Y0c1bkJqc0dWRG9SWTI5dWRHVnVkRjkwZVhCbFNTSUFCanNHVkRvUmMyVnlkbWxqWlY5dVlXMWxPZ3BzYjJOaGJBPT0iLCJleHAiOm51bGwsInB1ciI6ImJsb2Jfa2V5In19--d54da5bbb8b48c46a9bd2928ddf085a36f0a8ced/exo.png",
                width=300,
            )

        st.divider()

        # --- Features Section ---
        st.subheader("Key Features")
        # Add CSS for equal height cards
        st.markdown(
            """
            <style>
            .card {
                background:white;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                color: white;
                height: 220px; /* fixed equal height */
                display: flex;
                flex-direction: column;
                justify-content: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            }
            .card .icon {
                font-size: 40px;
                margin-bottom: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="card">
                    <div class="icon">🛰️</div>
                    <h4>Upload & Prepare Data</h4>
                    <p>Bring your own datasets or download directly from the NASA Exoplanet Archive.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="card">
                    <div class="icon">🚀</div>
                    <h4>Train ML Models</h4>
                    <p>Train a variety of machine learning models to classify exoplanets with a single click.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="card">
                    <div class="icon">🔭</div>
                    <h4>Analyze & Predict</h4>
                    <p>Use your trained models to predict the nature of new candidates and analyze model performance.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # --- Available Models Section ---
        st.subheader("Available Models for Training")
        model_list = list(classifier.available_models.keys())
        model_tags_html = "".join(
            [
                f'<div class="model-tag">{model.replace("_", " ").title()}</div>'
                for model in model_list
            ]
        )
        st.markdown(
            f'<div class="model-tags">{model_tags_html}</div>', unsafe_allow_html=True
        )

    elif page == "Guided Discovery":
        st.header("🔭 Guided Discovery")
        st.write("Explore the exoplanet data with these guided analyses.")

        if "combined_df" not in st.session_state:
            st.warning(
                "Please upload or download data on the 'Data Upload' page first!"
            )
            return

        df = st.session_state["combined_df"]

        # --- Section 1: Potentially Habitable Planets ---
        st.subheader("🌎 Potentially Habitable Planets")
        st.write(
            "Let's search for exoplanets that might be able to support life as we know it. We'll look for planets with temperatures where liquid water could exist and with a size that suggests a rocky composition."
        )

        col1, col2 = st.columns(2)
        with col1:
            temp_range = st.slider(
                "Equilibrium Temperature Range (K)",
                float(df["equilibrium_temp"].min()),
                float(df["equilibrium_temp"].max()),
                (200.0, 350.0),
            )
        with col2:
            radius_range = st.slider(
                "Planet Radius (Earth Radii)", 0.0, 10.0, (0.5, 2.0)
            )

        habitable_df = df[
            (df["equilibrium_temp"].between(temp_range[0], temp_range[1]))
            & (df["planet_radius"].between(radius_range[0], radius_range[1]))
        ]

        st.write(f"Found {len(habitable_df)} potentially habitable candidates.")

        if not habitable_df.empty:
            display_df = habitable_df[
                [
                    "object_name",
                    "classification",
                    "equilibrium_temp",
                    "planet_radius",
                    "mission",
                ]
            ].copy()
            for col in ["object_name", "classification", "mission"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].astype(str)
            st.dataframe(display_df)

            fig = px.scatter(
                habitable_df,
                x="equilibrium_temp",
                y="planet_radius",
                color="mission",
                hover_name="object_name",
                title="Potentially Habitable Candidates",
            )
            st.plotly_chart(fig)

        st.divider()

        # --- Section 2: Mission Comparison ---
        st.subheader("🛰️ Mission Comparison")
        st.write(
            "Compare the discoveries and candidates from the different NASA missions."
        )

        mission_counts = df["mission"].value_counts().reset_index()
        mission_counts.columns = ["Mission", "Count"]

        fig = px.bar(
            mission_counts,
            x="Mission",
            y="Count",
            color="Mission",
            title="Number of Objects per Mission",
        )
        st.plotly_chart(fig)

        st.divider()

        # --- Section 3: Extreme Exoplanets ---
        st.subheader("🔥 Extreme Exoplanets")
        st.write("Discover the outliers in the exoplanet catalog.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("#### Largest Planets")
            largest = df.nlargest(5, "planet_radius")[
                ["object_name", "planet_radius"]
            ].copy()
            largest["object_name"] = largest["object_name"].astype(str)
            st.dataframe(largest)
        with col2:
            st.write("#### Hottest Planets")
            hottest = df.nlargest(5, "equilibrium_temp")[
                ["object_name", "equilibrium_temp"]
            ].copy()
            hottest["object_name"] = hottest["object_name"].astype(str)
            st.dataframe(hottest)
        with col3:
            st.write("#### Fastest Orbits")
            fastest = (
                df[df["orbital_period"] > 0]
                .nsmallest(5, "orbital_period")[["object_name", "orbital_period"]]
                .copy()
            )
            fastest["object_name"] = fastest["object_name"].astype(str)
            st.dataframe(fastest)
    elif page == "Data Upload":
        st.header("Data Upload and Preparation")

        upload_tab, download_tab = st.tabs(
            ["Upload from files", "Download from NASA Archive"]
        )

        with upload_tab:
            st.subheader("Upload Your Datasets")
            st.write(
                "You can upload one or more datasets. "
                "If you only upload Kepler data, the model will be trained only on it."
            )
            k2_file = st.file_uploader(
                "Upload K2 Dataset (CSV)", type=["csv"], key="k2_upload"
            )
            tess_file = st.file_uploader(
                "Upload TESS Dataset (CSV)", type=["csv"], key="tess_upload"
            )
            kepler_file = st.file_uploader(
                "Upload Kepler Dataset (CSV)", type=["csv"], key="kepler_upload"
            )

            if st.button("Process Uploaded Files"):
                if not any([k2_file, tess_file, kepler_file]):
                    st.warning("Please upload at least one dataset.")
                else:
                    k2_df = pd.read_csv(k2_file) if k2_file else None
                    tess_df = pd.read_csv(tess_file) if tess_file else None
                    kepler_df = pd.read_csv(kepler_file) if kepler_file else None
                    process_and_display_data(
                        classifier, k2_df=k2_df, tess_df=tess_df, kepler_df=kepler_df
                    )

        with download_tab:
            st.subheader("Download from NASA Exoplanet Archive")
            st.write(
                "Click the button below to download the latest datasets directly from the NASA Exoplanet Archive."
            )

            data_links = {
                "kepler": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+q1_q17_dr25_koi&format=csv",
                "k2": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+k2names&format=csv",
                "tess": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv",
            }

            if st.button("Download and Prepare Data"):
                with st.spinner("Downloading data... This may take a moment."):
                    kepler_df = pd.read_csv(data_links["kepler"])
                    k2_df = pd.read_csv(data_links["k2"])
                    tess_df = pd.read_csv(data_links["tess"])

                process_and_display_data(classifier, k2_df, tess_df, kepler_df)

    elif page == "Model Training":
        st.header("Model Training and Evaluation")

        if (
            "combined_df" not in st.session_state
            or "usable_features" not in st.session_state
        ):
            st.warning("Please upload and prepare your data first!")
            return

        # Model selection
        st.subheader("Select Models to Train")
        available_models = list(classifier.available_models.keys())
        selected_models = st.multiselect(
            "Choose models to train",
            available_models,
            default=["random_forest", "gradient_boosting"],
        )

        train_col1, train_col2 = st.columns([2, 1])
        with train_col1:
            if st.button("Train Selected Models"):
                if not selected_models:
                    st.error("Please select at least one model to train")
                    return

                try:
                    # Prepare data for ML
                    with st.spinner("Preparing data for training..."):
                        X, y, ml_df = classifier.prepare_for_ml(
                            st.session_state["combined_df"],
                            st.session_state["usable_features"],
                        )

                        if X is not None and y is not None:
                            st.session_state["X_data"] = X
                            st.session_state["y_data"] = y
                            # Train models
                            with st.spinner("Training models..."):
                                results, best_model = classifier.train_multiple_models(
                                    X, y, selected_models
                                )

                                if results:
                                    # Save models and update session state
                                    classifier.save_models(selected_models)

                                    # Store all necessary model results
                                    for model_name, model_results in results.items():
                                        st.session_state["model_results"][
                                            model_name
                                        ] = {
                                            "accuracy": model_results["accuracy"],
                                            "f1_score": model_results["f1_score"],
                                            "cv_mean": model_results["cv_mean"],
                                            "cv_std": model_results["cv_std"],
                                            "classification_report": model_results.get(
                                                "classification_report"
                                            ),
                                            "confusion_matrix": model_results.get(
                                                "confusion_matrix"
                                            ),
                                            "y_test": model_results.get("y_test"),
                                            "y_pred": model_results.get("y_pred"),
                                            "y_pred_proba": model_results.get(
                                                "y_pred_proba"
                                            ),
                                        }

                                    st.session_state["best_model"] = best_model
                                    st.session_state.models_loaded = True
                                    st.session_state.available_models = list(
                                        classifier.models.keys()
                                    )

                                    st.success(
                                        f"Training complete! Best model: {best_model}"
                                    )
                                else:
                                    st.error(
                                        "Model training failed. Please check your data and try again."
                                    )
                        else:
                            st.error("Error preparing data for machine learning")

                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")

        with train_col2:
            if st.button("Load Saved Models"):
                with st.spinner("Loading saved models..."):
                    loaded_models = classifier.load_all_models()
                    if loaded_models:
                        st.session_state.models_loaded = True
                        st.session_state.available_models = loaded_models
                        st.success(f"Successfully loaded {len(loaded_models)} models")
                    else:
                        st.warning("No saved models found")

    elif page == "Predictions":
        st.header("Make Predictions")

        if not st.session_state.models_loaded:
            st.warning("No models are loaded. Please train or load models first!")
            return

        # Tabs for different prediction methods
        pred_tab1, pred_tab2, pred_tab3 = st.tabs(
            ["Single Prediction", "Batch Prediction", "Ensemble Prediction"]
        )

        with pred_tab1:
            st.subheader("Single Object Prediction")

            # Check available models
            available_models = [name for name in classifier.models.keys()]
            if not available_models:
                st.warning("No trained models available. Please train models first.")
                return

            # Model selection for prediction
            model_name = st.selectbox(
                "Select Model for Prediction", available_models, key="single_pred_model"
            )

            # Use columns for better layout
            col1, col2 = st.columns(2)

            # Create input fields for features with default ranges
            feature_values = {}
            feature_ranges = {
                "ra": (-90, 90),
                "dec": (-90, 90),
                "stellar_mag": (0, 20),
                "orbital_period": (0, 1000),
                "planet_radius": (0, 10),
                "transit_depth": (0, 0.1),
                "insolation": (0, 2000),
                "equilibrium_temp": (0, 3000),
                "stellar_teff": (2000, 12000),
                "stellar_logg": (0, 5),
                "stellar_radius": (0, 100),
            }

            for i, feature in enumerate(classifier.feature_cols):
                with col1 if i % 2 == 0 else col2:
                    min_val, max_val = feature_ranges.get(feature, (0, 100))
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(min_val),
                        help=f"Range: {min_val} to {max_val}",
                        format="%.3f",
                    )

            if st.button("Make Prediction", key="single_pred_button"):
                try:
                    with st.spinner("Making prediction..."):
                        prediction, probabilities = classifier.predict_single_object(
                            model_name, **feature_values
                        )

                    # Create columns for results
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.success(f"Predicted Class: {prediction}")
                        if probabilities:
                            probs_df = pd.DataFrame(
                                probabilities.items(), columns=["Class", "Probability"]
                            ).sort_values("Probability", ascending=False)
                            st.dataframe(
                                probs_df.style.format({"Probability": "{:.3f}"})
                            )

                    with result_col2:
                        if probabilities:
                            fig = go.Figure(
                                data=[
                                    go.Bar(
                                        x=list(probabilities.keys()),
                                        y=list(probabilities.values()),
                                        text=[
                                            f"{v:.3f}" for v in probabilities.values()
                                        ],
                                        textposition="auto",
                                    )
                                ]
                            )
                            fig.update_layout(
                                title="Prediction Probabilities",
                                xaxis_title="Class",
                                yaxis_title="Probability",
                                yaxis_range=[0, 1],
                            )
                            st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

        with pred_tab2:
            st.subheader("Batch Prediction")

            # File upload for batch prediction
            batch_file = st.file_uploader(
                "Upload CSV file for batch prediction",
                type=["csv"],
                help="CSV file should contain columns matching the feature names",
            )

            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)

                    # Display feature columns information
                    st.subheader("Feature Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Required features:")
                        for feature in classifier.feature_cols:
                            st.write(f"- {feature}")

                    with col2:
                        st.write("Uploaded data features:")
                        for col in batch_df.columns:
                            st.write(f"- {col}")

                    # Create a feature mapping dictionary (case-insensitive)
                    column_mapping = {col.lower(): col for col in batch_df.columns}
                    missing_features = []
                    for feature in classifier.feature_cols:
                        if feature.lower() not in column_mapping:
                            missing_features.append(feature)

                    if missing_features:
                        st.error(
                            f"Missing required features: {', '.join(missing_features)}"
                        )
                    else:
                        st.success(
                            "All required features are present in the uploaded data!"
                        )

                    st.write("Preview of uploaded data:")
                    st.dataframe(batch_df.head())

                    # Model selection for batch prediction
                    batch_model = st.selectbox(
                        "Select Model for Batch Prediction",
                        list(st.session_state.get("model_results", {}).keys()),
                        key="batch_pred_model",
                    )

                    if st.button("Run Batch Prediction"):
                        with st.spinner("Processing batch predictions..."):
                            # Ensure we have all required features
                            required_features = classifier.feature_cols
                            missing_features = [
                                f
                                for f in required_features
                                if f not in batch_df.columns
                            ]

                            if missing_features:
                                st.error(
                                    f"Missing required features: {', '.join(missing_features)}"
                                )
                                return

                            # Prepare the data with correct feature order
                            prediction_data = batch_df[required_features].copy()

                            # Run prediction
                            results_df = classifier.predict_batch(
                                batch_model, prediction_data
                            )

                            # Add back any additional columns from original data
                            for col in batch_df.columns:
                                if col not in results_df.columns:
                                    results_df[col] = batch_df[col]

                            st.success("Batch prediction completed!")

                            # Show results summary
                            st.subheader("Prediction Results")

                            # Display class distribution
                            class_dist = results_df["predicted_class"].value_counts()
                            fig = px.pie(
                                values=class_dist.values,
                                names=class_dist.index,
                                title="Distribution of Predicted Classes",
                            )
                            st.plotly_chart(fig)

                            # Display detailed results
                            st.write("Detailed Results:")
                            st.dataframe(results_df)

                            # Option to download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Predictions CSV",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key="download-csv",
                            )

                except Exception as e:
                    st.error(f"Error in batch prediction: {str(e)}")

        with pred_tab3:
            st.subheader("Ensemble Prediction")

            available_models = [name for name in classifier.models.keys()]
            if len(available_models) < 2:
                st.warning(
                    "At least 2 trained models are required for ensemble prediction. Please train more models."
                )
                return

            st.info(f"Available models for ensemble: {', '.join(available_models)}")

            # Select models for ensemble
            selected_models = st.multiselect(
                "Select Models for Ensemble",
                available_models,
                default=available_models[:2],
            )

            # Select ensemble method
            ensemble_method = st.radio(
                "Select Ensemble Method",
                ["voting", "averaging"],
                help="Voting: majority vote, Averaging: average of probabilities",
            )

            # Input method (same as single prediction)
            feature_values = {}
            col1, col2 = st.columns(2)

            for i, feature in enumerate(classifier.feature_cols):
                with col1 if i % 2 == 0 else col2:
                    min_val, max_val = feature_ranges.get(feature, (0, 100))
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(min_val),
                        help=f"Range: {min_val} to {max_val}",
                        key=f"ensemble_{feature}",
                        format="%.3f",
                    )

            if st.button("Run Ensemble Prediction"):
                if len(selected_models) < 2:
                    st.error("Please select at least 2 models for ensemble prediction")
                else:
                    try:
                        # Create a DataFrame with the single row of features
                        input_df = pd.DataFrame([feature_values])

                        with st.spinner("Running ensemble prediction..."):
                            ensemble_results = classifier.ensemble_predict(
                                input_df, selected_models, method=ensemble_method
                            )

                            st.success("Ensemble prediction completed!")

                            # Display results
                            prediction = ensemble_results["ensemble_prediction"].iloc[0]
                            st.write(f"### Ensemble Prediction: {prediction}")

                            # Show individual model predictions
                            st.subheader("Individual Model Predictions")
                            model_predictions = {
                                model: ensemble_results[f"pred_{model}"].iloc[0]
                                for model in selected_models
                            }

                            fig = go.Figure(
                                data=[
                                    go.Table(
                                        header=dict(
                                            values=["Model", "Prediction"],
                                            fill_color="paleturquoise",
                                            align="left",
                                        ),
                                        cells=dict(
                                            values=[
                                                list(model_predictions.keys()),
                                                list(model_predictions.values()),
                                            ],
                                            fill_color="lavender",
                                            align="left",
                                        ),
                                    )
                                ]
                            )
                            st.plotly_chart(fig)

                    except Exception as e:
                        st.error(f"Error in ensemble prediction: {str(e)}")

    elif page == "Analysis":
        st.header("Model Analysis and Visualization")

        if "model_results" not in st.session_state:
            st.warning("Please train models first!")
            return

        # Create tabs for different types of analysis
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
            ["Model Performance", "Feature Analysis", "Detailed Metrics"]
        )

        with analysis_tab1:
            st.subheader("Model Performance Comparison")

            # Check for available models
            available_models = [name for name in classifier.models.keys()]
            if not available_models:
                st.warning("No trained models available. Please train models first.")
                return

            # Prepare performance data
            model_names = available_models
            metrics = {
                "Accuracy": [
                    st.session_state["model_results"][m]["accuracy"]
                    for m in model_names
                ],
                "F1 Score": [
                    st.session_state["model_results"][m]["f1_score"]
                    for m in model_names
                ],
                "CV Score": [
                    st.session_state["model_results"][m]["cv_mean"] for m in model_names
                ],
            }

            # Create interactive performance comparison plot
            metric_to_plot = st.selectbox(
                "Select Metric to Visualize", list(metrics.keys())
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=model_names,
                        y=metrics[metric_to_plot],
                        text=[f"{v:.3f}" for v in metrics[metric_to_plot]],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title=f"{metric_to_plot} Comparison",
                xaxis_title="Model",
                yaxis_title=metric_to_plot,
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig)

            # ROC Curves
            st.subheader("ROC Curves Comparison")
            selected_models = st.multiselect(
                "Select Models to Compare",
                model_names,
                default=[model_names[0]] if model_names else None,
            )

            if selected_models:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash"),
                        name="Random",
                    )
                )

                for model_name in selected_models:
                    results = st.session_state["model_results"].get(model_name, {})
                    if not results:
                        continue

                    y_pred_proba = results.get("y_pred_proba")
                    y_test = results.get("y_test")

                    if (
                        y_pred_proba is not None
                        and y_test is not None
                        and hasattr(classifier, "label_encoder")
                    ):
                        from sklearn.preprocessing import label_binarize

                        classes = classifier.label_encoder.classes_
                        y_test_bin = label_binarize(y_test, classes=classes)
                        n_classes = len(classes)

                        if n_classes <= 2:
                            # roc_curve for binary case
                            fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
                            auc_score = auc(fpr, tpr)
                            fig.add_trace(
                                go.Scatter(
                                    x=fpr,
                                    y=tpr,
                                    name=f"{model_name} (AUC = {auc_score:.3f})",
                                )
                            )
                        else:  # Multiclass
                            # Compute ROC curve and ROC area for each class
                            fpr = dict()
                            tpr = dict()
                            for i in range(n_classes):
                                fpr[i], tpr[i], _ = roc_curve(
                                    y_test_bin[:, i], y_pred_proba[:, i]
                                )

                            # Compute macro-average ROC curve and ROC area
                            all_fpr = np.unique(
                                np.concatenate([fpr[i] for i in range(n_classes)])
                            )
                            mean_tpr = np.zeros_like(all_fpr)
                            for i in range(n_classes):
                                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                            mean_tpr /= n_classes

                            macro_roc_auc = auc(all_fpr, mean_tpr)

                            fig.add_trace(
                                go.Scatter(
                                    x=all_fpr,
                                    y=mean_tpr,
                                    name=f"{model_name} Macro-avg (AUC = {macro_roc_auc:.3f})",
                                    line=dict(width=4),
                                )
                            )

                fig.update_layout(
                    title="ROC Curves Comparison",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    yaxis_scaleanchor="x",
                    yaxis_scaleratio=1,
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                )
                st.plotly_chart(fig)

        with analysis_tab2:
            feature_tabs = st.tabs(
                [
                    "Feature Importance",
                    "Feature Correlations",
                    "Feature Distribution",
                    "Feature Interactions",
                ]
            )

            with feature_tabs[0]:
                st.subheader("Feature Importance Analysis")
                # Model selection for feature analysis
                model_name = st.selectbox(
                    "Select Model to Analyze", model_names, key="feature_analysis_model"
                )

                try:
                    analysis_results = classifier.model_interpretability_analysis(
                        model_name
                    )

                    if analysis_results and "feature_importance" in analysis_results:
                        # Interactive feature importance plot
                        col1, col2 = st.columns([3, 1])

                        with col2:
                            n_features = st.slider(
                                "Top Features",
                                min_value=5,
                                max_value=len(classifier.feature_cols),
                                value=10,
                            )

                            plot_type = st.radio(
                                "Plot Type", ["Bar", "Tree"], key="importance_plot_type"
                            )

                        with col1:
                            if plot_type == "Bar":
                                fig = px.bar(
                                    analysis_results["feature_importance"].head(
                                        n_features
                                    ),
                                    x="importance",
                                    y="feature",
                                    orientation="h",
                                    title=f"Top {n_features} Most Important Features",
                                )
                                fig.update_layout(
                                    yaxis={"categoryorder": "total ascending"},
                                    height=400,
                                )
                            else:
                                # Tree plot using plotly
                                df = analysis_results["feature_importance"].head(
                                    n_features
                                )
                                fig = go.Figure(
                                    go.Treemap(
                                        labels=df["feature"],
                                        parents=[""] * len(df),
                                        values=df["importance"],
                                        textinfo="label+value",
                                        texttemplate="%{label}<br>%{value:.3f}",
                                    )
                                )
                                fig.update_layout(
                                    title=f"Feature Importance Treemap (Top {n_features})",
                                    height=400,
                                )

                            st.plotly_chart(fig)

                        # Feature importance table with sorting
                        st.subheader("Detailed Feature Importance")
                        importance_df = analysis_results["feature_importance"].copy()
                        importance_df["importance"] = importance_df["importance"].round(
                            4
                        )
                        st.dataframe(
                            importance_df.style.background_gradient(
                                subset=["importance"], cmap="RdYlBu"
                            ),
                            hide_index=True,
                        )

                except Exception as e:
                    st.error(f"Error in feature importance analysis: {str(e)}")

            with feature_tabs[1]:
                st.subheader("Feature Correlation Analysis")
                if "combined_df" in st.session_state:
                    correlation_matrix = st.session_state["combined_df"][
                        classifier.feature_cols
                    ].corr()

                    # Correlation visualization options
                    col1, col2 = st.columns([3, 1])

                    with col2:
                        correlation_threshold = st.slider(
                            "Correlation Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.1,
                        )

                        plot_type = st.radio(
                            "Plot Type",
                            ["Heatmap", "Network"],
                            key="correlation_plot_type",
                        )

                    with col1:
                        if plot_type == "Heatmap":
                            fig = px.imshow(
                                correlation_matrix,
                                labels=dict(color="Correlation"),
                                color_continuous_scale="RdBu_r",
                            )
                        else:
                            # Create network graph of correlations
                            edges = []
                            for i in range(len(correlation_matrix.columns)):
                                for j in range(i + 1, len(correlation_matrix.columns)):
                                    corr = correlation_matrix.iloc[i, j]
                                    if abs(corr) >= correlation_threshold:
                                        edges.append(
                                            (
                                                correlation_matrix.columns[i],
                                                correlation_matrix.columns[j],
                                                abs(corr),
                                            )
                                        )

                            if edges:
                                edge_x = []
                                edge_y = []
                                edge_colors = []
                                for edge in edges:
                                    edge_x.extend([edge[0], edge[1], None])
                                    edge_y.extend([edge[2], edge[2], None])
                                    edge_colors.extend([edge[2], edge[2], None])

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=edge_x,
                                        y=edge_y,
                                        line=dict(width=1, color="rgb(150,150,150)"),
                                        hoverinfo="none",
                                        mode="lines",
                                    )
                                )

                                fig.add_trace(
                                    go.Scatter(
                                        x=correlation_matrix.columns,
                                        y=[1] * len(correlation_matrix.columns),
                                        mode="markers+text",
                                        text=correlation_matrix.columns,
                                        textposition="bottom center",
                                        marker=dict(size=20),
                                    )
                                )
                            else:
                                fig = go.Figure()
                                fig.add_annotation(
                                    text="No correlations above threshold",
                                    xref="paper",
                                    yref="paper",
                                    x=0.5,
                                    y=0.5,
                                    showarrow=False,
                                )

                        fig.update_layout(
                            title="Feature Correlations", height=600, showlegend=False
                        )
                        st.plotly_chart(fig)

                    # Show strongest correlations in a table
                    st.subheader("Strongest Feature Correlations")
                    correlations = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i + 1, len(correlation_matrix.columns)):
                            corr = correlation_matrix.iloc[i, j]
                            if abs(corr) >= correlation_threshold:
                                correlations.append(
                                    {
                                        "Feature 1": correlation_matrix.columns[i],
                                        "Feature 2": correlation_matrix.columns[j],
                                        "Correlation": corr,
                                    }
                                )

                    if correlations:
                        corr_df = pd.DataFrame(correlations)
                        corr_df = corr_df.sort_values(
                            "Correlation", key=abs, ascending=False
                        )
                        st.dataframe(
                            corr_df.style.format(
                                {"Correlation": "{:.3f}"}
                            ).background_gradient(
                                subset=["Correlation"], cmap="RdYlBu"
                            ),
                            hide_index=True,
                        )

            with feature_tabs[2]:
                st.subheader("Feature Distribution Analysis")
                if "combined_df" in st.session_state:
                    col1, col2 = st.columns([3, 1])

                    with col2:
                        selected_feature = st.selectbox(
                            "Select Feature", classifier.feature_cols
                        )

                        plot_type = st.radio(
                            "Plot Type",
                            ["Histogram", "Box Plot", "Violin Plot"],
                            key="distribution_plot_type",
                        )

                        if "classification" in st.session_state["combined_df"].columns:
                            show_by_class = st.checkbox("Show by Class", value=True)

                    with col1:
                        data = st.session_state["combined_df"]

                        if plot_type == "Histogram":
                            if show_by_class and "classification" in data.columns:
                                fig = px.histogram(
                                    data,
                                    x=selected_feature,
                                    color="classification",
                                    marginal="box",
                                    title=f"Distribution of {selected_feature}",
                                )
                            else:
                                fig = px.histogram(
                                    data,
                                    x=selected_feature,
                                    marginal="box",
                                    title=f"Distribution of {selected_feature}",
                                )

                        elif plot_type == "Box Plot":
                            if show_by_class and "classification" in data.columns:
                                fig = px.box(
                                    data,
                                    y=selected_feature,
                                    x="classification",
                                    title=f"Box Plot of {selected_feature}",
                                )
                            else:
                                fig = px.box(
                                    data,
                                    y=selected_feature,
                                    title=f"Box Plot of {selected_feature}",
                                )

                        else:  # Violin Plot
                            if show_by_class and "classification" in data.columns:
                                fig = px.violin(
                                    data,
                                    y=selected_feature,
                                    x="classification",
                                    box=True,
                                    points="all",
                                    title=f"Violin Plot of {selected_feature}",
                                )
                            else:
                                fig = px.violin(
                                    data,
                                    y=selected_feature,
                                    box=True,
                                    points="all",
                                    title=f"Violin Plot of {selected_feature}",
                                )

                        fig.update_layout(height=500)
                        st.plotly_chart(fig)

                        # Show basic statistics
                        stats_df = pd.DataFrame(
                            {
                                "Statistic": [
                                    "Mean",
                                    "Median",
                                    "Std Dev",
                                    "Min",
                                    "Max",
                                ],
                                "Value": [
                                    data[selected_feature].mean(),
                                    data[selected_feature].median(),
                                    data[selected_feature].std(),
                                    data[selected_feature].min(),
                                    data[selected_feature].max(),
                                ],
                            }
                        )

                        # Convert numeric values to float and handle formatting
                        try:
                            stats_df["Value"] = pd.to_numeric(
                                stats_df["Value"], errors="coerce"
                            )
                            st.dataframe(
                                stats_df.style.format(
                                    {
                                        "Value": lambda x: (
                                            "{:.3f}".format(float(x))
                                            if pd.notnull(x)
                                            else "N/A"
                                        )
                                    }
                                ),
                                hide_index=True,
                            )
                        except Exception as e:
                            # Fallback to displaying without formatting if there's an error
                            st.dataframe(stats_df, hide_index=True)

            with feature_tabs[3]:
                st.subheader("Feature Interactions")
                if "combined_df" in st.session_state:
                    col1, col2 = st.columns([3, 1])

                    with col2:
                        feature1 = st.selectbox(
                            "Select first feature", classifier.feature_cols, key="feat1"
                        )
                        feature2 = st.selectbox(
                            "Select second feature",
                            classifier.feature_cols,
                            index=1,
                            key="feat2",
                        )

                        plot_type = st.radio(
                            "Plot Type",
                            ["Scatter", "Hexbin", "Density"],
                            key="interaction_plot_type",
                        )

                        if "classification" in st.session_state["combined_df"].columns:
                            color_by_class = st.checkbox("Color by Class", value=True)

                    with col1:
                        data = st.session_state["combined_df"]

                        if plot_type == "Scatter":
                            if color_by_class and "classification" in data.columns:
                                fig = px.scatter(
                                    data,
                                    x=feature1,
                                    y=feature2,
                                    color="classification",
                                    title=f"Interaction between {feature1} and {feature2}",
                                )
                            else:
                                fig = px.scatter(
                                    data,
                                    x=feature1,
                                    y=feature2,
                                    title=f"Interaction between {feature1} and {feature2}",
                                )

                        elif plot_type == "Hexbin":
                            fig = px.density_heatmap(
                                data,
                                x=feature1,
                                y=feature2,
                                marginal_x="histogram",
                                marginal_y="histogram",
                                title=f"Density Plot of {feature1} vs {feature2}",
                            )

                        else:  # Density
                            if color_by_class and "classification" in data.columns:
                                fig = px.density_contour(
                                    data,
                                    x=feature1,
                                    y=feature2,
                                    color="classification",
                                    marginal_x="histogram",
                                    marginal_y="histogram",
                                    title=f"Density Contour of {feature1} vs {feature2}",
                                )
                            else:
                                fig = px.density_contour(
                                    data,
                                    x=feature1,
                                    y=feature2,
                                    marginal_x="histogram",
                                    marginal_y="histogram",
                                    title=f"Density Contour of {feature1} vs {feature2}",
                                )

                        fig.update_layout(height=600)
                        st.plotly_chart(fig)

                        # Show correlation information
                        correlation = data[feature1].corr(data[feature2])
                        st.info(
                            f"Correlation between {feature1} and {feature2}: {correlation:.3f}"
                        )

        with analysis_tab3:
            st.subheader("Detailed Model Metrics")

            # Add model comparison matrix section
            st.subheader("Model Comparison Matrix")
            comparison_metrics = ["accuracy", "f1_score", "cv_mean", "cv_std"]
            comparison_data = []

            for model_name in model_names:
                model_results = st.session_state["model_results"].get(model_name, {})
                row_data = {
                    "Model": model_name,
                    "Accuracy": model_results.get("accuracy", 0.0),
                    "F1 Score": model_results.get("f1_score", 0.0),
                    "CV Mean": model_results.get("cv_mean", 0.0),
                    "CV Std": model_results.get("cv_std", 0.0),
                }
                comparison_data.append(row_data)

            comparison_df = pd.DataFrame(comparison_data).set_index("Model")
            comparison_df.columns = ["Accuracy", "F1 Score", "CV Mean", "CV Std"]

            # Create heatmap for model comparison
            fig = go.Figure(
                data=go.Heatmap(
                    z=comparison_df.values,
                    x=comparison_df.columns,
                    y=comparison_df.index,
                    text=np.round(comparison_df.values, 3),
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorscale="RdYlBu",
                    colorbar=dict(title="Score"),
                )
            )

            fig.update_layout(
                title="Model Performance Comparison Matrix",
                xaxis_title="Metrics",
                yaxis_title="Models",
                width=800,
                height=400,
            )
            st.plotly_chart(fig)

    elif page == "Challenge Info":
        st.header("🚀 Exoplanet Classification Suite")
        st.write("An interactive platform for analyzing and classifying exoplanet data")

        st.subheader("NASA Space Apps Challenge 2025")
        st.write("A World Away: Hunting for Exoplanets with AI")

        st.markdown(
            """
        ### Challenge Overview
        This application addresses the 2025 NASA Space Apps Challenge for automated exoplanet detection using AI/ML.

        ### Challenge Objectives ✓
        - ✅ **AI/ML Model**: Trained on NASA's open-source exoplanet datasets
        - ✅ **Multi-Mission Data**: Supports Kepler, K2, and TESS datasets
        - ✅ **Web Interface**: Complete user interaction platform
        - ✅ **Accurate Classification**: Identifies confirmed planets, candidates, and false positives
        
        ### Key Features Implemented
        **Data Integration**
        - NASA dataset compatibility (Kepler, K2, TESS)
        - Real-time data preprocessing and cleaning
        - Multiple mission data merging capabilities
        
        **AI/ML Capabilities**
        - Multiple algorithm support (Random Forest, SVM, Neural Networks, etc.)
        - Cross-validation and performance metrics
        - Feature importance analysis
        - Hyperparameter optimization
        
        **User Interface**
        - Interactive web application
        - Single and batch prediction capabilities
        - Real-time model performance monitoring
        - Model save/load functionality
        - Data visualization and analysis tools
        
        **Research-Oriented Features**
        - Model comparison and benchmarking
        - Feature correlation analysis
        - Detailed classification reports
        - Export capabilities for further research
        
        ### NASA Space Apps Challenge Alignment
        """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **Required Features**
            - ✅ AI/ML model trained on NASA datasets
            - ✅ Web interface for user interaction
            - ✅ Analysis of Kepler, K2, and TESS data
            - ✅ Classification into confirmed/candidate/false positive
            """
            )
        with col2:
            st.markdown(
                """
            **Optional Enhancements**
            - ✅ Interface for researchers and novices
            - ✅ Model accuracy statistics display
            - ✅ Hyperparameter tweaking interface
            - ✅ Data upload and manual entry capabilities
            - ✅ Model training with user-provided data
            """
            )

        st.markdown(
            """
        ### Technical Specifications
        **Machine Learning Models:**
        - Random Forest, Gradient Boosting, SVM
        - Neural Networks (MLP), k-NN, Naive Bayes
        - XGBoost and LightGBM (when available)
        
        **Data Processing:**
        - Automated feature extraction and cleaning
        - Missing value imputation strategies
        - Cross-mission data standardization
        
        **Evaluation Metrics:**
        - Accuracy, F1-Score, Precision, Recall
        - Cross-validation with confidence intervals
        - ROC curves and confusion matrices
        
        **Deployment Ready:**
        - Streamlit web application
        - Model persistence and loading
        - Scalable architecture for real datasets
        """
        )

        st.success(
            "This application fully addresses the NASA Space Apps Challenge requirements!"
        )

    elif page == "Hyperparameter Tuning":
        st.header("Real-time Hyperparameter Tuning")
        st.write("Adjust model parameters and see immediate impacts on performance")

        if "X_data" not in st.session_state or "y_data" not in st.session_state:
            st.warning(
                "Training data not available. Please go to the 'Model Training' page and run training at least once to prepare the data."
            )
            return

        if not st.session_state.get("available_models"):
            st.warning(
                "No models available for tuning. Please train or load models first."
            )
            return

        # Model selection for tuning
        tuning_model = st.selectbox(
            "Select Model for Hyperparameter Tuning", st.session_state.available_models
        )

        st.subheader("Current Model Performance")
        if (
            "model_results" in st.session_state
            and tuning_model in st.session_state["model_results"]
        ):
            results = st.session_state["model_results"][tuning_model]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Accuracy", f"{results['accuracy']:.3f}")
            with col2:
                st.metric("Current F1 Score", f"{results['f1_score']:.3f}")
            with col3:
                st.metric("Current CV Score", f"{results['cv_mean']:.3f}")

        st.subheader("Hyperparameter Controls")

        model_obj = classifier.available_models[tuning_model]

        # Dynamic hyperparameter interface based on model type
        if tuning_model == "random_forest":
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider(
                    "Number of Estimators",
                    10,
                    500,
                    getattr(model_obj, "n_estimators", 100),
                )
                max_depth = st.slider(
                    "Max Depth", 1, 30, getattr(model_obj, "max_depth", 10) or 10
                )
            with col2:
                min_samples_split = st.slider(
                    "Min Samples Split",
                    2,
                    20,
                    getattr(model_obj, "min_samples_split", 2),
                )
                min_samples_leaf = st.slider(
                    "Min Samples Leaf", 1, 10, getattr(model_obj, "min_samples_leaf", 1)
                )

            new_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }

        elif tuning_model == "gradient_boosting":
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider(
                    "Number of Estimators",
                    50,
                    300,
                    getattr(model_obj, "n_estimators", 100),
                )
                learning_rate = st.slider(
                    "Learning Rate", 0.01, 0.3, getattr(model_obj, "learning_rate", 0.1)
                )
            with col2:
                max_depth = st.slider(
                    "Max Depth", 1, 10, getattr(model_obj, "max_depth", 3)
                )
                subsample = st.slider("Subsample", 0.5, 1.0, 1.0)

            new_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
            }

        elif tuning_model == "svm":
            col1, col2 = st.columns(2)
            with col1:
                C = st.selectbox("C (Regularization)", [0.1, 1, 10, 100], index=1)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
            with col2:
                gamma = st.selectbox(
                    "Gamma", ["scale", "auto", 0.001, 0.01, 0.1], index=0
                )

            new_params = {"C": C, "kernel": kernel, "gamma": gamma}

        else:
            st.info(
                f"Hyperparameter tuning interface not implemented for {tuning_model}"
            )
            new_params = {}

        # Apply hyperparameters and retrain
        if new_params and st.button(
            "Apply Hyperparameters and Retrain", type="primary"
        ):
            try:
                # Get old results
                if tuning_model in st.session_state["model_results"]:
                    old_results = st.session_state["model_results"][tuning_model]
                    old_cv_score = old_results.get("cv_mean", 0.0)
                else:
                    old_cv_score = 0.0
                    old_results = None

                with st.spinner("Retraining model with new hyperparameters..."):
                    # Update model with new parameters
                    classifier.available_models[tuning_model].set_params(**new_params)

                    # Retrain the specific model
                    X = st.session_state["X_data"]
                    y = st.session_state["y_data"]

                    results, _ = classifier.train_multiple_models(X, y, [tuning_model])

                # Check if training was successful
                if "error" in results[tuning_model]:
                    st.error(
                        f"Error retraining model: {results[tuning_model]['error']}"
                    )
                    return

                new_results = results[tuning_model]
                new_cv_score = new_results["cv_mean"]

                st.subheader("Updated Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("New Accuracy", f"{new_results['accuracy']:.3f}")
                with col2:
                    st.metric("New F1 Score", f"{new_results['f1_score']:.3f}")
                with col3:
                    st.metric("New CV Score", f"{new_cv_score:.3f}")

                # Compare and decide whether to save
                if new_cv_score > old_cv_score:
                    st.success(
                        f"Model performance improved! CV score from {old_cv_score:.3f} to {new_cv_score:.3f}. Model saved."
                    )
                    # Update session state with the new results
                    serializable_results = {
                        k: v
                        for k, v in new_results.items()
                        if k not in ["model", "scaler"]
                    }
                    st.session_state["model_results"][
                        tuning_model
                    ] = serializable_results

                    classifier.save_models([tuning_model])
                else:
                    st.info(
                        f"Model performance did not improve. CV score of {new_cv_score:.3f} is not better than {old_cv_score:.3f}. Model not saved."
                    )
                    # Revert model in memory to the previously saved state
                    classifier.load_model(tuning_model)
                    if old_results:
                        classifier.model_performance[tuning_model] = old_results

            except Exception as e:
                st.error(f"Error retraining model: {str(e)}")

    elif page == "Chatbot Assistant":
        # Embed the chatbot UI from streamlit_chatbot_app.py within this app
        st.header("💬 Chatbot Assistant")
        try:
            chatbot_main()
            chatbot_footer()
        except Exception as e:
            st.error(f"Error loading chatbot: {str(e)}")

if __name__ == "__main__":
    create_streamlit_app()
