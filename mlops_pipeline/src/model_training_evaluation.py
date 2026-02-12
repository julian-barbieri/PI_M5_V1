from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from ft_engineering import ft_engineering_procesado
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
import warnings
from sklearn.exceptions import ConvergenceWarning
import optuna
import time
import numpy as np
import joblib
import os

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def save_model(model, model_name, path="models"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    print(f"✅ Modelo guardado en: {file_path}")


def build_models():
    models = [
        ("RandomForestClassifier", RandomForestClassifier(
            n_estimators=100,          # Cantidad de árboles en el bosque
            max_depth=4,               # Profundidad máxima de cada árbol (controla overfitting)
            class_weight="balanced",   # Compensa el desbalance de clases (churn)
            random_state=42,           # Reproducibilidad de los resultados
            n_jobs=-1                  # Usa todos los núcleos del CPU
        )),
        
        ('XGBClassifier', XGBClassifier(
            n_estimators=100,          # Cantidad de árboles (boosting rounds)
            max_depth=4,               # Profundidad de cada árbol
            learning_rate=0.1,         # Tasa de aprendizaje (controla overfitting)
            eval_metric="logloss",     # Métrica de evaluación para clasificación binaria
            random_state=42,           # Reproducibilidad
            n_jobs=-1                  # Paralelización
        )),
        
        ('CatBoostClassifier', CatBoostClassifier(
            iterations=100,            # Cantidad de árboles (boosting rounds)
            learning_rate=0.1,         # Tasa de aprendizaje
            depth=4,                   # Profundidad de los árboles
            loss_function="Logloss",   # Función de pérdida para clasificación binaria
            random_seed=42,
            verbose=False,
            allow_writing_files=False 
        )),
    ]
    return models

def summarize_classification(y_true, y_pred):
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    return metrics

def tune_best_model_with_optuna(best_name, X_train, y_train, cv_folds, optimize_metric="recall_0", n_trials=50):

    # scorers (incluye clase 0)
    scorers = {
        "recall_0": make_scorer(recall_score, pos_label=0),
        "f1_0": make_scorer(f1_score, pos_label=0),
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }

    scoring = scorers[optimize_metric]

    def objective(trial):
        # -------------------------
        # RandomForest
        # -------------------------
        if best_name == "RandomForestClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        # -------------------------
        # XGBoost
        # -------------------------
        elif best_name == "XGBClassifier":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,  # ✅ para que no imprima líneas
            }
            model = XGBClassifier(**params)

        # -------------------------
        # CatBoost
        # -------------------------
        elif best_name == "CatBoostClassifier":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 2, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "loss_function": "Logloss",
                "random_seed": 42,
                "verbose": False,  # ✅ para que no imprima líneas
            }
            model = CatBoostClassifier(**params)

        else:
            raise ValueError(f"No tengo espacio de búsqueda para: {best_name}")

        scores = cross_val_score(
            model,
            X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value 

def train_and_select_model(X_train, y_train, X_test, y_test):

    models = build_models()
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # scorers estándar + scorers para clase 0
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",     # para clase 1 por defecto
        "recall": "recall",           # para clase 1 por defecto
        "f1": "f1",                   # para clase 1 por defecto
        "roc_auc": "roc_auc",

        # Clase 0
        "recall_0": make_scorer(recall_score, pos_label=0),
        "f1_0": make_scorer(f1_score, pos_label=0),
    }

    results = []

    for name, model in models:
        print(f"\nEvaluando {name}...")
        start_time = time.time()

        cv = cross_validate(
            model,
            X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        elapsed_time = time.time() - start_time

        result = {
            "name": name,
            "model": model,
            "cv_recall0_mean": cv["test_recall_0"].mean(),
            "cv_recall0_std": cv["test_recall_0"].std(),
            "cv_f10_mean": cv["test_f1_0"].mean(),
            "cv_f10_std": cv["test_f1_0"].std(),
            "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv["test_roc_auc"].std(),
            "time": elapsed_time
        }
        results.append(result)

        print(f"Recall clase 0: {result['cv_recall0_mean']:.4f} (±{result['cv_recall0_std']:.4f})")
        print(f"F1 clase 0:     {result['cv_f10_mean']:.4f} (±{result['cv_f10_std']:.4f})")
        print(f"ROC AUC:        {result['cv_roc_auc_mean']:.4f} (±{result['cv_roc_auc_std']:.4f})")
        print(f"Tiempo: {elapsed_time:.2f}s")

    # elegir mejor por recall_0 (robusto: mean - std)
    best = max(results, key=lambda x: x["cv_recall0_mean"] - x["cv_recall0_std"])
    print(f"\n Mejor modelo base: {best['name']}")


    # Aplicamos Optuna SOLO para el mejor
    # ------------------------------
    best_params, best_cv_score = tune_best_model_with_optuna(
        best_name=best["name"],
        X_train=X_train,
        y_train=y_train,
        cv_folds=cv_folds,
        optimize_metric="recall_0", 
        n_trials=50
    )
    
    print(f"\nOptuna mejor CV ({'recall_0'}): {best_cv_score:.4f}")
    print("Mejores hiperparámetros:", best_params)

    # construir el modelo final con esos parámetros
    if best["name"] == "RandomForestClassifier":
        tuned_model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42, n_jobs=-1)

    elif best["name"] == "XGBClassifier":
        tuned_model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)

    elif best["name"] == "CatBoostClassifier":
        tuned_model = CatBoostClassifier(**best_params, loss_function="Logloss", random_seed=42, verbose=False, allow_writing_files=False)

    tuned_model.fit(X_train, y_train)

    # Guardar orden exacto de columnas con el que se entrenó
    feature_names = list(X_train.columns)

    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_names, "models/feature_names.pkl")
    
    #guardamos el modelo como archivo .pkl
    save_model(model=tuned_model, model_name=f"{best['name']}_optuna")

    best["tuned_model"] = tuned_model
    best["optuna_best_params"] = best_params
    best["optuna_best_cv_score"] = best_cv_score

    
    
    return best, results

# Ejecutar (usa X_train_df, X_test_df, y_train, y_test preprocesados)

X_train_df, X_test_df, y_train, y_test = ft_engineering_procesado()

best_model, all_results = train_and_select_model(
    X_train_df,
    y_train,
    X_test_df,
    y_test
)






