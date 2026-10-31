import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    PolynomialFeatures,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    plot_tree
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
DARK_THEME = {
    "bg": "#0D0D0D",
    "card": "#171717",
    "input": "#111111",
    "primary": "#C9A227",
    "primary_hover": "#E0B83F",
    "primary_light": "#F0CF68",
    "error": "#E05252",
    "text": "#F5F5F5",
    "secondary": "#A3A3A3",
    "muted": "#737373",
    "border": "#2A2A2A",
    "row": "#1D1D1D",
    "button": "#292929",
    "button_hover": "#3A3A3A"
}
ACCENT_PALETTE = {
    "gold": ("#C9A227", "#E0B83F", "#F0CF68"),
    "purple": ("#8B5CF6", "#A78BFA", "#C4B5FD"),
    "pink": ("#EC4899", "#F472B6", "#F9A8D4"),
    "red": ("#EF4444", "#F87171", "#FCA5A5"),
    "orange": ("#F97316", "#FB923C", "#FDBA74"),
    "cyan": ("#06B6D4", "#22D3EE", "#67E8F9"),
    "indigo": ("#6366F1", "#818CF8", "#A5B4FC"),
    "white": ("#F5F5F5", "#D4D4D4", "#FFFFFF"),
    "bronze": ("#CD7F32", "#E6A04A", "#F2C078"),
    "crimson": ("#DC143C", "#F04462", "#FF8A9E"),
    "plum": ("#7E3F7C", "#A35BA0", "#C98BC6"),
    "silver": ("#A3A3A3", "#C4C4C4", "#E5E5E5")
}

LIGHT_THEME = {
    "bg": "#F4F4F4",
    "card": "#FFFFFF",
    "input": "#EEEEEE",
    "primary": "#B28A00",
    "primary_hover": "#D2A800",
    "primary_light": "#9A7600",
    "error": "#C62828",
    "text": "#171717",
    "secondary": "#5F5F5F",
    "muted": "#888888",
    "border": "#D6D6D6",
    "row": "#F0F0F0",
    "button": "#E4E4E4",
    "button_hover": "#D5D5D5"
}
BG_COLOR = DARK_THEME["bg"]
CARD_COLOR = DARK_THEME["card"]
INPUT_COLOR = DARK_THEME["input"]
PRIMARY = DARK_THEME["primary"]
PRIMARY_HOVER = DARK_THEME["primary_hover"]
PRIMARY_LIGHT = DARK_THEME["primary_light"]
ERROR = DARK_THEME["error"]
TEXT_PRIMARY = DARK_THEME["text"]
TEXT_SECONDARY = DARK_THEME["secondary"]
TEXT_MUTED = DARK_THEME["muted"]
BORDER = DARK_THEME["border"]
ROW_COLOR = DARK_THEME["row"]
BUTTON_COLOR = DARK_THEME["button"]
BUTTON_HOVER = DARK_THEME["button_hover"]
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
class Nexora(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Nexora")
        width = 1100
        height = 700
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(
            f"{width}x{height}+{x}+{y}"
        )
        self.minsize(
            700,
            500
        )
        self.current_theme = "dark"
        self.current_language = "en"
        self.font_scale = 1.0
        self.accent_color = "gold"
        self.color_selection_buttons = {}
        self.selected_file = None
        self.data = None
        self.feature_columns = []
        self.target_column = None
        self.selected_algorithm = None
        self.selected_algorithm_type = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.preprocessor = None
        self.target_encoder = None
        self.test_rmse = None
        self.test_mae = None
        self.test_r2 = None
        self.test_accuracy = None
        self.prediction_inputs = {}
        self.last_prediction_values = {}
        self.last_prediction = None
        self.last_prediction_text = None
        self.last_prediction_probability = None
        self.best_polynomial_degree = None
        self.polynomial_cv_results = {}
        self.progress = None
        self.loading_value = 0
        self.current_X = None
        self.auto_selection_results = {}
        self.auto_selection_running = False
        self.auto_selection_progress = None
        self.auto_selection_thread = None
        self.prediction_history = []
        self.current_result_saved = False
        self.current_result_label = None
        self.update_global_theme()
        self.main_container = ctk.CTkFrame(
            self,
            fg_color=BG_COLOR,
            corner_radius=0
        )
        self.main_container.pack(
            fill="both",
            expand=True
        )
        self.show_splash()
    def t(self, key):
        translations = {
            "en": {
                "nexora":
                    "Nexora",
                "subtitle":
                    "Intelligent Machine Learning Studio",
                "created":
                    "Created by Mahdi Raeisi",
                "settings":
                    "Settings",
                "theme":
                    "Theme",
                "language":
                    "Language",
                "dark":
                    "Dark",
                "light":
                    "Light",
                "english":
                    "English",
                "persian":
                    "Persian",
                "apply":
                    "Apply",
                "close":
                    "Close",
                "import_dataset":
                    "Import Dataset",
                "build_train_predict":
                    "Build. Train. Predict.",
                "no_file":
                    "No file selected",
                "import":
                    "Import",
                "history":
                    "History",
                "next":
                    "Next  →",
                "back_nexora":
                    "← Back to Nexora",
                "back_algorithms":
                    "← Back to Algorithms",
                "continue":
                    "Continue →",
                "finish":
                    "Finish",
                "return_start":
                    "↻ Return to Start",
                "save_result":
                    "💾 Save Result",
                "save_result_label":
                    "Save Result",
                "save_result_prompt":
                    "Enter a label for this result:",
                "save_result_empty":
                    "Please enter a label.",
                "save_result_success":
                    "Result saved successfully.",
                "unsaved_result_title":
                    "Unsaved Result",
                "unsaved_result_message":
                    "Are you sure you don't want to save this result?",
                "prediction_history":
                    "Prediction History",
                "previous_predictions":
                    "Your previous machine learning predictions",
                "no_predictions":
                    "No predictions yet",
                "history_message":
                    "Your prediction history will appear here.",
                "dataset_overview":
                    "Dataset Overview",
                "data_distribution":
                    "Data Distribution",
                "numerical_distribution":
                    "Numerical Feature Distribution",
                "no_numeric":
                    "No numerical data available",
                "feature_target":
                    "Feature & Target Selection",
                "feature_description":
                    "Select the columns used by your machine learning model",
                "features":
                    "Features",
                "select_all":
                    "Select All Columns",
                "select_column":
                    "Select a column",
                "selected_features":
                    "Selected Features",
                "target":
                    "Target",
                "select_target":
                    "Select target",
                "selected_target":
                    "Selected Target",
                "no_target":
                    "No target selected",
                "choose_algorithm":
                    "Choose Machine Learning Algorithm",
                "regression":
                    "Regression",
                "classification":
                    "Classification",
                "linear_regression":
                    "Linear Regression",
                "polynomial_regression":
                    "Polynomial Regression",
                "ridge_regression":
                    "Ridge Regression",
                "lasso_regression":
                    "Lasso Regression",
                "elastic_net":
                    "Elastic Net",
                "decision_tree_regressor":
                    "Decision Tree Regressor",
                "random_forest_regressor":
                    "Random Forest Regressor",
                "extra_trees_regressor":
                    "Extra Trees Regressor",
                "gradient_boosting_regressor":
                    "Gradient Boosting Regressor",
                "xgboost_regressor":
                    "XGBoost Regressor",
                "svr":
                    "SVR",
                "knn_regressor":
                    "KNN Regressor",
                "logistic_regression":
                    "Logistic Regression",
                "knn":
                    "K-Nearest Neighbors",
                "naive_bayes":
                    "Naive Bayes",
                "decision_tree":
                    "Decision Tree",
                "random_forest":
                    "Random Forest",
                "extra_trees":
                    "Extra Trees",
                "gradient_boosting":
                    "Gradient Boosting",
                "adaboost":
                    "AdaBoost",
                "svm":
                    "Support Vector Machine",
                "xgboost":
                    "XGBoost",
                "lightgbm":
                    "LightGBM",
                "catboost":
                    "CatBoost",
                "linear_baseline":
                    "Linear baseline",
                "automatic_degree":
                    "Automatic degree selection",
                "l2_regularization":
                    "L2 regularization",
                "l1_regularization":
                    "L1 regularization",
                "l1_l2":
                    "L1 + L2",
                "tree_model":
                    "Tree model",
                "tree_ensemble":
                    "Tree ensemble",
                "randomized_ensemble":
                    "Randomized ensemble",
                "boosting":
                    "Boosting",
                "advanced_boosting":
                    "Advanced boosting",
                "support_vector_regression":
                    "Support Vector Regression",
                "nearest_neighbors":
                    "Nearest neighbors",
                "linear_classifier":
                    "Linear classifier",
                "distance_based":
                    "Distance-based",
                "probabilistic":
                    "Probabilistic",
                "adaptive_boosting":
                    "Adaptive boosting",
                "margin_classifier":
                    "Margin classifier",
                "fast_boosting":
                    "Fast boosting",
                "categorical_boosting":
                    "Categorical boosting",
                "use_this":
                    "Use This",
                "no_algorithm":
                    "No algorithm selected",
                "selected":
                    "Selected",
                "auto_select":
                    "⚡ Auto Select Best Model",
                "analyzing":
                    "Analyzing...",
                "comparing_models":
                    "Comparing models using Cross Validation...",
                "model_visualization":
                    "Model Visualization",
                "automatic_polynomial":
                    "Automatic Polynomial Regression",
                "target_label":
                    "Target:",
                "best_degree":
                    "Best Degree:",
                "actual_predicted":
                    "Actual vs Predicted",
                "automatic_curve":
                    "Automatic Polynomial Curve",
                "decision_tree_structure":
                    "Decision Tree Structure",
                "tree_structure_ensemble":
                    "Tree Structure from Ensemble",
                "confusion_matrix":
                    "Confusion Matrix",
                "actual_vs_predicted":
                    "Actual vs Predicted Values",
                "actual_target":
                    "Actual Target",
                "predicted_target":
                    "Predicted Target",
                "actual_data":
                    "Actual Data",
                "classification_confusion":
                    "Classification Confusion Matrix",
                "prediction_input":
                    "Prediction Input",
                "prediction_description":
                    "Enter values for the selected features",
                "numeric_placeholder":
                    "Enter numeric value",
                "predict_target":
                    "Predict Target",
                "prediction_result":
                    "Prediction Result",
                "prediction_processed":
                    "Your input has been processed by the selected model",
                "prediction":
                    "PREDICTION",
                "model_confidence":
                    "Model confidence:",
                "probability_unavailable":
                    "Model probability is not available",
                "estimated_error":
                    "Estimated model error (RMSE): ±",
                "average_error":
                    "Average absolute error (MAE):",
                "input_features":
                    "Input Features",
                "algorithm":
                    "Algorithm",
                "type":
                    "Type",
                "confidence":
                    "Confidence",
                "rmse":
                    "RMSE",
                "mae":
                    "MAE",
                "r2":
                    "R²",
                "accuracy":
                    "Accuracy",
                "invalid_file":
                    "Invalid file format. Please select an Excel file.",
                "excel_loaded":
                    "✓ Excel file loaded successfully",
                "excel_error":
                    "Could not read Excel file:",
                "empty_excel":
                    "The Excel file is empty.",
                "please_features_target":
                    "Please select features and target first.",
                "not_enough_rows":
                    "The dataset does not contain enough valid rows.",
                "two_classes":
                    "The target must contain at least two classes.",
                "at_least_three":
                    "At least 3 samples are required.",
                "at_least_two_each_class":
                    "Each class must contain at least two samples.",
                "no_model":
                    "No suitable model could be evaluated.",
                "model_error":
                    "Model Error",
                "prediction_error":
                    "Prediction Error",
                "auto_error":
                    "Auto Selection Error",
                "auto_selection_title":
                    "Auto Model Selection",
                "best_model":
                    "Best Model:",
                "task":
                    "Task:",
                "cv_rmse":
                    "CV RMSE:",
                "cv_accuracy":
                    "CV Accuracy:",
                "model_ranking":
                    "Model Ranking:",
                "features_label":
                    "Features:",
                "enter_value":
                    "Please enter a value for",
                "estimated_model_error":
                    "Estimated model error (RMSE): ±",
                "automatic_degree_selected":
                    "Automatically Selected Polynomial Degree =",
                "degree_label":
                    "Degree",
                "delete":
                    "Delete",
                "delete_selected":
                    "Delete Selected",
                "delete_all":
                    "Delete All",
                "select_all_history":
                    "Select All",
                "deselect_all_history":
                    "Deselect All",
                "delete_history_title":
                    "Delete History",
                "delete_history_message":
                    "Are you sure you want to delete the selected history items?",
                "no_history_selected":
                    "Please select at least one history item.",
                "history_deleted":
                    "History deleted successfully.",
                "font_size":
                    "Font Size",
                "font_size_hint":
                    "Safe range: 85% to 115%",
                "accent_color":
                    "Accent Color",
                "color_gold":
                    "Gold",
                "color_purple":
                    "Purple",
                "color_pink":
                    "Pink",
                "color_red":
                    "Red",
                "color_orange":
                    "Orange",
                "color_cyan":
                    "Cyan",
                "color_indigo":
                    "Indigo",
                "color_white":
                    "White",
                "color_bronze":
                    "Bronze",
                "color_crimson":
                    "Crimson",
                "color_plum":
                    "Plum",
                "color_silver":
                    "Silver"
            },
            "fa": {
                "nexora":
                    "نکسورا",
                "subtitle":
                    "استودیو هوشمند یادگیری ماشین",
                "created":
                    "ساخته شده توسط مهدی رئیسی",
                "settings":
                    "تنظیمات",
                "theme":
                    "پوسته",
                "language":
                    "زبان",
                "dark":
                    "تیره",
                "light":
                    "روشن",
                "english":
                    "انگلیسی",
                "persian":
                    "فارسی",
                "apply":
                    "اعمال",
                "close":
                    "بستن",
                "import_dataset":
                    "وارد کردن دیتاست",
                "build_train_predict":
                    "ساخت. آموزش. پیش‌بینی.",
                "no_file":
                    "فایلی انتخاب نشده",
                "import":
                    "وارد کردن",
                "history":
                    "تاریخچه",
                "next":
                    "بعدی  →",
                "back_nexora":
                    "← بازگشت به نکسورا",
                "back_algorithms":
                    "← بازگشت به الگوریتم‌ها",
                "continue":
                    "ادامه →",
                "finish":
                    "پایان",
                "return_start":
                    "↻ بازگشت به ابتدا",
                "save_result":
                    "💾 ذخیره نتیجه",
                "save_result_label":
                    "ذخیره نتیجه",
                "save_result_prompt":
                    "یک نام برای این نتیجه وارد کنید:",
                "save_result_empty":
                    "لطفاً یک نام وارد کنید.",
                "save_result_success":
                    "نتیجه با موفقیت ذخیره شد.",
                "unsaved_result_title":
                    "نتیجه ذخیره‌نشده",
                "unsaved_result_message":
                    "آیا مطمئن هستید که نمی‌خواهید این نتیجه را ذخیره کنید؟",
                "prediction_history":
                    "تاریخچه پیش‌بینی",
                "previous_predictions":
                    "پیش‌بینی‌های قبلی یادگیری ماشین شما",
                "no_predictions":
                    "هنوز پیش‌بینی‌ای انجام نشده",
                "history_message":
                    "تاریخچه پیش‌بینی‌های شما در اینجا نمایش داده می‌شود.",
                "dataset_overview":
                    "نمای کلی دیتاست",
                "data_distribution":
                    "توزیع داده",
                "numerical_distribution":
                    "توزیع ویژگی‌های عددی",
                "no_numeric":
                    "داده عددی وجود ندارد",
                "feature_target":
                    "انتخاب ویژگی و هدف",
                "feature_description":
                    "ستون‌های مورد استفاده در مدل یادگیری ماشین را انتخاب کنید",
                "features":
                    "ویژگی‌ها",
                "select_all":
                    "انتخاب همه ستون‌ها",
                "select_column":
                    "انتخاب یک ستون",
                "selected_features":
                    "ویژگی‌های انتخاب‌شده",
                "target":
                    "هدف",
                "select_target":
                    "انتخاب هدف",
                "selected_target":
                    "هدف انتخاب‌شده",
                "no_target":
                    "هدفی انتخاب نشده",
                "choose_algorithm":
                    "انتخاب الگوریتم یادگیری ماشین",
                "regression":
                    "رگرسیون",
                "classification":
                    "دسته‌بندی",
                "linear_regression":
                    "رگرسیون خطی",
                "polynomial_regression":
                    "رگرسیون چندجمله‌ای",
                "ridge_regression":
                    "رگرسیون Ridge",
                "lasso_regression":
                    "رگرسیون Lasso",
                "elastic_net":
                    "Elastic Net",
                "decision_tree_regressor":
                    "درخت تصمیم رگرسیونی",
                "random_forest_regressor":
                    "جنگل تصادفی رگرسیونی",
                "extra_trees_regressor":
                    "Extra Trees رگرسیونی",
                "gradient_boosting_regressor":
                    "Gradient Boosting رگرسیونی",
                "xgboost_regressor":
                    "XGBoost رگرسیونی",
                "svr":
                    "SVR",
                "knn_regressor":
                    "KNN رگرسیونی",
                "logistic_regression":
                    "رگرسیون لجستیک",
                "knn":
                    "K-Nearest Neighbors",
                "naive_bayes":
                    "Naive Bayes",
                "decision_tree":
                    "درخت تصمیم",
                "random_forest":
                    "جنگل تصادفی",
                "extra_trees":
                    "Extra Trees",
                "gradient_boosting":
                    "Gradient Boosting",
                "adaboost":
                    "AdaBoost",
                "svm":
                    "ماشین بردار پشتیبان",
                "xgboost":
                    "XGBoost",
                "lightgbm":
                    "LightGBM",
                "catboost":
                    "CatBoost",
                "linear_baseline":
                    "مدل پایه خطی",
                "automatic_degree":
                    "انتخاب خودکار درجه",
                "l2_regularization":
                    "منظم‌سازی L2",
                "l1_regularization":
                    "منظم‌سازی L1",
                "l1_l2":
                    "L1 + L2",
                "tree_model":
                    "مدل درختی",
                "tree_ensemble":
                    "مجموعه درختی",
                "randomized_ensemble":
                    "مجموعه تصادفی",
                "boosting":
                    "Boosting",
                "advanced_boosting":
                    "Boosting پیشرفته",
                "support_vector_regression":
                    "رگرسیون بردار پشتیبان",
                "nearest_neighbors":
                    "نزدیک‌ترین همسایه‌ها",
                "linear_classifier":
                    "دسته‌بندی‌کننده خطی",
                "distance_based":
                    "مبتنی بر فاصله",
                "probabilistic":
                    "احتمالاتی",
                "adaptive_boosting":
                    "Boosting تطبیقی",
                "margin_classifier":
                    "دسته‌بندی‌کننده حاشیه‌ای",
                "fast_boosting":
                    "Boosting سریع",
                "categorical_boosting":
                    "Boosting دسته‌ای",
                "use_this":
                    "استفاده",
                "no_algorithm":
                    "هیچ الگوریتمی انتخاب نشده",
                "selected":
                    "انتخاب شد",
                "auto_select":
                    "⚡ انتخاب خودکار بهترین مدل",
                "analyzing":
                    "در حال تحلیل...",
                "comparing_models":
                    "در حال مقایسه مدل‌ها با Cross Validation...",
                "model_visualization":
                    "نمایش مدل",
                "automatic_polynomial":
                    "رگرسیون چندجمله‌ای خودکار",
                "target_label":
                    "هدف:",
                "best_degree":
                    "بهترین درجه:",
                "actual_predicted":
                    "واقعی در برابر پیش‌بینی‌شده",
                "automatic_curve":
                    "منحنی چندجمله‌ای خودکار",
                "decision_tree_structure":
                    "ساختار درخت تصمیم",
                "tree_structure_ensemble":
                    "ساختار یک درخت از مجموعه",
                "confusion_matrix":
                    "ماتریس اغتشاش",
                "actual_vs_predicted":
                    "مقادیر واقعی و پیش‌بینی‌شده",
                "actual_target":
                    "مقدار واقعی هدف",
                "predicted_target":
                    "مقدار پیش‌بینی‌شده",
                "actual_data":
                    "داده واقعی",
                "classification_confusion":
                    "ماتریس اغتشاش دسته‌بندی",
                "prediction_input":
                    "ورودی پیش‌بینی",
                "prediction_description":
                    "مقادیر ویژگی‌های انتخاب‌شده را وارد کنید",
                "numeric_placeholder":
                    "مقدار عددی را وارد کنید",
                "predict_target":
                    "پیش‌بینی هدف",
                "prediction_result":
                    "نتیجه پیش‌بینی",
                "prediction_processed":
                    "ورودی شما توسط مدل انتخاب‌شده پردازش شد",
                "prediction":
                    "پیش‌بینی",
                "model_confidence":
                    "اطمینان مدل:",
                "probability_unavailable":
                    "احتمال مدل در دسترس نیست",
                "estimated_error":
                    "خطای تخمینی مدل (RMSE): ±",
                "average_error":
                    "میانگین خطای مطلق (MAE):",
                "input_features":
                    "ویژگی‌های ورودی",
                "algorithm":
                    "الگوریتم",
                "type":
                    "نوع",
                "confidence":
                    "اطمینان",
                "rmse":
                    "RMSE",
                "mae":
                    "MAE",
                "r2":
                    "R²",
                "accuracy":
                    "دقت",
                "invalid_file":
                    "فرمت فایل نامعتبر است. یک فایل Excel انتخاب کنید.",
                "excel_loaded":
                    "✓ فایل Excel با موفقیت بارگذاری شد",
                "excel_error":
                    "خواندن فایل Excel امکان‌پذیر نبود:",
                "empty_excel":
                    "فایل Excel خالی است.",
                "please_features_target":
                    "ابتدا ویژگی‌ها و هدف را انتخاب کنید.",
                "not_enough_rows":
                    "دیتاست دارای تعداد کافی ردیف معتبر نیست.",
                "two_classes":
                    "هدف باید حداقل دو کلاس داشته باشد.",
                "at_least_three":
                    "حداقل ۳ نمونه لازم است.",
                "at_least_two_each_class":
                    "هر کلاس باید حداقل دو نمونه داشته باشد.",
                "no_model":
                    "هیچ مدل مناسبی قابل ارزیابی نبود.",
                "model_error":
                    "خطای مدل",
                "prediction_error":
                    "خطای پیش‌بینی",
                "auto_error":
                    "خطای انتخاب خودکار",
                "auto_selection_title":
                    "انتخاب خودکار مدل",
                "best_model":
                    "بهترین مدل:",
                "task":
                    "نوع مسئله:",
                "cv_rmse":
                    "RMSE اعتبارسنجی:",
                "cv_accuracy":
                    "دقت اعتبارسنجی:",
                "model_ranking":
                    "رتبه‌بندی مدل‌ها:",
                "features_label":
                    "ویژگی‌ها:",
                "enter_value":
                    "لطفاً مقداری برای ستون وارد کنید:",
                "estimated_model_error":
                    "خطای تخمینی مدل (RMSE): ±",
                "automatic_degree_selected":
                    "درجه چندجمله‌ای انتخاب‌شده:",
                "degree_label":
                    "درجه",
                "delete":
                    "حذف",
                "delete_selected":
                    "حذف انتخاب‌شده‌ها",
                "delete_all":
                    "حذف همه",
                "select_all_history":
                    "انتخاب همه",
                "deselect_all_history":
                    "لغو انتخاب همه",
                "delete_history_title":
                    "حذف تاریخچه",
                "delete_history_message":
                    "آیا مطمئن هستید که می‌خواهید موارد انتخاب‌شده را حذف کنید؟",
                "no_history_selected":
                    "حداقل یک مورد از تاریخچه را انتخاب کنید.",
                "history_deleted":
                    "تاریخچه با موفقیت حذف شد.",
                "font_size":
                    "اندازه فونت",
                "font_size_hint":
                    "محدوده امن: ۸۵٪ تا ۱۱۵٪",
                "accent_color":
                    "رنگ اصلی",
                "color_gold":
                    "طلایی",
                "color_purple":
                    "بنفش",
                "color_pink":
                    "صورتی",
                "color_red":
                    "قرمز",
                "color_orange":
                    "نارنجی",
                "color_cyan":
                    "فیروزه‌ای",
                "color_indigo":
                    "نیلی",
                "color_white":
                    "سفید",
                "color_bronze":
                    "برنزی",
                "color_crimson":
                    "زرشکی",
                "color_plum":
                    "آلویی",
                "color_silver":
                    "نقره‌ای"
            }
        }
        return translations[
            self.current_language
        ].get(
            key,
            key
        )
    def title_font(self, size):
        base = size + (3 if self.current_language == "fa" else 0)
        return ctk.CTkFont(
            family="Segoe UI",
            size=max(8, int(round(base * self.font_scale))),
            weight="bold"
        )
    def body_font(self, size):
        base = size + (2 if self.current_language == "fa" else 0)
        return ctk.CTkFont(
            family="Segoe UI",
            size=max(8, int(round(base * self.font_scale)))
        )
    def small_font(self, size):
        base = size + (1 if self.current_language == "fa" else 0)
        return ctk.CTkFont(
            family="Segoe UI",
            size=max(7, int(round(base * self.font_scale)))
        )
    def update_global_theme(self):
        global BG_COLOR
        global CARD_COLOR
        global INPUT_COLOR
        global PRIMARY
        global PRIMARY_HOVER
        global PRIMARY_LIGHT
        global ERROR
        global TEXT_PRIMARY
        global TEXT_SECONDARY
        global TEXT_MUTED
        global BORDER
        global ROW_COLOR
        global BUTTON_COLOR
        global BUTTON_HOVER
        theme = (
            DARK_THEME
            if self.current_theme == "dark"
            else LIGHT_THEME
        )
        BG_COLOR = theme["bg"]
        CARD_COLOR = theme["card"]
        INPUT_COLOR = theme["input"]
        PRIMARY, PRIMARY_HOVER, PRIMARY_LIGHT = ACCENT_PALETTE.get(self.accent_color, ACCENT_PALETTE["gold"])
        ERROR = theme["error"]
        TEXT_PRIMARY = theme["text"]
        TEXT_SECONDARY = theme["secondary"]
        TEXT_MUTED = theme["muted"]
        BORDER = theme["border"]
        ROW_COLOR = theme["row"]
        BUTTON_COLOR = theme["button"]
        BUTTON_HOVER = theme["button_hover"]
        ctk.set_appearance_mode(
            "dark"
            if self.current_theme == "dark"
            else "light"
        )
        self.configure(
            fg_color=BG_COLOR
        )
    def open_settings(self):
        if getattr(self, "settings_panel", None) is not None:
            self.close_settings()
            return

        self.settings_opening = True

        self.settings_panel = ctk.CTkFrame(
            self.main_container,
            width=350,
            height=0,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        self.settings_panel.place(
            relx=0.96,
            rely=0.105,
            anchor="ne"
        )
        self.settings_panel.pack_propagate(False)

        ctk.CTkLabel(
            self.settings_panel,
            text="⚙  " + self.t("settings"),
            font=self.title_font(20),
            text_color=PRIMARY
        ).pack(
            pady=(18, 8)
        )

        content_frame = ctk.CTkScrollableFrame(
            self.settings_panel,
            fg_color="transparent",
            scrollbar_button_color=BUTTON_COLOR,
            scrollbar_button_hover_color=BUTTON_HOVER,
            width=320
        )
        content_frame.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=(0, 4)
        )

        ctk.CTkLabel(
            content_frame,
            text=self.t("theme"),
            font=self.body_font(13),
            text_color=TEXT_PRIMARY
        ).pack(
            anchor="w",
            padx=18
        )

        theme_values = [
            self.t("dark"),
            self.t("light")
        ]
        theme_var = ctk.StringVar(
            value=(
                self.t("dark")
                if self.current_theme == "dark"
                else self.t("light")
            )
        )

        ctk.CTkComboBox(
            content_frame,
            values=theme_values,
            variable=theme_var,
            width=290,
            height=38,
            fg_color=INPUT_COLOR,
            border_color=BORDER,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color=BUTTON_HOVER,
            font=self.body_font(12),
            dropdown_font=self.body_font(12)
        ).pack(
            pady=(7, 10),
            padx=4
        )

        ctk.CTkLabel(
            content_frame,
            text=self.t("language"),
            font=self.body_font(13),
            text_color=TEXT_PRIMARY
        ).pack(
            anchor="w",
            padx=18
        )

        language_values = [
            self.t("english"),
            self.t("persian")
        ]
        language_var = ctk.StringVar(
            value=(
                self.t("english")
                if self.current_language == "en"
                else self.t("persian")
            )
        )

        ctk.CTkComboBox(
            content_frame,
            values=language_values,
            variable=language_var,
            width=290,
            height=38,
            fg_color=INPUT_COLOR,
            border_color=BORDER,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color=BUTTON_HOVER,
            font=self.body_font(12),
            dropdown_font=self.body_font(12)
        ).pack(
            pady=(7, 10),
            padx=4
        )

        ctk.CTkLabel(
            content_frame,
            text=self.t("font_size"),
            font=self.body_font(13),
            text_color=TEXT_PRIMARY
        ).pack(
            anchor="w",
            padx=18
        )

        font_var = ctk.StringVar(
            value=f"{int(round(self.font_scale * 100))}%"
        )
        font_values = [
            "85%",
            "90%",
            "95%",
            "100%",
            "105%",
            "110%",
            "115%"
        ]

        ctk.CTkComboBox(
            content_frame,
            values=font_values,
            variable=font_var,
            width=290,
            height=38,
            fg_color=INPUT_COLOR,
            border_color=BORDER,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color=BUTTON_HOVER,
            font=self.body_font(12),
            dropdown_font=self.body_font(12)
        ).pack(
            pady=(7, 2),
            padx=4
        )

        ctk.CTkLabel(
            content_frame,
            text=self.t("font_size_hint"),
            font=self.small_font(9),
            text_color=TEXT_MUTED
        ).pack(
            pady=(0, 12)
        )

        ctk.CTkLabel(
            content_frame,
            text=self.t("accent_color"),
            font=self.body_font(13),
            text_color=TEXT_PRIMARY
        ).pack(
            anchor="w",
            padx=18
        )

        accent_var = ctk.StringVar(value=self.accent_color)

        color_frame = ctk.CTkFrame(
            content_frame,
            fg_color="transparent"
        )
        color_frame.pack(
            pady=(8, 8),
            padx=8
        )

        color_names = [
            ("gold", self.t("color_gold")),
            ("purple", self.t("color_purple")),
            ("pink", self.t("color_pink")),
            ("red", self.t("color_red")),
            ("orange", self.t("color_orange")),
            ("cyan", self.t("color_cyan")),
            ("indigo", self.t("color_indigo")),
            ("white", self.t("color_white")),
            ("bronze", self.t("color_bronze")),
            ("crimson", self.t("color_crimson")),
            ("plum", self.t("color_plum")),
            ("silver", self.t("color_silver"))
        ]

        self.color_selection_buttons = {}

        for index, (color_name, color_label) in enumerate(color_names):
            color_value, hover_value, light_value = ACCENT_PALETTE[color_name]

            ring = ctk.CTkFrame(
                color_frame,
                width=54,
                height=54,
                corner_radius=27,
                fg_color="transparent",
                border_width=3 if color_name == self.accent_color else 0,
                border_color=color_value
            )
            ring.grid(
                row=index // 4,
                column=index % 4,
                padx=5,
                pady=5
            )
            ring.grid_propagate(False)

            button = ctk.CTkButton(
                ring,
                text="",
                width=40,
                height=40,
                corner_radius=20,
                fg_color=color_value,
                hover_color=hover_value,
                border_width=0,
                command=lambda value=color_name: select_accent(value)
            )
            button.place(
                relx=0.5,
                rely=0.5,
                anchor="center"
            )

            self.color_selection_buttons[color_name] = (ring, button)

        def select_accent(value):
            accent_var.set(value)
            for name, (ring_widget, button_widget) in self.color_selection_buttons.items():
                selected_value = ACCENT_PALETTE[name][0]
                ring_widget.configure(
                    border_width=3 if name == value else 0,
                    border_color=selected_value
                )

        ctk.CTkLabel(
            content_frame,
            textvariable=accent_var,
            font=self.small_font(9),
            text_color=TEXT_MUTED
        ).pack(
            pady=(0, 7)
        )

        ctk.CTkLabel(
            content_frame,
            text="English / فارسی",
            font=self.small_font(10),
            text_color=TEXT_SECONDARY
        ).pack(
            pady=(0, 10)
        )

        button_row = ctk.CTkFrame(
            self.settings_panel,
            fg_color=CARD_COLOR
        )
        button_row.pack(
            fill="x",
            padx=12,
            pady=(4, 14)
        )

        button_inner = ctk.CTkFrame(
            button_row,
            fg_color="transparent"
        )
        button_inner.pack()

        def apply_settings():
            self.current_theme = (
                "dark"
                if theme_var.get() == self.t("dark")
                else "light"
            )
            self.current_language = (
                "en"
                if language_var.get() == self.t("english")
                else "fa"
            )
            self.font_scale = max(
                0.85,
                min(
                    1.15,
                    int(font_var.get().replace("%", "")) / 100
                )
            )
            self.accent_color = (
                accent_var.get()
                if accent_var.get() in ACCENT_PALETTE
                else "gold"
            )
            self.update_global_theme()
            self.close_settings()
            self.show_page_one()

        ctk.CTkButton(
            button_inner,
            text=self.t("apply"),
            width=140,
            height=40,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(12),
            command=apply_settings
        ).pack(
            side="left",
            padx=4
        )

        ctk.CTkButton(
            button_inner,
            text=self.t("close"),
            width=100,
            height=40,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(12),
            command=self.close_settings
        ).pack(
            side="left",
            padx=4
        )

        self.settings_theme_var = theme_var
        self.settings_language_var = language_var
        self.settings_font_var = font_var
        self.settings_accent_var = accent_var

        available_height = max(
            400,
            self.winfo_height() - 110
        )
        self.settings_target_height = min(
            590,
            available_height
        )
        self.settings_current_height = 0

        self.settings_button.configure(
            text="✕",
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111"
        )

        self.animate_settings_open()

    def animate_settings_open(self):
        if getattr(self, "settings_panel", None) is None:
            return
        self.settings_current_height += 35
        if self.settings_current_height >= self.settings_target_height:
            self.settings_current_height = self.settings_target_height
            self.settings_panel.configure(height=self.settings_current_height)
            self.settings_opening = False
            return
        self.settings_panel.configure(height=self.settings_current_height)
        self.after(15, self.animate_settings_open)
    def close_settings(self):
        if getattr(self, "settings_panel", None) is None:
            return
        self.settings_opening = False
        self.settings_closing = True
        self.animate_settings_close()
    def animate_settings_close(self):
        if getattr(self, "settings_panel", None) is None:
            return
        self.settings_current_height -= 35
        if self.settings_current_height <= 0:
            self.settings_panel.destroy()
            self.settings_panel = None
            self.settings_closing = False
            self.settings_button.configure(
                text="⚙",
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=TEXT_PRIMARY
            )
            return
        self.settings_panel.configure(height=self.settings_current_height)
        self.after(15, self.animate_settings_close)
    def clear_page(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()
    def text_anchor(self):
        return (
            "e"
            if self.current_language == "fa"
            else "w"
        )
    def draw_nexora_logo(
        self,
        canvas,
        cx,
        cy,
        scale=1
    ):
        gold = PRIMARY
        w = 32 * scale
        h = 42 * scale
        line_width = max(
            4,
            int(8 * scale)
        )
        canvas.create_line(
            cx - w,
            cy + h,
            cx - w,
            cy - h,
            fill=gold,
            width=line_width
        )
        canvas.create_line(
            cx - w,
            cy - h,
            cx + w,
            cy + h,
            fill=gold,
            width=line_width
        )
        canvas.create_line(
            cx + w,
            cy + h,
            cx + w,
            cy - h,
            fill=gold,
            width=line_width
        )
    def create_next_button(
        self,
        parent,
        command,
        text=None
    ):
        if text is None:
            text = self.t("next")
        button = ctk.CTkButton(
            parent,
            text=text,
            width=150,
            height=42,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(14),
            command=command
        )
        button.pack(
            anchor="center"
        )
        return button
    def show_splash(self):
        self.clear_page()
        splash = ctk.CTkFrame(
            self.main_container,
            fg_color=BG_COLOR,
            corner_radius=0
        )
        splash.pack(
            fill="both",
            expand=True
        )
        self.splash_canvas = ctk.CTkCanvas(
            splash,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.splash_canvas.pack(
            fill="both",
            expand=True
        )
        self.splash_frame = 0
        self.after(
            80,
            self.animate_splash
        )
    def animate_splash(self):
        self.splash_frame += 1
        frame = self.splash_frame
        canvas = self.splash_canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1:
            self.after(
                30,
                self.animate_splash
            )
            return
        canvas.delete("all")
        cx = width / 2
        cy = height / 2 - 35
        intro = min(
            1,
            frame / 40
        )
        pulse = (
            np.sin(
                frame * 0.12
            ) + 1
        ) / 2
        orbit_radius = (
            175 -
            105 * intro
        )
        core_radius = (
            8 +
            20 * intro +
            pulse * 3
        )
        for ring in range(3):
            radius = (
                70 +
                ring * 30 +
                pulse * 8
            )
            canvas.create_oval(
                cx - radius,
                cy - radius * 0.58,
                cx + radius,
                cy + radius * 0.58,
                outline=BORDER,
                width=1
            )
        nodes = []
        for i in range(8):
            angle = (
                frame * 0.035 +
                i * (np.pi / 4)
            )
            px = (
                cx +
                np.cos(angle) *
                orbit_radius
            )
            py = (
                cy +
                np.sin(angle) *
                orbit_radius *
                0.58
            )
            nodes.append(
                (px, py)
            )
            size = (
                3 +
                i % 3
            )
            canvas.create_oval(
                px - size,
                py - size,
                px + size,
                py + size,
                fill=PRIMARY,
                outline=""
            )
            canvas.create_line(
                px,
                py,
                cx,
                cy,
                fill=BORDER,
                width=1
            )
        for i in range(
            len(nodes)
        ):
            x1, y1 = nodes[i]
            x2, y2 = nodes[
                (i + 1) % len(nodes)
            ]
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=BORDER,
                width=1
            )
        canvas.create_oval(
            cx - core_radius - 10,
            cy - core_radius - 10,
            cx + core_radius + 10,
            cy + core_radius + 10,
            outline=BORDER,
            width=2
        )
        canvas.create_oval(
            cx - core_radius,
            cy - core_radius,
            cx + core_radius,
            cy + core_radius,
            outline=PRIMARY,
            width=3
        )
        inner_radius = core_radius * 0.33
        canvas.create_oval(
            cx - inner_radius,
            cy - inner_radius,
            cx + inner_radius,
            cy + inner_radius,
            fill=PRIMARY,
            outline=""
        )
        if frame >= 30:
            logo_progress = min(
                1,
                (frame - 30) / 25
            )
            scale = (
                0.15 +
                0.85 * logo_progress
            )
            self.draw_nexora_logo(
                canvas,
                cx,
                cy,
                scale
            )
        if frame >= 47:
            title_progress = min(
                1,
                (frame - 47) / 25
            )
            title_size = max(
                1,
                int(
                    40 * title_progress
                )
            )
            canvas.create_text(
                cx,
                cy + 108,
                text=self.t("nexora"),
                fill=PRIMARY,
                font=(
                    "Segoe UI",
                    title_size,
                    "bold"
                )
            )
        if frame >= 70:
            subtitle_progress = min(
                1,
                (frame - 70) / 23
            )
            subtitle_size = max(
                1,
                int(
                    12 * subtitle_progress
                )
            )
            canvas.create_text(
                cx,
                cy + 153,
                text=self.t("subtitle"),
                fill=TEXT_SECONDARY,
                font=(
                    "Segoe UI",
                    subtitle_size
                )
            )
        if frame >= 78:
            author_progress = min(
                1,
                (frame - 78) / 20
            )
            author_size = max(
                1,
                int(
                    10 * author_progress
                )
            )
            canvas.create_text(
                cx,
                cy + 177,
                text=self.t("created"),
                fill=TEXT_MUTED,
                font=(
                    "Segoe UI",
                    author_size
                )
            )
        if frame >= 108:
            progress = min(
                1,
                (frame - 108) / 18
            )
            scale = max(
                0,
                1 - progress
            )
            if scale > 0:
                self.draw_nexora_logo(
                    canvas,
                    cx,
                    cy,
                    scale
                )
        if frame < 125:
            self.after(
                33,
                self.animate_splash
            )
        else:
            self.show_page_one()
    def show_page_one(self):
        self.clear_page()
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            padx=60,
            pady=(25, 0)
        )
        history_button = ctk.CTkButton(
            self.main_container,
            text="◷",
            width=44,
            height=44,
            corner_radius=12,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(21),
            command=self.show_history_page
        )
        history_button.place(
            relx=0.04,
            rely=0.045,
            anchor="nw"
        )
        settings_button = ctk.CTkButton(
            self.main_container,
            text="⚙",
            width=44,
            height=44,
            corner_radius=12,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(20),
            command=self.open_settings
        )
        settings_button.place(
            relx=0.96,
            rely=0.045,
            anchor="ne"
        )
        self.settings_button = settings_button
        self.settings_panel = None
        self.settings_opening = False
        self.settings_closing = False
        self.settings_current_height = 0
        ctk.CTkLabel(
            header,
            text=self.t("nexora"),
            font=self.title_font(30),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("subtitle"),
            font=self.body_font(14),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center",
            pady=(4, 0)
        )
        ctk.CTkLabel(
            header,
            text=self.t("created"),
            font=self.small_font(10),
            text_color=TEXT_MUTED
        ).pack(
            anchor="center",
            pady=(3, 0)
        )
        center = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        center.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=15
        )
        import_card = ctk.CTkFrame(
            center,
            width=650,
            height=330,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        import_card.place(
            relx=0.5,
            rely=0.48,
            anchor="center"
        )
        import_card.pack_propagate(False)
        logo_frame = ctk.CTkFrame(
            import_card,
            fg_color="transparent",
            width=70,
            height=70
        )
        logo_frame.pack(
            pady=(18, 2)
        )
        logo_canvas = ctk.CTkCanvas(
            logo_frame,
            width=70,
            height=70,
            bg=CARD_COLOR,
            highlightthickness=0
        )
        logo_canvas.pack()
        self.draw_nexora_logo(
            logo_canvas,
            35,
            35,
            0.75
        )
        ctk.CTkLabel(
            import_card,
            text=self.t("import_dataset"),
            font=self.title_font(22),
            text_color=TEXT_PRIMARY
        ).pack()
        ctk.CTkLabel(
            import_card,
            text=self.t("build_train_predict"),
            font=self.body_font(13),
            text_color=TEXT_SECONDARY
        ).pack(
            pady=(6, 18)
        )
        file_box = ctk.CTkFrame(
            import_card,
            width=520,
            height=55,
            fg_color=INPUT_COLOR,
            corner_radius=9,
            border_width=1,
            border_color=BORDER
        )
        file_box.pack()
        file_box.pack_propagate(False)
        self.file_label = ctk.CTkLabel(
            file_box,
            text=self.t("no_file"),
            font=self.body_font(13),
            text_color=TEXT_SECONDARY
        )
        self.file_label.pack(
            side="left",
            padx=18,
            expand=True,
            fill="x"
        )
        ctk.CTkButton(
            file_box,
            text=self.t("import"),
            width=100,
            height=36,
            corner_radius=8,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(13),
            command=self.import_file
        ).pack(
            side="right",
            padx=9
        )
        self.error_label = ctk.CTkLabel(
            import_card,
            text="",
            font=self.body_font(12),
            text_color=ERROR
        )
        self.error_label.pack(
            pady=(8, 0)
        )
        self.next_area = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=70
        )
        self.next_area.pack(
            side="bottom",
            fill="x",
            pady=(0, 10)
        )
        self.next_button = ctk.CTkButton(
            self.next_area,
            text=self.t("next"),
            width=150,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_MUTED,
            font=self.body_font(14),
            command=self.start_page_two_loading,
            state="disabled"
        )
        self.next_button.pack(
            anchor="center"
        )
    def import_file(self):
        file_path = filedialog.askopenfilename(
            title=self.t("import_dataset"),
            filetypes=[
                ("Excel Files", "*.xlsx *.xls")
            ]
        )
        if not file_path:
            return
        extension = os.path.splitext(
            file_path
        )[1].lower()
        if extension not in [
            ".xlsx",
            ".xls"
        ]:
            self.error_label.configure(
                text=self.t("invalid_file"),
                text_color=ERROR
            )
            return
        try:
            data = pd.read_excel(
                file_path
            )
            if data.empty:
                raise ValueError(
                    self.t("empty_excel")
                )
            self.selected_file = file_path
            self.data = data
            self.file_label.configure(
                text=os.path.basename(
                    file_path
                ),
                text_color=TEXT_PRIMARY
            )
            self.error_label.configure(
                text=self.t("excel_loaded"),
                text_color=PRIMARY_LIGHT
            )
            self.next_button.configure(
                state="normal",
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
                text_color="#111111"
            )
        except Exception as error:
            self.selected_file = None
            self.data = None
            self.file_label.configure(
                text=self.t("no_file")
            )
            self.error_label.configure(
                text=f'{self.t("excel_error")} {error}',
                text_color=ERROR
            )
    def save_prediction_history(
        self,
        values,
        prediction_text,
        probability,
        label
    ):
        record = {
            "label":
                label,
            "dataset":
                os.path.basename(
                    self.selected_file
                )
                if self.selected_file
                else "Unknown Dataset",
            "algorithm":
                self.selected_algorithm,
            "task":
                self.selected_algorithm_type,
            "target":
                self.target_column,
            "prediction":
                prediction_text,
            "probability":
                probability,
            "features":
                list(values.keys()),
            "time":
                datetime.now().strftime(
                    "%Y-%m-%d  %H:%M"
                ),
            "rmse":
                self.test_rmse
                if self.selected_algorithm_type == "Regression"
                else None,
            "mae":
                self.test_mae
                if self.selected_algorithm_type == "Regression"
                else None,
            "r2":
                self.test_r2
                if self.selected_algorithm_type == "Regression"
                else None,
            "accuracy":
                self.test_accuracy
                if self.selected_algorithm_type == "Classification"
                else None
        }
        self.prediction_history.append(
            record
        )
    def show_history_page(self):
        self.clear_page()
        self.history_selected = {}
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(20, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_history"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("previous_predictions"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center",
            pady=(3, 10)
        )
        if self.prediction_history:
            action_row = ctk.CTkFrame(
                self.main_container,
                fg_color="transparent"
            )
            action_row.pack(
                fill="x",
                padx=70,
                pady=(0, 8)
            )
            self.history_select_all_button = ctk.CTkButton(
                action_row,
                text=self.t("select_all_history"),
                width=145,
                height=38,
                corner_radius=9,
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=TEXT_PRIMARY,
                font=self.body_font(12),
                command=self.toggle_select_all_history
            )
            self.history_select_all_button.pack(
                side="left",
                padx=4
            )
            ctk.CTkButton(
                action_row,
                text=self.t("delete_selected"),
                width=155,
                height=38,
                corner_radius=9,
                fg_color=ERROR,
                hover_color="#B83C3C",
                text_color="#FFFFFF",
                font=self.body_font(12),
                command=self.delete_selected_history
            ).pack(
                side="left",
                padx=4
            )
            ctk.CTkButton(
                action_row,
                text=self.t("delete_all"),
                width=120,
                height=38,
                corner_radius=9,
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=ERROR,
                font=self.body_font(12),
                command=self.delete_all_history
            ).pack(
                side="right",
                padx=4
            )
        content = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=70,
            pady=10
        )
        if not self.prediction_history:
            empty_card = ctk.CTkFrame(
                content,
                fg_color=CARD_COLOR,
                corner_radius=16,
                border_width=1,
                border_color=BORDER,
                height=190
            )
            empty_card.pack(
                fill="x",
                pady=20
            )
            empty_card.pack_propagate(False)
            ctk.CTkLabel(
                empty_card,
                text="◇",
                font=self.title_font(40),
                text_color=PRIMARY
            ).pack(
                pady=(28, 3)
            )
            ctk.CTkLabel(
                empty_card,
                text=self.t("no_predictions"),
                font=self.title_font(18),
                text_color=TEXT_PRIMARY
            ).pack()
            ctk.CTkLabel(
                empty_card,
                text=self.t("history_message"),
                font=self.body_font(12),
                text_color=TEXT_SECONDARY
            ).pack(
                pady=(5, 0)
            )
        else:
            for record in reversed(self.prediction_history):
                card = ctk.CTkFrame(
                    content,
                    fg_color=CARD_COLOR,
                    corner_radius=14,
                    border_width=1,
                    border_color=BORDER
                )
                card.pack(
                    fill="x",
                    pady=5
                )
                select_var = ctk.BooleanVar(value=False)
                self.history_selected[id(record)] = (select_var, record)
                ctk.CTkCheckBox(
                    card,
                    text="",
                    variable=select_var,
                    width=25,
                    height=25,
                    checkbox_width=20,
                    checkbox_height=20,
                    fg_color=PRIMARY,
                    hover_color=PRIMARY_HOVER,
                    border_color=BORDER
                ).pack(
                    side="left",
                    padx=(12, 4)
                )
                info_frame = ctk.CTkFrame(
                    card,
                    fg_color="transparent"
                )
                info_frame.pack(
                    side="left",
                    fill="x",
                    expand=True
                )
                ctk.CTkButton(
                    info_frame,
                    text=record.get("label", record.get("dataset", "History")),
                    height=45,
                    corner_radius=10,
                    fg_color="transparent",
                    hover_color=BUTTON_HOVER,
                    text_color=TEXT_PRIMARY,
                    font=self.title_font(15),
                    anchor=self.text_anchor(),
                    command=lambda r=record: self.show_saved_prediction(r)
                ).pack(
                    fill="x",
                    padx=5,
                    pady=(7, 0)
                )
                summary = (
                    f'{record.get("algorithm", "")}  •  '
                    f'{record.get("time", "")}'
                )
                ctk.CTkLabel(
                    info_frame,
                    text=summary,
                    font=self.small_font(10),
                    text_color=TEXT_MUTED,
                    anchor=self.text_anchor()
                ).pack(
                    fill="x",
                    padx=10,
                    pady=(0, 8)
                )
                ctk.CTkButton(
                    card,
                    text="✕",
                    width=38,
                    height=38,
                    corner_radius=8,
                    fg_color="transparent",
                    hover_color=BUTTON_HOVER,
                    text_color=ERROR,
                    font=self.body_font(16),
                    command=lambda r=record: self.delete_single_history(r)
                ).pack(
                    side="right",
                    padx=10
                )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=65
        )
        bottom.pack(
            side="bottom",
            fill="x"
        )
        ctk.CTkButton(
            bottom,
            text=self.t("back_nexora"),
            width=180,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13),
            command=self.show_page_one
        ).pack(
            anchor="center",
            pady=(0, 8)
        )
    def toggle_select_all_history(self):
        if not self.prediction_history:
            return
        all_selected = all(
            var.get()
            for var, record in self.history_selected.values()
        )
        new_value = not all_selected
        for var, record in self.history_selected.values():
            var.set(new_value)
        if hasattr(self, "history_select_all_button"):
            self.history_select_all_button.configure(
                text=(
                    self.t("deselect_all_history")
                    if new_value
                    else self.t("select_all_history")
                )
            )
    def delete_single_history(self, record):
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        self.prediction_history = [
            item
            for item in self.prediction_history
            if item is not record
        ]
        self.show_history_page()
    def delete_selected_history(self):
        selected_records = [
            record
            for var, record in self.history_selected.values()
            if var.get()
        ]
        if not selected_records:
            messagebox.showwarning(
                self.t("delete_history_title"),
                self.t("no_history_selected")
            )
            return
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        selected_ids = {
            id(record)
            for record in selected_records
        }
        self.prediction_history = [
            record
            for record in self.prediction_history
            if id(record) not in selected_ids
        ]
        self.show_history_page()
    def delete_all_history(self):
        if not self.prediction_history:
            return
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        self.prediction_history.clear()
        self.show_history_page()
    def show_saved_prediction(self, record):
        details = [
            (
                self.t("target"),
                record.get("target", "-")
            ),
            (
                self.t("algorithm"),
                record.get("algorithm", "-")
            ),
            (
                self.t("type"),
                record.get("task", "-")
            ),
            (
                self.t("prediction"),
                record.get("prediction", "-")
            ),
            (
                self.t("time"),
                record.get("time", "-")
            )
        ]
        if record.get("task") == "Regression":
            if record.get("rmse") is not None:
                details.append(
                    (
                        self.t("rmse"),
                        f'{record["rmse"]:.4f}'
                    )
                )
            if record.get("mae") is not None:
                details.append(
                    (
                        self.t("mae"),
                        f'{record["mae"]:.4f}'
                    )
                )
            if record.get("r2") is not None:
                details.append(
                    (
                        self.t("r2"),
                        f'{record["r2"]:.4f}'
                    )
                )
        else:
            if record.get("accuracy") is not None:
                details.append(
                    (
                        self.t("accuracy"),
                        f'{record["accuracy"]:.2%}'
                    )
                )
            if record.get("probability") is not None:
                details.append(
                    (
                        self.t("confidence"),
                        f'{record["probability"]:.2%}'
                    )
                )
        message = "\n\n".join(
            f"{label}: {value}"
            for label, value in details
        )
        messagebox.showinfo(
            record.get(
                "label",
                record.get(
                    "dataset",
                    "History"
                )
            ),
            message
        )
    def start_page_two_loading(self):
        self.next_button.configure(
            state="disabled"
        )
        self.loading_value = 0
        self.progress = ctk.CTkProgressBar(
            self.next_area,
            width=320,
            height=14,
            corner_radius=6,
            fg_color=BUTTON_COLOR,
            progress_color=PRIMARY
        )
        self.progress.pack(
            anchor="center",
            pady=(8, 0)
        )
        self.progress.set(0)
        self.animate_page_two_loading()
    def animate_page_two_loading(self):
        self.loading_value += 0.01
        self.progress.set(
            self.loading_value
        )
        if self.loading_value >= 1:
            self.progress.destroy()
            self.progress = None
            self.show_page_two()
        else:
            self.after(
                40,
                self.animate_page_two_loading
            )
    def show_page_two(self):
        self.clear_page()
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(20, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("dataset_overview"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        content = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=5
        )
        left = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=14,
            border_width=1,
            border_color=BORDER
        )
        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6)
        )
        right = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=14,
            border_width=1,
            border_color=BORDER
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(6, 0)
        )
        rows_text = "rows" if self.current_language == "en" else "ردیف"
        columns_text = "columns" if self.current_language == "en" else "ستون"
        numeric_count = len(self.data.select_dtypes(include="number").columns)
        categorical_count = len(self.data.select_dtypes(exclude="number").columns)
        missing_count = int(self.data.isna().sum().sum())
        duplicate_count = int(self.data.duplicated().sum())
        stats = [
            ("Rows" if self.current_language == "en" else "ردیف", f"{self.data.shape[0]:,}"),
            ("Columns" if self.current_language == "en" else "ستون", f"{self.data.shape[1]:,}"),
            ("Numeric" if self.current_language == "en" else "عددی", f"{numeric_count:,}"),
            ("Categorical" if self.current_language == "en" else "دسته‌ای", f"{categorical_count:,}"),
            ("Missing" if self.current_language == "en" else "خالی", f"{missing_count:,}"),
            ("Duplicates" if self.current_language == "en" else "تکراری", f"{duplicate_count:,}")
        ]
        stats_frame = ctk.CTkFrame(
            left,
            fg_color="transparent"
        )
        stats_frame.pack(
            fill="x",
            padx=12,
            pady=(12, 8)
        )
        for index, (label, value) in enumerate(stats):
            card = ctk.CTkFrame(
                stats_frame,
                fg_color=INPUT_COLOR,
                corner_radius=9,
                border_width=1,
                border_color=BORDER
            )
            card.grid(
                row=index // 3,
                column=index % 3,
                padx=4,
                pady=4,
                sticky="nsew"
            )
            ctk.CTkLabel(
                card,
                text=label,
                font=self.small_font(9),
                text_color=TEXT_SECONDARY
            ).pack(
                pady=(7, 0)
            )
            ctk.CTkLabel(
                card,
                text=value,
                font=self.title_font(15),
                text_color=PRIMARY
            ).pack(
                pady=(1, 7)
            )
        for column in range(3):
            stats_frame.grid_columnconfigure(
                column,
                weight=1
            )
        table_title = ctk.CTkFrame(
            left,
            fg_color="transparent"
        )
        table_title.pack(
            fill="x",
            padx=16,
            pady=(0, 6)
        )
        ctk.CTkLabel(
            table_title,
            text="Columns" if self.current_language == "en" else "ستون‌ها",
            font=self.title_font(15),
            text_color=TEXT_PRIMARY
        ).pack(
            side="left"
        )
        ctk.CTkLabel(
            table_title,
            text=f"{self.data.shape[1]:,} {columns_text}",
            font=self.small_font(10),
            text_color=TEXT_MUTED
        ).pack(
            side="right"
        )
        table = ctk.CTkScrollableFrame(
            left,
            fg_color=INPUT_COLOR
        )
        table.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 10)
        )
        for column in self.data.columns:
            row = ctk.CTkFrame(
                table,
                fg_color=ROW_COLOR,
                corner_radius=7
            )
            row.pack(
                fill="x",
                pady=2
            )
            missing = int(self.data[column].isna().sum())
            dtype_text = str(self.data[column].dtype)
            column_button = ctk.CTkButton(
                row,
                text=str(column),
                text_color=TEXT_PRIMARY,
                anchor="e" if self.current_language == "fa" else "w",
                font=self.body_font(11),
                fg_color="transparent",
                hover_color=BUTTON_HOVER,
                command=lambda selected_column=column: self.show_column_chart(selected_column)
            )
            column_button.pack(
                side="left" if self.current_language == "en" else "right",
                padx=2,
                pady=2,
                expand=True,
                fill="x"
            )
            ctk.CTkLabel(
                row,
                text=(f"{dtype_text}  •  {missing} missing" if self.current_language == "en" else f"{dtype_text}  •  {missing} خالی"),
                text_color=ERROR if missing else TEXT_SECONDARY,
                font=self.small_font(9)
            ).pack(
                side="right" if self.current_language == "en" else "left",
                padx=10
            )
        ctk.CTkLabel(
            right,
            text=self.t("data_distribution"),
            font=self.title_font(17),
            text_color=TEXT_PRIMARY
        ).pack(
            pady=(10, 2)
        )
        ctk.CTkLabel(
            right,
            text=("Click a column name to view its most suitable chart" if self.current_language == "en" else "روی نام هر ستون کلیک کنید تا مناسب‌ترین نمودار آن نمایش داده شود"),
            font=self.small_font(10),
            text_color=TEXT_SECONDARY
        ).pack(
            pady=(0, 2)
        )
        self.chart_container = ctk.CTkFrame(
            right,
            fg_color="transparent"
        )
        self.chart_container.pack(
            fill="both",
            expand=True,
            padx=4,
            pady=2
        )
        self.create_overview_chart(
            self.chart_container
        )
        self.next_area = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=65
        )
        self.next_area.pack(
            side="bottom",
            fill="x",
            pady=(0, 8)
        )
        self.create_next_button(
            self.next_area,
            self.show_page_three
        )
    def create_overview_chart(
        self,
        parent
    ):
        if self.data is None or self.data.empty or len(self.data.columns) == 0:
            ctk.CTkLabel(
                parent,
                text=self.t("no_numeric"),
                text_color=TEXT_SECONDARY,
                font=self.body_font(12)
            ).pack(
                expand=True
            )
            return
        self.show_column_chart(
            self.data.columns[0],
            parent
        )
    def show_column_chart(
        self,
        column,
        parent=None
    ):
        if parent is None:
            parent = self.chart_container
        for widget in parent.winfo_children():
            widget.destroy()
        series = self.data[column]
        valid = series.dropna()
        if valid.empty:
            ctk.CTkLabel(
                parent,
                text=("No data available for this column" if self.current_language == "en" else "برای این ستون داده‌ای وجود ندارد"),
                text_color=TEXT_SECONDARY,
                font=self.body_font(12)
            ).pack(
                expand=True
            )
            return
        if pd.api.types.is_datetime64_any_dtype(series):
            chart_type = ("Time Series" if self.current_language == "en" else "نمودار زمانی")
            plot_series = valid.sort_values()
            figure, ax = plt.subplots(
                figsize=(6.5, 4.8),
                dpi=100
            )
            figure.patch.set_facecolor(
                CARD_COLOR
            )
            ax.set_facecolor(
                CARD_COLOR
            )
            ax.plot(
                plot_series.index,
                plot_series.values,
                marker="o",
                linewidth=1.8,
                markersize=3,
                color=PRIMARY
            )
            ax.set_xlabel(
                "Row" if self.current_language == "en" else "ردیف",
                color=TEXT_SECONDARY
            )
            ax.set_ylabel(
                str(column),
                color=TEXT_SECONDARY
            )
        elif pd.api.types.is_numeric_dtype(series):
            chart_type = ("Histogram" if self.current_language == "en" else "هیستوگرام")
            values = pd.to_numeric(
                valid,
                errors="coerce"
            ).dropna()
            figure, ax = plt.subplots(
                figsize=(6.5, 4.8),
                dpi=100
            )
            figure.patch.set_facecolor(
                CARD_COLOR
            )
            ax.set_facecolor(
                CARD_COLOR
            )
            bins = min(
                20,
                max(
                    5,
                    int(np.sqrt(len(values)))
                )
            )
            ax.hist(
                values,
                bins=bins,
                alpha=0.8,
                color=PRIMARY
            )
            ax.set_xlabel(
                str(column),
                color=TEXT_SECONDARY
            )
            ax.set_ylabel(
                "Frequency" if self.current_language == "en" else "فراوانی",
                color=TEXT_SECONDARY
            )
        else:
            category_frame = ctk.CTkFrame(
                parent,
                fg_color=CARD_COLOR,
                corner_radius=16,
                border_width=1,
                border_color=BORDER
            )
            category_frame.pack(
                fill="both",
                expand=True,
                padx=20,
                pady=20
            )
            ctk.CTkLabel(
                category_frame,
                text=(
                    "Categorical Column"
                    if self.current_language == "en"
                    else "این ستون کتگوری است"
                ),
                font=self.title_font(24),
                text_color=PRIMARY
            ).pack(
                pady=(70, 12)
            )
            ctk.CTkLabel(
                category_frame,
                text=(
                    f'"{column}" is a categorical column.'
                    if self.current_language == "en"
                    else f'ستون «{column}» از نوع کتگوری است.'
                ),
                font=self.body_font(14),
                text_color=TEXT_SECONDARY
            ).pack(
                pady=5
            )
            ctk.CTkLabel(
                category_frame,
                text=(
                    "No numerical chart is displayed for categorical columns."
                    if self.current_language == "en"
                    else "برای ستون‌های کتگوری نمودار عددی نمایش داده نمی‌شود."
                ),
                font=self.small_font(11),
                text_color=TEXT_MUTED
            ).pack(
                pady=5
            )
            return
        ax.set_title(
            f"{column} • {chart_type}",
            color=TEXT_PRIMARY,
            fontsize=12,
            pad=12
        )
        ax.tick_params(
            colors=TEXT_SECONDARY,
            labelsize=8
        )
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(
            axis="y",
            alpha=0.18
        )
        figure.tight_layout(
            pad=1.8
        )
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=14,
            pady=8
        )
    def show_page_three(self):
        self.clear_page()
        self.feature_columns = []
        self.target_column = None
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(20, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("feature_target"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("feature_description"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center",
            pady=(3, 0)
        )
        content = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=35,
            pady=8
        )
        feature_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        feature_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 7)
        )
        target_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        target_card.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(7, 0)
        )
        ctk.CTkLabel(
            feature_card,
            text=self.t("features"),
            font=self.title_font(19),
            text_color=TEXT_PRIMARY
        ).pack(
            pady=(15, 3)
        )
        self.select_all_var = ctk.BooleanVar(
            value=False
        )
        ctk.CTkCheckBox(
            feature_card,
            text=self.t("select_all"),
            variable=self.select_all_var,
            command=self.toggle_select_all,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13)
        ).pack(
            anchor="w",
            padx=25,
            pady=(4, 10)
        )
        self.feature_menu = ctk.CTkComboBox(
            feature_card,
            values=list(
                self.data.columns
            ),
            width=280,
            height=40,
            fg_color=INPUT_COLOR,
            border_color=BORDER,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color=BUTTON_HOVER,
            font=self.body_font(13),
            dropdown_font=self.body_font(13),
            command=self.add_feature
        )
        self.feature_menu.pack(
            padx=25,
            pady=(0, 10)
        )
        self.feature_menu.set(
            self.t("select_column")
        )
        ctk.CTkLabel(
            feature_card,
            text=self.t("selected_features"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor=self.text_anchor(),
            padx=25
        )
        self.feature_box = ctk.CTkScrollableFrame(
            feature_card,
            fg_color=INPUT_COLOR
        )
        self.feature_box.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=(5, 15)
        )
        ctk.CTkLabel(
            target_card,
            text=self.t("target"),
            font=self.title_font(19),
            text_color=TEXT_PRIMARY
        ).pack(
            pady=(15, 3)
        )
        self.target_menu = ctk.CTkComboBox(
            target_card,
            values=list(
                self.data.columns
            ),
            width=280,
            height=40,
            fg_color=INPUT_COLOR,
            border_color=BORDER,
            button_color=PRIMARY,
            button_hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            dropdown_fg_color=CARD_COLOR,
            dropdown_hover_color=BUTTON_HOVER,
            font=self.body_font(13),
            dropdown_font=self.body_font(13),
            command=self.select_target
        )
        self.target_menu.pack(
            padx=25,
            pady=25
        )
        self.target_menu.set(
            self.t("select_target")
        )
        ctk.CTkLabel(
            target_card,
            text=self.t("selected_target"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor=self.text_anchor(),
            padx=25
        )
        self.target_box = ctk.CTkFrame(
            target_card,
            height=65,
            fg_color=INPUT_COLOR,
            corner_radius=10,
            border_width=1,
            border_color=BORDER
        )
        self.target_box.pack(
            fill="x",
            padx=25,
            pady=5
        )
        self.target_box.pack_propagate(False)
        self.target_value = ctk.CTkLabel(
            self.target_box,
            text=self.t("no_target"),
            text_color=TEXT_MUTED,
            font=self.body_font(12)
        )
        self.target_value.pack(
            expand=True
        )
        self.next_area = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=65
        )
        self.next_area.pack(
            side="bottom",
            fill="x",
            pady=(0, 8)
        )
        self.page_three_next = self.create_next_button(
            self.next_area,
            self.start_page_four_loading
        )
        self.page_three_next.configure(
            state="disabled",
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_MUTED
        )
    def add_feature(
        self,
        choice
    ):
        if choice == self.t("select_column"):
            return
        if (
            choice not in self.feature_columns
            and choice != self.target_column
        ):
            self.feature_columns.append(
                choice
            )
        self.feature_menu.set(
            self.t("select_column")
        )
        self.update_feature_box()
        self.update_next_button()
    def update_feature_box(self):
        for widget in self.feature_box.winfo_children():
            widget.destroy()
        for column in self.feature_columns:
            row = ctk.CTkFrame(
                self.feature_box,
                fg_color=ROW_COLOR,
                corner_radius=7
            )
            row.pack(
                fill="x",
                pady=3
            )
            ctk.CTkLabel(
                row,
                text=str(column),
                text_color=TEXT_PRIMARY,
                font=self.body_font(12)
            ).pack(
                side="left",
                padx=10,
                pady=7
            )
            ctk.CTkButton(
                row,
                text="×",
                width=28,
                height=26,
                fg_color="transparent",
                hover_color="#7A2828",
                text_color=ERROR,
                font=self.body_font(14),
                command=lambda c=column:
                    self.remove_feature(c)
            ).pack(
                side="right",
                padx=5
            )
    def remove_feature(
        self,
        column
    ):
        if column in self.feature_columns:
            self.feature_columns.remove(
                column
            )
        self.select_all_var.set(False)
        self.update_feature_box()
        self.update_next_button()
    def toggle_select_all(self):
        if self.select_all_var.get():
            self.feature_columns = [
                column
                for column in self.data.columns
                if column != self.target_column
            ]
        else:
            self.feature_columns = []
        self.update_feature_box()
        self.update_next_button()
    def select_target(
        self,
        choice
    ):
        if choice == self.t("select_target"):
            return
        if choice in self.feature_columns:
            self.feature_columns.remove(
                choice
            )
        self.target_column = choice
        self.target_value.configure(
            text=choice,
            text_color=PRIMARY
        )
        self.update_feature_box()
        self.update_next_button()
    def update_next_button(self):
        if (
            self.feature_columns
            and self.target_column
        ):
            self.page_three_next.configure(
                state="normal",
                fg_color=PRIMARY,
                hover_color=PRIMARY_HOVER,
                text_color="#111111"
            )
        else:
            self.page_three_next.configure(
                state="disabled",
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=TEXT_MUTED
            )
    def start_page_four_loading(self):
        if (
            not self.feature_columns
            or not self.target_column
        ):
            return
        self.page_three_next.configure(
            state="disabled"
        )
        self.loading_value = 0
        self.progress = ctk.CTkProgressBar(
            self.next_area,
            width=320,
            height=14,
            corner_radius=6,
            fg_color=BUTTON_COLOR,
            progress_color=PRIMARY
        )
        self.progress.pack(
            anchor="center",
            pady=(7, 0)
        )
        self.progress.set(0)
        self.animate_page_four_loading()
    def animate_page_four_loading(self):
        self.loading_value += 0.01
        self.progress.set(
            self.loading_value
        )
        if self.loading_value >= 1:
            self.progress.destroy()
            self.progress = None
            self.show_page_four()
        else:
            self.after(
                30,
                self.animate_page_four_loading
            )
    def show_page_four(self):
        self.clear_page()
        self.selected_algorithm = None
        self.selected_algorithm_type = None
        self.auto_selection_results = {}
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(15, 3)
        )
        ctk.CTkLabel(
            header,
            text=self.t("choose_algorithm"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        content = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=5
        )
        target_task_type = None
        if self.target_column and self.data is not None:
            try:
                target_task_type = self.infer_auto_task_type(
                    self.data[self.target_column].dropna()
                )
            except Exception:
                target_task_type = None

        regression_enabled = target_task_type in (None, "Regression")
        classification_enabled = target_task_type in (None, "Classification")

        regression_frame = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR if regression_enabled else BUTTON_COLOR,
            corner_radius=15,
            border_width=1,
            border_color=BORDER if regression_enabled else TEXT_MUTED
        )
        regression_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 5)
        )
        classification_frame = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR if classification_enabled else BUTTON_COLOR,
            corner_radius=15,
            border_width=1,
            border_color=BORDER if classification_enabled else TEXT_MUTED
        )
        classification_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(5, 0)
        )
        ctk.CTkLabel(
            regression_frame,
            text=self.t("regression"),
            font=self.title_font(20),
            text_color=TEXT_PRIMARY if regression_enabled else TEXT_MUTED
        ).pack(
            pady=(10, 2)
        )
        regression_scroll = ctk.CTkScrollableFrame(
            regression_frame,
            fg_color="transparent"
        )
        regression_scroll.pack(
            fill="both",
            expand=True,
            padx=8
        )
        regression_algorithms = [
            (
                "Linear Regression",
                self.t("linear_baseline")
            ),
            (
                "Polynomial Regression",
                self.t("automatic_degree")
            ),
            (
                "Ridge Regression",
                self.t("l2_regularization")
            ),
            (
                "Lasso Regression",
                self.t("l1_regularization")
            ),
            (
                "Elastic Net",
                self.t("l1_l2")
            ),
            (
                "Decision Tree Regressor",
                self.t("tree_model")
            ),
            (
                "Random Forest Regressor",
                self.t("tree_ensemble")
            ),
            (
                "Extra Trees Regressor",
                self.t("randomized_ensemble")
            ),
            (
                "Gradient Boosting Regressor",
                self.t("boosting")
            ),
            (
                "XGBoost Regressor",
                self.t("advanced_boosting")
            ),
            (
                "SVR",
                self.t("support_vector_regression")
            ),
            (
                "KNN Regressor",
                self.t("nearest_neighbors")
            )
        ]
        for name, description in regression_algorithms:
            self.create_algorithm_card(
                regression_scroll,
                name,
                description,
                "Regression",
                regression_enabled
            )
        ctk.CTkLabel(
            classification_frame,
            text=self.t("classification"),
            font=self.title_font(20),
            text_color=TEXT_PRIMARY if classification_enabled else TEXT_MUTED
        ).pack(
            pady=(10, 2)
        )
        classification_scroll = ctk.CTkScrollableFrame(
            classification_frame,
            fg_color="transparent"
        )
        classification_scroll.pack(
            fill="both",
            expand=True,
            padx=8
        )
        classification_algorithms = [
            (
                "Logistic Regression",
                self.t("linear_classifier")
            ),
            (
                "K-Nearest Neighbors",
                self.t("distance_based")
            ),
            (
                "Naive Bayes",
                self.t("probabilistic")
            ),
            (
                "Decision Tree",
                self.t("tree_model")
            ),
            (
                "Random Forest",
                self.t("tree_ensemble")
            ),
            (
                "Extra Trees",
                self.t("randomized_ensemble")
            ),
            (
                "Gradient Boosting",
                self.t("boosting")
            ),
            (
                "AdaBoost",
                self.t("adaptive_boosting")
            ),
            (
                "Support Vector Machine",
                self.t("margin_classifier")
            ),
            (
                "XGBoost",
                self.t("advanced_boosting")
            ),
            (
                "LightGBM",
                self.t("fast_boosting")
            ),
            (
                "CatBoost",
                self.t("categorical_boosting")
            )
        ]
        for name, description in classification_algorithms:
            self.create_algorithm_card(
                classification_scroll,
                name,
                description,
                "Classification",
                classification_enabled
            )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=95
        )
        bottom.pack(
            side="bottom",
            fill="x",
            pady=(0, 5)
        )
        self.confirmation_label = ctk.CTkLabel(
            bottom,
            text=self.t("no_algorithm"),
            text_color=TEXT_MUTED,
            font=self.body_font(12)
        )
        self.confirmation_label.pack(
            pady=(0, 5)
        )
        button_row = ctk.CTkFrame(
            bottom,
            fg_color="transparent"
        )
        button_row.pack(
            anchor="center"
        )
        self.auto_select_button = ctk.CTkButton(
            button_row,
            text=self.t("auto_select"),
            width=220,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=PRIMARY_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(12),
            command=self.start_auto_selection
        )
        self.auto_select_button.pack(
            side="left",
            padx=6
        )
        self.auto_selection_progress = ctk.CTkProgressBar(
            button_row,
            width=180,
            height=12,
            corner_radius=6,
            fg_color=BUTTON_COLOR,
            progress_color=PRIMARY
        )
        self.auto_selection_progress.set(0)
        self.auto_selection_progress.pack(
            side="left",
            padx=6
        )
        self.auto_selection_progress.pack_forget()
        self.page_four_next = ctk.CTkButton(
            button_row,
            text=self.t("next"),
            width=150,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_MUTED,
            font=self.body_font(14),
            command=self.start_page_five,
            state="disabled"
        )
        self.page_four_next.pack(
            side="left",
            padx=6
        )
    def create_algorithm_card(
        self,
        parent,
        name,
        description,
        algorithm_type,
        enabled=True
    ):
        card_fg = INPUT_COLOR if enabled else BUTTON_COLOR
        card_border = BORDER if enabled else TEXT_MUTED
        card_text = TEXT_PRIMARY if enabled else TEXT_MUTED
        card_secondary = TEXT_SECONDARY if enabled else TEXT_MUTED

        card = ctk.CTkFrame(
            parent,
            fg_color=card_fg,
            corner_radius=10,
            border_width=1,
            border_color=card_border
        )
        card.pack(
            fill="x",
            pady=3
        )
        info = ctk.CTkFrame(
            card,
            fg_color="transparent"
        )
        info.pack(
            side="left",
            fill="both",
            expand=True,
            padx=10,
            pady=6
        )
        ctk.CTkLabel(
            info,
            text=name,
            font=self.body_font(12),
            text_color=card_text
        ).pack(
            anchor=self.text_anchor()
        )
        ctk.CTkLabel(
            info,
            text=description,
            font=self.small_font(9),
            text_color=card_secondary
        ).pack(
            anchor=self.text_anchor()
        )
        ctk.CTkButton(
            card,
            text=self.t("use_this"),
            width=82,
            height=30,
            corner_radius=7,
            fg_color=BUTTON_COLOR if enabled else CARD_COLOR,
            hover_color=PRIMARY if enabled else CARD_COLOR,
            text_color=TEXT_PRIMARY if enabled else TEXT_MUTED,
            font=self.small_font(10),
            state="normal" if enabled else "disabled",
            command=lambda:
            self.select_algorithm(
                name,
                algorithm_type
            )
        ).pack(
            side="right",
            padx=8
        )
    def select_algorithm(
        self,
        name,
        algorithm_type
    ):
        self.selected_algorithm = name
        self.selected_algorithm_type = algorithm_type
        self.confirmation_label.configure(
            text=(
                f"✓ {self.t('selected')}: "
                f"{name}  |  "
                f"{algorithm_type}"
            ),
            text_color=PRIMARY
        )
        self.page_four_next.configure(
            state="normal",
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111"
        )
    def start_auto_selection(self):
        if (
            not self.feature_columns
            or not self.target_column
        ):
            messagebox.showwarning(
                self.t("auto_selection_title"),
                self.t("please_features_target")
            )
            return
        if self.auto_selection_running:
            return
        self.auto_selection_running = True
        self.auto_select_button.configure(
            state="disabled",
            text=self.t("analyzing")
        )
        self.page_four_next.configure(
            state="disabled"
        )
        self.confirmation_label.configure(
            text=self.t("comparing_models"),
            text_color=TEXT_SECONDARY
        )
        self.auto_selection_progress.pack(
            side="left",
            padx=6
        )
        self.auto_selection_progress.set(0)
        self.auto_selection_thread = threading.Thread(
            target=self.run_auto_selection,
            daemon=True
        )
        self.auto_selection_thread.start()
    def infer_auto_task_type(self, y):
        if (
            y.dtype == "object"
            or str(y.dtype).startswith("category")
            or pd.api.types.is_bool_dtype(y)
        ):
            return "Classification"
        return (
            "Classification"
            if y.nunique(
                dropna=True
            ) <= 2
            else "Regression"
        )
    def prepare_auto_data(self, task):
        X = self.data[
            self.feature_columns
        ].copy()
        y = self.data[
            self.target_column
        ].copy()
        valid_indices = y.notna()
        X = X.loc[
            valid_indices
        ]
        y = y.loc[
            valid_indices
        ]
        target_encoder = None
        if task == "Classification":
            if (
                y.dtype == "object"
                or str(y.dtype).startswith("category")
                or pd.api.types.is_bool_dtype(y)
            ):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(
                    y.astype(str)
                )
            else:
                y = np.asarray(y)
        else:
            y = pd.to_numeric(
                y,
                errors="coerce"
            )
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid]
        return X, y, target_encoder
    def build_preprocessor_for_auto(self, X):
        numeric_columns = list(
            X.select_dtypes(
                include=np.number
            ).columns
        )
        categorical_columns = list(
            X.select_dtypes(
                exclude=np.number
            ).columns
        )
        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )
        transformers = []
        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            )
        return ColumnTransformer(
            transformers=transformers
        )
    def get_auto_models(
        self,
        task_type,
        n_samples
    ):
        if task_type == "Regression":
            models = {
                "Linear Regression":
                    LinearRegression(),
                "Ridge Regression":
                    Ridge(),
                "Random Forest Regressor":
                    RandomForestRegressor(
                        n_estimators=60,
                        random_state=42,
                        n_jobs=-1
                    ),
                "Extra Trees Regressor":
                    ExtraTreesRegressor(
                        n_estimators=60,
                        random_state=42,
                        n_jobs=-1
                    ),
                "Gradient Boosting Regressor":
                    GradientBoostingRegressor(
                        random_state=42
                    )
            }
            if XGBOOST_AVAILABLE:
                models[
                    "XGBoost Regressor"
                ] = XGBRegressor(
                    n_estimators=60,
                    random_state=42,
                    n_jobs=-1
                )
            if n_samples > 20000:
                models.pop(
                    "Gradient Boosting Regressor",
                    None
                )
        else:
            models = {
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),
                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),
                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=60,
                        random_state=42,
                        n_jobs=-1
                    ),
                "Extra Trees":
                    ExtraTreesClassifier(
                        n_estimators=60,
                        random_state=42,
                        n_jobs=-1
                    ),
                "Gradient Boosting":
                    GradientBoostingClassifier(
                        random_state=42
                    )
            }
            if XGBOOST_AVAILABLE:
                models[
                    "XGBoost"
                ] = XGBClassifier(
                    n_estimators=60,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                )
            if n_samples > 20000:
                models.pop(
                    "Gradient Boosting",
                    None
                )
        return models
    def create_auto_pipeline(
        self,
        X,
        model
    ):
        return Pipeline(
            [
                (
                    "preprocessor",
                    self.build_preprocessor_for_auto(X)
                ),
                (
                    "model",
                    model
                )
            ]
        )
    def get_auto_cv(
        self,
        X,
        y,
        task_type
    ):
        folds = 3
        if task_type == "Classification":
            _, counts = np.unique(
                y,
                return_counts=True
            )
            if (
                len(counts) == 0
                or np.min(counts) < 2
            ):
                raise ValueError(
                    self.t("at_least_two_each_class")
                )
            return StratifiedKFold(
                n_splits=min(
                    folds,
                    int(np.min(counts))
                ),
                shuffle=True,
                random_state=42
            )
        if len(X) < 3:
            raise ValueError(
                self.t("at_least_three")
            )
        return KFold(
            n_splits=min(
                folds,
                len(X)
            ),
            shuffle=True,
            random_state=42
        )
    def run_auto_selection(self):
        try:
            raw_y = self.data[
                self.target_column
            ].dropna()
            task_type = self.infer_auto_task_type(
                raw_y
            )
            X, y, target_encoder = self.prepare_auto_data(
                task_type
            )
            if len(X) < 5:
                raise ValueError(
                    self.t("not_enough_rows")
                )
            if (
                task_type == "Classification"
                and len(np.unique(y)) < 2
            ):
                raise ValueError(
                    self.t("two_classes")
                )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
                if task_type == "Classification"
                else None
            )
            models = self.get_auto_models(
                task_type,
                len(X_train)
            )
            cv = self.get_auto_cv(
                X_train,
                y_train,
                task_type
            )
            results = {}
            total_models = len(models)
            for index, (name, model) in enumerate(
                models.items(),
                start=1
            ):
                try:
                    pipeline = self.create_auto_pipeline(
                        X_train,
                        model
                    )
                    if task_type == "Regression":
                        scores = cross_val_score(
                            pipeline,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=-1
                        )
                        score = -float(
                            np.mean(scores)
                        )
                    else:
                        scores = cross_val_score(
                            pipeline,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring="accuracy",
                            n_jobs=-1
                        )
                        score = float(
                            np.mean(scores)
                        )
                    if np.isfinite(score):
                        results[
                            name
                        ] = score
                except Exception as error:
                    print(
                        f"Auto model '{name}' failed: {error}"
                    )
                progress_value = (
                    index /
                    max(
                        1,
                        total_models
                    )
                )
                self.after(
                    0,
                    lambda value=progress_value:
                    self.update_auto_progress(value)
                )
            if not results:
                raise ValueError(
                    self.t("no_model")
                )
            if task_type == "Regression":
                best_model_name = min(
                    results,
                    key=results.get
                )
            else:
                best_model_name = max(
                    results,
                    key=results.get
                )
            best_score = results[
                best_model_name
            ]
            self.after(
                0,
                lambda:
                self.finish_auto_selection(
                    task_type,
                    best_model_name,
                    best_score,
                    results,
                    target_encoder
                )
            )
        except Exception as error:
            self.after(
                0,
                lambda:
                self.auto_selection_failed(
                    str(error)
                )
            )
    def update_auto_progress(self, value):
        if (
            self.auto_selection_progress
            and self.auto_selection_progress.winfo_exists()
        ):
            self.auto_selection_progress.set(value)
    def finish_auto_selection(
        self,
        task_type,
        best_model_name,
        best_score,
        results,
        target_encoder
    ):
        self.auto_selection_running = False
        self.auto_selection_thread = None
        self.auto_selection_progress.pack_forget()
        self.auto_selection_progress.set(0)
        self.auto_selection_results = results
        self.selected_algorithm = best_model_name
        self.selected_algorithm_type = task_type
        self.target_encoder = target_encoder
        if task_type == "Regression":
            metric_text = (
                f'{self.t("cv_rmse")} '
                f'{best_score:.4f}'
            )
        else:
            metric_text = (
                f'{self.t("cv_accuracy")} '
                f'{best_score:.2%}'
            )
        self.confirmation_label.configure(
            text=(
                f'✓ {self.t("selected")}: '
                f'{best_model_name}'
                f'  |  '
                f'{task_type}'
                f'  |  '
                f'{metric_text}'
            ),
            text_color=PRIMARY
        )
        self.auto_select_button.configure(
            state="normal",
            text=self.t("auto_select")
        )
        self.page_four_next.configure(
            state="normal",
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111"
        )
        sorted_results = sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=(
                task_type == "Classification"
            )
        )
        result_lines = []
        for rank, (name, score) in enumerate(
            sorted_results,
            start=1
        ):
            if task_type == "Regression":
                result_lines.append(
                    f"{rank}. {name} → RMSE = {score:.4f}"
                )
            else:
                result_lines.append(
                    f"{rank}. {name} → Accuracy = {score:.2%}"
                )
        messagebox.showinfo(
            self.t("auto_selection_title"),
            (
                f'{self.t("best_model")}\n'
                f'{best_model_name}\n\n'
                f'{self.t("task")}\n'
                f'{task_type}\n'
                f'{metric_text}\n\n'
                f'{self.t("model_ranking")}\n\n'
                +
                "\n".join(result_lines)
            )
        )
    def auto_selection_failed(self, error):
        self.auto_selection_running = False
        self.auto_selection_thread = None
        self.auto_selection_progress.pack_forget()
        self.auto_selection_progress.set(0)
        self.auto_select_button.configure(
            state="normal",
            text=self.t("auto_select")
        )
        self.page_four_next.configure(
            state="disabled"
        )
        self.confirmation_label.configure(
            text=self.t("auto_error"),
            text_color=ERROR
        )
        messagebox.showerror(
            self.t("auto_error"),
            error
        )
    def start_page_five(self):
        self.page_four_next.configure(
            state="disabled"
        )
        self.train_selected_model()
    def prepare_data(self):
        X = self.data[
            self.feature_columns
        ].copy()
        y = self.data[
            self.target_column
        ].copy()
        valid_indices = y.notna()
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        if self.selected_algorithm_type == "Classification":
            if (
                y.dtype == "object"
                or str(y.dtype).startswith("category")
                or pd.api.types.is_bool_dtype(y)
            ):
                encoder = LabelEncoder()
                y = encoder.fit_transform(
                    y.astype(str)
                )
                self.target_encoder = encoder
            else:
                y = np.asarray(y)
        else:
            y = pd.to_numeric(
                y,
                errors="coerce"
            )
            valid = y.notna()
            X = X.loc[valid]
            y = y.loc[valid]
        return X, y
    def create_model(self):
        name = self.selected_algorithm
        models = {
            "Linear Regression":
                LinearRegression(),
            "Ridge Regression":
                Ridge(),
            "Lasso Regression":
                Lasso(),
            "Elastic Net":
                ElasticNet(),
            "Decision Tree Regressor":
                DecisionTreeRegressor(
                    random_state=42
                ),
            "Random Forest Regressor":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),
            "Extra Trees Regressor":
                ExtraTreesRegressor(
                    n_estimators=100,
                    random_state=42
                ),
            "Gradient Boosting Regressor":
                GradientBoostingRegressor(
                    random_state=42
                ),
            "SVR":
                SVR(),
            "KNN Regressor":
                KNeighborsRegressor(),
            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000
                ),
            "K-Nearest Neighbors":
                KNeighborsClassifier(),
            "Naive Bayes":
                GaussianNB(),
            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),
            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),
            "Extra Trees":
                ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=42
                ),
            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                ),
            "AdaBoost":
                AdaBoostClassifier(
                    random_state=42
                ),
            "Support Vector Machine":
                SVC(
                    probability=True
                )
        }
        if (
            name == "XGBoost Regressor"
            and XGBOOST_AVAILABLE
        ):
            return XGBRegressor(
                n_estimators=100,
                random_state=42
            )
        if (
            name == "XGBoost"
            and XGBOOST_AVAILABLE
        ):
            return XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss"
            )
        if (
            name == "LightGBM"
            and LIGHTGBM_AVAILABLE
        ):
            return LGBMClassifier(
                n_estimators=100,
                random_state=42,
                verbose=-1
            )
        if (
            name == "CatBoost"
            and CATBOOST_AVAILABLE
        ):
            return CatBoostClassifier(
                iterations=100,
                verbose=False,
                random_seed=42
            )
        if name not in models:
            raise ValueError(
                f"{name} is not available."
            )
        return models[name]
    def make_preprocessor(
        self,
        X,
        polynomial=False,
        degree=1
    ):
        numeric_columns = list(
            X.select_dtypes(
                include=np.number
            ).columns
        )
        categorical_columns = list(
            X.select_dtypes(
                exclude=np.number
            ).columns
        )
        transformers = []
        if numeric_columns:
            numeric_steps = [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
            if polynomial:
                numeric_steps.append(
                    (
                        "polynomial",
                        PolynomialFeatures(
                            degree=degree,
                            include_bias=False
                        )
                    )
                )
            numeric_steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )
            transformers.append(
                (
                    "numeric",
                    Pipeline(numeric_steps),
                    numeric_columns
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                )
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False
                                )
                            )
                        ]
                    ),
                    categorical_columns
                )
            )
        return ColumnTransformer(
            transformers=transformers
        )
    def create_polynomial_model(
        self,
        degree
    ):
        preprocessor = self.make_preprocessor(
            self.current_X,
            True,
            degree
        )
        return Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        )
    def find_best_polynomial_degree(
        self,
        X,
        y
    ):
        folds = (
            5
            if len(X) >= 50
            else 4
            if len(X) >= 20
            else 3
        )
        max_degree = (
            10
            if len(X) >= 10
            else 5
        )
        cv = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )
        self.polynomial_cv_results = {}
        best_degree = None
        best_score = np.inf
        for degree in range(
            1,
            max_degree + 1
        ):
            try:
                pipeline = self.create_polynomial_model(
                    degree
                )
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1
                )
                score = -scores.mean()
                if np.isfinite(score):
                    self.polynomial_cv_results[
                        degree
                    ] = score
                    if score < best_score:
                        best_score = score
                        best_degree = degree
            except Exception as error:
                print(
                    f"Polynomial degree {degree} failed: {error}"
                )
        if best_degree is None:
            raise ValueError(
                "Could not determine a suitable polynomial degree."
            )
        self.best_polynomial_degree = best_degree
        return best_degree
    def train_selected_model(self):
        try:
            X, y = self.prepare_data()
            if len(X) < 5:
                raise ValueError(
                    self.t("not_enough_rows")
                )
            if (
                self.selected_algorithm_type
                == "Classification"
                and len(np.unique(y)) < 2
            ):
                raise ValueError(
                    self.t("two_classes")
                )
            self.current_X = X
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
                if self.selected_algorithm_type
                == "Classification"
                else None
            )
            self.X_test = X_test
            self.y_test = y_test
            if (
                self.selected_algorithm
                == "Polynomial Regression"
            ):
                degree = self.find_best_polynomial_degree(
                    X_train,
                    y_train
                )
                self.model = self.create_polynomial_model(
                    degree
                )
            else:
                self.preprocessor = self.make_preprocessor(
                    X
                )
                self.model = Pipeline(
                    [
                        (
                            "preprocessor",
                            self.preprocessor
                        ),
                        (
                            "model",
                            self.create_model()
                        )
                    ]
                )
            self.model.fit(
                X_train,
                y_train
            )
            self.y_pred = self.model.predict(
                X_test
            )
            if self.selected_algorithm_type == "Regression":
                self.test_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        self.y_pred
                    )
                )
                self.test_mae = mean_absolute_error(
                    y_test,
                    self.y_pred
                )
                self.test_r2 = r2_score(
                    y_test,
                    self.y_pred
                )
            else:
                self.test_accuracy = np.mean(
                    np.asarray(y_test)
                    ==
                    np.asarray(self.y_pred)
                )
            self.show_page_five()
        except Exception as error:
            messagebox.showerror(
                self.t("model_error"),
                str(error)
            )
            self.show_page_four()
    def get_chart_title(self):
        if (
            self.selected_algorithm
            == "Polynomial Regression"
        ):
            if len(self.feature_columns) == 1:
                return self.t("automatic_curve")
            return self.t("actual_predicted")
        if self.selected_algorithm in [
            "Decision Tree Regressor",
            "Decision Tree"
        ]:
            return self.t("decision_tree_structure")
        if self.selected_algorithm in [
            "Random Forest Regressor",
            "Random Forest",
            "Extra Trees Regressor",
            "Extra Trees"
        ]:
            return self.t("tree_structure_ensemble")
        if self.selected_algorithm_type == "Regression":
            return self.t("actual_predicted")
        return self.t("confusion_matrix")
    def show_page_five(self):
        self.clear_page()
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(15, 5)
        )
        title = (
            self.t("automatic_polynomial")
            if self.selected_algorithm
            == "Polynomial Regression"
            else self.t("model_visualization")
        )
        ctk.CTkLabel(
            header,
            text=title,
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        subtitle = (
            f"{self.selected_algorithm}"
            f"  •  "
            f"{self.t('target_label')} "
            f"{self.target_column}"
        )
        if self.selected_algorithm == "Polynomial Regression":
            subtitle += (
                f"  •  "
                f"{self.t('best_degree')} "
                f"{self.best_polynomial_degree}"
            )
        ctk.CTkLabel(
            header,
            text=subtitle,
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center"
        )
        content = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=35,
            pady=8
        )
        chart_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )
        chart_card.pack(
            fill="both",
            expand=True
        )
        ctk.CTkLabel(
            chart_card,
            text=self.get_chart_title(),
            font=self.title_font(18),
            text_color=TEXT_PRIMARY
        ).pack(
            pady=(10, 0)
        )
        self.create_model_chart(
            chart_card
        )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=75
        )
        bottom.pack(
            side="bottom",
            fill="x",
            pady=(0, 5)
        )
        ctk.CTkLabel(
            bottom,
            text=self.calculate_metrics(),
            font=self.body_font(11),
            text_color=TEXT_SECONDARY
        ).pack(
            pady=(0, 5)
        )
        row = ctk.CTkFrame(
            bottom,
            fg_color="transparent"
        )
        row.pack(
            anchor="center"
        )
        ctk.CTkButton(
            row,
            text=self.t("back_algorithms"),
            width=180,
            height=40,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13),
            command=self.back_to_algorithms
        ).pack(
            side="left",
            padx=5
        )
        ctk.CTkButton(
            row,
            text=self.t("continue"),
            width=150,
            height=40,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(13),
            command=self.show_prediction_page
        ).pack(
            side="left",
            padx=5
        )
    def back_to_algorithms(self):
        self.selected_algorithm = None
        self.selected_algorithm_type = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.test_rmse = None
        self.test_mae = None
        self.test_r2 = None
        self.test_accuracy = None
        self.best_polynomial_degree = None
        self.polynomial_cv_results = {}
        self.show_page_four()
    def create_model_chart(
        self,
        parent
    ):
        if self.selected_algorithm == "Polynomial Regression":
            if len(self.feature_columns) == 1:
                self.draw_polynomial_curve(parent)
            else:
                self.draw_regression_chart(parent)
        elif self.selected_algorithm_type == "Regression":
            if self.selected_algorithm == "Decision Tree Regressor":
                self.draw_decision_tree(
                    parent,
                    False
                )
            elif self.selected_algorithm in [
                "Random Forest Regressor",
                "Extra Trees Regressor"
            ]:
                self.draw_ensemble_tree(
                    parent,
                    False
                )
            else:
                self.draw_regression_chart(parent)
        else:
            if self.selected_algorithm == "Decision Tree":
                self.draw_decision_tree(
                    parent,
                    True
                )
            elif self.selected_algorithm in [
                "Random Forest",
                "Extra Trees"
            ]:
                self.draw_ensemble_tree(
                    parent,
                    True
                )
            else:
                self.draw_classification_chart(parent)
    def draw_regression_chart(
        self,
        parent
    ):
        figure, ax = plt.subplots(
            figsize=(8, 5),
            dpi=90
        )
        figure.patch.set_facecolor(
            CARD_COLOR
        )
        ax.set_facecolor(
            CARD_COLOR
        )
        ax.scatter(
            self.y_test,
            self.y_pred,
            alpha=0.7
        )
        minimum = min(
            np.min(self.y_test),
            np.min(self.y_pred)
        )
        maximum = max(
            np.max(self.y_test),
            np.max(self.y_pred)
        )
        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            linewidth=2
        )
        ax.set_xlabel(
            self.t("actual_target"),
            color=TEXT_SECONDARY
        )
        ax.set_ylabel(
            self.t("predicted_target"),
            color=TEXT_SECONDARY
        )
        ax.set_title(
            self.t("actual_vs_predicted"),
            color=TEXT_PRIMARY
        )
        ax.tick_params(
            colors=TEXT_SECONDARY
        )
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(alpha=0.15)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )
    def draw_polynomial_curve(
        self,
        parent
    ):
        feature = self.feature_columns[0]
        temp = self.data[
            [
                feature,
                self.target_column
            ]
        ].copy()
        temp[feature] = pd.to_numeric(
            temp[feature],
            errors="coerce"
        )
        temp[
            self.target_column
        ] = pd.to_numeric(
            temp[
                self.target_column
            ],
            errors="coerce"
        )
        temp = temp.dropna()
        if len(temp) < 2:
            self.draw_regression_chart(
                parent
            )
            return
        x = temp[feature].values
        y = temp[self.target_column].values
        x_values = np.linspace(
            np.min(x),
            np.max(x),
            400
        )
        y_curve = self.model.predict(
            pd.DataFrame(
                {
                    feature: x_values
                }
            )
        )
        figure, ax = plt.subplots(
            figsize=(8, 5),
            dpi=90
        )
        figure.patch.set_facecolor(
            CARD_COLOR
        )
        ax.set_facecolor(
            CARD_COLOR
        )
        ax.scatter(
            x,
            y,
            alpha=0.55,
            label=self.t("actual_data")
        )
        ax.plot(
            x_values,
            y_curve,
            linewidth=2.5,
            label=(
                f"{self.t('degree_label')} "
                f"{self.best_polynomial_degree}"
            )
        )
        ax.set_xlabel(
            feature,
            color=TEXT_SECONDARY
        )
        ax.set_ylabel(
            self.target_column,
            color=TEXT_SECONDARY
        )
        ax.set_title(
            (
                f"{self.t('automatic_degree_selected')} "
                f"{self.best_polynomial_degree}"
            ),
            color=TEXT_PRIMARY
        )
        ax.tick_params(
            colors=TEXT_SECONDARY
        )
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(alpha=0.15)
        legend = ax.legend(
            frameon=False
        )
        if legend:
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )
    def draw_classification_chart(
        self,
        parent
    ):
        figure, ax = plt.subplots(
            figsize=(7, 5),
            dpi=90
        )
        figure.patch.set_facecolor(
            CARD_COLOR
        )
        ax.set_facecolor(
            CARD_COLOR
        )
        matrix = confusion_matrix(
            self.y_test,
            self.y_pred
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix
        )
        display.plot(
            ax=ax,
            colorbar=False
        )
        ax.set_title(
            self.t("classification_confusion"),
            color=TEXT_PRIMARY
        )
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )
    def draw_decision_tree(
        self,
        parent,
        classifier=False
    ):
        model = self.model.named_steps[
            "model"
        ]
        figure, ax = plt.subplots(
            figsize=(10, 5),
            dpi=80
        )
        figure.patch.set_facecolor(
            CARD_COLOR
        )
        ax.set_facecolor(
            CARD_COLOR
        )
        try:
            feature_names = self.model.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            feature_names = [
                str(x)
                for x in feature_names
            ]
        except Exception:
            feature_names = None
        if classifier:
            plot_tree(
                model,
                ax=ax,
                max_depth=3,
                feature_names=feature_names,
                class_names=[
                    str(x)
                    for x in model.classes_
                ],
                filled=False,
                fontsize=6
            )
        else:
            plot_tree(
                model,
                ax=ax,
                max_depth=3,
                feature_names=feature_names,
                filled=False,
                fontsize=6
            )
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=5,
            pady=3
        )
    def draw_ensemble_tree(
        self,
        parent,
        classifier=False
    ):
        model = self.model.named_steps[
            "model"
        ]
        tree = model.estimators_[0]
        figure, ax = plt.subplots(
            figsize=(10, 5),
            dpi=80
        )
        figure.patch.set_facecolor(
            CARD_COLOR
        )
        ax.set_facecolor(
            CARD_COLOR
        )
        try:
            feature_names = self.model.named_steps[
                "preprocessor"
            ].get_feature_names_out()
            feature_names = [
                str(x)
                for x in feature_names
            ]
        except Exception:
            feature_names = None
        if classifier:
            plot_tree(
                tree,
                ax=ax,
                max_depth=3,
                feature_names=feature_names,
                class_names=[
                    str(x)
                    for x in model.classes_
                ],
                filled=False,
                fontsize=6
            )
        else:
            plot_tree(
                tree,
                ax=ax,
                max_depth=3,
                feature_names=feature_names,
                filled=False,
                fontsize=6
            )
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(
            figure,
            master=parent
        )
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=5,
            pady=3
        )
    def calculate_metrics(self):
        if self.selected_algorithm_type == "Regression":
            result = (
                f'{self.t("r2")}: {self.test_r2:.4f}'
                f'    |    '
                f'{self.t("rmse")}: {self.test_rmse:.4f}'
                f'    |    '
                f'{self.t("mae")}: {self.test_mae:.4f}'
            )
            if self.selected_algorithm == "Polynomial Regression":
                result += (
                    f'    |    '
                    f'{self.t("best_degree")} '
                    f'{self.best_polynomial_degree}'
                )
            return result
        return (
            f'{self.t("accuracy")}: '
            f'{self.test_accuracy:.2%}'
        )
    def show_prediction_page(self):
        self.clear_page()
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(15, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_input"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_description"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center"
        )
        content = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=100,
            pady=8
        )
        self.prediction_inputs = {}
        for column in self.feature_columns:
            row = ctk.CTkFrame(
                content,
                fg_color=CARD_COLOR,
                corner_radius=12,
                border_width=1,
                border_color=BORDER
            )
            row.pack(
                fill="x",
                pady=4
            )
            ctk.CTkLabel(
                row,
                text=str(column),
                width=200,
                anchor=(
                    "e"
                    if self.current_language == "fa"
                    else "w"
                ),
                font=self.body_font(13),
                text_color=TEXT_PRIMARY
            ).pack(
                side="left",
                padx=15,
                pady=10
            )
            dtype = self.data[column].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                entry = ctk.CTkEntry(
                    row,
                    height=36,
                    fg_color=INPUT_COLOR,
                    border_color=BORDER,
                    text_color=TEXT_PRIMARY,
                    placeholder_text=self.t("numeric_placeholder"),
                    font=self.body_font(12)
                )
                entry.pack(
                    side="right",
                    fill="x",
                    expand=True,
                    padx=15,
                    pady=8
                )
                self.prediction_inputs[
                    column
                ] = (
                    "numeric",
                    entry
                )
            else:
                values = sorted(
                    self.data[
                        column
                    ].dropna().astype(str).unique().tolist()
                )
                combo = ctk.CTkComboBox(
                    row,
                    values=values,
                    height=36,
                    fg_color=INPUT_COLOR,
                    border_color=BORDER,
                    button_color=PRIMARY,
                    button_hover_color=PRIMARY_HOVER,
                    text_color=TEXT_PRIMARY,
                    dropdown_fg_color=CARD_COLOR,
                    dropdown_hover_color=BUTTON_HOVER,
                    font=self.body_font(12),
                    dropdown_font=self.body_font(12)
                )
                combo.pack(
                    side="right",
                    fill="x",
                    expand=True,
                    padx=15,
                    pady=8
                )
                if values:
                    combo.set(values[0])
                self.prediction_inputs[
                    column
                ] = (
                    "categorical",
                    combo
                )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=65
        )
        bottom.pack(
            side="bottom",
            fill="x",
            pady=(0, 8)
        )
        ctk.CTkButton(
            bottom,
            text=self.t("predict_target"),
            width=180,
            height=42,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(14),
            command=self.predict_target
        ).pack(
            anchor="center"
        )
    def predict_target(self):
        values = {}
        try:
            for column in self.feature_columns:
                input_type, widget = (
                    self.prediction_inputs[column]
                )
                value = widget.get().strip()
                if value == "":
                    raise ValueError(
                        f'{self.t("enter_value")} "{column}".'
                    )
                if input_type == "numeric":
                    value = float(value)
                    if pd.api.types.is_integer_dtype(
                        self.data[column].dtype
                    ):
                        value = int(value)
                values[column] = value
            input_data = pd.DataFrame(
                [values],
                columns=self.feature_columns
            )
            prediction = self.model.predict(
                input_data
            )[0]
            probability = None
            if self.selected_algorithm_type == "Classification":
                if self.target_encoder is not None:
                    try:
                        prediction = (
                            self.target_encoder
                            .inverse_transform(
                                [int(prediction)]
                            )[0]
                        )
                    except Exception:
                        pass
                try:
                    probabilities = self.model.predict_proba(
                        input_data
                    )[0]
                    probability = float(
                        probabilities[
                            np.argmax(probabilities)
                        ]
                    )
                except Exception:
                    probability = None
            if isinstance(
                prediction,
                (
                    np.integer,
                    int
                )
            ):
                prediction_text = str(
                    prediction
                )
            elif isinstance(
                prediction,
                (
                    np.floating,
                    float
                )
            ):
                prediction_text = f"{prediction:.6f}"
            else:
                prediction_text = str(
                    prediction
                )
            self.last_prediction = prediction
            self.last_prediction_text = prediction_text
            self.last_prediction_probability = probability
            self.last_prediction_values = values.copy()
            self.current_result_saved = False
            self.current_result_label = None
            self.show_prediction_result()
        except Exception as error:
            messagebox.showerror(
                self.t("prediction_error"),
                str(error)
            )
    def show_history_page(self):
        self.clear_page()
        self.history_selected = {}
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(20, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_history"),
            font=self.title_font(25),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("previous_predictions"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center",
            pady=(3, 10)
        )
        if self.prediction_history:
            action_row = ctk.CTkFrame(
                self.main_container,
                fg_color="transparent"
            )
            action_row.pack(
                fill="x",
                padx=70,
                pady=(0, 8)
            )
            self.history_select_all_button = ctk.CTkButton(
                action_row,
                text=self.t("select_all_history"),
                width=145,
                height=38,
                corner_radius=9,
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=TEXT_PRIMARY,
                font=self.body_font(12),
                command=self.toggle_select_all_history
            )
            self.history_select_all_button.pack(
                side="left",
                padx=4
            )
            ctk.CTkButton(
                action_row,
                text=self.t("delete_selected"),
                width=155,
                height=38,
                corner_radius=9,
                fg_color=ERROR,
                hover_color="#B83C3C",
                text_color="#FFFFFF",
                font=self.body_font(12),
                command=self.delete_selected_history
            ).pack(
                side="left",
                padx=4
            )
            ctk.CTkButton(
                action_row,
                text=self.t("delete_all"),
                width=120,
                height=38,
                corner_radius=9,
                fg_color=BUTTON_COLOR,
                hover_color=BUTTON_HOVER,
                text_color=ERROR,
                font=self.body_font(12),
                command=self.delete_all_history
            ).pack(
                side="right",
                padx=4
            )
        content = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=70,
            pady=10
        )
        if not self.prediction_history:
            empty_card = ctk.CTkFrame(
                content,
                fg_color=CARD_COLOR,
                corner_radius=16,
                border_width=1,
                border_color=BORDER,
                height=190
            )
            empty_card.pack(
                fill="x",
                pady=20
            )
            empty_card.pack_propagate(False)
            ctk.CTkLabel(
                empty_card,
                text="◇",
                font=self.title_font(40),
                text_color=PRIMARY
            ).pack(
                pady=(28, 3)
            )
            ctk.CTkLabel(
                empty_card,
                text=self.t("no_predictions"),
                font=self.title_font(18),
                text_color=TEXT_PRIMARY
            ).pack()
            ctk.CTkLabel(
                empty_card,
                text=self.t("history_message"),
                font=self.body_font(12),
                text_color=TEXT_SECONDARY
            ).pack(
                pady=(5, 0)
            )
        else:
            for record in reversed(self.prediction_history):
                card = ctk.CTkFrame(
                    content,
                    fg_color=CARD_COLOR,
                    corner_radius=14,
                    border_width=1,
                    border_color=BORDER
                )
                card.pack(
                    fill="x",
                    pady=5
                )
                select_var = ctk.BooleanVar(value=False)
                self.history_selected[id(record)] = (select_var, record)
                ctk.CTkCheckBox(
                    card,
                    text="",
                    variable=select_var,
                    width=25,
                    height=25,
                    checkbox_width=20,
                    checkbox_height=20,
                    fg_color=PRIMARY,
                    hover_color=PRIMARY_HOVER,
                    border_color=BORDER
                ).pack(
                    side="left",
                    padx=(12, 4)
                )
                info_frame = ctk.CTkFrame(
                    card,
                    fg_color="transparent"
                )
                info_frame.pack(
                    side="left",
                    fill="x",
                    expand=True
                )
                ctk.CTkButton(
                    info_frame,
                    text=record.get("label", record.get("dataset", "History")),
                    height=45,
                    corner_radius=10,
                    fg_color="transparent",
                    hover_color=BUTTON_HOVER,
                    text_color=TEXT_PRIMARY,
                    font=self.title_font(15),
                    anchor=self.text_anchor(),
                    command=lambda r=record: self.show_saved_prediction(r)
                ).pack(
                    fill="x",
                    padx=5,
                    pady=(7, 0)
                )
                summary = (
                    f'{record.get("algorithm", "")}  •  '
                    f'{record.get("time", "")}'
                )
                ctk.CTkLabel(
                    info_frame,
                    text=summary,
                    font=self.small_font(10),
                    text_color=TEXT_MUTED,
                    anchor=self.text_anchor()
                ).pack(
                    fill="x",
                    padx=10,
                    pady=(0, 8)
                )
                ctk.CTkButton(
                    card,
                    text="✕",
                    width=38,
                    height=38,
                    corner_radius=8,
                    fg_color="transparent",
                    hover_color=BUTTON_HOVER,
                    text_color=ERROR,
                    font=self.body_font(16),
                    command=lambda r=record: self.delete_single_history(r)
                ).pack(
                    side="right",
                    padx=10
                )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=65
        )
        bottom.pack(
            side="bottom",
            fill="x"
        )
        ctk.CTkButton(
            bottom,
            text=self.t("back_nexora"),
            width=180,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13),
            command=self.show_page_one
        ).pack(
            anchor="center",
            pady=(0, 8)
        )
    def toggle_select_all_history(self):
        if not self.prediction_history:
            return
        all_selected = all(
            var.get()
            for var, record in self.history_selected.values()
        )
        new_value = not all_selected
        for var, record in self.history_selected.values():
            var.set(new_value)
        if hasattr(self, "history_select_all_button"):
            self.history_select_all_button.configure(
                text=(
                    self.t("deselect_all_history")
                    if new_value
                    else self.t("select_all_history")
                )
            )
    def delete_single_history(self, record):
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        self.prediction_history = [
            item
            for item in self.prediction_history
            if item is not record
        ]
        self.show_history_page()
    def delete_selected_history(self):
        selected_records = [
            record
            for var, record in self.history_selected.values()
            if var.get()
        ]
        if not selected_records:
            messagebox.showwarning(
                self.t("delete_history_title"),
                self.t("no_history_selected")
            )
            return
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        selected_ids = {
            id(record)
            for record in selected_records
        }
        self.prediction_history = [
            record
            for record in self.prediction_history
            if id(record) not in selected_ids
        ]
        self.show_history_page()
    def delete_all_history(self):
        if not self.prediction_history:
            return
        confirmed = messagebox.askyesno(
            self.t("delete_history_title"),
            self.t("delete_history_message")
        )
        if not confirmed:
            return
        self.prediction_history.clear()
        self.show_history_page()
    def show_saved_prediction(self, record):
        details = [
            (
                self.t("target"),
                record.get("target", "-")
            ),
            (
                self.t("algorithm"),
                record.get("algorithm", "-")
            ),
            (
                self.t("type"),
                record.get("task", "-")
            ),
            (
                self.t("prediction"),
                record.get("prediction", "-")
            ),
            (
                self.t("time"),
                record.get("time", "-")
            )
        ]
        if record.get("task") == "Regression":
            if record.get("rmse") is not None:
                details.append(
                    (
                        self.t("rmse"),
                        f'{record["rmse"]:.4f}'
                    )
                )
            if record.get("mae") is not None:
                details.append(
                    (
                        self.t("mae"),
                        f'{record["mae"]:.4f}'
                    )
                )
            if record.get("r2") is not None:
                details.append(
                    (
                        self.t("r2"),
                        f'{record["r2"]:.4f}'
                    )
                )
        else:
            if record.get("accuracy") is not None:
                details.append(
                    (
                        self.t("accuracy"),
                        f'{record["accuracy"]:.2%}'
                    )
                )
            if record.get("probability") is not None:
                details.append(
                    (
                        self.t("confidence"),
                        f'{record["probability"]:.2%}'
                    )
                )
        message = "\n\n".join(
            f"{label}: {value}"
            for label, value in details
        )
        messagebox.showinfo(
            record.get(
                "label",
                record.get(
                    "dataset",
                    "History"
                )
            ),
            message
        )
    def show_prediction_result(self):
        self.clear_page()
        header = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent"
        )
        header.pack(
            fill="x",
            pady=(12, 5)
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_result"),
            font=self.title_font(26),
            text_color=PRIMARY
        ).pack(
            anchor="center"
        )
        ctk.CTkLabel(
            header,
            text=self.t("prediction_processed"),
            font=self.body_font(12),
            text_color=TEXT_SECONDARY
        ).pack(
            anchor="center"
        )
        content = ctk.CTkScrollableFrame(
            self.main_container,
            fg_color="transparent",
            scrollbar_button_color=BUTTON_COLOR,
            scrollbar_button_hover_color=BUTTON_HOVER
        )
        content.pack(
            fill="both",
            expand=True,
            padx=50,
            pady=5
        )
        result_card = ctk.CTkFrame(
            content,
            fg_color=CARD_COLOR,
            corner_radius=18,
            border_width=1,
            border_color=BORDER
        )
        result_card.pack(
            fill="x",
            padx=5,
            pady=5
        )
        ctk.CTkLabel(
            result_card,
            text=self.t("prediction").upper(),
            font=self.body_font(10),
            text_color=TEXT_SECONDARY
        ).pack(
            pady=(15, 2)
        )
        ctk.CTkLabel(
            result_card,
            text=str(self.target_column),
            font=self.body_font(13),
            text_color=TEXT_SECONDARY
        ).pack()
        ctk.CTkLabel(
            result_card,
            text=str(
                self.last_prediction_text
            ),
            font=self.title_font(32),
            text_color=PRIMARY
        ).pack(
            pady=(2, 8)
        )
        if self.selected_algorithm_type == "Classification":
            if self.last_prediction_probability is not None:
                confidence_text = (
                    f'{self.t("model_confidence")} '
                    f'{self.last_prediction_probability:.2%}'
                )
            else:
                confidence_text = self.t(
                    "probability_unavailable"
                )
            ctk.CTkLabel(
                result_card,
                text=confidence_text,
                font=self.body_font(13),
                text_color=TEXT_PRIMARY
            ).pack(
                pady=(0, 10)
            )
        else:
            ctk.CTkLabel(
                result_card,
                text=(
                    f'{self.t("estimated_error")} '
                    f'{self.test_rmse:.4f}'
                ),
                font=self.body_font(13),
                text_color=TEXT_PRIMARY
            ).pack(
                pady=(0, 3)
            )
            ctk.CTkLabel(
                result_card,
                text=(
                    f'{self.t("average_error")} '
                    f'{self.test_mae:.4f}'
                ),
                font=self.body_font(11),
                text_color=TEXT_SECONDARY
            ).pack(
                pady=(0, 8)
            )
        ctk.CTkLabel(
            result_card,
            text=self.t("input_features"),
            font=self.body_font(12),
            text_color=TEXT_PRIMARY
        ).pack(
            pady=(8, 5)
        )
        feature_frame = ctk.CTkFrame(
            result_card,
            fg_color=INPUT_COLOR,
            corner_radius=8
        )
        feature_frame.pack(
            fill="x",
            padx=25,
            pady=(0, 10)
        )
        for column, value in (
            self.last_prediction_values.items()
        ):
            row = ctk.CTkFrame(
                feature_frame,
                fg_color=ROW_COLOR,
                corner_radius=5
            )
            row.pack(
                fill="x",
                padx=6,
                pady=2
            )
            ctk.CTkLabel(
                row,
                text=str(column),
                text_color=TEXT_SECONDARY,
                anchor=(
                    "e"
                    if self.current_language == "fa"
                    else "w"
                ),
                font=self.small_font(10)
            ).pack(
                side="left",
                padx=8,
                pady=4
            )
            ctk.CTkLabel(
                row,
                text=str(value),
                text_color=TEXT_PRIMARY,
                anchor="e",
                font=self.small_font(10)
            ).pack(
                side="right",
                padx=8,
                pady=4
            )
        ctk.CTkLabel(
            result_card,
            text=(
                f'{self.t("algorithm")}: '
                f'{self.selected_algorithm}'
            ),
            font=self.small_font(10),
            text_color=TEXT_MUTED
        ).pack(
            pady=(2, 12)
        )
        bottom = ctk.CTkFrame(
            self.main_container,
            fg_color=BG_COLOR,
            height=70
        )
        bottom.pack(
            side="bottom",
            fill="x"
        )
        bottom.pack_propagate(False)
        row = ctk.CTkFrame(
            bottom,
            fg_color="transparent"
        )
        row.pack(
            anchor="center",
            pady=12
        )
        ctk.CTkButton(
            row,
            text=self.t("return_start"),
            width=180,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13),
            command=self.try_return_to_start
        ).pack(
            side="left",
            padx=6
        )
        ctk.CTkButton(
            row,
            text=self.t("save_result"),
            width=180,
            height=42,
            corner_radius=9,
            fg_color=PRIMARY,
            hover_color=PRIMARY_HOVER,
            text_color="#111111",
            font=self.body_font(13),
            command=self.save_current_result
        ).pack(
            side="left",
            padx=6
        )
        ctk.CTkButton(
            row,
            text=self.t("finish"),
            width=150,
            height=42,
            corner_radius=9,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            font=self.body_font(13),
            command=self.try_finish_program
        ).pack(
            side="left",
            padx=6
        )
    def save_current_result(self):
        dialog = ctk.CTkInputDialog(
            text=self.t("save_result_prompt"),
            title=self.t("save_result_label")
        )
        label = dialog.get_input()
        if label is None:
            return
        label = label.strip()
        if not label:
            messagebox.showwarning(
                self.t("save_result_label"),
                self.t("save_result_empty")
            )
            return
        if any(
            item.get("label", "").strip().lower() == label.lower()
            for item in self.prediction_history
        ):
            messagebox.showerror(
                self.t("save_result_label"),
                "This name has already been used."
            )
            return
        self.save_prediction_history(
            self.last_prediction_values,
            self.last_prediction_text,
            self.last_prediction_probability,
            label
        )
        self.current_result_saved = True
        self.current_result_label = label
        messagebox.showinfo(
            self.t("save_result_label"),
            self.t("save_result_success")
        )
    def confirm_unsaved_result(self):
        if self.current_result_saved:
            return True
        return messagebox.askyesno(
            self.t("unsaved_result_title"),
            self.t("unsaved_result_message")
        )
    def try_return_to_start(self):
        if self.confirm_unsaved_result():
            self.return_to_start()
    def try_finish_program(self):
        if self.confirm_unsaved_result():
            self.finish_program()
    def return_to_start(self):
        self.current_result_saved = False
        self.current_result_label = None
        self.selected_file = None
        self.data = None
        self.feature_columns = []
        self.target_column = None
        self.selected_algorithm = None
        self.selected_algorithm_type = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.preprocessor = None
        self.target_encoder = None
        self.test_rmse = None
        self.test_mae = None
        self.test_r2 = None
        self.test_accuracy = None
        self.best_polynomial_degree = None
        self.polynomial_cv_results = {}
        self.prediction_inputs = {}
        self.last_prediction_values = {}
        self.last_prediction = None
        self.last_prediction_text = None
        self.last_prediction_probability = None
        self.auto_selection_results = {}
        self.auto_selection_running = False
        self.auto_selection_thread = None
        self.loading_value = 0
        self.progress = None
        self.auto_selection_progress = None
        self.show_page_one()
    def finish_program(self):
        self.destroy()
if __name__ == "__main__":
    ctk.set_appearance_mode(
        "dark"
    )
    app = Nexora()
    app.mainloop()
