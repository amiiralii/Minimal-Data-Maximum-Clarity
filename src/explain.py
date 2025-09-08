import sys
from ezr import *
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from FS_experiment import *
from sklearn.inspection import permutation_importance
from dalex import Explainer

def global_explanation(X_train, X_test, y_train, y_test, features):
    ## Global Explanation
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    result = permutation_importance(
        estimator=model,            # trained model
        X=X_test,                   # validation features
        y=y_test.values.ravel(),                   # validation labels
        n_repeats=10,              # number of shuffles per feature
        random_state=42,           # for reproducibility
        scoring='neg_mean_squared_error',        # scoring metric
        n_jobs=-1                  # use all CPUs
    )
    res = result.importances_mean

    importances = pd.DataFrame({
        'feature': sorted(features),
        'PI': res,
    })
    the.Stop = 32
    model = actLearn(data,shuffle=True)
    root = tree(model.best.rows + model.rest.rows,data)
    print("Global EZR Tree:")
    showTree(root)
    ezr_xplain, _ = run_ezr_explainer(data, print_budget=False)
    del ezr_xplain['explainer']
    del ezr_xplain['run_time']
    importances['ezr'] = dict(sorted(ezr_xplain.items())).values()

    anova_xplain = run_anova_explainer(data, features)
    del anova_xplain['explainer']
    del anova_xplain['run_time']
    importances['anova'] = dict(sorted(anova_xplain.items())).values()

    shap_xplain = run_shap_explainer(data, test_data, features)
    del shap_xplain['explainer']
    del shap_xplain['run_time']
    importances['shap'] = dict(sorted(shap_xplain.items())).values()

    print("Feature importance values by methods:")
    print(importances)

    # # Create bar chart of feature importances
    # plt.figure(figsize=(8, 4))
    # # Get the explainer columns (excluding 'feature')
    # explainer_cols = [col for col in importances.columns if col != 'feature']
    # # Set up the bar positions
    # x = np.arange(len(importances))
    # width = 0.4  # Width of bars (reduced from 0.2 to make them thinner)
    # offset = np.linspace(-(len(explainer_cols)-1)/2, (len(explainer_cols)-1)/2, len(explainer_cols))
    # # Create bars for each explainer
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each explainer
    # for i, (explainer, color) in enumerate(zip(explainer_cols, colors)):
    #     values = importances[explainer].fillna(0)  # Fill NaN values with 0
    #     plt.bar(x + offset[i] * width, values*1000, width, label=explainer, color=color, alpha=0.8)
    # # Customize the plot
    # plt.ylabel('Global Feature Importance Score', fontsize=10)
    # plt.xticks(x, importances['feature'], rotation=45, ha='right', fontsize=8)
    # plt.grid(axis='y', alpha=0.3)
    # plt.tight_layout()

    # # Save the plot
    # plt.savefig(f'explanations/feature_importance_comparison_{sys.argv[1].split("/")[-1][:-4]}.png', 
    #             dpi=300, bbox_inches='tight')

def run_local_explainer(X_train, X_test, y_train, y_test, features, idx):
    """Run Local explainer and return feature importance"""
    
    # Preprocess data
    le = LabelEncoder()
    for c in features:
        if c[0].isupper():
            X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
            X_test[c] = pd.to_numeric(X_test[c], errors='coerce')
        else:
            X_train[c] = X_train[c].astype('category')
            X_train[c] = le.fit_transform(X_train[c])
            X_test[c] = X_test[c].astype('category')
            X_test[c] = le.fit_transform(X_test[c])
    
    # Train LGBM model
    model = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        num_leaves=8,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        min_data_in_leaf=2,
        min_data_in_bin=1,
        max_bin=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        force_col_wise=True,
        verbose=-1,
        random_state=42
    )
    model.fit(X_train[features], y_train)
    
    explainee = X_test.iloc[idx]

    print('-------BreakDown:-------')
    explainer = Explainer(model, X_train, y_train, label="my_model")
    bd = explainer.predict_parts(
    new_observation = explainee,
    type            = "break_down",    # select the BreakDown method
    B               = 0                # disable bootstrapping for exact path
    )
    bd.plot()
    plt.savefig(f"explanations/{sys.argv[1].split('/')[-1][:-4]}_breakdown.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(bd.result)


    print('-------SHAP:-------')
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train[features])
    
    # Create SHAP plots
    shap.summary_plot(
        shap_values,
        X_train[features],
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"explanations/{sys.argv[1].split("/")[-1][:-4]}_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.clf()
    shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=explainee
    ),
    show=False
    )
    plt.tight_layout()
    plt.savefig(f"explanations/{sys.argv[1].split("/")[-1][:-4]}_shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.clf()

    print('-------LIME:-------')
    # LIME explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train[features].values,
        feature_names=features,
        categorical_features=[c for c in features if not c[0].isupper()],
        mode='regression',
        discretize_continuous=True
    )
    
    instance = explainee.values.reshape(1, -1)
    true_value = y_test.iloc[idx]
    predicted_value = model.predict(instance)[0]
    
    print(f"True target = {true_value['d2h']}, Model prediction = {predicted_value}")
    
    exp = explainer.explain_instance(
        data_row=instance.flatten(),
        predict_fn=model.predict,
        num_features=6
    )
    
    print("LIME Explain:")
    for feature, weight in exp.as_list():
        direction = "↑" if weight > 0 else "↓"
        print(f"{feature:>20}: {weight:+.3f} ({direction})")
    
    exp.save_to_file(f'explanations/{sys.argv[1].split("/")[-1][:-4]}_lime.html')
    return


random_seed = 42
dataset = sys.argv[1]
raw_data = Data(csv(dataset))
data, test_data = split_data(raw_data)
stp = min(30, len(data.rows) // 10)
the.Stop = stp
the.acq = "xploit"

cols = [d.txt for d in data.cols.all]
targets = [c for c in cols if c[-1] in ["+", "-"]]
features = [c for c in cols if c[-1] not in ["+", "-", "X"]]

X_train = pd.DataFrame(data.rows, columns=cols)
X_test = pd.DataFrame(test_data.rows, columns=cols)
X_train = X_train[features]
X_test = X_test[features]
y_train = pd.DataFrame([ydist(row, data) for row in data.rows], columns=["d2h"])
y_test = pd.DataFrame([ydist(row, data) for row in test_data.rows], columns=["d2h"])

print("--------- Global Explanation ---------")
global_explanation(X_train, X_test, y_train, y_test, features)

print("--------- Local Explanation ---------")
idx = 16 ## Random instance to explain the decision
print("Instance to explain:")
print(X_test.iloc[idx])

the.Stop = 32
model = actLearn(data)
root = tree(model.best.rows + model.rest.rows,data)
pred = leaf(root, test_data.rows[idx])
run_local_explainer(X_train, X_test, y_train, y_test, features, idx)