def plot_loss_valid():
    from matplotlib import pyplot
    from sklearn.metrics import accuracy_score
    from numpy import loadtxt

    data_interface = generate_data_interface("psiK_1000.root")
    data = generate_data_mixing(data_interface)
    X_train, y_train, X_test, y_test = generate_prepared_data(data)
    # fit model no training data
    model = XGBClassifier(use_label_encoder=False)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train,
        y_train,
        eval_metric=["error", "logloss"],
        eval_set=eval_set,
        verbose=False,
    )

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results["validation_0"]["error"])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
    ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
    ax.legend()

    pyplot.ylabel("Log Loss")
    pyplot.title("XGBoost Log Loss")
    pyplot.show()

    # plot classification error
    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(x_axis, results["validation_0"]["error"], label="Train")
    ax.plot(x_axis, results["validation_1"]["error"], label="Test")
    ax.legend()

    pyplot.ylabel("Classification Error")
    pyplot.title("XGBoost Classification Error")
    pyplot.show()


def roc_curve_error(X_train, y_train, X_test, y_test, default=False, ratio=1):
    n_repeats = 10
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=101)
    folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

    metrics = ["auc", "fpr", "tpr", "thresholds"]
    results = {
        "train": {m: [] for m in metrics},
        "val": {m: [] for m in metrics},
        "test": {m: [] for m in metrics},
    }
    if default:
        params = {
            "early_stopping_rounds": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "silent": True,
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "base_score": 0.5,
            "use_label_encoder": False,
            "random_state": 42,
            "seed": 42,
        }

    else:
        params = {
            "early_stopping_rounds": 20,
            "max_depth": 9,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "silent": True,
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "base_score": 0.5,
            "use_label_encoder": False,
            "random_state": 42,
            "seed": 42,
        }

    dtest = (X_test, y_test)
    for train, test in tqdm(folds, total=len(folds)):
        dtrain = (X_train.iloc[train, :], y_train.iloc[train])
        dval = (X_train.iloc[test, :], y_train.iloc[test])

        model = (
            XGBClassifier()
            .set_params(**params)
            .fit(dtrain[0], dtrain[1], eval_set=[dval], verbose=0)
        )
        sets = [dtrain, dval, dtest]
        for i, ds in enumerate(results.keys()):
            y_preds = model.predict_proba(sets[i][0])[:, 1]
            labels = sets[i][1]
            fpr, tpr, thresholds = roc_curve(labels, y_preds)
            results[ds]["fpr"].append(fpr)
            results[ds]["tpr"].append(tpr)
            results[ds]["thresholds"].append(thresholds)
            results[ds]["auc"].append(roc_auc_score(labels, y_preds))

    c_fill_train = "rgba(128, 252, 128, 0.2)"
    c_line_train = "rgba(128, 152, 128, 0.5)"
    c_line_main_train = "rgba(128, 0, 128, 1.0)"

    c_fill_val = "rgba(52, 152, 0, 0.2)"
    c_line_val = "rgba(52, 152, 0, 0.5)"
    c_line_main_val = "rgba(41, 128, 0, 1.0)"

    c_fill_test = "rgba(0, 152, 219, 0.2)"
    c_line_test = "rgba(0, 152, 219, 0.5)"
    c_line_main_test = "rgba(0, 128, 185, 1.0)"

    c_grid = "rgba(189, 195, 199, 0.5)"
    c_annot = "rgba(149, 165, 166, 0.5)"
    c_highlight = "rgba(192, 57, 43, 1.0)"
    fpr_mean = np.linspace(0, 1, 100)

    def tp_rates(kind, results):
        interp_tprs = []
        for i in range(n_repeats):
            fpr = results[kind]["fpr"][i]
            tpr = results[kind]["tpr"][i]
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std = 2 * np.std(interp_tprs, axis=0)
        tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
        tpr_lower = tpr_mean - tpr_std
        auc = np.mean(results[kind]["auc"])
        return tpr_upper, tpr_mean, tpr_lower, auc

    kind = "val"
    try:
        title = "ROC Curve with signal to background ratio of: {:.2f}".format(ratio)
    except:
        title = "ROC Curve"
    train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
        "train", results
    )
    val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
    test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates("test", results)
    fig = go.Figure(
        [
            go.Scatter(
                x=fpr_mean,
                y=train_tpr_upper,
                line=dict(color=c_line_train, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=fpr_mean,
                y=train_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_train,
                line=dict(color=c_line_train, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=fpr_mean,
                y=train_tpr_mean,
                line=dict(color=c_line_main_train, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Train_AUC: {train_auc:.3f}",
            ),
            go.Scatter(
                x=fpr_mean,
                y=val_tpr_upper,
                line=dict(color=c_line_val, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=fpr_mean,
                y=val_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_val,
                line=dict(color=c_line_val, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=fpr_mean,
                y=val_tpr_mean,
                line=dict(color=c_line_main_val, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Val_AUC: {val_auc:.3f}",
            ),
            go.Scatter(
                x=fpr_mean,
                y=test_tpr_upper,
                line=dict(color=c_line_test, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=fpr_mean,
                y=test_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_test,
                line=dict(color=c_line_test, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=fpr_mean,
                y=test_tpr_mean,
                line=dict(color=c_line_main_test, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Test_AUC: {test_auc:.3f}",
            ),
        ]
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(
        # title=title,
        template="plotly_white",
        title_x=0.5,
        xaxis_title="1 - Specificity",
        yaxis_title="Sensitivity",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
        ),
        yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        ),
    )

    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor="black",
    )
    fig.update_xaxes(
        range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
    )
    # import os

    # if not os.path.exists("images"):
    #     os.mkdir("images")

    fig.write_image("images/roc_curve_test2.svg")
    fig.show()
    return results, model


def precision_recall(X_train, y_train, X_test, y_test, default=False, ratio=1):
    n_repeats = 10
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=101)
    folds = [(train, val_temp) for train, val_temp in cv.split(X_train, y_train)]

    metrics = ["ap", "precision", "recall", "thresholds"]
    results = {
        "train": {m: [] for m in metrics},
        "val": {m: [] for m in metrics},
        "test": {m: [] for m in metrics},
    }
    if default:
        params = {
            "early_stopping_rounds": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "silent": True,
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "base_score": 0.5,
            "use_label_encoder": False,
            "random_state": 42,
            "seed": 42,
        }

    else:
        params = {
            "early_stopping_rounds": 20,
            "max_depth": 9,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "silent": True,
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "n_jobs": -1,
            "nthread": None,
            "gamma": 0,
            "min_child_weight": 1,
            "max_delta_step": 0,
            "subsample": 1,
            "colsample_bytree": 1,
            "colsample_bylevel": 1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "base_score": 0.5,
            "use_label_encoder": False,
            "random_state": 42,
            "seed": 42,
        }

    dtest = (X_test, y_test)
    for train, test in tqdm(folds, total=len(folds)):
        dtrain = (X_train.iloc[train, :], y_train.iloc[train])
        dval = (X_train.iloc[test, :], y_train.iloc[test])

        model = (
            XGBClassifier()
            .set_params(**params)
            .fit(dtrain[0], dtrain[1], eval_set=[dval], verbose=0)
        )
        sets = [dtrain, dval, dtest]
        for i, ds in enumerate(results.keys()):
            y_preds = model.predict_proba(sets[i][0])[:, 1]
            labels = sets[i][1]
            precision, recall, thresholds = precision_recall_curve(labels, y_preds)
            results[ds]["precision"].append(precision)
            results[ds]["recall"].append(recall)
            results[ds]["thresholds"].append(thresholds)
            results[ds]["ap"].append(average_precision_score(labels, y_preds))

    c_fill_train = "rgba(128, 252, 128, 0.2)"
    c_line_train = "rgba(128, 152, 128, 0.5)"
    c_line_main_train = "rgba(128, 0, 128, 1.0)"

    c_fill_val = "rgba(52, 152, 0, 0.2)"
    c_line_val = "rgba(52, 152, 0, 0.5)"
    c_line_main_val = "rgba(41, 128, 0, 1.0)"

    c_fill_test = "rgba(0, 152, 219, 0.2)"
    c_line_test = "rgba(0, 152, 219, 0.5)"
    c_line_main_test = "rgba(0, 128, 185, 1.0)"

    c_grid = "rgba(189, 195, 199, 0.5)"
    c_annot = "rgba(149, 165, 166, 0.5)"
    c_highlight = "rgba(192, 57, 43, 1.0)"
    recall_mean = np.linspace(0, 1, 1000)

    def tp_rates(kind, results):
        interp_precisions = []
        for i in range(n_repeats):
            recall = np.array(results[kind]["recall"][i])
            precision = np.array(results[kind]["precision"][i])
            inds = recall.argsort()
            recall_s = recall[inds]
            precision_s = precision[inds]
            interp_precision = np.interp(recall_mean, recall_s, precision_s)
            # interp_precision[0] = 0.0
            interp_precisions.append(interp_precision)
        precision_mean = np.mean(interp_precisions, axis=0)
        # precision_mean[-1] = 1.0
        precision_std = 2 * np.std(interp_precisions, axis=0)
        precision_upper = np.clip(precision_mean + precision_std, 0, 1)
        precision_lower = precision_mean - precision_std
        ap = np.mean(results[kind]["ap"])
        return precision_upper, precision_mean, precision_lower, ap

    kind = "val"
    try:
        title = "Precision-Recall Curve with signal to background ratio of: {:.2f}".format(
            ratio
        )
    except:
        title = "Precision-Recall Curve"

    train_tpr_upper, train_tpr_mean, train_tpr_lower, train_auc = tp_rates(
        "train", results
    )
    val_tpr_upper, val_tpr_mean, val_tpr_lower, val_auc = tp_rates("val", results)
    test_tpr_upper, test_tpr_mean, test_tpr_lower, test_auc = tp_rates("test", results)
    fig = go.Figure(
        [
            go.Scatter(
                x=recall_mean,
                y=train_tpr_upper,
                line=dict(color=c_line_train, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=recall_mean,
                y=train_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_train,
                line=dict(color=c_line_train, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=recall_mean,
                y=train_tpr_mean,
                line=dict(color=c_line_main_train, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Train_AP: {train_auc:.3f}",
            ),
            go.Scatter(
                x=recall_mean,
                y=val_tpr_upper,
                line=dict(color=c_line_val, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=recall_mean,
                y=val_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_val,
                line=dict(color=c_line_val, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=recall_mean,
                y=val_tpr_mean,
                line=dict(color=c_line_main_val, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Val_AP: {val_auc:.3f}",
            ),
            go.Scatter(
                x=recall_mean,
                y=test_tpr_upper,
                line=dict(color=c_line_test, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="upper",
            ),
            go.Scatter(
                x=recall_mean,
                y=test_tpr_lower,
                fill="tonexty",
                fillcolor=c_fill_test,
                line=dict(color=c_line_test, width=1),
                hoverinfo="skip",
                showlegend=False,
                name="lower",
            ),
            go.Scatter(
                x=recall_mean,
                y=test_tpr_mean,
                line=dict(color=c_line_main_test, width=2),
                hoverinfo="skip",
                showlegend=True,
                name=f"Test_AP: {test_auc:.3f}",
            ),
        ]
    )
    # fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(
        # title=title,
        template="plotly_white",
        title_x=0.5,
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=800,
        height=800,
        legend=dict(
            yanchor="bottom", xanchor="right", x=0.95, y=0.01, font=dict(size=24)
        ),
        yaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=20)),
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        ),
    )

    fig.update_yaxes(
        range=[0, 1],
        gridcolor=c_grid,
        scaleanchor="x",
        scaleratio=1,
        linecolor="black",
    )
    fig.update_xaxes(
        range=[0, 1], gridcolor=c_grid, constrain="domain", linecolor="black"
    )
    # import os

    # if not os.path.exists("images"):
    #     os.mkdir("images")

    fig.write_image("images/Precision_recall_test.svg")
    fig.show()
    return results, model


def optimise_model(model, X_train, y_train, X_test, y_test):
    from skopt.space import Real, Integer
    from skopt import gp_minimize
    from sklearn.metrics import f1_score

    # defining the space
    search_space = [
        Real(0.5, 1.0, name="colsample_bylevel"),
        Real(0.5, 1.0, name="colsample_bytree"),
        Real(0.0, 1.0, name="gamma"),
        Real(0.0001, 0.01, name="learning_rate"),
        Real(0.1, 10, name="max_delta_step"),
        Integer(3, 15, name="max_depth"),
        Real(1, 50, name="min_child_weight"),
        Integer(10, 1500, name="n_estimators"),
        Real(0.1, 100, name="reg_alpha"),
        Real(0.1, 100, name="reg_lambda"),
        Real(0.5, 1.0, name="subsample"),
    ]

    # collecting the fitted models and model performance
    models = []
    train_scores = []
    test_scores = []
    curr_model_hyper_params = [
        "colsample_bylevel",
        "colsample_bytree",
        "gamma",
        "learning_rate",
        "max_delta_step",
        "max_depth",
        "min_child_weight",
        "n_estimators",
        "reg_alpha",
        "reg_lambda",
        "subsample",
    ]

    # function to fit the model and return the performance of the model
    def return_model_assessment(args, X_train, y_train, X_test, y_test):
        # global models, train_scores, test_scores, curr_model_hyper_params
        params = {
            curr_model_hyper_params[i]: args[i]
            for i, j in enumerate(curr_model_hyper_params)
        }
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            early_stopping_rounds=20,
        )
        model.set_params(**params)
        fitted_model = model.fit(X_train, y_train, sample_weight=None)
        models.append(fitted_model)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_score = f1_score(train_predictions, y_train)
        test_score = f1_score(test_predictions, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        return 1 - test_score

    objective_function = partial(
        return_model_assessment,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # running the algorithm
    n_calls = 2000  # number of times you want to train your model
    results = gp_minimize(
        objective_function,
        search_space,
        base_estimator=None,
        n_calls=n_calls,
        n_random_starts=n_calls - 1,
        random_state=42,
        verbose=True,
        n_jobs=-1,
    )

    return results, models, train_scores, test_scores

    # # get_uncertainty_graphs("1000ev.root")
    # # data_interface = generate_data_interface("psiK_1000.root")
    # # data = generate_data_mixing(data_interface,1)
    # # (
    # #     training_data,
    # #     training_labels,
    # #     validation_data,
    # #     validation_labels,
    # # ) = generate_prepared_data(data, 0.9)
    # # # classifier = train_xgboost(
    # # #     training_data, training_labels, validation_data, validation_labels
    # # # )

    # # data_interface_pp = generate_data_interface("Bu2JpsiK_ee_mu1.1_1000_events.root")
    # # data_pp = generate_data_mixing(data_interface_pp, 1)
    # # (
    # #     test_data,
    # #     test_labels,
    # #     validation_data2,
    # #     validation_labels2,
    # # ) = generate_prepared_data(data_pp, 0.9)

    # # roc_curve_error(
    # #     pd.DataFrame(training_data),
    # #     pd.DataFrame(training_labels),
    # #     pd.DataFrame(test_data),
    # #     pd.DataFrame(test_labels),
    # #     default = False
    # # )

    # # pr = precision_recall(
    # #     pd.DataFrame(training_data),
    # #     pd.DataFrame(training_labels),
    # #     pd.DataFrame(test_data),
    # #     pd.DataFrame(test_labels),
    # #     default = False
    # # )
    # # xgb = XGBClassifier()
    # # opt_model = optimise_model(xgb, training_data, training_labels, validation_data, validation_labels)
    # print('2')
