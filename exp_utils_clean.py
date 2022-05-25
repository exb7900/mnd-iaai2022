import itertools
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
torch.manual_seed(0)


def scoring(metric, x_test, y_test, model, domain_adaptation=False):
    """
    Returns a requested metric or metrics.

    Args:
        metric: string representing metric to collect
        x_test: features for test set
        y_test: labels for test set
        model: trained model
        domain_adaptation: runs model.predict instead of just model if True

    Returns:
        returned_metrics: dictionary with metric name and value
    """
    if domain_adaptation: 
        y_pred = model.predict(x_test)
    else:
        y_pred = model(x_test)

    y_pred = y_pred.detach().numpy()
    y_pred_thresh = np.where(y_pred > 0.5, 1, 0).astype(np.uint8)

    y_test = y_test.detach().numpy()

    returned_metrics = {}

    # Classification
    if 'Accuracy' in metric:
        y_test = y_test.astype(np.uint8)
        returned_metrics['Accuracy'] = sklearn.metrics.accuracy_score(y_test, 
                                                                y_pred_thresh)

    if 'F1' in metric: 
        y_test = y_test.astype(np.uint8)
        returned_metrics['F1'] = sklearn.metrics.f1_score(y_test, 
                                                          y_pred_thresh)

    if 'Precision' in metric:
        y_test = y_test.astype(np.uint8)
        returned_metrics['Precision'] = sklearn.metrics.precision_score(y_test,
                                                                 y_pred_thresh)

    if 'Recall' in metric:
        y_test = y_test.astype(np.uint8)
        returned_metrics['Recall'] = sklearn.metrics.recall_score(y_test, 
                                                                 y_pred_thresh)

    if 'AUC' in metric:
        y_test = y_test.astype(np.uint8)
        if np.unique(y_test).shape[0] >= 2:
            returned_metrics['AUC'] = sklearn.metrics.roc_auc_score(y_test,
                                                                        y_pred)
        else:
            returned_metrics['AUC'] = np.nan
    
    # Regression
    if 'RMSE' in metric:
        returned_metrics['RMSE'] = sklearn.metrics.mean_squared_error(y_test, 
                                                                      y_pred, 
                                                                 squared=False)

    if 'MAE' in metric:
        returned_metrics['MAE'] = sklearn.metrics.mean_absolute_error(y_test, 
                                                                      y_pred)

    if 'Weights' in metric:
        returned_metrics['Weights'] = \
            next(model.parameters()).data.numpy().flatten()

    return returned_metrics

# Logistic Regression
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def train_lr_pytorch(x_data, y_data, regression=False):
    """
    Train a logistic regression model on data with one batch.

    Args:
        x_data: values from rasters, pixels in rows, bands in cols
        y_data: labels (either regression or classification)
        regression: set to True if running regression instead of classification

    Returns:
        trained model
    """
    model = LogisticRegression(x_data.shape[1])
    if regression:
        criterion = torch.nn.MSELoss(reduction='sum')
    else:
        criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=0.01)
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_data)
        # Compute Loss
        loss = criterion(y_pred, y_data)
        # Backward pass
        loss.backward()
        optimizer.step()
    return model.eval()


# Running LR
def k_fold_cross_validation_lr(dataset, 
                               label_col, 
                               data_cols, 
                               metric,
                               k=4, 
                               regression=False,
                               write_path=None):
    """
    Run k-fold cross validation for logistic regression model. Note that we do
    nanmeans and nanstds, as ROC AUC could be nan (see scoring).

    Args:
        dataset: pandas array containing both data and labels
        label_col: index of column with labels of interest
        data_cols: list of column indices with raster data, not labels
        metric: how to evaluate (list -- see scoring, above)
        k: number of folds
        regression: set to True if running regression instead of classification
        write_path: path to save trained models in, or None if not saving

    Returns:
        mean_metrics: dictionary with mean for each metric requested
    
    """

    # Do cross validation.
    stored_metrics = []
    stored_models = []
    kf = KFold(k, shuffle=True, random_state=5)
    for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):
        # Select data for fold.
        train_data = dataset.iloc[train_ids].to_numpy()
        test_data = dataset.iloc[test_ids].to_numpy()

        # Select features data (x).
        train_x = train_data[:, data_cols]
        test_x = test_data[:, data_cols]

        # Select label data (y).
        train_y = np.expand_dims(train_data[:, label_col], 1)
        test_y = np.expand_dims(test_data[:, label_col], 1)

        # Scale features data.
        scaler = StandardScaler().fit(train_x)
        scaled_train_x = scaler.transform(train_x)
        scaled_test_x = scaler.transform(test_x)

        # Convert to tensors.
        train_x_tensor = torch.FloatTensor(scaled_train_x)
        test_x_tensor = torch.FloatTensor(scaled_test_x)
        train_y_tensor = torch.FloatTensor(train_y)
        test_y_tensor = torch.FloatTensor(test_y)

        # Train.
        trained_model = train_lr_pytorch(train_x_tensor, 
                                         train_y_tensor,
                                         regression=regression)

        # Evaluate.
        returned_metrics = scoring(metric, 
                                   test_x_tensor, 
                                   test_y_tensor, 
                                   trained_model)

        # Store.
        stored_metrics.append(returned_metrics)
        stored_models.append(trained_model)
    
    # Take means.
    mean_metrics = {}
    for m in stored_metrics[0].keys():
        metrics_values = [sub[m] for sub in stored_metrics] 
        if m == 'Weights':
            mean_metrics[m] = np.nanmean(metrics_values, axis=0)
        else:
            mean_metrics[m] = np.nanmean(metrics_values)

    # Save model if applicable.
    if write_path is not None:
        with open(write_path, 'wb') as o:
            pickle.dump(stored_models, o)

    return mean_metrics


# Domain Adaptation
class DA(torch.nn.Module):
    def __init__(self, input_dim):
        super(DA, self).__init__()
        self.base_network = torch.nn.Linear(input_dim, 5)
        self.classifier_layer = torch.nn.Linear(5, 1)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = torch.sigmoid(self.classifier_layer(source))
        tgt_clf = torch.sigmoid(self.classifier_layer(target))
        transfer_loss = self.adapt_loss(source, target)
        return source_clf, tgt_clf, transfer_loss
    
    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf
        
    def adapt_loss(self, source, target):
        """
        Compute adaptation loss with CORAL. Code adapted from: 
        https://github.com/jindongwang/transferlearning/blob/master/notebooks/deep_transfer_tutorial.ipynb
        https://colab.research.google.com/drive/1MVuk95mMg4ecGyUAIG94vedF81HtWQAr?usp=sharing

        Args:
            source: source tensor
            target: target tensor

        Returns:
            adaptation loss tensor
        """
        return CORAL(source, target)


def CORAL(source, target):
    """
    Calculate CORAL loss. Adapted from 
    https://github.com/tim-learn/ATDOC/blob/main/loss.py/
    https://github.com/jindongwang/transferlearning/blob/master/notebooks/deep_transfer_tutorial.ipynb
    https://colab.research.google.com/drive/1MVuk95mMg4ecGyUAIG94vedF81HtWQAr?usp=sharing.
    """
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss


def train_da_pytorch(x_data_src, 
                     y_data_src, 
                     x_data_tgt, 
                     y_data_tgt, 
                     lambda_param=10, 
                     alpha=1, 
                     verbose=False):
    """
    Train a domain adaptation model on data with one batch.

    Args:
        x_data_src: values from rasters, pixels in rows, bands in cols,
                    from all regions where we aren't predicting
        y_data_src: labels (either regression or classification),
                    from all regions where we aren't predicting
        x_data_tgt: values from rasters, pixels in rows, bands in cols,
                    from region where we are predicting
        y_data_tgt: labels (either regression or classification),
                    from region where we are predicting
        lambda_param: hyperparameter, weight for transfer loss in overall loss
        alpha: hyperparameter, weight for source loss
        verbose: optionally print loss value

    Returns:
        trained model
    """
    model = DA(x_data_src.shape[1])
    criterion = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=0.01)
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()

        label_source_pred, label_tgt_pred, transfer_loss = model(x_data_src, x_data_tgt)
        clf_src_loss = criterion(label_source_pred, y_data_src)
        clf_tgt_loss = criterion(label_tgt_pred, y_data_tgt)
        loss = alpha * clf_src_loss + clf_tgt_loss + lambda_param * transfer_loss
        if verbose:
            print(loss)
        loss.backward()
        optimizer.step()
    return model.eval()

# Running DA
def k_fold_cross_validation_da(selected_csvs, 
                               label_col, 
                               data_cols, 
                               metric,
                               k=4, 
                               write_path=None,
                               lambda_param=10, 
                               alpha=1, 
                               verbose=False):
    """
    Run k-fold cross validation for domain adaptation model. Note that we do
    nanmeans and nanstds, as ROC AUC could be nan (see scoring).

    Args:
        selected_csvs: list of dicts, each dict has 'data' with raw csv data,
            'region' with corresponding region, and 'mnd', one of which was
            selected (i.e., when this arrives, it will only have 1 mnd in all)
        label_col: index of column with labels of interest
        data_cols: list of column indices with raster data, not labels
        metric: how to evaluate (list -- see scoring, above)
        k: number of folds
        write_path: path to save trained models in, or None if not saving

    Returns:
        mean_metrics: dictionary with mean for each metric requested
    """
    # Initialize cross validation.
    kf = KFold(k, shuffle=True, random_state=5)

    # Gather data for naively combine baseline (combine all train data in 
    # current train folds, save each test set for current region).
    folds_all_train = {}
    folds_regions_test = []
    for il in range(len(selected_csvs)):
        current_data = selected_csvs[il]['data'].to_numpy()
        current_region = selected_csvs[il]['region']
        current_mnd = selected_csvs[il]['mnd']
        folds_regions_test_tmp = {}
        for index, (train_ids, test_ids) in enumerate(kf.split(current_data)):
            if il == 0:
                folds_all_train[index] = current_data[train_ids]
            else:
                folds_all_train[index] = np.vstack((folds_all_train[index],
                                                    current_data[train_ids]))
            folds_regions_test_tmp[index] = current_data[test_ids]
        folds_regions_test.append(folds_regions_test_tmp)

    # Gather data for domain adaptation (for current region, collect 
    # training data from current region - tgt / other regions - src, in 
    # current fold).
    folds_src_da_train = []
    folds_tgt_da_train = []
    folds_tgt_da_test = []
    for roi in range(len(selected_csvs)):
        folds_src_da_train_tmp = {}
        folds_tgt_da_train_tmp = {}
        folds_tgt_da_test_tmp = {}
        counter_tgt = 0
        counter_src = 0
        for il in range(len(selected_csvs)):
            current_data = selected_csvs[il]['data'].to_numpy()
            current_region = selected_csvs[il]['region']
            current_mnd = selected_csvs[il]['mnd']
            if roi == il:
                for index, (train_ids, test_ids) in \
                    enumerate(kf.split(current_data)):
                    if counter_tgt == 0:
                        folds_tgt_da_train_tmp[index] = current_data[train_ids]
                        folds_tgt_da_test_tmp[index] = current_data[test_ids]
                    else:
                        folds_tgt_da_train_tmp[index] = np.vstack((
                                                folds_tgt_da_train_tmp[index],
                                                current_data[train_ids]))
                        folds_tgt_da_test_tmp[index] = np.vstack((
                                                folds_tgt_da_test_tmp[index],
                                                current_data[test_ids]))
                counter_tgt += 1
            else:
                for index, (train_ids, test_ids) in \
                    enumerate(kf.split(current_data)):
                    if counter_src == 0:
                        folds_src_da_train_tmp[index] = current_data[train_ids]
                    else:
                        folds_src_da_train_tmp[index] = np.vstack((
                                                folds_src_da_train_tmp[index],
                                                current_data[train_ids]))
                counter_src += 1
        folds_src_da_train.append(folds_src_da_train_tmp)
        folds_tgt_da_train.append(folds_tgt_da_train_tmp)
        folds_tgt_da_test.append(folds_tgt_da_test_tmp)

    # Do cross validation.
    stored_models_all = []
    stored_models_da = []
    mean_metrics = {}
    counter = 0
    for roi in range(len(selected_csvs)):
        stored_metrics_all = []
        stored_metrics_da = []
        for fold in range(k):

            # Select data for fold.
            src_train_data = folds_src_da_train[roi][fold]
            tgt_train_data = folds_tgt_da_train[roi][fold]
            tgt_test_data = folds_tgt_da_test[roi][fold]

            all_train_data = folds_all_train[fold]
            all_test_data = folds_regions_test[roi][fold]

            # Select features data (x).
            src_train_x = src_train_data[:, data_cols]
            tgt_train_x = tgt_train_data[:, data_cols]
            tgt_test_x = tgt_test_data[:, data_cols]

            all_train_x = all_train_data[:, data_cols]
            all_test_x = all_test_data[:, data_cols]

            # Select label data (y).
            src_train_y = np.expand_dims(src_train_data[:, label_col], 1)
            tgt_train_y = np.expand_dims(tgt_train_data[:, label_col], 1)
            tgt_test_y = np.expand_dims(tgt_test_data[:, label_col], 1)

            all_train_y = np.expand_dims(all_train_data[:, label_col], 1)
            all_test_y = np.expand_dims(all_test_data[:, label_col], 1)

            # Scale features data.
            scaler_src = StandardScaler().fit(src_train_x)
            scaler_tgt = StandardScaler().fit(tgt_train_x)
            scaler_all = StandardScaler().fit(all_train_x)

            scaled_src_train_x = scaler_src.transform(src_train_x)
            scaled_tgt_train_x = scaler_tgt.transform(tgt_train_x)
            scaled_all_train_x = scaler_all.transform(all_train_x)

            scaled_tgt_test_x = scaler_tgt.transform(tgt_test_x)
            scaled_all_test_x = scaler_all.transform(all_test_x)

            # Convert to tensors.
            tensor_src_train_x = torch.FloatTensor(scaled_src_train_x)
            tensor_tgt_train_x = torch.FloatTensor(scaled_tgt_train_x)
            tensor_all_train_x = torch.FloatTensor(scaled_all_train_x)

            tensor_tgt_test_x = torch.FloatTensor(scaled_tgt_test_x)
            tensor_all_test_x = torch.FloatTensor(scaled_all_test_x)

            tensor_src_train_y = torch.FloatTensor(src_train_y)
            tensor_tgt_train_y = torch.FloatTensor(tgt_train_y)
            tensor_tgt_test_y = torch.FloatTensor(tgt_test_y)

            tensor_all_train_y = torch.FloatTensor(all_train_y)
            tensor_all_test_y = torch.FloatTensor(all_test_y)

            # Train DA.
            trained_da_model = train_da_pytorch(tensor_src_train_x,
                                                tensor_src_train_y,
                                                tensor_tgt_train_x,
                                                tensor_tgt_train_y,
                                                lambda_param=lambda_param, 
                                                alpha=alpha, 
                                                verbose=verbose)

            # Train all.
            trained_all_model = train_lr_pytorch(tensor_all_train_x,
                                                 tensor_all_train_y,
                                                 regression=False)

            # Evaluate DA.
            returned_metrics_da = scoring(metric, 
                                          tensor_tgt_test_x,
                                          tensor_tgt_test_y,
                                          trained_da_model,
                                          domain_adaptation=True)

            # Evaluate all.
            returned_metrics_all = scoring(metric, 
                                           tensor_all_test_x,
                                           tensor_all_test_y,
                                           trained_all_model)

            # Store.
            stored_metrics_da.append(returned_metrics_da)
            stored_models_da.append(trained_da_model)
            stored_metrics_all.append(returned_metrics_all)
            stored_models_all.append(trained_all_model)
    
        # Take means.
        for m in stored_metrics_da[0].keys():
            metrics_values_da = [sub[m] for sub in stored_metrics_da]
            metrics_values_all = [sub[m] for sub in stored_metrics_all]
            if counter == 0:
                mean_metrics[m] = np.zeros((len(selected_csvs), 2)) 
            mean_metrics[m][roi, 1] = np.nanmean(metrics_values_da)
            mean_metrics[m][roi, 0] = np.nanmean(metrics_values_all)
        counter += 1

    # Save model if applicable.
    if write_path is not None:
        with open(write_path+'_da.pkl', 'wb') as o:
            pickle.dump(stored_models_da, o)
        with open(write_path+'_nc.pkl', 'wb') as o:
            pickle.dump(stored_models_all, o)

    return mean_metrics


# Plotting methods below.
def make_bar_plot(means, 
                  col_names, 
                  regions=['SE', 'SW', 'WCO', 'CP'],
                  ylabel='AUC', 
                  write_path=None, 
                  hatches=itertools.cycle(['/','+', 'x']), 
                  colors=itertools.cycle(['C0','C1','C2'])):
    """
    Makes Fig. 3 plots.

    Args:
        means: np array with regions data in rows, methods data in columns, 
               MND in bands
        col_names: name of each method
        regions: list of region abbreviations
        ylabel: what is the metric for the y axis label
        write_path: path to save trained models in, or None if not saving
        hatches: itertools cycle for hatches to use in bars
        colors: itertools cycle for colors to use in bars
    """
    x = np.arange(len(regions))  # the label locations
    num_methods = means.shape[1]

    width = 0.08  # the width of the bars
    coords = list(np.arange(-np.floor(num_methods / 2), num_methods / 2))

    for mnd in range(means.shape[2]):
        fig, ax = plt.subplots()
        for col in range(num_methods):    
            rects1 = ax.bar(x + coords[col] * (width), 
                            means[:, col, mnd],
                            width, 
                            label = col_names[col], 
                            hatch = next(hatches), 
                            color = next(colors))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                    loc=3, 
                    ncol=1, 
                    mode="expand", 
                    borderaxespad=0.)

        fig.tight_layout()

        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams.update({'font.size':28})
        plt.ylim([0,1])

        if write_path is not None:
            plt.savefig(write_path+'_'+str(mnd)+'.pdf')
        else:
            plt.show()


def make_horizontal_bar_plot(weights, 
                             features, 
                             exponential=False, 
                             fontsize=28, 
                             write_path=None):
    """
    Makes Fig. 5 to plot coefficients of logistic regression.

    Args:
        exponential: if True, take the exponential of weights first, based on
            https://christophm.github.io/interpretable-ml-book/logistic.html
            [default is False to preserve directionality]
    """
    if len(weights) != len(features):
        raise ValueError('Features and mean_metrics are not same size.')
    
    y_pos = np.arange(len(weights))
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size':fontsize})
    if exponential:
        ax.barh(y_pos, np.exp(weights), align='center')
    else:
        ax.barh(y_pos, weights, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.yaxis.grid()
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Weight')
    fig.tight_layout()
    if write_path is not None:
        plt.savefig(write_path)
    else:
        plt.show()


def make_dot_plot(means, 
                  col_names,
                  xlabel='AUC',
                  mnd_labels=['Iron', 'B12', 'A'], 
                  regions=['SE', 'SW', 'WCO', 'CP'], 
                  region_choice=['WCO', 'WCO', 'WCO'], 
                  marker=['o', '^', 's'], 
                  markersize=15, 
                  colors=['C0','C1','C2'], 
                  write_path=None):
    """
    Makes Fig. 4 & 6 plots. 
    Adapted from https://gist.github.com/jhykes/d6f1577313a7c6eccfeb.

    Args:
        means: np array with regions data in rows, methods data in columns, 
               MND in bands
        col_names: name of each method
        xlabel: what is the metric for the x axis label
        mnd_labels: list of MND abbreviations
        regions: list of region abbreviations
        region_choice: which region to pull data from for plot
        marker: list for marker shapes to use
        markersize: markersize parameter for plt
        colors: list for colors to use in shapes
        write_path: path to save trained models in, or None if not saving
    """
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(1,1,1)

    n = means.shape[1] * means.shape[2]

    marker = marker[:means.shape[1]] #markers will represent methods
    colors = colors[:means.shape[0]] #colors will be for regions
    filler = ['' for i in np.arange(means.shape[1]-1)]
    mnd_labelsP = []
    for d in mnd_labels:
        mnd_labelsP += [d]+filler

    y = np.arange(n)[::-1]

    accumulateSize = 0
    for id in range(means.shape[2]): #mnds
        for m in range(means.shape[1]): #methods
            for r in range(means.shape[0]): #regions

                if regions[r] == region_choice[id]:
                    if id == 0:
                        ax.plot(means[r,m,id],
                                     y[accumulateSize], 
                                     marker=marker[m], 
                                     linestyle='', 
                                     markersize=markersize, 
                                     markeredgewidth=0, 
                                     color=colors[m], 
                                     label=col_names[m])
                    else:
                        ax.plot(means[r,m,id], 
                                y[accumulateSize], 
                                marker=marker[m], 
                                linestyle='', 
                                markersize=markersize, 
                                markeredgewidth=0, 
                                color=colors[m])

            accumulateSize += 1 
    ticks = ax.yaxis.set_ticks(y)
    text = ax.yaxis.set_ticklabels(mnd_labelsP)

    ax.tick_params(axis='y', which='major', right='on', left='on', color='0.8')
    ax.grid(axis='y', which='major', color='0.7', zorder=-10, linestyle=(0, (5, 10)))
    ax.set_xlabel(xlabel)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim(0, 1.1)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),
              loc=3, 
              ncol=1, 
              mode="expand", 
              borderaxespad=0.,
              frameon=False)
    fig.tight_layout()
    if write_path is not None:
        plt.savefig(write_path)
    else:
        plt.show()


def determine_cols(data_cols, meth_idx, dataset):
    """
    Helper function for run_and_plot, finds columns to use based on provided 
    data_cols (see run_and_plot for details).
    """
    if data_cols[meth_idx][0] == 'consecutive':
        use_data_cols = list(range(len(dataset.columns) + \
                                    data_cols[meth_idx][1]))
    else:
        use_data_cols = data_cols[meth_idx][1]
    return use_data_cols

def select_methods(means_lr, methods, fig_methods):
    """
    Helper function for run_and_plot, selects methods in means_lr and 
    methods during plotting of each figure (see run_and_plot for details).
    """
    means_lr_selected = np.zeros((means_lr.shape[0], 
                                  len(fig_methods), 
                                  means_lr.shape[2], 
                                  means_lr.shape[3]))
    for i,f in enumerate(fig_methods):
        meth_idx = methods.index(f)
        means_lr_selected[:,i,:,:] = means_lr[:,meth_idx,:,:]

    return means_lr_selected

def run_and_plot(csvs, 
                 csvs_metadata, 
                 label_col,
                 data_cols,
                 metric,
                 da_method='Satellite Auto FS',
                 fig3_methods=['survey-fs', 'survey-full', 'auto'],
                 fig4_methods=['remove0', 'expert', 'auto'],
                 features=None,
                 where_weights=[0],
                 method_weights=0,
                 exponential_weights=False,
                 fontsize_weights=28,
                 k=4, 
                 regression=False, 
                 model_path=None,
                 plot_path=None):
    """
    Run all of the above components and plot in Colab.

    Args:
        csvs: list of paths
        csvs_metadata: list where each element corresponds to [MND, 
            region, method]
        label_col: list of index of column with labels of interest, 
            corresponding to each method (since CSV could be different for each)
        data_cols: list corresponding to label_col; if ('consecutive',x), then 
            column indices with raster data, not labels, are all those leading
            up to index x; if ('listed',list), then use indices in list
        metric: how to evaluate (list -- see scoring, above)
        da_method: name of baseline method from logistic regression for domain
            adaptation evaluation
        fig3_methods: methods to generate fig 3 in IAAI 22 paper
        fig4_methods: methods to generate fig 4 in IAAI 22 paper
        features: list of human-readable column names if desire to replace 
            given, else None (and will just use column names)
        where_weights: list of indices corresponding to csvs/csvs_metadata for
            where to plot weights (Fig. 6)
        method_weights: csvs index where to collect weights (should be a csv 
            from method where weights will be collected)
        k: number of folds
        regression: set to True if running regression instead of classification
        model_path: *beginning of* path to save trained models in, or None if
            not saving
        plot_path: path to save plot pdfs in, or None if not saving
        

    """
    # Get number of methods, MNDs, regions.
    mnds = list(dict.fromkeys([x[0] for x in csvs_metadata]))
    regions = list(dict.fromkeys([x[1] for x in csvs_metadata]))
    methods =  list(dict.fromkeys([x[2] for x in csvs_metadata]))

    # Run logistic regression cross validation and collect data.
    if 'Weights' in metric:
        len_metric = len(metric) - 1
    else:
        len_metric = len(metric)
    means_lr = np.zeros((len(regions), len(methods), len(mnds), len_metric))
    use_data_cols = determine_cols(data_cols, 
                                   method_weights, 
                                   pd.read_csv(csvs[method_weights], 
                                               header=0, 
                                               index_col=0))
    mean_weights = np.zeros((len(csvs), len(use_data_cols)))
    store_csvs = []
    for index, csv in enumerate(csvs):
        dataset = pd.read_csv(csv, header=0, index_col=0)
        mnd, region, method = csvs_metadata[index]
        mnd_idx = mnds.index(mnd)
        reg_idx = regions.index(region)
        meth_idx = methods.index(method)
        if method == da_method:
            store_csvs.append({'data': dataset, 'region': region, 'mnd': mnd})

        if model_path is None:
            to_write = None
        else:
            to_write = model_path + '_' + str(mnd) + '_' + str(region) + \
                '_' + str(method) + '.pkl'
        use_data_cols = determine_cols(data_cols, 
                                   meth_idx, 
                                   dataset)
        mean_metrics = k_fold_cross_validation_lr(dataset, 
                                                  label_col[meth_idx], 
                                                  use_data_cols, 
                                                  metric,
                                                  k=k, 
                                                  regression=regression,
                                                  write_path=to_write)
        for m, met in enumerate(metric):
            if met == 'Weights':
                if meth_idx == method_weights:
                    mean_weights[index] = mean_metrics[met]
                
            else:
                means_lr[reg_idx, meth_idx, mnd_idx, m] = mean_metrics[met]

    # Run domain adaptation cross validation and collect data.
    # 3 b/c 2 fixed baselines: combining all regions naively, proposed LR above
    means_da = np.zeros((len(regions), 3, len(mnds), len_metric))  
    meth_idx = methods.index(da_method)
    means_da[:, -1, :, :] = means_lr[:, meth_idx, :, :]  # Pulls LR baseline
    
    for mi, mnd in enumerate(mnds):
        selected_csvs = [x for x in store_csvs if x['mnd'] == mnd]
        if model_path is None:
            to_write = None
        else:
            to_write = model_path + '_' + str(mnd) + '_' + str(region) + \
                '_' + str(method) + '.pkl'

        for index, csv in enumerate(csvs):
            mnd, region, method = csvs_metadata[index]
            meth_idx = methods.index(method)
            if method == da_method:
                dataset = pd.read_csv(csv, header=0, index_col=0)
        use_data_cols = determine_cols(data_cols, 
                                   meth_idx, 
                                   dataset)
        mean_metrics_da = k_fold_cross_validation_da(selected_csvs, 
                                                     label_col[meth_idx], 
                                                     use_data_cols, 
                                                     metric,
                                                     k=4, 
                                                     write_path=to_write,
                                                     lambda_param=0.005, 
                                                     alpha=0.2, 
                                                     verbose=False)
        for m, met in enumerate(metric):
            if met == 'Weights':
                continue
            else:
                means_da[:, :-1, mi, m] = mean_metrics_da[met]  
    
    # # Fig. 3 (compare survey and satellite-based MND prediction)
    means_lr_3 = select_methods(means_lr, methods, fig3_methods)
    for m, met in enumerate(metric):
        if met == 'Weights':
            continue
        make_bar_plot(means_lr_3[:,:,:,m], 
                      fig3_methods, 
                      regions=regions,
                      ylabel=met, 
                      write_path=None, #plot_path+'fig3_'+met, 
                      hatches=itertools.cycle(['/','+', 'x']), 
                      colors=itertools.cycle(['C0','C1','C2']))

    # Fig. 4 (compare feature selections)
    means_lr_4 = select_methods(means_lr, methods, fig4_methods)
    for m, met in enumerate(metric):
        if met == 'Weights':
            continue
        make_dot_plot(means_lr_4[:,:,:,m], 
                      fig4_methods, 
                      xlabel=met,
                      mnd_labels=mnds,
                      regions=regions,
                      region_choice=['WCO', 'WCO', 'WCO'], 
                      marker=['o', '^', 's'], 
                      markersize=15, 
                      colors=['C3','C7','C2','C2'],
                      write_path=plot_path+'fig4_'+met+'.pdf')

    # Fig. 5 (logistic regression coefficients)
    if 'Weights' in metric:
        # Get features if needed.
        if features is None:
            dataset = pd.read_csv(csvs[0], header=0, index_col=0)
            features_preprocessing = dataset.columns[data_cols].tolist()
            features = [x.split('__')[0] for x in features_preprocessing]
        for w in where_weights:
            make_horizontal_bar_plot(mean_weights[w], 
                                     features, 
                                     fontsize=fontsize_weights, 
                                     exponential=exponential_weights,
                                     write_path=plot_path+'fig5_'+str(w)+'.pdf')

    # Fig. 6 (compare logistic regression and domain adaptation)
    for m, met in enumerate(metric):
        if met == 'Weights':
            continue
        make_dot_plot(means_da[:,:,:,m], 
                      ['Naively Combine', 'Domain Adaptation', 
                        da_method],
                      xlabel=met,
                      mnd_labels=mnds,
                      regions=regions,
                      region_choice=['CP', 'CP', 'CP'], 
                      marker=['o', '^', 's'], 
                      markersize=15, 
                      colors=['C4','C5','C2','C2'],
                      write_path=plot_path+'fig6_'+met+'.pdf')