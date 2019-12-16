import numpy as np
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances, roc_curve, roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals.six import StringIO
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV
from sklearn import preprocessing
import seaborn as sns

# Set up plotting properties
sns.set_style('whitegrid')
sns.set_context("paper")

RAND_STATE = 1
RAND_STATE_CL = 42
NSPLITS = 5


def scale_features(data):
    """
    Scale features by their mean and std
    Args:
        data (pandas dataframe): dataframe containing only the numeric
            radiomic values

    Returns:

    """

    df = data.copy()
    for key in data.keys():

        d = data[key]

        d = (d - d.mean()) / d.std()

        df[key] = d

    return df


def bar_plot(data):

    plt.figure(figsize=(25, 10)), data.iloc[0].plot.bar(), plt.show()


def svm_cross(x, y):

    # Attempt an SVM classifier
    plt.close('all')
    fig = plt.figure()

    cv = StratifiedKFold(n_splits=NSPLITS, random_state=RAND_STATE)

    clfsvm = SVC(kernel='rbf', gamma=0.01, C=5, probability=True, random_state=RAND_STATE_CL, verbose=0)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', clfsvm)
        ])

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Grid search
    param = [
        {
            "svm_clf__kernel": ["sigmoid", "rbf"],
            "svm_clf__C": [0.05, 0.1, 0.5, 0.75, 1, 10, 100, 1000],
            "svm_clf__gamma": [5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    grid = GridSearchCV(clf, param_grid=param, n_jobs=4, cv=cv, verbose=0, iid=False)
    grid.fit(x, y)

    print("\nBest parameters set: ")
    print(grid.best_params_)

    # Assign best estimator
    clf = grid.best_estimator_

    i = 0
    train_score = []
    test_score = []
    for train, test in cv.split(x, y):

        print('Size of train set: %d\t\tSize of test set: %d\n' %(len(train), len(test)))

        # Compute probabilities
        probas_ = clf.fit(x[train], y[train]).predict_proba(x[test])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        train_score.append(clf.score(x[train], y[train]))
        test_score.append(clf.score(x[test], y[test]))
        print("Training set score: %f" % train_score[i])
        print("Test set score: %f\n\n" % test_score[i])

        i += 1

    print('Mean training score %f' % np.mean(train_score))
    print('Mean testing  score %f' % np.mean(test_score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    return fig  #plt.savefig(sname, dpi=300)


def grid_search(clf, x, y, cv, params):

    net = []
    for layers in range(2, 4):

        for nodes in range(20, 110, 20):
            cur = []
            for n in range(layers - 1):

                cur.append(nodes)

            cur.append(1)

            net.append(cur)

    # net = [(50, 25, 1),
    #        (75, 50, 25, 1),
    #        (10, 25, 10, 1),
    #        (100, 100, 1),
    #        (100, 150, 1),
    #        (100, 50, 1),
    #        (50, 100, 50, 1),
    #        (25, 50, 25, 10, 1),
    #        (10, 20, 20, 10, 1)]
    # 
    # params = {'nn_clf__hidden_layer_sizes': net, #[(100, 100, 1), (50, 50, 1), (25, 25, 1), (10, 10, 1)],
    #           # 'nn_clf__learning_rate': ['adaptive', 'invscaling'],
    #           'nn_clf__learning_rate_init': [1e-4, 1e-3, 1e-2]}

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score, make_scorer, roc_auc_score
    scorer = make_scorer(roc_auc_score)
    # Define grid search
    search = GridSearchCV(clf, params, cv=cv, scoring=scorer, n_jobs=-1)

    # Fit
    search.fit(x, y)

    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres['params']):
        print(np.sqrt(mean_score), params)

    print('Best params: ', search.best_params_)
    print('Best score: ', search.best_score_)


def neural_network_cross(x, y):

    plt.close('all')
    fig = plt.figure() #figsize = (20, 15))

    sz = x.shape

    cv = StratifiedKFold(n_splits=NSPLITS, random_state=RAND_STATE)

    hidden_layer_sizes = (50, 25, )
    clfnn = MLPClassifier(solver='sgd', max_iter=5000, verbose=0, tol=1e-4, alpha=1e-4,
                        hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=0.001, random_state=RAND_STATE_CL)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('nn_clf', clfnn)
                    ])

    # Grid search
    net = [(100, ),
           (50, 25, )]
    param = [
        {
            "nn_clf__solver": ["sgd", "adam"],
            "nn_clf__alpha": [1e-5, 1e-4, 1e-3],
            "nn_clf__learning_rate_init": [1e-3, 1e-4],
            "nn_clf__hidden_layer_sizes": net,
        }
    ]

    grid = GridSearchCV(clf, param_grid=param, n_jobs=4, cv=cv, verbose=0, iid=False)
    grid.fit(x, y)

    print("\nBest parameters set: ")
    print(grid.best_params_)

    # Assign best estimator
    clf = grid.best_estimator_

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    train_score = []
    test_score = []
    i = 0
    for train, test in cv.split(x, y):

        print('Size of train set: %d\t\tSize of test set: %d\n' %(len(train), len(test)))

        # Compute probablilities
        probas_ = clf.fit(x[train], y[train]).predict_proba(x[test])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Get training scores
        train_score.append(clf.score(x[train], y[train]))
        test_score.append(clf.score(x[test], y[test]))
        print("Training set score: %f" % train_score[i])
        print("Test set score: %f\n\n" % test_score[i])

        i += 1

    print('Mean training score %f' % np.mean(train_score))
    print('Mean testing  score %f' % np.mean(test_score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.nanstd(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # Perform a grid search
    # net = [(50, 25, 1),
    #        (75, 50, 25, 1),
    #        (10, 25, 10, 1),
    #        (100, 100, 1),
    #        (100, 150, 1),
    #        (100, 50, 1),
    #        (50, 100, 50, 1),
    #        (25, 50, 25, 10, 1),
    #        (10, 20, 20, 10, 1)]
    #
    # params = {'nn_clf__hidden_layer_sizes': net, #[(100, 100, 1), (50, 50, 1), (25, 25, 1), (10, 10, 1)],
    #           # 'nn_clf__learning_rate': ['adaptive', 'invscaling'],
    #           'nn_clf__learning_rate_init': [1e-4, 1e-3, 1e-2]}
    # grid_search(clf, x, y, cv, params)

    return fig


def svm_classifier(data, inds_1, inds_2):

    # Convert pandas df to array
    np.random.seed(42)
    array = data.copy()
    num_samps = array.shape[0]
    num_test = int(len(inds_1) * 0.25)
    test_inds = np.arange(0, num_samps)
    np.random.shuffle(test_inds)
    train_inds = test_inds[num_test:]
    test_inds = test_inds[:num_test]
    train_inds_array = np.zeros(num_samps, dtype=np.int)
    train_inds_array[train_inds] = 1

    # Create labels
    y = np.zeros(shape=(array.shape[0]))
    y[inds_2] = 1

    print('Test distribution: True: %d and False: %d' %(np.sum(y[test_inds]), len(test_inds) - np.sum(y[test_inds])))

    # Attempt an SVM classifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    # poly_kernel_svm_clf = SVC()
    poly_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm_clf', SVC(kernel='rbf', probability=True, random_state=RAND_STATE_CL))
        ])
    poly_kernel_svm_clf.fit(array[train_inds_array == 1], y[train_inds_array==1])

    correct = 0
    for z in train_inds:
        ans = poly_kernel_svm_clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            correct += 1

    pcorrect = correct / len(train_inds)
    print('\tTraining data fit: %f' % pcorrect)

    correct = 0
    for z in test_inds:
        ans = poly_kernel_svm_clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            correct += 1

    pcorrect = correct / len(test_inds)
    print('\tTest data fit: %f' % pcorrect)

    # Compute probabilities
    # Run classifier with cross-validation and plot ROC curves
    plt.close('all')
    fig = plt.figure(figsize = (20, 15))

    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(array, y):
        probas_ = poly_kernel_svm_clf.fit(array[train], y[train]).predict_proba(array[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig('roc.png', dpi=300)
    # plt.show()
    # fig.close()

    # Plot SVM
    if data.shape[1] == 2:
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2).fit(array)
        # X = pca.transform(array)
        X = data.copy()

        # CT
        fig = plt.figure(21)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                    edgecolor='k', s=20)

        # Circle out the test data
        # plt.scatter(lle_CT[:, 0], lle_CT[:, 1], s=80, facecolors='none',
        #             zorder=10, edgecolor='k')

        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = poly_kernel_svm_clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
        plt.title('CT: SVM on 2 Principal Components')
        # plt.savefig('Figures/LLE_CT_SVM_classifier.png')
        # plt.show()

    else:
        fig = []

    return fig, poly_kernel_svm_clf


def decision_tree_classifier(data, inds_1, inds_2, num_test, max_depth, export):

    # Convert pandas df to array
    array = data.copy()
    num_samps = array.shape[0]
    # num_test =
    test_inds = np.arange(0, num_samps)
    np.random.shuffle(test_inds)
    train_inds = test_inds[num_test:]
    test_inds = test_inds[:num_test]
    train_inds_array = np.zeros(num_samps, dtype=np.int)
    train_inds_array[train_inds] = 1

    # Create labels
    y = np.zeros(shape=(array.shape[0]), dtype=np.int)
    y[inds_2] = 1

    # Set up tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(array[train_inds_array == 1], y[train_inds_array == 1])

    # Evaluate the fit of the training data
    ans = clf.predict_proba(array[train_inds_array == 1])
    cor = y[train_inds_array == 1]

    train_correct = 0
    for z in range(len(cor)):
        a = np.argmax(ans[z])

        if a == cor[z]:
            train_correct += 1

    print('\tTraining data fit: %f' % (train_correct / sum(train_inds_array ==
                                                          1)))

    # Evaluate the fit of the test data
    test_correct = 0
    for z in test_inds:
        ans = clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            test_correct += 1

    pcorrect = test_correct / len(test_inds)
    print('\tTest data fit: %f' % pcorrect)

    # # Export decision tree graph
    # if export:
    #     dot_data = StringIO()
    #     export_graphviz(clf, out_file=dot_data)
    #     graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #     graph.write_pdf("Figures/decisiontree.pdf")

    return clf


def random_forest(data, inds_1, inds_2, num_test, n_estimators, max_depth):

    array = data.copy()
    num_samps = array.shape[0]
    # num_test =
    test_inds = np.arange(0, num_samps)
    np.random.shuffle(test_inds)
    train_inds = test_inds[num_test:]
    test_inds = test_inds[:num_test]
    train_inds_array = np.zeros(num_samps, dtype=np.int)
    train_inds_array[train_inds] = 1

    # Create labels
    y = np.zeros(shape=(array.shape[0]), dtype=np.int)
    y[inds_2] = 1

    # Set up random forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 n_jobs=-1)
    clf.fit(array[train_inds_array == 1], y[train_inds_array == 1])

    # Evaluate the fit of the training data
    ans = clf.predict_proba(array[train_inds_array == 1])
    cor = y[train_inds_array == 1]

    train_correct = 0
    for z in range(len(cor)):
        a = np.argmax(ans[z])

        if a == cor[z]:
            train_correct += 1

    print('\tTraining data fit: %f' %
          (train_correct / sum(train_inds_array == 1)))

    # Evaluate the fit of the test data
    test_correct = 0
    for z in test_inds:
        ans = clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            test_correct += 1

    pcorrect = test_correct / len(test_inds)
    print('\tTest data fit: %f' % pcorrect)

    return clf


def gb_random_forest(data, inds_1, inds_2, num_test, n_estimators, max_depth):

    array = data.copy()
    num_samps = array.shape[0]
    # num_test =
    test_inds = np.arange(0, num_samps)
    np.random.shuffle(test_inds)
    train_inds = test_inds[num_test:]
    test_inds = test_inds[:num_test]
    train_inds_array = np.zeros(num_samps, dtype=np.int)
    train_inds_array[train_inds] = 1

    # Create labels
    y = np.zeros(shape=(array.shape[0]), dtype=np.int)
    y[inds_2] = 1

    # Set up random forest classifier
    clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     learning_rate=1.0)
    clf.fit(array[train_inds_array == 1], y[train_inds_array == 1])

    # Evaluate the fit of the training data
    ans = clf.predict_proba(array[train_inds_array == 1])
    cor = y[train_inds_array == 1]

    train_correct = 0
    for z in range(len(cor)):
        a = np.argmax(ans[z])

        if a == cor[z]:
            train_correct += 1

    print('\tTraining data fit: %f' %
          (train_correct / sum(train_inds_array == 1)))

    # Evaluate the fit of the test data
    test_correct = 0
    for z in test_inds:
        ans = clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            test_correct += 1

    pcorrect = test_correct / len(test_inds)
    print('\tTest data fit: %f' % pcorrect)

    return clf


def adb_random_forest(data, inds_1, inds_2, num_test, n_estimators, max_depth):

    array = data.copy()
    num_samps = array.shape[0]
    # num_test =
    test_inds = np.arange(0, num_samps)
    np.random.shuffle(test_inds)
    train_inds = test_inds[num_test:]
    test_inds = test_inds[:num_test]
    train_inds_array = np.zeros(num_samps, dtype=np.int)
    train_inds_array[train_inds] = 1

    # Create labels
    y = np.zeros(shape=(array.shape[0]), dtype=np.int)
    y[inds_2] = 1

    # Set up random forest classifier
    clf = AdaBoostClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     learning_rate=1.0)
    clf.fit(array[train_inds_array == 1], y[train_inds_array == 1])

    # Evaluate the fit of the training data
    ans = clf.predict_proba(array[train_inds_array == 1])
    cor = y[train_inds_array == 1]

    train_correct = 0
    for z in range(len(cor)):
        a = np.argmax(ans[z])

        if a == cor[z]:
            train_correct += 1

    print('\tTraining data fit: %f' %
          (train_correct / sum(train_inds_array == 1)))

    # Evaluate the fit of the test data
    test_correct = 0
    for z in test_inds:
        ans = clf.predict([array[z, :]])
        # print('Predicted: %d' % ans)
        # print('Actual: %d' % y[z])

        if ans == y[z]:
            test_correct += 1

    pcorrect = test_correct / len(test_inds)
    print('\tTest data fit: %f' % pcorrect)

    return clf


def local_linear_embedding(data, n_components=2, n_neighbors=5):

    from sklearn.manifold import LocallyLinearEmbedding, locally_linear_embedding
    # lle = LocallyLinearEmbedding(n_components=n_components,
    #                              n_neighbors=n_neighbors)

    [lle, er] = locally_linear_embedding(data, n_neighbors=n_neighbors, n_components=n_components, method='modified')

    return lle


def plot_euclidean_distances(rad_data, inds_1, inds_2):
    # Convert to numpy array
    if type(rad_data) != type(np.array([])):
        array = rad_data.get_values()
    else:
        array = rad_data.copy()

    similarities = euclidean_distances(array)

    similarities = np.nan_to_num(similarities)

    seed = np.random.RandomState(seed=3)

    mds = manifold.MDS(n_components=2, max_iter=5000, eps=1e-12,
                       random_state=seed,
                       n_init=2,
                       dissimilarity="precomputed", n_jobs=1, metric=False)
    pos = mds.fit_transform(similarities)

    from matplotlib.collections import LineCollection
    import matplotlib.cm as cm

    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    s = 100

    plt.scatter(pos[inds_1, 0], pos[inds_1, 1], color='navy', alpha=1.0, s=s,
                lw=5,
                label='0Gy')
    plt.scatter(pos[inds_2, 0], pos[inds_2, 1], color='turquoise', alpha=1.0,
                s=s,
                lw=2, label='20Gy')

    plt.legend(scatterpoints=1, loc=1, shadow=False)

    # plt.show()

    return fig


def pca_analyis(data, threshold=0.95):

    n_pcomps = min(data.shape)

    pca = PCA(n_components=n_pcomps)
    pca_dat = pca.fit(data.T)

    # Get components
    # comps = pca_dat.components_
    # print(len(comps))

    # Variance of the data along eac principal dimension
    cumsum = np.cumsum(pca_dat.explained_variance_ratio_)

    # Find the number of components to keep
    d = np.argmax(cumsum >= threshold)
    print('Use %d principal components to preserve %0.2f of the data '
          'variance' % (d, threshold))

    fig = plt.figure()

    line1, = plt.plot(list(range(0, n_pcomps + 1)), [0] + list(cumsum), '-*',
                      label='CT - 0Gy')
    line2, = plt.plot([0, n_pcomps+1], [threshold, threshold], '--k')
    line3, = plt.plot([d, d], [0, 1], '--k')
    line4, = plt.plot(d, threshold, '.k')

    plt.xlabel('Number of principal components')
    plt.title('Cumulative sum of variance')
    plt.ylim([0, 1])
    plt.xlim([0, n_pcomps])
    # plt.legend(loc=4)
    # plt.savefig('Figures/Cumulative Variance')
    # plt.show()

    return d, fig


def pca_reduction(data, npcomps):

    pca = PCA(n_components=npcomps)
    # pca = KernelPCA(n_components=npcomps)
    return pca.fit_transform(data)


def heatmap(data, inds_1, inds_2):

    train_inds_array = np.zeros(data.shape[1], dtype=np.int)
    train_inds_array[inds_1] = 1
    set1 = data.iloc[inds_1].corr()

    f, ax = plt.subplots(figsize=(15, 10))
    # Draw the heatmap using seaborn
    sns.heatmap(set1, vmax=.8, square=True)
    plt.title('Set1')
    plt.yticks(rotation=0)
    ax.get_xaxis().set_ticks([])
    plt.xticks(rotation=90)

    plt.show()

    set2 = data.iloc[inds_2].corr()

    f, ax = plt.subplots(figsize=(15, 10))
    # Draw the heatmap using seaborn
    sns.heatmap(set2, vmax=.8, square=True)
    plt.title('Set2')
    plt.yticks(rotation=0)
    ax.get_xaxis().set_ticks([])
    plt.xticks(rotation=90)

    plt.show()

    set3 = set1 - set2

    f, ax = plt.subplots(figsize=(15, 10))
    # Draw the heatmap using seaborn
    sns.heatmap(set3, vmax=.8, square=True)
    plt.title('Diff')
    plt.yticks(rotation=0)
    ax.get_xaxis().set_ticks([])
    plt.xticks(rotation=90)

    plt.show()


def concate_contrasts(data):
    """
    Takes a dataframe containing radiomcis data for all animals and contrasta.
    Returns a dataframe with contrasts of the same animal concatenated together.
    Args:
        data:

    Returns:

    """

    # Get file names and before/after classification
    filenames = data['Image']
    inds_1 = [i for (i, name) in enumerate(filenames) if '_1.nii' in name]
    inds_2 = [i for (i, name) in enumerate(filenames) if '_2.nii' in name]

    # Turn multiple contrast images into a single vector of data
    unique_ids = np.unique(data['ID'])
    keys = data.keys()
    cat_cols = [i + '_T1' for i in keys] + \
               [i + '_T1C' for i in keys] + \
               [i + '_T2' for i in keys]
    df_cat = pd.DataFrame(columns=cat_cols, index=range(2*len(unique_ids)))

    ind = 0
    for (i, id) in enumerate(unique_ids):

        tmp = data[data['ID'] == id]

        # Make pre RT data
        tmp_pre = tmp[tmp.index < min(inds_2)]
        tmp_post = tmp[tmp.index >= min(inds_2)]

        # Append data into larger array
        df_cat.iloc[ind] = [n for n in tmp_pre.get_values().ravel()]
        ind += 1

        df_cat.iloc[ind] = [n for n in tmp_post.get_values().ravel()]
        ind += 1

    return df_cat


def nn_classifier():
    hidden_layer_sizes = (50, 25, )
    clf = MLPClassifier(solver='sgd', max_iter=5000, verbose=0, tol=1e-4, alpha=1e-4,
                        hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=0.001, random_state=RAND_STATE_CL)
    return clf


def svm_classifier():
    return SVC(kernel='rbf', gamma=0.01, C=5, probability=True, random_state=RAND_STATE_CL, verbose=0)


def rnd_forest_classifier():

    clf = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=4, verbose=0)

    params = {'clf__max_depth': [2, 3, 4],
              'clf__n_estmiators': [6, 8, 10, 12]}

    return clf


def lasso_classifier():

    from sklearn.linear_model import Lasso

    clf = Lasso()

    return clf


def RBM_classifier():

    from sklearn.neural_network import BernoulliRBM

    clf = BernoulliRBM()

    return clf


def classifier(x, y, clfer):

    plt.close('all')
    fig = plt.figure() #figsize = (20, 15))

    sz = x.shape

    cv = StratifiedKFold(n_splits=NSPLITS, random_state=RAND_STATE)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clfer)])

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(x, y):

        # print('Size of train set: %d\t\tSize of test set: %d\n' %(len(train), len(test)))

        probas_ = clf.fit(x[train], y[train]).predict_proba(x[test])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # print("Training set score: %f" % clf.score(x[train], y[train]))
        # print("Test set score: %f\n\n" % clf.score(x[test], y[test]))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.nanstd(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # Perform a grid search
    # net = [(50, 25, 1),
    #        (75, 50, 25, 1),
    #        (10, 25, 10, 1),
    #        (100, 100, 1),
    #        (100, 150, 1),
    #        (100, 50, 1),
    #        (50, 100, 50, 1),
    #        (25, 50, 25, 10, 1),
    #        (10, 20, 20, 10, 1)]
    #
    # params = {'clf__hidden_layer_sizes': net, #[(100, 100, 1), (50, 50, 1), (25, 25, 1), (10, 10, 1)],
    #           # 'nn_clf__learning_rate': ['adaptive', 'invscaling'],
    #           'clf__learning_rate_init': [1e-4, 1e-3, 1e-2]}
    # grid_search(clf, x, y, cv, params)

    print('Mean AUC: {:0.2f}'.format(mean_auc))
    print('Std  AUC: {:0.2f}'.format(std_auc))

    return fig, mean_auc, std_auc