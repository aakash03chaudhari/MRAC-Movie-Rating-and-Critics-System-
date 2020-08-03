
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn
#import seaborn as sns
from six.moves import range
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score




def call_NB(usermovie,userreview):
    # Setup Pandas
    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.notebook_repr_html', True)

    # Setup Seaborn
    # sns.set_style("whitegrid")
    # sns.set_context("poster")

    #usermovie = 'tony'
    #userreview = 'this movie is awesome'
    freshness = 0
    avgfreshness = 0

    critics = pd.read_csv('mysite/critics.csv')
    # let's drop rows with missing quotes
    critics = critics[~critics.quote.isnull()]
    critics.head()

    n_reviews = len(critics)
    n_movies = critics.rtid.unique().size
    n_critics = critics.critic.unique().size

    print("Number of reviews: {:d}".format(n_reviews))
    print("Number of critics: {:d}".format(n_critics))
    print("Number of movies:  {:d}".format(n_movies))

    df = critics.copy()
    df['fresh'] = df.fresh == 'fresh'
    grp = df.groupby('critic')
    counts = grp.critic.count()  # number of reviews by each critic
    means = grp.fresh.mean()  # average freshness for each critic

    means[counts > 100].hist(bins=10, edgecolor='w', lw=1)
    plt.xlabel("Average Rating per critic")
    plt.ylabel("Number of Critics")
    plt.yticks([0, 2, 4, 6, 8, 10]);

    vectorizer = CountVectorizer(min_df=0)


    X, y = make_xy(critics)

    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]

    roc_auc_score(y_val, y_pred)

    clf_new = MultinomialNB(alpha=1.0)
    clf_new.fit(X_train, y_train)
    y_pred_new = clf_new.predict_proba(X_val)[:, 1]
    roc_auc_score(y_val, y_pred_new)
    print("The accuracy score for the test set is %f" % clf.score(X_val, y_val))
    print("The accuracy score for the training set is %f" % clf.score(X_train, y_train))






    LL = {}
    alphas = [.1, 1, 5, 10, 50]
    best_min_df = 0

    x, y = make_xy(critics, vectorizer)

    prob = clf.predict_proba(x)[:, 0]
    predict = clf.predict(x)
    bad_rotten = np.argsort(prob[y == 0])[:5]
    bad_fresh = np.argsort(prob[y == 1])[-5:]

    # for row in bad_rotten:
    # print(critics[y == 0].quote.iloc[row])
    # print("")

    # for row in bad_fresh:
    # print(critics[y == 1].quote.iloc[row])
    # print("")

    v = vectorizer.transform([userreview])
    freshness = clf.predict_proba(v)[0][1]
    #return clf.predict_proba(v)[0][1]
    print('Reviews percentange : ', clf.predict_proba(v)[0][1] * 100)

    row = [usermovie, userreview, str(freshness)]
    print(row)

    with open('mysite/review.csv', 'a', newline='') as writeFile:
        writer = csv.writer(writeFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)
        writeFile.close()

    review = pd.read_csv('mysite/review.csv')

    df = review[review.usermovie == usermovie]
    df1 = df['freshness'].mean()
    print('Overall Freshness :', df1 * 100)
    return(usermovie,userreview,str(round(freshness*100,2)),str(round(df1*100,2)))

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

def cv_score(clf, X, y, scorefunc):
    result = 0.
    nfold = 5
    for train, test in KFold(nfold).split(X):  # split data into train/test groups, 5 times
        clf.fit(X[train], y[train])  # fit the classifier, passed is as clf.
        result += scorefunc(clf, X[test], y[test])  # evaluate score function on held-out data
    return result / nfold  # average

def make_xy(critics, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
    X = X.tocsc()  # some versions of sklearn return COO format
    y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, y

# NB_RESULT= call_NB('jumanji 3','bore')
# print(NB_RESULT[2])

