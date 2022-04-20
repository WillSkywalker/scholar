import os
import sys
from optparse import OptionParser
import re
import gensim
import numpy as np
import pandas as pd
from pprint import pprint
import pandas

import file_handling as fh
from scholar import Scholar
import string

from collections import Counter
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')



class LDAModel(BaseEstimator):

    def __init__(self, K, alpha, beta, id2word, corpus, train_text):
        self.lda_model = None
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.id2word = id2word
        self.corpus = corpus
        self.train_text = train_text

    def fit(self, x, Y=None):
        print('Start training: K=%d, alpha=%s, beta=%s' % (self.K, self.alpha, self.beta))
        if self.alpha == 'auto' or self.beta == 'auto':
            self.lda_model = gensim.models.ldamodel.LdaModel(x, num_topics=self.K,
                alpha=self.alpha, eta=self.beta, id2word=self.id2word)
        else:
            self.lda_model = gensim.models.ldamulticore.LdaMulticore(x, 
                num_topics=self.K, alpha=self.alpha, eta=self.beta, id2word=self.id2word, workers=16)

    def transform(self, x):
        doc_topic_distr = []
        for doc in x:
            doc_topic_distr.append(get_document_topics(doc))
        return doc_topic_distr

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def coherence(self, x):
        coherence_score = compute_npmi(self.lda_model, self.corpus, self.id2word)
        return coherence_score

    def perplexity(self, x):
        return np.exp(-1. * self.lda_model.log_perplexity(x))

    def score(self, x, y=None):
        print('Generating scores: K=%d, alpha=%s, beta=%s' % (self.K, self.alpha, self.beta))
        scores = {'perplexity': self.perplexity(x),
                'coherence': self.coherence(x)}
        pprint(scores)
        print('====================\n\n\n')
        return scores





def main(args):
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-k', dest='n_topics', type=int, default=20,
                      help='Size of latent representation (~num topics): default=%default')

    parser.add_option('--train-prefix', type=str, default='train',
                      help='Prefix of train set: default=%default')
    parser.add_option('--test-prefix', type=str, default=None,
                      help='Prefix of test set: default=%default')

    parser.add_option('--alpha', type=float, default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser.add_option('--no-bg', action="store_true", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--dev-folds', type=int, default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', type=int, default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--seed', type=int, default=None,
                      help='Random seed: default=%default')

    options, args = parser.parse_args(args)

    input_dir = args[0]


    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None


    # load the training data
    train_X, vocab, row_selector, train_ids = load_word_counts(input_dir, options.train_prefix)
    options.n_train, vocab_size = train_X.shape



    # load the test data
    if options.test_prefix is not None:
        test_X, _, row_selector, test_ids = load_word_counts(input_dir, options.test_prefix, vocab=vocab)
        n_test, _ = test_X.shape

    else:
        test_X = None
        n_test = 0
        test_prior_covars = None
        test_topic_covars = None

    stopword_list = fh.read_text(os.path.join('stopwords', 'nl_stopwords.txt'))
    stopword_set = {s.strip() for s in stopword_list}
    # stopword_set = set([])

    corpus = gensim.matutils.Sparse2Corpus(train_X)

    train_items = fh.read_jsonlist('data/kinderwens/all.jsonlist')
    train_text = [tokenize(t['text'], stopwords=stopword_set)[0] for t in train_items]
    train_texts = [clean_text(t['text']) for t in train_items]
    vectorizer = CountVectorizer(analyzer='word',       
                                 lowercase=True,                   # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                                )

    data_vectorized = vectorizer.fit_transform(train_texts)

    id2word = gensim.corpora.Dictionary(train_text)
    id2word.filter_extremes(no_below=2, no_above=0.4)
    id2word.compactify()
    text_corpus = [id2word.doc2bow(text, allow_update=True) for text in train_text]
    tfidf_model = gensim.models.TfidfModel(text_corpus)
    tfidf_corpus = tfidf_model[text_corpus]


    dictionary = {i:w for i, w in enumerate(vocab)}
    dictionary = gensim.corpora.Dictionary.from_corpus(corpus, id2word=dictionary)
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=options.n_topics, id2word=dictionary)
    # ldamodel = gensim.models.ldamodel.LdaModel(text_corpus, num_topics=options.n_topics, id2word=id2word)

    search_params = {'K': list(range(3, 4)), 
        # 'alpha': ['symmetric', 'auto', 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 5],
        # 'beta': ['symmetric', 'auto', 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 5]
    }

    test_corpus = gensim.matutils.Sparse2Corpus(test_X)


    lda = LDAModel(3, 'symmetric', 'symmetric', id2word, corpus=text_corpus ,train_text=train_text)
    lda1 = LDAModel(6, 0.3, 0.7, id2word, corpus=text_corpus ,train_text=train_text)
    lda1.fit(text_corpus)
    lda1.score(test_corpus)
    lda2 = LDAModel(6, 'auto', 'auto', id2word, corpus=text_corpus ,train_text=train_text)
    lda2.fit(text_corpus)
    lda2.score(test_corpus)

    lda3 = LDAModel(6, 0.3, 0.7, id2word, corpus=tfidf_corpus,train_text=train_text)
    lda3.fit(tfidf_corpus)
    lda3.score(test_corpus)
    lda4 = LDAModel(6, 'auto', 5, id2word, corpus=tfidf_corpus,train_text=train_text)
    lda4.fit(tfidf_corpus)
    lda4.score(test_corpus)

    # grid_model = GridSearchCV(lda, param_grid=search_params, scoring=LDAModel.score, refit='perplexity')
    # grid_model.fit(text_corpus)
    # pprint(grid_model.score(test_corpus))

    # pandas.DataFrame(grid_model.cv_results_).to_csv('LDA_gridsearch.csv')




    # if options.test_prefix:
    #     test_corpus = gensim.matutils.Sparse2Corpus(test_X)
    #     perplexity = np.exp(-1. * ldamodel.log_perplexity(test_corpus))
    #     print('perplexity:', perplexity)


    # coherence = gensim.models.coherencemodel.CoherenceModel(ldamodel, 
    #     texts=train_text, dictionary=id2word, coherence='c_npmi').get_coherence()

    # print('coherence:', coherence)
    # pprint(ldamodel.print_topics(num_words=10))


    # # Visualize the topics
    LDAvis_prepared1 = pyLDAvis.gensim_models.prepare(lda1.lda_model, text_corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared1, 'lda1.html')

    LDAvis_prepared2 = pyLDAvis.gensim_models.prepare(lda2.lda_model, text_corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared2, 'lda2.html')

    LDAvis_prepared3 = pyLDAvis.gensim_models.prepare(lda3.lda_model, text_corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared3, 'lda3.html')

    LDAvis_prepared4 = pyLDAvis.gensim_models.prepare(lda4.lda_model, text_corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared4, 'lda4.html')


def run_lda(alpha, beta, K):
    stopword_list = fh.read_text(os.path.join('stopwords', 'nl_stopwords.txt'))
    stopword_set = {s.strip() for s in stopword_list}
    stopword_set = set([])

    # corpus = gensim.matutils.Sparse2Corpus(train_X)

    train_items = fh.read_jsonlist('data/kinderwens/train.jsonlist')
    train_text = [tokenize(t['text'], stopwords=stopword_set)[0] for t in train_items]

    id2word = gensim.corpora.Dictionary(train_text)
    text_corpus = [id2word.doc2bow(text) for text in train_text]

    ldamodel = gensim.models.ldamodel.LdaModel(text_corpus, num_topics=options.n_topics, id2word=id2word)
    return ldamodel



def load_word_counts(input_dir, input_prefix, vocab=None):
    print("Loading data")
    # laod the word counts and convert to a dense matrix
    #temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    #X = np.array(temp, dtype='float32')
    X = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).tocsr()
    # load the vocabulary
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    ids = fh.read_json(os.path.join(input_dir, input_prefix + '.ids.json'))

    # filter out empty documents and return a boolean selector for filtering labels and covariates
    #row_selector = np.array(X.sum(axis=1) > 0, dtype=bool)
    row_sums = np.array(X.sum(axis=1)).reshape((n_items,))
    row_selector = np.array(row_sums > 0, dtype=bool)

    print("Found %d non-empty documents" % np.sum(row_selector))
    X = X[row_selector, :]
    ids = [doc_id for i, doc_id in enumerate(ids) if row_selector[i]]

    return X, vocab, row_selector, ids


def load_labels(input_dir, input_prefix, row_selector, options):
    labels = None
    label_type = None
    label_names = None
    n_labels = 0
    # load the label file if given
    if options.labels is not None:
        label_file = os.path.join(input_dir, input_prefix + '.' + options.labels + '.csv')
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            # select the rows that match the non-empty documents (from load_word_counts)
            labels = labels[row_selector, :]
            n, n_labels = labels.shape
            print("Found %d labels" % n_labels)
        else:
            raise(FileNotFoundError("Label file {:s} not found".format(label_file)))

    return labels, label_type, label_names, n_labels



def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[n_dev * options.dev_fold: n_dev * (options.dev_fold +1)]
        else:
            dev_indices = indices[n_dev * options.dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def compute_npmi(model, corpus, id2word, n=10):

    n_terms = len(id2word)
    n_docs = id2word.num_docs
    ref_counts = gensim.matutils.corpus2dense(corpus, num_terms=n_terms, num_docs=n_docs).T

    npmi_means = []
    for idx in range(model.num_topics):
        words = [x[0] for x in model.get_topic_terms(idx)]
        npmi_vals = []
        for word_i, word1 in enumerate(words[:n]):
            for word2 in words[word_i+1:n]:

                col1 = np.array(ref_counts[:, word1] > 0, dtype=int)
                col2 = np.array(ref_counts[:, word2] > 0, dtype=int)
                c1 = col1.sum()
                c2 = col2.sum()
                c12 = np.sum(col1 * col2)
                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        print(str(np.mean(npmi_vals)) + ': ' + ' '.join([id2word[w] for w in words[:n]]))
        npmi_means.append(np.mean(npmi_vals))
    print(np.mean(npmi_means))
    return np.mean(npmi_means)




def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()
    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]
    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else '_' for t in tokens]
    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]
    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]
    counts = Counter()
    unigrams = [t for t in tokens if t != '_']
    counts.update(unigrams)
    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams
    return tokens, counts


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main(sys.argv[1:])
