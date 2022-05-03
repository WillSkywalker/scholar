import os
import sys
from optparse import OptionParser
import re
import gensim
import numpy as np
import pandas as pd
from pprint import pprint
import pandas
from gensim.models import Phrases

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

from nltk.tokenize import RegexpTokenizer

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')

STOPWORD_LIST = fh.read_text(os.path.join('stopwords', 'nl_stopwords.txt'))
STOPWORD_SET = {s.strip() for s in STOPWORD_LIST}

T_R = RegexpTokenizer(r'\w+')

zwanger = []
with open('zwanger.txt', 'r') as f:
    for line in f:
        words = T_R.tokenize(line.lower())
        words = [w for w in words if len(w) > 1]
        zwanger.append(words)

nzwanger = zwanger[:]
zwanger_bigram = Phrases(nzwanger, min_count=2)
for idx in range(len(nzwanger)):
    for token in zwanger_bigram[nzwanger[idx]]:
        if '_' in token:
            nzwanger[idx].append(token)

zwanger = [[w for w in t if w not in STOPWORD_SET] for t in zwanger]
nzwanger = [[w for w in t if w not in STOPWORD_SET] for t in nzwanger]
zwanger_dict = gensim.corpora.Dictionary(zwanger)
zwanger_dict.filter_extremes(no_below=2, no_above=0.5)
zwanger_dict.compactify()
nzwanger_dict = gensim.corpora.Dictionary(nzwanger)
nzwanger_dict.filter_extremes(no_below=2, no_above=0.5)
nzwanger_dict.compactify()
zwanger = [zwanger_dict.doc2bow(text) for text in zwanger]
nzwanger = [nzwanger_dict.doc2bow(text) for text in nzwanger]


kinderwens = []
with open('kinderwens.txt', 'r') as f:
    for line in f:
        words = T_R.tokenize(line.lower())
        words = [w for w in words if len(w) > 1]
        kinderwens.append(words)

nkinderwens = kinderwens[:]
kind_bigram = Phrases(nkinderwens, min_count=2)
for idx in range(len(nkinderwens)):
    for token in kind_bigram[nkinderwens[idx]]:
        if '_' in token:
            nkinderwens[idx].append(token)

kinderwens = [[w for w in t if w not in STOPWORD_SET] for t in kinderwens]
nkinderwens = [[w for w in t if w not in STOPWORD_SET] for t in nkinderwens]

kinderwens_dict = gensim.corpora.Dictionary(kinderwens)
kinderwens_dict.filter_extremes(no_below=2, no_above=0.5)
kinderwens_dict.compactify()
kinderwens = [kinderwens_dict.doc2bow(text) for text in kinderwens]

nkinderwens_dict = gensim.corpora.Dictionary(nkinderwens)
nkinderwens_dict.filter_extremes(no_below=2, no_above=0.5)
nkinderwens_dict.compactify()
nkinderwens = [nkinderwens_dict.doc2bow(text) for text in nkinderwens]


class LDAModel(BaseEstimator):



    def __init__(self, K, alpha, beta, corpuses, train_text, passes=1, iterations=50):
        self.lda_model = None
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.corpuses = corpuses
        self.id2word = corpuses[2]
        self.corpus = corpuses[1]
        self.input_corpus = corpuses[0]
        self.train_text = train_text
        self.passes = passes
        self.iterations = iterations

        if 'ngram' in corpuses[0]:
            self.zwanger = nzwanger
            self.zwanger_dict = nzwanger_dict
            self.kinderwens = nkinderwens
            self.kinderwens_dict = nkinderwens_dict
        else:
            self.zwanger = zwanger
            self.zwanger_dict = zwanger_dict
            self.kinderwens = kinderwens
            self.kinderwens_dict = kinderwens_dict




    def fit(self, x, Y=None):
        self.id2word = self.corpuses[2]
        self.corpus = self.corpuses[1]
        self.input_corpus = self.corpuses[0]
        x = self.corpus
        print('Start training: K=%d, corpus=%s, alpha=%s, beta=%s, passes=%s' % (self.K, self.input_corpus,
            self.alpha, self.beta, str(self.passes)))
        if self.alpha == 'auto' or self.beta == 'auto':
            self.lda_model = gensim.models.ldamodel.LdaModel(x, num_topics=self.K,
                alpha=self.alpha, eta=self.beta, id2word=self.id2word, passes=self.passes, iterations=self.iterations)
        else:
            self.lda_model = gensim.models.ldamulticore.LdaMulticore(x, 
                num_topics=self.K, alpha=self.alpha, eta=self.beta, 
                id2word=self.id2word, passes=self.passes, iterations=self.iterations, workers=2)

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

    def coherence_zwanger(self, x):
        coherence_score = compute_npmi(self.lda_model, self.zwanger, self.zwanger_dict)
        return coherence_score

    def coherence_kinderwens(self, x):
        coherence_score = compute_npmi(self.lda_model, self.kinderwens, self.kinderwens_dict)
        return coherence_score


    def perplexity(self, x):
        return np.exp(-1. * self.lda_model.log_perplexity(x))

    def score(self, x, y=None):
        print('Generating scores: K=%d, corpus=%s, alpha=%s, beta=%s' % (self.K, self.input_corpus,self.alpha, self.beta))
        scores = {'perplexity': self.perplexity(x),
                'coherence_internal': self.coherence(x),
                'coherence_zwanger': self.coherence_zwanger(x),
                'coherence_kinderwens': self.coherence_kinderwens(x)}
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
    ngrams_text = [tokenize(t['text'])[0] for t in train_items]
    bigram = Phrases(ngrams_text, min_count=2)
    for idx in range(len(ngrams_text)):
        for token in bigram[ngrams_text[idx]]:
            if '_' in token:
                ngrams_text[idx].append(token)
    ngrams_text = [[w for w in t if w not in stopword_set] for t in ngrams_text]
    id2word_ngram = gensim.corpora.Dictionary(ngrams_text)
    id2word_ngram.filter_extremes(no_below=2, no_above=0.5)
    id2word_ngram.compactify()
    ngram_corpus = [id2word_ngram.doc2bow(text) for text in ngrams_text]
    ngram_tfidf_model = gensim.models.TfidfModel(ngram_corpus)
    ngram_tfidf_corpus = ngram_tfidf_model[ngram_corpus]

    train_text = [tokenize(t['text'], stopwords=stopword_set)[0] for t in train_items]
    train_texts = [clean_text(t['text']) for t in train_items]
    vectorizer = CountVectorizer(analyzer='word',       
                                 lowercase=True,                   # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{2,}',  # num chars > 2
                                )

    id2word = gensim.corpora.Dictionary(train_text)
    id2word.filter_extremes(no_below=2, no_above=0.5)
    id2word.compactify()
    text_corpus = [id2word.doc2bow(text) for text in train_text]
    tfidf_model = gensim.models.TfidfModel(text_corpus)
    tfidf_corpus = tfidf_model[text_corpus]


# ======= Q1

    q1_train_items = fh.read_jsonlist('data/kinderwens_q1/all.jsonlist')
    q1_ngrams_text = [tokenize(t['text'])[0] for t in q1_train_items]
    q1_bigram = Phrases(q1_ngrams_text, min_count=2)
    for idx in range(len(q1_ngrams_text)):
        for token in q1_bigram[q1_ngrams_text[idx]]:
            if '_' in token:
                q1_ngrams_text[idx].append(token)
    q1_ngrams_text = [[w for w in t if w not in stopword_set] for t in q1_ngrams_text]
    q1_id2word_ngram = gensim.corpora.Dictionary(q1_ngrams_text)
    q1_id2word_ngram.filter_extremes(no_below=2, no_above=0.5)
    q1_id2word_ngram.compactify()
    q1_ngram_corpus = [q1_id2word_ngram.doc2bow(text) for text in q1_ngrams_text]
    q1_ngram_tfidf_model = gensim.models.TfidfModel(q1_ngram_corpus)
    q1_ngram_tfidf_corpus = q1_ngram_tfidf_model[q1_ngram_corpus]

    q1_train_text = [tokenize(t['text'], stopwords=stopword_set)[0] for t in q1_train_items]
    q1_train_texts = [clean_text(t['text']) for t in q1_train_items]


    q1_id2word = gensim.corpora.Dictionary(q1_train_text)
    q1_id2word.filter_extremes(no_below=2, no_above=0.5)
    q1_id2word.compactify()
    q1_text_corpus = [q1_id2word.doc2bow(text) for text in q1_train_text]
    q1_tfidf_model = gensim.models.TfidfModel(q1_text_corpus)
    q1_tfidf_corpus = q1_tfidf_model[q1_text_corpus]


    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=options.n_topics, id2word=dictionary)
    # ldamodel = gensim.models.ldamodel.LdaModel(text_corpus, num_topics=options.n_topics, id2word=id2word)

    search_params = {'K': list(range(3, 13)), 
        'alpha': ['symmetric', 'auto', 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 5],
        'beta': ['symmetric', 'auto', 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 5],
        'passes': [1, 2, 5, 10, 20, 50, 100],
        'iterations': [50, 100, 200, 400],
        'corpuses': [('all-original', text_corpus, id2word),
                     ('all-tfidf', tfidf_corpus, id2word),
                     ('all-original-ngrams', ngram_corpus, id2word_ngram),
                     ('all-tfidf-ngrams', ngram_tfidf_corpus, id2word_ngram),
                     ('q1-original', q1_text_corpus, q1_id2word),
                     ('q1-tfidf', q1_tfidf_corpus, q1_id2word),
                     ('q1-original-ngrams', q1_ngram_corpus, q1_id2word_ngram),
                     ('q1-tfidf-ngrams', q1_ngram_tfidf_corpus, q1_id2word_ngram),]

    }

    test_corpus = gensim.matutils.Sparse2Corpus(test_X)


    lda = LDAModel(3, 'symmetric', 'symmetric', ('all-original', text_corpus, id2word),
        train_text=train_text)
    # lda1 = LDAModel(6, 0.3, 0.7, id2word, corpus=text_corpus ,train_text=train_text)
    # lda1.fit(text_corpus)
    # lda1.score(test_corpus)
    # lda2 = LDAModel(6, 'auto', 'auto', id2word, corpus=text_corpus ,train_text=train_text)
    # lda2.fit(text_corpus)
    # lda2.score(test_corpus)

    # lda3 = LDAModel(6, 0.3, 0.7, id2word, corpus=tfidf_corpus,train_text=train_text)
    # lda3.fit(tfidf_corpus)
    # lda3.score(test_corpus)
    # lda4 = LDAModel(6, 'auto', 5, id2word, corpus=tfidf_corpus,train_text=train_text)
    # lda4.fit(tfidf_corpus)
    # lda4.score(test_corpus)

    grid_model = GridSearchCV(lda, param_grid=search_params, scoring=LDAModel.score, refit='perplexity')
    grid_model.fit(text_corpus)
    # pprint(grid_model.score(test_corpus))

    pandas.DataFrame(grid_model.cv_results_).to_csv('LDA_gridsearch_0522.csv')




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
        words = [id2word.token2id[x[0]] for x in model.show_topic(idx)]
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
