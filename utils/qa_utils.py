from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn


LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')


def get_lemmas(phrase, pos=wn.VERB):
    return [LEMMATIZER.lemmatize(w, pos) for w in phrase.split(' ')]

def get_lemmas_vo(phrase, pos=wn.VERB):
    return [LEMMATIZER.lemmatize(w, pos) if len(wn.synsets(w, pos)) > 0 else w for w in phrase.split(' ')]

def get_lemmas_only_verbs(phrase, pos=wn.VERB):
    return set([w for w in get_lemmas(phrase, pos) if len(wn.synsets(w, pos)) > 0])

def get_only_verbs(phrase, pos=wn.VERB):
    return [w for w in phrase.split(' ') if len(wn.synsets(w, pos)) > 0]

def get_lemmas_no_stopwords(phrase, pos=wn.VERB):
    return set([w for w in get_lemmas(phrase, pos) if w not in STOPWORDS])


def aligned_args(q, a):
    q_arg = get_lemmas_no_stopwords(q[2], wn.NOUN)
    if q_arg == get_lemmas_no_stopwords(a[2], wn.NOUN):
        return True
    if q_arg == get_lemmas_no_stopwords(a[0], wn.NOUN):
        return False
    raise Exception('HORRIBLE BUG!!!')


def diff(q, a):
    q_tokens = q.split(' ')
    a_tokens = a.split(' ')
    min_len = min(len(q_tokens), len(a_tokens))
    
    for start, (qw, qa) in enumerate(zip(q_tokens[:min_len], a_tokens[:min_len])):
        if qw != qa:
            break
    
    for end, (qw, qa) in enumerate(zip(q_tokens[::-1][:min_len], a_tokens[::-1][:min_len])):
        if qw != qa:
            break
    
    if end > 0:
        q_tokens = q_tokens[start:-end]
        a_tokens = a_tokens[start:-end]
    else:
        q_tokens = q_tokens[start:]
        a_tokens = a_tokens[start:]
    
    return ' '.join(q_tokens), ' '.join(a_tokens)

