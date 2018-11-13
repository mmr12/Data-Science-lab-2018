from gensim.parsing.preprocessing import preprocess_string
import re
import string

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import strip_numeric


def remove_ip(s):
    # Replace all ip adresses with '<ip>' tag
    ip_regexp = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    return re.sub(ip_regexp, '<ip>', s)


def remove_email(s):
    # Replace all email adresses with '<email>' tag
    email_regexp = r"([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})"
    return re.sub(email_regexp, '<email>', s)


def remove_mailto(s):
    # Replace all "<mailto:<email>>" with <email>. Email adresses should be replaced by remove_email first.
    return s.replace("<mailto:<email>>", "<email>")


def remove_url(s):
    # Replace all url's with '<url>' tag
    url_regexp = r"((http|ftp|https):\/\/)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    s = re.sub(url_regexp, '<url>', s)
    # Sometimes url's are inside <> so we need to replace <<url>> with <url>
    return s.replace("<<url>>", "<url>")


def remove_punc(s, exceptions):
    # Remove all punctuation from string with exceptions in list exceptions
    remove = string.punctuation
    for exception in exceptions:
        remove = remove.replace(exception, "")
    # Create the pattern
    pattern = r"[{}]".format(remove)

    return re.sub(pattern, "", s)


def remove_custom_stopwords(s, stopwords):
    for stopword in stopwords:
        s = s.replace(stopword, "")
    return s


def lower_case(s):
    return s.lower()


def preprocess_sentence_fn(s):
    # Preprocess a single sentence to a list of tokens
    punc_exceptions = ['<', '>']
    custom_stopwords = ['dear', 'sincerely', 'thanks', 'yours', 'regards']

    filters = [lower_case,
               remove_ip,
               remove_email,
               remove_mailto,
               remove_url,
               lambda x: remove_punc(x, punc_exceptions),
               remove_stopwords,
               lambda x: remove_custom_stopwords(x, custom_stopwords),
               strip_multiple_whitespaces,
               stem_text,
               strip_numeric]
    out = preprocess_string(s, filters=filters)

    return out


def preprocess_docs_fn(docs):
    # Apply preprocess_sentence_fn to a list of sentances (docs) to get a list of lists
    return [preprocess_sentence_fn(s) for s in docs]
