import nltk

'''
Мы порешаем сейчас POS Tagging для английского.

Будем работать с таким набором тегов:

ADJ - adjective (new, good, high, ...)
ADP - adposition (on, of, at, ...)
ADV - adverb (really, already, still, ...)
CONJ - conjunction (and, or, but, ...)
DET - determiner, article (the, a, some, ...)
NOUN - noun (year, home, costs, ...)
NUM - numeral (twenty-four, fourth, 1991, ...)
PRT - particle (at, on, out, ...)
PRON - pronoun (he, their, her, ...)
VERB - verb (is, say, told, ...)
. - punctuation marks (. , ;)
X - other (ersatz, esprit, dunno, ...)
'''


def load_brown_dataset():
    '''
    Функция возвращает какой то текстовый датасет
    Пример использование:
    for word, tag in data[0]:
       print('{:15}\t{}'.format(word, tag))
    '''
    nltk.download('brown')
    nltk.download('universal_tagset')
    data = nltk.corpus.brown.tagged_sents(tagset='universal')
    
    return data


def tokenize_srt(text):
    '''
    Токенизация предложения
    Пример использования:
    text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        print(sentence)
        print()
        
    '''
    
    sentences = nltk.sent_tokenize(text)
    
    return sentences


