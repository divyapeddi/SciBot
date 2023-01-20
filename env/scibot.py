import io
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from nltk.stem import WordNetLemmatizer

f=open('content\science.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts the text to lowercase
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "hey",)
GREETING_RESPONSES = ("hi", "hey", "hi there", "hello")
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response = robo_response + "I am sorry! I am not able to help you with this question."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


flag=True
print("SciBot: I can help you with questions about Science. If you want to exit, type quit.")
while flag:
    user_response = input("You: ")
    user_response = user_response.lower()
    if user_response != 'quit':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("SciBot: You are welcome.")
        else:
            if greeting(user_response) is not None:
                print("SciBot: " + greeting(user_response))
            else:
                print("SciBot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("SciBot: Bye!")