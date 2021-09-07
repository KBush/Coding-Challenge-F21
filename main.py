import stanza
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')
from senticnet.senticnet import SenticNet
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')
sn = SenticNet()

#Regex match for sentences
f = open('input.txt', 'r')
regex = "[A-Z].*?(?<!Dr)(?<!Mr)[\.!?][\',\", ]"
sentences = re.compile(regex, re.MULTILINE  ).findall(f.read().replace('\n',' ')+' ')
f.close()

total = 0
for s in sentences:
    #Finding sentiment via sentiwordnet
    tokens = nltk.word_tokenize(s[:-1])
    tagged = pos_tag(tokens)
    synsets=None
    pscore = 0
    oscore = 0
    nscore = 0
    for t in tagged:
        if 'NN' in t:
            synsets=wn.synsets(t[0], wn.NOUN)
        elif 'VB' in t:
            synsets=wn.synsets(t[0], wn.VERB)
        elif 'JJ' in t:
            synsets=wn.synsets(t[0], wn.ADJ)
        elif 'RB' in t:
            synsets=wn.synsets(t[0], wn.ADV)
        if not synsets:
            continue
        l = swn.senti_synset(synsets[0].name())
        pscore+=l.pos_score()
        oscore+=l.obj_score()
        nscore+=l.neg_score()
    #Finding seintiment via core NLP
    doc = nlp(s)
    #Combining results
    sscore = 0
    if(pscore>oscore): sscore+=2
    elif(nscore>oscore): sscore+=0
    else: sscore+=1
    sscore+=doc.sentences[0].sentiment
    print(s+': ', sscore)
    total+=sscore
#Output average of methods
print(total/len(sentences))
