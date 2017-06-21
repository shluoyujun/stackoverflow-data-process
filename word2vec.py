import gensim
import nltk
import jieba
import multiprocessing
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

punctuations = ['','\n','\t',',','.',':',';','?','(',')','[',']','&','!','*','@','$','%',"?","'","\"","/","`","\\"]
#jieba.load_userdict("user.txt")
if __name__== "__main__":
  file = open("C://data//w2vdata.txt",encoding="utf-8")
  fw = open("w2vdatamini.re.txt","w",encoding="utf-8")
  
  lines = file.readlines()
  count=0
  for line in lines:
    count=count+1
    if (count%10000==0):
      print (count)
    line=line.lower()
    l1 = nltk.word_tokenize(line)
    #l1 = WhitespaceTokenizer().tokenize(line) #tokenize only by whitespace
    #l1 = jieba.cut(line,cut_all=False)
    l2 = [w for w in l1 if (w not in stopwords.words('english'))] #remove stopping words
    l3 = [w for w in l2 if (w not in punctuations)] #remove punctuations
    lw=""
    for i in l3:
      lw=lw+i+" "
    fw.write(lw+"\n")
    
  fw.close()
  
  model = Word2Vec(LineSentence("w2vdatamini.re.txt"), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
  model.save("SO.model")
  model.wv.save_word2vec_format("SO.vector", binary=False)
