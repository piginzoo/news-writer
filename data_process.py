
# coding: utf-8

# In[13]:


import jieba
from jieba import analyse 


import os
for filename in os.listdir("./data"):
    print (filename)
    
    with open("./data/"+filename,"r") as f:
        content = f.read()
        seg_list = jieba.cut(content,cut_all=False)    
        seg_content = " ".join([word for word in seg_list])
        with open("./data_cut/"+filename,"w") as d:
            d.write(seg_content)
    


# In[17]:


def loadWord2Vec(filename):
    vocab = []
    embd = []
    cnt = 0
    fr = open(filename,'r')
    line = fr.readline().strip()
    #print line
    word_dim = int(line.split(' ')[1])    
    vocab.append("unk")
    embd.append([0]*word_dim)
    for line in fr :
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print ("loaded word2vec")
    fr.close()
    return vocab,embd


# In[18]:



# vocab,embd = loadWord2Vec("./sgns.financial.word")
# vocab_size = len(vocab)
# embedding_dim = len(embd[0])
# embedding = np.asarray(embd)
# print(vocab_size)
# print(embedding_dim)

def get_inputs():
    
    # inputs和targets的类型都是整数的
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs, targets, learning_rate

    
