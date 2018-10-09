# -*- coding: utf-8 -*-
import numpy as np
import seaborn
import time
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import wordcloud
from sklearn.decomposition import LatentDirichletAllocation,PCA
from sklearn.manifold import Isomap,LocallyLinearEmbedding,SpectralEmbedding,TSNE,MDS

########### data preprocessing
def remove(word):
    word_remove=[i.strip('"') for i in word]
    return word_remove

word_article=np.loadtxt("D:\\yuany_course\\project1\\NIPS_1987-2015.csv",dtype=np.str,delimiter=',')
word=remove(word_article[1:,0].tolist())
word_count_per=word_article[1:,1:].astype(np.float)
word_count=word_count_per.sum(axis=1)
## clean data
doc_count=word_count_per.sum(axis=0)
doc_count_0_idx=np.arange(len(doc_count))[doc_count==0]
article=remove(np.delete(word_article[0,1:],doc_count_0_idx))
article_year=[i[:4] for i in article]
years=sorted(list(set(article_year)))
doc_word=np.delete(word_article[1:,1:],doc_count_0_idx,axis=1).astype(np.float).T


########### some basic analysis
######### 1. wordcloud
### one year
year_one='1988'
year_idx=np.arange(len(article_year))[np.array(article_year)==year_one]
word_count_one_year=word_article[1:,year_idx+1].astype(np.float).sum(axis=1)
word_freq=dict((word[i],word_count_one_year[i]) for i in range(len(word)))
# ### all year
word_freq=dict((word[i],word_count[i]) for i in range(len(word)))
wc=wordcloud.WordCloud(max_words=1000,background_color='white',max_font_size=40,random_state=42)
wc.generate_from_frequencies(word_freq)
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud.png')
# plt.show()
# ######### 2.number of accpted paper by year
paper_year=Counter(article_year).values()
plt.bar(range(len(years)),paper_year)
plt.xticks(range(len(years)),np.arange(87,87+len(years))%100)
plt.xlabel('year')
plt.ylabel('Accepted Paper')
plt.title('NIPS')
plt.savefig('AcceptedPaper.pdf')
# ######## 3.the comparision between 1987 and 2015
year_idx=np.arange(len(article_year))[np.array(article_year)=='1987']
word_count_1987=word_article[1:,year_idx+1].astype(np.float).sum(axis=1)
year_idx=np.arange(len(article_year))[np.array(article_year)=='2015']
word_count_2015=word_article[1:,year_idx+1].astype(np.float).sum(axis=1)
top_10_word_2015_idx=word_count_2015.argsort()[:-11:-1]
bar_width=0.2
plt.bar(np.arange(10),word_count_2015[top_10_word_2015_idx],bar_width,color='b',label='2015')
plt.bar(np.arange(10)+bar_width,word_count_1987[top_10_word_2015_idx],bar_width,color='r',label='1987')
plt.xlabel('word')
plt.ylabel('count')
plt.title('WordCount Comparision between 1987 and 2015')
plt.xticks(np.arange(10)+bar_width,np.array(word)[top_10_word_2015_idx])
plt.legend(loc='best')
plt.savefig('1987_2015.png')

############ lda
k=5
lda=LatentDirichletAllocation(n_topics=k,max_iter=50,evaluate_every=1,n_jobs=1)
doc_top=lda.fit_transform(doc_word)
top_word=lda.components_
print(doc_top.shape)
print(top_word.shape)

def get_topN(topic,n):
    return topic.argsort()[:-n-1:-1]

with open('5TopicTop10words.txt','a+') as f:
    f.readline()
    for top_idx,topic in enumerate(top_word):
        f.write('top_index:{}:'.format(top_idx))
        for top in get_topN(topic,10):
            f.write(word[top]+' ')
        f.write('\n')

############# topic change by year
n=3
doc_top_n=doc_top.argsort(axis=1)[:,:-n-1:-1]
year_top={}
for i in range(k):
    for year in article_year:
        year_top.setdefault(year+'_'+str(i),0)
for year,top in zip(article_year,doc_top_n):
    for top_id in top:
        year_top[year+'_'+str(top_id)]+=1
for year,count in year_top.items():
    if 'top_{}_year'.format(year[-1]) not in locals().keys():
        exec('top_{}_year=[]'.format(year[-1]))
    eval('top_{}_year.append({})'.format(year[-1],count))
plt.figure(1)
for i in range(k):
    exec('plt.plot(years,top_{}_year,label={})'.format(i,i))
plt.xlabel('year')
plt.ylabel('times')
plt.title('Trend of Topics')
plt.legend(loc='best')
plt.savefig('topByYear.pdf',dpi=4180)
# plt.show()


############# select topic number by perplexity
# log_perplexity=[]
# for i in range(2,16):
#     print('{} topic'.format(i))
#     lda=LatentDirichletAllocation(n_topics=i,max_iter=10,n_jobs=1)
#     lda.fit(doc_word)
#     log_per=np.log(lda.perplexity(doc_word))
#     print(log_per)
#     log_perplexity.append(log_per)
#
# plt.figure(2)
# plt.plot(range(2,16),log_perplexity)
# plt.xlabel('topic_number')
# plt.ylabel('log_perplexity')
# plt.title('The log_perplexity for Different Topic Number')
# plt.xticks(range(2,16))
# plt.savefig('log_perplexity.png')
# plt.show()
#
###################### 8种可视化 LLE,LTSA,HLLE,MLLE,ISO,MDS,SEM,pca
LLE_dict=dict(zip(['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE'],
                    ['standard', 'ltsa', 'hessian', 'modified']))
methods=['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE','Isomap','MDS','SpectralEmbedding','PCA']
doc_top=np.load('doc_top.npy')
top_word=np.load('top_word.npy')
fig_doc=plt.figure(1)
fig_doc.suptitle('Manifold Learning for document in NIPS(after LDA)')
fig_word=plt.figure(2)
fig_word.suptitle('Manifold Learning for word in NIPS(after LDA)')
for i,method in enumerate(methods):
    print('{} starts......'.format(method))
    if method in LLE_dict.keys():
        manifold=LocallyLinearEmbedding(n_neighbors=6,method=LLE_dict[method],eigen_solver='dense')
    elif method=='MDS':
        manifold=MDS(n_init=1,max_iter=100)
    elif method=='PCA':
        manifold=PCA(n_components=2)
    else:
        exec('manifold={}(n_neighbors=6)'.format(method))
    doc_2D=manifold.fit_transform(doc_top)
    np.save('doc_2d_{}'.format(method),doc_2D)
    word_2D=manifold.fit_transform(top_word.T)
    np.save('word_2d_{}'.format(method),word_2D)
    ax=fig_doc.add_subplot(241+i)
    ax.scatter(doc_2D[:,0],doc_2D[:,1],s=1)
    ax.set_title(method)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    ax=fig_word.add_subplot(241+i)
    ax.scatter(word_2D[:,0],word_2D[:,1],s=1,c='black')
    ax.set_title(method)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
fig_doc.savefig('Manifold_of_doc.png')
fig_word.savefig('Manifold_of_word.png')

# plt.figure(5)
# plt.plot(doc_top_pca[:,0],doc_top_pca[:,1],'rx',label='doc')
# for i,article_i in enumerate(article):
#     if i%10==0:
#         plt.text(doc_top_pca[i,0],doc_top_pca[i,1],article_i)
# plt.legend(loc='best')
# plt.savefig('year_pca.pdf',dpi=4180)
# plt.figure(6)
# plt.plot(word_top_pca[:,0],word_top_pca[:,1],'x',label='word')
# for i,word_i in enumerate(word):
#     if i%10==0:
#         plt.text(word_top_pca[i,0],word_top_pca[i,1],word_i)
# plt.legend(loc='best')
# plt.savefig('word_pca.pdf',dpi=4180)
# plt.show()
# plt.close()
