
# coding: utf-8

# ## Data Visualization

# In[ ]:


#import the data file after data cleaning
import pandas as pd
data = pd.read_csv('general_movie_data.csv')


# ## Explore the difference brought by movie genres 

genre_set = ['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western']
genre =data[['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western','Release_Day']]


genre_by_year = genre.groupby('Release_Day').sum()  
genresum_by_year = genre_by_year.sum()/len(genre)
genresum_by_year = genresum_by_year.sort_values(ascending=False)
drama = genresum_by_year[0]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize=(15,11))
ax = plt.subplot(1,1,1)     
ax = genresum_by_year.plot.bar()
plt.xticks(rotation=60)
plt.title('Film genre by year', fontsize=18)   
plt.xlabel('count', fontsize=18)   
plt.ylabel('genre', fontsize=18)    

# just a few movies that are not included into the Drara genre, so when we go on further analysis, we may delete drama from the genre_set
df = pd.read_csv('film_list.csv')
df[df['genre']== 'Drama']


# ## Analysis the calender(year,month) difference for different movie genre

genre_by_year = genre_by_year[['Drama','Comedy','Thriller','Action',
                               'Adventure','Crime', 'Romance','Horror']]
new_year = genre_by_year
fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot(1,1,1)
plt.plot(genre_by_year)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Film count', fontsize=12)
plt.title('Film count by year', fontsize=15)
#plt.legend(loc='best',ncol=2) #https://blog.csdn.net/you_are_my_dream/article/details/53440964
plt.legend(['Drama','Comedy','Thriller','Action', 'Adventure','Crime', 'Romance','Horror'], loc='best',ncol=2)


# Because the data only includes movies released before 2018.06, so we double the values for 2018. Then we replot the figure
new_year.iloc[5,:] = new_year.iloc[4].values*2
new_year_1 = new_year.drop(index = 2018.0)
new_year_1


fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot(1,1,1)
plt.plot(new_year_1)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Film count', fontsize=12)
plt.title('Film count by year', fontsize=15)
#plt.legend(loc='best',ncol=2) #https://blog.csdn.net/you_are_my_dream/article/details/53440964
plt.legend(['Drama','Comedy','Thriller','Action', 'Adventure','Crime', 'Romance','Horror'], loc='best',ncol=2) 

month =  data[['name', 'Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western','Release_Day','Release_Month']]
month = month[month['Release_Day'].isin([2014.0,2015.0,2016.0,2017.0])]
genre_by_month = month.groupby('Release_Month').sum()  
genresum_by_month = genre_by_month.sum()/len(genre)
genresum_by_month = genresum_by_month.sort_values(ascending=False)
genresum_by_month = genresum_by_month[1:]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize=(15,11))  
ax = plt.subplot(1,1,1)
ax = genresum_by_month.plot.bar()
plt.xticks(rotation=60)
plt.title('Film genre by month', fontsize=18)    
plt.xlabel('count', fontsize=18)   
plt.ylabel('genre', fontsize=18)    

genre_by_month = genre_by_month[['Drama','Comedy','Thriller','Action',
                               'Adventure','Crime', 'Romance','Horror']]
new_month = genre_by_month

fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot(1,1,1)
plt.plot(new_month)
plt.xlabel('month', fontsize=12)
plt.ylabel('Film count', fontsize=12)
plt.title('Film count by month', fontsize=15)
plt.legend(['Drama','Comedy','Thriller','Action', 'Adventure','Crime', 'Romance','Horror'], loc='best',ncol=2)


# ## Analysis the profitbility for different movie genre

profit_data = data
profit_data['Budget'] = profit_data['Budget']/(10**6)
profit_data['profit'] =  profit_data['Gross'] - profit_data['Budget']

genre_df = data[['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western']]
profit_df = pd.DataFrame()
profit_df = pd.concat([genre_df,profit_data['profit']],axis=1)


profit_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    profit_by_genre.loc[genre]=profit_df[[genre,'profit']].groupby(genre, as_index=False).mean().loc[1,'profit']
print(profit_by_genre)

budget_df = pd.concat([genre_df,profit_data['Budget']],axis=1)
budget_df.head(2)
budget_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    budget_by_genre.loc[genre]=budget_df.loc[:,[genre,'Budget']].groupby(genre,as_index=False).mean().loc[1,'Budget']
print(budget_by_genre)

profit_rate = pd.concat([profit_by_genre, budget_by_genre],axis=1)
profit_rate.columns=['profit','budget']  

profit_rate['profit_rate'] = (profit_rate['profit']/profit_rate['budget'])*100
profit_rate.sort_values(by=['profit','profit_rate'], ascending=False, inplace=True)

x = list(range(len(profit_rate.index)))
labels = profit_rate.index
profit_rate

fig = plt.figure(figsize=(18,13))
ax1 = fig.add_subplot(111)
ax1.bar(x, profit_rate['profit'],label='profit',alpha=0.7)
plt.xticks(x,labels,rotation=60,fontsize=12)
plt.yticks(fontsize=12)
ax1.set_title('Profit by genres', fontsize=20)
ax1.set_ylabel('Film Profit',fontsize=18)
ax1.set_xlabel('Genre',fontsize=18)
#ax1.set_ylim(0,1.2e11)
ax1.legend(loc=2,fontsize=15)

import matplotlib.ticker as mtick
ax2 = ax1.twinx()
ax2.plot(x, profit_rate['profit_rate'],'ro-',lw=2,label='profit_rate')
fmt='%.2f%%'
yticks = mtick.FormatStrFormatter(fmt)
ax2.yaxis.set_major_formatter(yticks)
plt.xticks(x,labels,fontsize=12,rotation=60)
plt.yticks(fontsize=15)
ax2.set_ylabel('Profit_rate',fontsize=18)
ax2.legend(loc=1,fontsize=15)


# ## Analysis the Open Weekend Revenue

studio = ['Lionsgate',
 'Paramount Pictures',
 'Twentieth Century Fox',
 'Blumhouse Productions',
 'Warner Bros.',
 'Columbia Pictures Corporation',
 'LStar Capital',
 'Walt Disney Pictures']


def if_famous_studio(string):
    try:
        for i in string.split(','):
            if i.strip() in studio:
                return 1
        return 0
    except:
        return None
data['famous_studio'] = data.apply(lambda x :if_famous_studio(x['Studio']),axis = 1)


data.to_csv('final_data_v1.csv')

genre_df = data[['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western']]
open_df = pd.DataFrame()
open_df = pd.concat([genre_df,open_data['Open']],axis=1)

open_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    open_by_genre.loc[genre]=open_df[[genre,'Open']].groupby(genre, as_index=False).mean().loc[1,'Open']
open_by_genre = open_by_genre.sort_values(ascending=False)
open_by_genre


# plot average open weekend revenue for different genre
fig = plt.figure(figsize=(18,13))
ax1 = fig.add_subplot(111)
plt.bar(x, open_by_genre,label='open',alpha=0.7)
plt.xticks(x,open_by_genre.index,rotation=60,fontsize=12)
plt.yticks(fontsize=12)
ax1.set_title('open by genres', fontsize=20)
ax1.set_ylabel('film open',fontsize=18)
ax1.set_xlabel('Genre',fontsize=18)
#ax1.set_ylim(0,1.2e11)
ax1.legend(loc=2,fontsize=15)


data_open = data[['Open','Release_Month']]
mean_open = data_open.groupby('Release_Month').mean()
mean_open.plot()

# Budget & Open Relationship
import seaborn as sns
sns.set(color_codes=True)
fig = plt.figure(figsize=(17,5))
ax1 = sns.regplot(x='Budget', y='Open', data=data)
#ax1.text(400,2e9,'r=0.64',fontsize=15)
plt.title('Budget vs Open',fontsize=15)
plt.xlabel('Budget',fontsize=13)
plt.ylabel('Open',fontsize=13)

# Open Weekend Revenue VS. Famous_studio
open_studio = data[['Open','famous_studio']]
mean_open_studio = open_studio.groupby('famous_studio').mean()
mean_open_studio.plot(kind='bar')


# Open Weekend Revenue VS. Famous_director
open_director = data[['Open','famous_director']]
mean_open_director = open_director.groupby('famous_director').mean()
mean_open_director.plot(kind='bar')


# ## Coun, Gross, Open Weekend Revenue, Meta Score and Release Month Plot


#run these code in the terminal with command bokeh serve --show plot.py
def process_data():
    import pandas as pd
    import numpy as np
    gross = pd.read_csv('gross.csv',index_col=0)
    open_week = pd.read_csv('open_week.csv',index_col=0)
    count = pd.read_csv('count.csv',index_col=0)
    group= pd.read_csv('group.csv',index_col=0)
    
    # Make the column names ints not strings for handling
    columns = list(gross.columns)
    months = [(i+1) for i in range(12)]
    rename_dict = dict(zip(columns, months))

    gross = gross.rename(columns=rename_dict)
    open_week = open_week.rename(columns=rename_dict)
    count = count.rename(columns=rename_dict)
    group = group.rename(columns=rename_dict)

    group_list = list(group.Group.unique())

    # Turn population into bubble sizes. Use min_size and factor to tweak.
    scale_factor = 10
    count_size = np.sqrt(count/ np.pi)*50
    min_size = 0.1
    count_size = count_size.where(count_size >= min_size).fillna(min_size)

    return gross, open_week, count_size, group, months, group_list


import pandas as pd
import numpy as np
from bokeh.core.properties import field
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import (ColumnDataSource, HoverTool, SingleIntervalTicker,
                          Slider, Button, Label, CategoricalColorMapper)
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, show


gross_df, open_week_df, count_df_size, group_df, months, group_list = process_data()

df = pd.concat({'gross': gross_df,
                'open_week': open_week_df,
                'count': count_df_size},
               axis=1)

data = {}

group_df.rename({'Group':'Quality'}, axis='columns', inplace=True)
for month in months:
    df_month = df.iloc[:,df.columns.get_level_values(1)==month]
    df_month.columns = df_month.columns.droplevel(1)
    data[month] = df_month.join(group_df.Quality).reset_index().to_dict('series')

source = ColumnDataSource(data=data[months[0]])

plot = figure(x_range=(-10, 100), y_range=(-5, 45), title='Movie Data', plot_height=200)
plot.xaxis.ticker = SingleIntervalTicker(interval=10)
plot.xaxis.axis_label = "Average Box Office"
plot.yaxis.ticker = SingleIntervalTicker(interval=5)
plot.yaxis.axis_label = "Average Open Week Revenue"

label = Label(x=1.1, y=18, text=str(months[0]), text_font_size='70pt', text_color='#000000')
plot.add_layout(label)

color_mapper = CategoricalColorMapper(palette=Spectral4, factors=group_list)
plot.circle(
    x='gross',
    y='open_week',
    size='count',
    source=source,
    fill_color={'field': 'Quality', 'transform': color_mapper},
    fill_alpha=0.8,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
    legend=field('Quality'),
)
#plot.add_tools(HoverTool(tooltips="@Country", show_arrow=False, point_policy='follow_mouse'))


def animate_update():
    month = slider.value + 1
    if month > months[-1]:
        month = months[0]
    slider.value = month


def slider_update(attrname, old, new):
    month = slider.value
    label.text = str(month)
    source.data = data[month]

slider = Slider(start=months[0], end=months[-1], value=months[0], step=1, title="Month")
slider.on_change('value', slider_update)

callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 200)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60)
button.on_click(animate)

layout = layout([
    [plot],
    [slider, button],
], sizing_mode='scale_width')

curdoc().add_root(layout)
curdoc().title = "Movie"

df2 = pd.read_csv('release_date.csv')
result = pd.merge(df1[['Studio','name']], df2[['Release_date','Release_Day','name']], on='name')
df3 = result
# split data by its released day, and find the top 20 studios in each year.
df_2014 = df3[df3['Release_Day']==2014]
df_2015 = df3[df3['Release_Day']==2015]
df_2016 = df3[df3['Release_Day']==2016]
df_2017 = df3[df3['Release_Day']==2017]
all_studio = list()
for i in range(len(df_2014)):
    all_studio.extend(str(df_2014.iloc[i]['Studio']).split(','))
    c = collections.Counter(all_studio)
    studio_common_2014 = c.most_common(20)
m_2014 = dict(studio_common_2014)
value_2014 = list(m_2014.values())
studio_2014 = list(m_2014.keys())
all_studio = list()
for i in range(len(df_2015)):
    all_studio.extend(str(df_2015.iloc[i]['Studio']).split(','))
    c = collections.Counter(all_studio)
    studio_common_2015 = c.most_common(20)
m_2015 = dict(studio_common_2015)
value_2015 = list(m_2015.values())
studio_2015 = list(m_2015.keys())
all_studio = list()
for i in range(len(df_2016)):
    all_studio.extend(str(df_2016.iloc[i]['Studio']).split(','))
    c = collections.Counter(all_studio)
    studio_common_2016 = c.most_common(20)
m_2016 = dict(studio_common_2016[1:])
value_2016 = list(m_2016.values())
studio_2016 = list(m_2016.keys())
all_studio = list()
for i in range(len(df_2017)):
    all_studio.extend(str(df_2017.iloc[i]['Studio']).split(','))
    c = collections.Counter(all_studio)
    studio_common_2017 = c.most_common(20)
m_2017 = dict(studio_common_2017)
value_2017 = list(m_2017.values())
studio_2017 = list(m_2017.keys())

df_2017 = pd.DataFrame({'number_2017':value_2017,'studio':studio_2017})
df_2016 = pd.DataFrame({'number_2016':value_2016,'studio':studio_2016})
df_2015 = pd.DataFrame({'number_2015':value_2015,'studio':studio_2015})
df_2014 = pd.DataFrame({'number_2015':value_2014,'studio':studio_2014})

# Then merge them to find how the amount of movies released by these studios changes through years.
dfm = pd.merge(df_2015,df_2016,on = 'studio')
dfm = pd.merge(dfm,df_2014,on = 'studio')
dfm = pd.merge(dfm,df_2017,on = 'studio')

# plot the trend
dfm = dfm.drop(1)
dff = dfm.set_index('studio').T
fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot(1,1,1)
plt.plot(dff)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Film count', fontsize=12)
plt.title('Film count by year', fontsize=15)
plt.legend(dfm['studio'], loc='best',ncol=2)

# get average revenue in open weeks for a genre
data = pd.read_csv('final_data_v2.csv')
genre_df = data[['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western']]
open_df = pd.DataFrame()#创建空的数据框
open_df = pd.concat([genre_df,data['meta_score']],axis=1)
genre_set = ['Adventure', 'Action','Family', 'Fantasy', 'Animation', 'Comedy', 'Biography', 'Drama','Musical', 'Crime', 'Mystery', 'Thriller', 'Sport', 'Horror', 'Sci-Fi','History', 'Romance', 'Music', 'Western']
open_by_genre = pd.Series(index=genre_set)
for genre in genre_set:
    open_by_genre.loc[genre]=open_df[[genre,'meta_score']].groupby(genre, as_index=False).mean().loc[1,'meta_score']
open_by_genre = open_by_genre.sort_values(ascending=False)
open_by_genre

# plot the trend for each genre
fig = plt.figure(figsize=(9,9))
plt.bar(open_by_genre.index,open_by_genre,label='open',alpha=0.7)
# plt.xticks(x,open_by_genre.index,rotation=60,fontsize=12)
plt.yticks(fontsize=12)
ax1.set_title('opent by genres', fontsize=20)
ax1.set_ylabel('film open',fontsize=18)
ax1.set_xlabel('Genre',fontsize=10)
ax1.legend(loc=2,fontsize=15)

# plot the average revenue in open week for each month
data_open = data[['meta_score','Release_Month']]
mean_open = data_open.groupby('Release_Month').mean()
mean_open.plot()

# compare movies' meta score between famous and normal studios
data_open = data[['meta_score','famous_studio']]
mean_open = data_open.groupby('famous_studio')
mean_open.boxplot()

# compare movies' meta score between famous and normal directors
data_open = data[['meta_score','famous_director']]
mean_open = data_open.groupby('famous_director')
mean_open.boxplot()

# compare movies' meta score between famous and normal actors
data_open = data[['meta_score','famous_actor']]
mean_open = data_open.groupby('famous_actor')
mean_open.boxplot()


# ## Review Comparision and Word Cloud

# In[ ]:


from scipy.misc import imread
from tqdm import tqdm
import json
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import inaugural
from nltk import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import collections
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import OrderedDict
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis.gensim
import pprint
import re
import warnings
warnings.filterwarnings('ignore')


# In[11]:


#Read reviews
audience_df = pd.read_csv('au_text.csv')
critic_df = pd.read_csv('cr_text.csv')


# In[12]:


audience_df.info()


# In[13]:


critic_df.info()


# In[14]:


def cleaned_word_list(df):    
    comment_ls = df['content']
    text = ''.join(comment_ls)
    
    #update film stop words
    update_stop_word = ['film','movie','films','movies','one','never','may',
                        'even','much','well','would','feels','makes','make','time','full',
                       'enough','review','franchise','really','going','go']
    stop_words = stopwords.words() + update_stop_word
    
    #Remove words not printable
    ascii_chars = set(string.printable)  
    def remove_non_ascii_prinatble_from_list(list_of_words):
        return [word for word in list_of_words 
                if all(char in ascii_chars for char in word)]
    tw_tokenizer = TweetTokenizer()
    words = tw_tokenizer.tokenize(text)
    lowercase_words = [word.lower() for word in words
                      if word.lower() not in stop_words and word.isalpha()]
    low_words = remove_non_ascii_prinatble_from_list(lowercase_words)
    low_words = [word for word in low_words if word not in stop_words]
    return low_words


# In[15]:


critic_word_list = cleaned_word_list(critic_df)


# In[ ]:


critic_dict = {'content':critic_word_list}


# In[ ]:


#Save .json file
with open('audience_good_cr_text_cleaned.json', 'w') as fp:
            json.dump(critic_dict, fp)


# In[6]:


audience_word_list = cleaned_word_list(audience_df)


# In[ ]:


audience_dict = {'content':audience_word_list}


# In[ ]:


#Save .json file
with open('audience_good_au_text_cleaned.json', 'w') as fp:
            json.dump(audience_dict, fp)


# In[ ]:


#Try to filter the films for which the critics give low ratings while the audience give high ratings
#By drawing the wordcloud of these filtered films, we want to figure out on which types of films two sides have disagreements


# In[ ]:


#Read total data
total_df = pd.read_csv('final_data_v1.csv')


# In[ ]:


df = total_df[['name','meta_score','rate','Budget','profit','Adventure',
               'Action','Family','Fantasy','Comedy','Biography',
               'Drama','Crime','Mystery','Thriller','Sport','Horror',
               'Sci-Fi','History','Romance','Western']]


# In[ ]:


part = df[ (df['meta_score']<60) & (df['rate']>6) & (df['profit']>50) ]


# In[ ]:


name_list = list(part['name'])


# In[ ]:


#Comparison analysis
#Load the json data for processing
with open('critic_review_data.json', 'r') as fp:
    critic_dict = json.load(fp)
with open('audience_review_data.json', 'r') as fp:
    audience_dict = json.load(fp)


# In[ ]:



for name in tqdm(df['name']):
    if (name in audience_dict) & (name not in audience_good_name_list):
        audience_dict.pop(name)


# In[ ]:


for name in tqdm(df['name']):
    if (name in critic_dict) & (name not in audience_good_name_list):
        critic_dict.pop(name)
len(critic_dict)


# In[ ]:


audience_word_list = audience_dict['content']


# In[ ]:


critic_word_list = critic_dict['content']


# In[16]:


#Draw word cloud
def word_cloud(word_list):
    wc = WordCloud(background_color="white",
               mask= imread('bg2.png'),
               width=2000,
               height=2000,
               max_words=100,
               collocations=False,
               contour_width=1,
               contour_color='black',
                min_font_size=10,
                font_path='arial.ttf')
    wc.generate(' '.join(word_list))
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud.png')
    plt.show()


# In[17]:


word_cloud(critic_word_list)


# In[10]:


word_cloud(audience_word_list)

