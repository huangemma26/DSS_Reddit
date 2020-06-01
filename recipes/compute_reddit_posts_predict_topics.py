%pylab inline
import warnings                         # Disable some warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd,  seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text 

from sklearn.decomposition import LatentDirichletAllocation,NMF
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

mydataset = dataiku.Dataset("reddit_posts_predict")
df = mydataset.get_dataframe()

# Get the column names
numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
categorical_columns = list(df.select_dtypes(include=[object]).columns)
date_columns = list(df.select_dtypes(include=['<M8[ns]']).columns)
raw_text_col = categorical_columns[3]
raw_text = df[raw_text_col]
custom_stop_words = ['incel',
                     'chad',
                     'alpha',
                     'beta',
                     've',
                     'gt',
                     'amp',
                     'nbsp',
                     'rp',
                     'pm',
                     'ampnbsp',
                     'th',
                     'aa',
                     'adam',
                     'john',
                     'really',
                     'just'
                    ]

stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)

cnt_vectorizer = CountVectorizer(strip_accents = 'unicode',stop_words = stop_words,lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.8, min_df = 0.025)

cnt_vectorizer.fit(raw_text)
cnt_vectorizer.vocabulary_['man'] = cnt_vectorizer.vocabulary_['men']
cnt_vectorizer.vocabulary_['kid'] = cnt_vectorizer.vocabulary_['child']
text_cnt = cnt_vectorizer.transform(raw_text)

n_topics= 15
topics_model = LatentDirichletAllocation(n_topics, random_state=0)

topics_model.fit(text_cnt)

lda_docs = list(topics_model.transform(text_cnt))
lda_topic = [list(i).index(max(i)) for i in lda_docs]
df['lda_topic'] = lda_topic





reddit_posts_predict_topics_df = reddit_posts_predict_df # For this sample code, simply copy input to output


# Write recipe outputs
reddit_posts_predict_topics = dataiku.Dataset("reddit_posts_predict_topics")
reddit_posts_predict_topics.write_with_schema(reddit_posts_predict_topics_df)
