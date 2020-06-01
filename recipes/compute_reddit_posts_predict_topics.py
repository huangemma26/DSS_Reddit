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


reddit_posts_predict = dataiku.Dataset("reddit_posts_predict")
reddit_posts_predict_df = reddit_posts_predict.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

reddit_posts_predict_topics_df = reddit_posts_predict_df # For this sample code, simply copy input to output


# Write recipe outputs
reddit_posts_predict_topics = dataiku.Dataset("reddit_posts_predict_topics")
reddit_posts_predict_topics.write_with_schema(reddit_posts_predict_topics_df)
