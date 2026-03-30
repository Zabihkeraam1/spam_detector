import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

data = pd.read_csv(
    'data/raw/spam.csv',
    encoding='latin-1',
    usecols=[0, 1]
)

data.columns = ['label', 'message']
data = data.dropna(subset=['message'])
data['message'] = data['message'].astype(str)
data = data[data['message'].str.strip() != ""]
data = data.dropna(subset=['label'])

# First map
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Then clean any failed mappings
data = data.dropna(subset=['label'])

# Then convert to int
data['label'] = data['label'].astype(int)

def preprocess(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    tokens = word_tokenize(text)
    
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

data['message'] = data['message'].apply(preprocess)

data = data.drop_duplicates()

data.to_csv('data/processed/spam_cleaned.csv', index=False)

print(data.head())