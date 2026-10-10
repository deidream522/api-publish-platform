import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

for _pkg in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(_pkg, quiet=True)

def clean_text(text, is_baseline=True):
    #clean the text,based on model to choose the clean degree(light clean-False)
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)

    if is_baseline:
        text = text.lower()
        #only save the letter
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for word in tokens:
            #fliter the stop words and word len <=1
            if word not in stop_words and len(word) > 1:
                #resore word form
                clean_word = lemmatizer.lemmatize(word)
                clean_tokens.append(clean_word)

        #re-concatenate it into a string
        text = ' '.join(clean_tokens)

    else:
        text = text.lower()
        #save the letter,number,symbol
        text = re.sub(r'[^a-z0-9\s\!\?\.\,\']', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_test_dataset(test_file, test_labels_file, output_file):
    #kaggle's test data clean
    print("loading the data")
    test_df = pd.read_csv(test_file)
    labels_df = pd.read_csv(test_labels_file)

    #connect two labels
    merged_df = pd.merge(test_df, labels_df, on='id')

    #remove the -1 tags
    valid_test_df = merged_df[merged_df['toxic'] != -1].copy()

    print(f"original data have {len(merged_df)}, then the vali data have {len(valid_test_df)}")
    print("begin cleaning..")
    #can change the clean degree
    valid_test_df['cleaned_text'] = valid_test_df['comment_text'].apply(lambda x: clean_text(x, is_baseline=True))
    valid_test_df = valid_test_df[valid_test_df['cleaned_text'].str.strip() != '']
    print(f"save in {output_file} ...")
    final_df = valid_test_df[['cleaned_text', 'toxic']]
    final_df.to_csv(output_file, index=False)
    print("finish the test data clean")


def process_train_dataset(input_file, output_file, text_col='comment_text', label_col='toxic'):
    #kaggle's test data clean,batch processing
    print(f"loading the file:{input_file} ...")
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file, engine='pyarrow')
        else:
            print(f"can not process this type file {input_file}")
            return
    except FileNotFoundError:
        print(f"wrong file path in {input_file}")
        return
    except Exception as e:
        print(f"some error in load the data {e}")
        return

    df = df.dropna(subset=[text_col])

    print("begin cleaning..")
    #can change the clean degree
    df['cleaned_text'] = df[text_col].apply(lambda x: clean_text(x, is_baseline=True))
    #clean the null string and repeat data
    df = df[df['cleaned_text'].str.strip() != '']
    df = df.drop_duplicates(subset=['cleaned_text', label_col])

    print(f"finish, save in {output_file}")
    #only save the useful text and label
    final_df = df[['cleaned_text', label_col]]
    final_df.to_csv(output_file, index=False)
    print("already saved")


def process_civil_comments_parquet(input_files, output_file, sample_size=None, is_baseline=True):
    #process the parquet file data, True->deep clean
    print(f"load the data")
    dfs = []
    for file in input_files:
        try:
            df = pd.read_parquet(file, engine='pyarrow')
            dfs.append(df)
        except Exception as e:
            print(f"can not read this file {e}")
            return

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"the original data have {len(combined_df)}")

    #make subsampling
    if sample_size and len(combined_df) > sample_size:
        print(f"subsampling {sample_size} numbers data")
        combined_df = combined_df.sample(n=sample_size, random_state=111).copy()

    #remove the missing values
    combined_df = combined_df.dropna(subset=['text', 'toxicity'])

    #make the label binarization
    print("make the label binarization..")
    combined_df['toxic'] = (combined_df['toxicity'] >= 0.5).astype(int)

    #clean the data
    print("begin cleaning..")
    combined_df['cleaned_text'] = combined_df['text'].apply(lambda x: clean_text(x, is_baseline=is_baseline))
    combined_df = combined_df[combined_df['cleaned_text'].str.strip() != '']
    combined_df = combined_df.drop_duplicates(subset=['cleaned_text', 'toxic'])

    print(f"finish, save in  {output_file}")
    final_df = combined_df[['cleaned_text', 'toxic']]
    final_df.to_csv(output_file, index=False)
    print(f"finish get {len(final_df)} numbers data")



if __name__ == "__main__":
    process_train_dataset('dataset/main/main_dataset.csv', 'dataset/main/main_dataset_deep.csv')
r'''
kaggle(open /Applications/Python\ 3.12/Install\ Certificates.command)
test1
    process_train_dataset('kaggle/train.csv', 'kaggle/cleaned_train_deep.csv')
test2
    process_test_dataset('kaggle/test.csv', 'kaggle/test_labels.csv', 'kaggle/cleaned_test_deep.csv')
test3
    def process_train_dataset--->(ture-false)
        df['cleaned_text'] = df[text_col].apply(lambda x: clean_text(x, is_baseline=False))
    process_train_dataset('kaggle/train.csv', 'kaggle/cleaned_train_light.csv')
test4
    def process_test_dataset--->(ture-false)
        valid_test_df['cleaned_text'] = valid_test_df['comment_text'].apply(lambda x: clean_text(x, is_baseline=False))
    process_test_dataset('kaggle/test.csv', 'kaggle/test_labels.csv', 'kaggle/cleaned_test_light.csv')
parquet(pip3 install pyarrow)
test1
    (ture--deep_cleaning)
    data_dir = 'google/'

    # process the train data(subsampling 50000 data)
    train_files = [
        os.path.join(data_dir, 'train-00000-of-00002.parquet'),
        os.path.join(data_dir, 'train-00001-of-00002.parquet')
    ]
    process_civil_comments_parquet(
        input_files=train_files,
        output_file=os.path.join(data_dir, 'civil_train_cleaned_deep.csv'),
        sample_size=50000,
        is_baseline=True
    )

    # process the validation data(subsampling 5000 data)
    val_files = [os.path.join(data_dir, 'validation-00000-of-00001.parquet')]
    process_civil_comments_parquet(
        input_files=val_files,
        output_file=os.path.join(data_dir, 'civil_val_cleaned_deep.csv'),
        sample_size=5000,
        is_baseline=True
    )

    # process the test data(subsampling 5000 data)
    test_files = [os.path.join(data_dir, 'test-00000-of-00001.parquet')]
    process_civil_comments_parquet(
        input_files=test_files,
        output_file=os.path.join(data_dir, 'civil_test_cleaned_deep.csv'),
        sample_size=5000,
        is_baseline=True
    )
'''