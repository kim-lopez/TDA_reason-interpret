# == IMPORTS == #
# spacy
import spacy
import textdescriptives as td

# nltk
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer

# for heatmaps
from dcor import distance_correlation
import seaborn as sns
import matplotlib.pyplot as plt
import dcor
import os
import gc
import math
from scipy import stats


## == EXTRACT LINGUISTIC FEATURES == ##
# load nlp model
def get_desc_spacy(data):
  nlp = spacy.load("en_core_web_sm")
  nlp.add_pipe("textdescriptives/all")

  doc = nlp(data.iloc[0])
  df1 = td.extract_df(doc)
  
  for text in tqdm(data.iloc[1:]):
    text = nlp(text)
    df2 = td.extract_df(text)
    df1 = df1._append(df2)
  return df1

def sen_feat_nltk(sen):
    tokens = word_tokenize(sen)
    lemmatizer = WordNetLemmatizer()
    tokens = [t.lower() for t in tokens if t.isalpha()]  # keep words only
    stop_words = set(stopwords.words("english"))

    if len(tokens) == 0:
        return {}

    freq = FreqDist(tokens)
    content_words = [t for t in tokens if t not in stop_words]

    # entropy :0
    total = sum(freq.values())
    entropy = -sum((f/total) * math.log2(f/total) for f in freq.values())

    # lexical sophistication
    polysemy = [len(wordnet.synsets(lemmatizer.lemmatize(w))) for w in set(content_words)]
    avg_polysemy = sum(polysemy) / len(polysemy) if polysemy else 0

    long_words = [w for w in tokens if len(w) >= 7]
    rare_words = freq.hapaxes()

    types = len(set(tokens))
    total = len(tokens)
    simpson_div = 1 - sum((c / total) ** 2 for c in freq.values())
    
    yules_k_term = sum(v**2 for v in freq.values())
    yules_k = 10000 * (yules_k_term - total) / (total ** 2)

    # POS
    tagged = nltk.pos_tag(tokens)

    num_nouns = sum(tag.startswith("NN") for _, tag in tagged)
    num_verbs = sum(tag.startswith("VB") for _, tag in tagged)
    num_adjs = sum(tag.startswith("JJ") for _, tag in tagged)
    num_advs = sum(tag.startswith("RB") for _, tag in tagged)

    # bag of words
    top_k = 100
    bow_features = {
    f"bow_{word}": count
    for word, count in freq.most_common(top_k)
      }

    features = {
        # length/stats
        "num_tokens": total,
        "num_types": types,
        "avg_word_length": sum(len(t) for t in tokens) / len(tokens),
        "char_count": sum(len(t) for t in tokens),

        # lexical diversity
        "ttr": types / total,
        "cttr": types / math.sqrt(2 * total),
        "herdan_c": math.log(types) / math.log(total),
        "simpson_div": simpson_div,
        "yules_k": yules_k,

        # lexical density
        "num_content_words": len(content_words),
        "lexical_density": len(content_words) / total,

        # lexical sophistication
        "avg_polysemy": avg_polysemy,
        "num_long_words": len(long_words),
        "long_word_ratio": len(long_words) / total,
        "hapax_legomena": len(rare_words),
        "hapax_ratio": len(rare_words) / total,

        # stopwords
        "num_stopwords": len([t for t in tokens if t in stop_words]),
        "stopword_ratio": len([t for t in tokens if t in stop_words]) / len(tokens),

        # POS
        "noun_ratio": num_nouns / total,
        "verb_ratio": num_verbs / total,
        "adj_ratio": num_adjs / total,
        "adv_ratio": num_advs / total,

        # frequency
        "entropy": entropy,
        "top_word_freq": freq.most_common(1)[0][1],

        # N-grams
        "unique_bigram_approx": len(list(nltk.bigrams(tokens))),
    }

    return features

def list_feat_nltk(list):
  rows = [sen_feat_nltk(sen) for sen in list]
  return pd.DataFrame(rows)

def get_desc_nltk(data):
  df_feat = list_feat_nltk(data.tolist())
  return df_feat


## == EXTRACT LINGUISTIC FEATURES == ##
def get_ling_feat(dataset, create = False):
    # get dataset labels
    data_name, data_csv = data_label(dataset)
    df_prompts = pd.read_csv(data_csv)
    ling_feat_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpet/ling_feat/{data_name}_ling_feat.csv")

    if create:
        prompt = df_prompts["prompt"]
        ling_feat = get_desc_nltk(prompt)
        
        ling_feat.to_csv(ling_feat_path)
        print(f"Added ling features for questions from {data_name}!")
    
    else:
        ling_feat = pd.read_csv(ling_feat_path)

    return ling_feat


## == CORRELATION HEATMAP == ##
def make_heatmap(dataset, model_id, linguistic_label = linguistic_labels[0], tda_label = tda_labels[0], save = True):
    # get the topological and linguistic features
    top_feat = get_top_feat(model_id, dataset)
    ling_feat = get_ling_feat(dataset, model_id)
    
    # create masks for csvs
    correct = top_feat["correctness"] == 1
    incorrect = top_feat["correctness"] == 0
    
    # focus the heatmap
    correct_top_feat = top_feat[correct][tda_label[1]]
    correct_ling_feat = ling_feat[correct][linguistic_label[1]]

    incorrect_top_feat = top_feat[incorrect][tda_label[1]]
    incorrect_ling_feat = ling_feat[incorrect][linguistic_label[1]]


    # convert features to numpy
    tda_features_correct = correct_top_feat.to_numpy()
    linguistic_features_correct = correct_ling_feat.to_numpy()

    tda_features_incorrect = incorrect_top_feat.to_numpy()
    linguistic_features_incorrect = incorrect_ling_feat.to_numpy()

    # randomly sample 100 entries from each
    srs = np.random.default_rng(8)
    srs_rows = srs.choice(tda_filtered.shape[0], size=100, replace=False)

    correct_top_feat = correct_top_feat[srs_rows]
    correct_ling_feat = correct_ling_feat[srs_rows]

    incorrect_top_feat = incorrect_top_feat[srs_rows]
    incorrect_ling_feat = incorrect_ling_feat[srs_rows]

    # compute pairwise distance correlation matrix
    corr_matrix_correct = np.zeros((tda_features.shape[1], linguistic_features.shape[1]))
    corr_matrix_incorrect = np.zeros((tda_features.shape[1], linguistic_features.shape[1]))
    
    for i in range(correct_top_feat.shape[1]):
        for j in range(correct_ling_feat.shape[1]):
            corr_correct = dcor.distance_correlation(correct_top_feat[:, i], correct_ling_feat[:, j])
            corr_incorrect = dcor.distance_correlation(incorrect_top_feat[:, i], incorrect_ling_feat[:, j])
            if np.isnan(corr_correct):
                corr_correct = np.nan_to_num(corr_correct)
            corr_correct[i, j] = corr_correct

            if np.isnan(corr_incorrect):
                corr_incorrect = np.nan_to_num(corr_incorrect)
            corr_matrix_incorrect[i, j] = corr_incorrect

    # get labels for title!
    dataset_name, _, = data_label(dataset)
    _, _, model_short = which_model(model_id)

    # plot heatmap correct
    plt.figure(figsize=(len(linguistic_label[1]), len(tda_label[1])))
    sns.set_theme(font_scale=1.5) # make label size bigger
    sns.heatmap(corr_matrix_correct, annot=True, cmap="coolwarm",
                xticklabels= linguistic_label[1],
                yticklabels= tda_label[1],
                annot_kws={"fontsize": 15})
    plt.title(f"{dataset_name}: {tda_label[0]} vs {linguistic_label[0]} ({model_short}, correct)", fontsize=23)

    # plot heatmap incorrect
    plt.figure(figsize=(len(linguistic_label[1]), len(tda_label[1])))
    sns.set_theme(font_scale=1.5) # make label size bigger
    sns.heatmap(corr_matrix_incorrect, annot=True, cmap="coolwarm",
                xticklabels= linguistic_label[1],
                yticklabels= tda_label[1],
                annot_kws={"fontsize": 15})
    plt.title(f"{dataset_name}: {tda_label[0]} vs {linguistic_label[0]} ({model_short}, incorrect)", fontsize=23)
    
    # save the heatmaps as pdf files
    if save:
        path_correct = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpet/{dataset_name}/{model_short}_{dataset_name}_correct.pdf")
        plt.savefig(path_correct)
        
        path_incorrect = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpet/{dataset_name}/{model_short}_{dataset_name}_incorrect.pdf")
        plt.savefig(path_incorrect)
        
    plt.show()

    return


## == p-VALUE HEATMAP == ##
def make_pval_heatmap(dataset, model_id, linguistic_label = linguistic_labels[0], tda_label = tda_labels[0], save = True, extra_label = "", spa = False, show = True):
    # get the topological and linguistic features
    top_feat, _ = get_top_feat(model_id, "a", dataset)
    ling_feat, _ = get_ling_feat(dataset, model_id)

    # get labels for title!
    dataset_name, data_csv, eng_label, _ = data_label(dataset)
    _, _, model_short, _ = which_model(model_id)

    if spa:
        top_feat, _ = get_top_feat(model_id, "a", dataset, spa = True)
        ling_feat, _ = get_ling_feat(model_id, dataset, spa = True)

    # get N!
    df_dataset = pd.read_csv(data_csv)
    n = len(df_dataset[f'{eng_label}'])
    print("n is:", n)
    
    # focus the heatmap
    focus_top_feat = top_feat[tda_label[1]]
    focus_ling_feat = ling_feat[linguistic_label[1]]

    # convert features to numpy
    tda_features = focus_top_feat.to_numpy()
    linguistic_features = focus_ling_feat.to_numpy()

    # compute pairwise distance correlation matrix
    pval_matrix = np.zeros((tda_features.shape[1], linguistic_features.shape[1]))

    # determine significance threshold
    threshold = 0.05
    num_sig = 0
    sum_pval = 0

    for i in range(tda_features.shape[1]):
        for j in range(linguistic_features.shape[1]):
            corr = dcor.distance_correlation(tda_features[:, i], linguistic_features[:, j])
            if np.isnan(corr):
                corr = np.nan_to_num(corr)
            tval = (corr * math.sqrt(n-2)) / math.sqrt(1 - corr**2)
            pval = stats.t.sf(np.abs(tval), n-1)*2
            pval_matrix[i, j] = pval
            sum_pval += pval
            if pval < threshold:
                num_sig += 1

    # average percent significant
    num_cells = tda_features.shape[1] * linguistic_features.shape[1]
    avg_pval = sum_pval / num_cells
    per_sig = num_sig / num_cells
    print("The percent of significant p-values is", per_sig, "with an average p-value of", avg_pval)

    # plot heatmap
    plt.figure(figsize=(len(linguistic_label[1]), len(tda_label[1])))
    sns.set_theme(font_scale=1.5) # make label size bigger
    sns.heatmap(pval_matrix, annot=True, cmap="coolwarm",
                xticklabels= linguistic_label[1],
                yticklabels= tda_label[1],
                annot_kws={"fontsize": 10}, vmin=0, vmax=0.05
                )

    plt.title(f"{dataset_name}: {tda_label[0]} vs {linguistic_label[0]} p-values ({model_short})", fontsize=23)

    if save:
        eng_path = os.path.expanduser(f"~/mitll/TDA_reason-interpet/{model_short}/{dataset_name}/{model_short}_{dataset_name}_eng_pval{extra_label}.pdf")
        plt.savefig(eng_path)

    if show: 
        plt.show()

    return per_sig, avg_pval