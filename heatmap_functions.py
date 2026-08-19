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
def get_ling_feat(dataset, num_sen = 500, spa = False, create = False):
    # get dataset labels
    dataset_name, data_csv, label_eng, label_spa = data_label(dataset)
    sen_pairs = pd.read_csv(data_csv)

    if create:
        eng_sen = sen_pairs[label_eng][0:num_sen]
        eng_feat = get_desc_nltk(eng_sen)
        
        eng_feat_path = os.path.expanduser(f"~/mitll/TDA_reason-interpet/ling_feat/{dataset_name}_eng_nlp.csv")
        eng_feat.to_csv(eng_feat_path)
        print(f"Added ling features for (eng) texts from {dataset_name}!")
    else:
        eng_feat = pd.read_csv(f"~/mitll/TDA_reason-interpet/ling_feat/{dataset_name}_eng_nlp.csv")

    spa_feat = ""
    if (spa and create):
        spa_sen = sen_pairs[label_spa][0:num_sen]
        spa_feat = get_desc_nltk(spa_sen)

        spa_feat_path = os.path.expanduser(f"~/mitll/TDA_reason-interpet/ling_feat/{dataset_name}_spa_nlp.csv")
        spa_feat.to_csv(spa_feat_path)
        print(f"Added ling features for (spa) texts from {dataset_name}!")
    elif spa:
        spa_feat = pd.read_csv(f"~/mitll/TDA_reason-interpet/ling_feat/{dataset_name}_spa_nlp.csv")

    return eng_feat, spa_feat


## == CORRELATION HEATMAP == ##
def make_heatmap(dataset, model_id, linguistic_label = linguistic_labels[0], tda_label = tda_labels[0], save = True, extra_label = "", spa = False, red = False):
    # get the topological and linguistic features
    top_feat, _ = get_top_feat(model_id, "a", dataset)
    ling_feat, _ = get_ling_feat(dataset, model_id)

    if spa:
        top_feat, _ = get_top_feat(model_id, "a", dataset, spa = True)
        ling_feat, _ = get_ling_feat(model_id, dataset, spa = True)
    
    # focus the heatmap
    focus_top_feat = top_feat[tda_label[1]]
    focus_ling_feat = ling_feat[linguistic_label[1]]

    # convert features to numpy
    tda_features = focus_top_feat.to_numpy()
    linguistic_features = focus_ling_feat.to_numpy()

    # compute pairwise distance correlation matrix
    corr_matrix = np.zeros((tda_features.shape[1], linguistic_features.shape[1]))

    # determine significance threshold
    threshold = 0.6
    num_sig = 0
    sum_corr = 0
    
    for i in range(tda_features.shape[1]):
        for j in range(linguistic_features.shape[1]):
            corr = dcor.distance_correlation(tda_features[:, i], linguistic_features[:, j])
            if np.isnan(corr):
                corr = np.nan_to_num(corr)
            corr_matrix[i, j] = corr
            sum_corr += corr
            if corr > threshold:
                num_sig += 1

    # average percent significant
    num_cells = tda_features.shape[1] * linguistic_features.shape[1]
    avg_corr = sum_corr / num_cells
    per_sig = num_sig / num_cells
    print("The percent of significant correlations is", per_sig, "with an average correlation of", avg_corr)

    # plot heatmap
    plt.figure(figsize=(len(linguistic_label[1]), len(tda_label[1])))
    sns.set_theme(font_scale=1.5) # make label size bigger
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
                xticklabels= linguistic_label[1],
                yticklabels= tda_label[1],
                annot_kws={"fontsize": 15}
                )
    # get labels for title!
    dataset_name, _, _, _ = data_label(dataset)
    _, _, model_short, _ = which_model(model_id)

    plt.title(f"{dataset_name}: {tda_label[0]} vs {linguistic_label[0]} ({model_short})", fontsize=23)

    if red:
        # find highest correlated values
        red_threshold=0.5
        red_cells = [(tda_label[1][i], linguistic_label[1][j], corr_matrix[i, j]) 
                    for i in range(corr_matrix.shape[0])
                    for j in range(corr_matrix.shape[1])
                    if (corr_matrix[i, j] > red_threshold)]

        print("Red cells (row, col, value):")
        for cell in red_cells:
            print(cell)
    if save:
        eng_path = os.path.expanduser(f"~/mitll/TDA_reason-interpet/{dataset_name}/{model_short}_{dataset_name}_eng_corr{extra_label}.pdf")
        plt.savefig(eng_path)
        
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