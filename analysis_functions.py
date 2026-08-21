## == IMPORTS == ##
# for attention extraction + analysis
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm
import math
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prettytable import PrettyTable

# to evaluate model
from lm_eval.tasks import TaskManager

## == LOAD IN MODELS == ##
def load_model(model_id = "meta-llama/Llama-3.1-8B-Instruct", device = "cuda"):
    """
    Load model and tokenizer, handling different architectures.
    """
    from transformers import AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # for encoder-only models (BERT, RoBERTa, etc.)
        model = AutoModel.from_pretrained(
            model_id, 
            output_attentions=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception:
        try:
            # For causal LM models (Llama, Qwen, GPT, etc.)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                output_attentions=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
        except Exception:
            # for seq2seq models (BART, T5, etc.)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                output_attentions=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

# get information from dataset
def data_label(dataset):
    """Extracts information from dataset strings"""
    name = dataset[0]
    csv = dataset[1]
    return name, csv

# get information from model
def which_model(model):
    """Extracts information from model strings"""
    name = model[0]
    hf_id = model[1]
    short = model[2]
    return name, hf_id, short

# check if answer is correct
def evaluate_model(text_cat, q_index, llm_answer):
    task_dict = TaskManager.load_task_or_group(task_list=[text_cat])
    task_obj = task_dict["tasks"][text_cat]

    docs = list(task_obj.eval_docs())
    doc = docs[q_index]

    if task.startswith("mmlu_"):
        question = doc["question"]
        choices = doc["choices"]
        correct_index = int(doc["answer"])

    elif task == "hellaswag":
        question = doc["ctx"]
        choices = doc["endings"]
        correct_index = int(doc["label"])

    else:
        raise ValueError(f"Unsupported task: {task}")

    answer = llm_answer.strip().upper()

    if answer in "ABCD":
        predicted_index = ord(answer) - ord("A")
    elif answer in ("0", "1", "2", "3"):
        predicted_index = int(answer)
    else:
        raise ValueError(
            f"Expected A/B/C/D or 0/1/2/3, got: {llm_answer!r}"
        )
    
    correctness = (predicted_index == correct_index)
    
    return correctness


## == TDA HELPERS == ##
## function to assist with finding highest h0/h1 TDA feature
def find_highest_finite_value_comprehension(data):
    """Finds the highest value in a list, ignoring inf values, using list comprehension."""
    finite_values = [x for x in data if not math.isinf(x)]
    return max(finite_values) if finite_values else -math.inf

## function to assist with finding second highest h0/h1 TDA feature
def get_second_value_ignoring_inf(data):
    """
    Returns the second non-inf value in a list.

    Args:
      data: A list of numerical values.

    Returns:
      The second non-inf value in the list, or None if not found.
    """
    non_inf_values = [x for x in data if not math.isinf(x)]
    if len(non_inf_values) < 2:
        return None
    return non_inf_values[1]


## == EXTRACT ATTENTION MAP == #
# extract attention from model
def get_attention(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True
        )
    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    attention_matrices = torch.stack(outputs.attentions).mean(dim=(0, 2)).squeeze(0).cpu().numpy()   # torch.stack(outputs.attentions).mean(dim=0).squeeze().cpu().numpy()
    avg_attention = np.mean(attention_matrices, axis=0)
    return avg_attention, answer # Averaging across heads

def build_graph(attention_matrix, threshold=0.1):
    graph = nx.Graph()
    num_nodes = attention_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if attention_matrix[i, j] > threshold:
                graph.add_edge(i, j, weight=attention_matrix[i, j])

    return graph

# aquire TDA features from model
def compute_tda_features(graph):
    adjacency_matrix = nx.to_numpy_array(graph)
    diagrams = ripser(adjacency_matrix, maxdim=1)['dgms']

    h0 = diagrams[0]
    h1 = diagrams[1] if len(diagrams) > 1 else np.array([])

    num_h0 = geek.count_nonzero(np.round(h0)) #count_nonzero(h0) # len(h0)
    highest_h0 = find_highest_finite_value_comprehension(h0[:, 1] - h0[:, 0]) if num_h0 > 0 else 0
    Second_highest_h0 = get_second_value_ignoring_inf(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else 0
    highest_minus_second_h0 = highest_h0 - Second_highest_h0 if num_h0 > 1 else 0

    # Replace inf values with 0
    h0[np.isinf(h0)] = 0
    mean_h0 = np.mean(h0) if num_h0 > 0 else 0
    # print("mean h0: ", mean_h0)


    num_h1 = geek.count_nonzero(np.round(h1))
    highest_h1 = find_highest_finite_value_comprehension(h1[:, 1] - h1[:, 0]) if num_h1 > 0 else 0
    second_highest_h1 = get_second_value_ignoring_inf(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else 0
    highest_minus_second_h1 = highest_h1 - second_highest_h1 if num_h1 > 1 else 0

    # Replace inf values with 0
    h1[np.isinf(h1)] = 0
    mean_h1 = np.mean(h1) if num_h1 > 0 else 0
    # print("mean h1: ", mean_h1)

    h0_persistences = np.sort(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else np.array([0])

    h1_persistences = np.sort(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else np.array([0])

    # Additional TDA features for linguistic correlation
    sum_persistence_0 = np.sum(h0_persistences) if len(h0_persistences) > 0 else 0
    sum_persistence_1 = np.sum(h1_persistences) if len(h1_persistences) > 0 else 0
    persistence_entropy_0 = -np.sum(h0_persistences * np.log(h0_persistences + 1e-10)) if len(h0_persistences) > 0 else 0
    persistence_entropy_1 = -np.sum(h1_persistences * np.log(h1_persistences + 1e-10)) if len(h1_persistences) > 0 else 0
    betti_curve_0 = len(h0_persistences)
    betti_curve_1 = len(h1_persistences)


    return [num_h0, highest_h0, highest_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
            num_h1, highest_h1, highest_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1]

# analyzes text from model
def process_texts(texts, text_cat, model_id):
    model, tokenizer = load_model(model_id)
    data = []
    index = 0
    for text in tqdm(texts):
        attention_matrix, answer = get_attention(text, model, tokenizer)
        graph = build_graph(attention_matrix)
        tda_features = compute_tda_features(graph)
        data.append(tda_features)

        correctness = evaluate_model(text_cat, index, answer)
        data.append(answer)
        index += 1

    columns = ["Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
               "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1",
               "corectness"]
    return pd.DataFrame(data, columns=columns)


## == EXTRACT TDA FEATURES FROM MODEL == ##
# extract top feats from models
def get_top_feat(model, dataset, create = False):
    # get labels for model
    model_name, model_id, model_short  = which_model(model)

    # labels for dataset
    data_name, data_csv = data_label(dataset)
    questions = pd.read_csv(data_csv)
    
    tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpet/{model_short}/{model_short}_{data_name}_tda.csv")

    # either create or load data
    if create:
        feats_sen = questions["prompt"]
        feats_tda = process_texts(feats_sen, model_id)
        feats_tda.to_csv(tda_path, index=False)
        print(f"Added questions from {data_name} for {model_short}!")
   
    else:
        feats_tda = pd.read_csv(tda_path)

    return feats_tda

# analyze the h0 and h1 features
def analyze_feats(model, dataset):
    feats_tda = get_top_feat(model, dataset)
    
    # isolate correct/incorrect answers
    correct_feats = feats_tda[feats_tda["correctness"] == 1]
    incorrect_feats = feats_tda[feats_tda["correctness"] == 0]

    avg_correct_0dim = [correct_feats["Num_0dim"].mean(), correct_feats["Max_0dim"].mean(),
                        correct_feats["Max_0dim_Minus_Second"].mean(), correct_feats["Mean_0dim"].mean(),
                        correct_feats["betti_curve_0"].mean(), correct_feats["persistence_entropy_0"].mean()]
    avg_correct_1dim = [correct_feats["Num_1dim"].mean(), correct_feats["Max_1dim"].mean(),
                        correct_feats["Max_1dim_Minus_Second"].mean(), correct_feats["Mean_1dim"].mean(),
                        correct_feats["betti_curve_1"].mean(), correct_feats["persistence_entropy_1"].mean()]
    
    avg_incorrect_0dim = [incorrect_feats["Num_0dim"].mean(), incorrect_feats["Max_0dim"].mean(),
                        incorrect_feats["Max_0dim_Minus_Second"].mean(), incorrect_feats["Mean_0dim"].mean(),
                        incorrect_feats["betti_curve_0"].mean(), incorrect_feats["persistence_entropy_0"].mean()]
    avg_incorrect_1dim = [incorrect_feats["Num_1dim"].mean(), incorrect_feats["Max_1dim"].mean(),
                        incorrect_feats["Max_1dim_Minus_Second"].mean(), incorrect_feats["Mean_1dim"].mean(),
                        incorrect_feats["betti_curve_1"].mean(), incorrect_feats["persistence_entropy_1"].mean()]
    

    # display info
    table = PrettyTable()
    table.field_names = ["label", "num_feat", "max_feat", "max_feat_minus_second", "mean_feat", "betti_curve", "persistence_entropy"]
    table.add_row = ["correct_0dim"] + avg_correct_0dim
    table.add_row = ["correct_1dim"] + avg_correct_1dim
    table.add_row = ["incorrect_0dim"] + avg_incorrect_0dim
    table.add_row = ["incorrect_1dim"] + avg_incorrect_1dim

    print(table)

    return 0


## == old functions for extracting attention + embeddings == ##
# def get_attention(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_attentions=True)
#     attention_matrices = torch.stack(outputs.attentions).mean(dim=0).squeeze(0).cpu().numpy()
#     return np.mean(attention_matrices, axis=0)  # Averaging across heads

# def get_embeddings(text, model, tokenizer, last_layer = False):
#     # get the intial tokenization (layer 0 embeddings)
#     inputs = tokenizer(text=text, truncation=True, return_tensors="pt").to(model.device)

#     # get hidden state information
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     index = 0
#     if last_layer:
#         index = -1
#     embeddings = outputs.hidden_states[index]

#     return embeddings

# def build_graph_attention(attention_matrix, threshold=0.1):
#     graph = nx.Graph()
#     num_nodes = attention_matrix.shape[0]

#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if attention_matrix[i, j] > threshold:
#                 graph.add_edge(i, j, weight=attention_matrix[i, j])

#     if graph.number_of_nodes() == 0:
#         print("transposed matrix")
#         for i in range(num_nodes):
#             for j in range(i + 1, num_nodes):
#                 if attention_matrix[j, i] > threshold:
#                     graph.add_edge(i, j, weight=attention_matrix[i, j])
    
#     adjacency_matrix = nx.to_numpy_array(graph)
#     return adjacency_matrix

# def build_graph_embeddings(embeddings):
#     # ensure embeddings are numpy array
#     embeddings = np.asarray(embeddings)

#     # standardize embeddings
#     if embeddings.ndim > 2:
#         embeddings = embeddings.reshape(-1, embeddings.shape[-1])

#     # check if empty
#     if embeddings.shape[1] == 0:
#         raise ValueError(f"embeddings have shape: {embeddings.shape}")

#     # remove values if they are NaN
#     if np.isnan(embeddings).any():
#         imputer = SimpleImputer(strategy='mean')
#         embeddings = imputer.fit_transform(embeddings)
    
#     # fit + transform PCA with 2 components
#     n_components = min(2, embeddings.shape[1])
#     graph_embeddings = PCA(n_components=n_components).fit_transform(embeddings)

#     return graph_embeddings

# def compute_tda_features(graph):
#     diagrams = ripser(graph, maxdim=1)['dgms']

#     h0 = diagrams[0]
#     h1 = diagrams[1] if len(diagrams) > 1 else np.array([])

#     num_h0 = np.count_nonzero(np.round(h0))
#     highest_h0 = find_highest_finite_value_comprehension(h0[:, 1] - h0[:, 0]) if num_h0 > 0 else 0
#     Second_highest_h0 = get_second_value_ignoring_inf(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else 0
#     highest_minus_second_h0 = highest_h0 - Second_highest_h0 if num_h0 > 1 else 0

#     # Replace inf values with 0
#     h0[np.isinf(h0)] = 0
#     mean_h0 = np.mean(h0) if num_h0 > 0 else 0

#     num_h1 = np.count_nonzero(np.round(h1))
#     highest_h1 = find_highest_finite_value_comprehension(h1[:, 1] - h1[:, 0]) if num_h1 > 0 else 0
#     second_highest_h1 = get_second_value_ignoring_inf(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else 0
    
#     # set none values to 0 to avoid error
#     highest_h1 = highest_h1 if highest_h1 is not None else 0
#     second_highest_h1 = second_highest_h1 if second_highest_h1 is not None else 0
#     highest_minus_second_h1 = highest_h1 - second_highest_h1 if num_h1 > 1 else 0

#     # Replace inf values with 0
#     h1[np.isinf(h1)] = 0
#     mean_h1 = np.mean(h1) if num_h1 > 0 else 0
#     # print("mean h1: ", mean_h1)

#     h0_persistences = np.sort(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else np.array([0])

#     h1_persistences = np.sort(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else np.array([0])

#     # Additional TDA features for linguistic correlation
#     sum_persistence_0 = np.sum(h0_persistences) if len(h0_persistences) > 0 else 0
#     sum_persistence_1 = np.sum(h1_persistences) if len(h1_persistences) > 0 else 0
#     persistence_entropy_0 = -np.sum(h0_persistences * np.log(h0_persistences + 1e-10)) if len(h0_persistences) > 0 else 0
#     persistence_entropy_1 = -np.sum(h1_persistences * np.log(h1_persistences + 1e-10)) if len(h1_persistences) > 0 else 0
#     betti_curve_0 = len(h0_persistences)
#     betti_curve_1 = len(h1_persistences)


#     return [num_h0, highest_h0, highest_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
#             num_h1, highest_h1, highest_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1]
    
# def process_texts(texts, lat_rep, model_id, last_layer = False):
#     """
#     input: - texts: array of strings
#            - lat_rep: string of which latent representation to use for PH analysis,
#              valid arguments are "hs" for hidden states or "a" for attention
#            - model_id: which model to use
#     """
#     model, tokenizer = load_model(model_id)
    
#     data = []
#     for text in tqdm(texts):
#         if lat_rep == "hs":
#             hidden_states = get_embeddings(text, model, tokenizer, last_layer).cpu().detach().numpy()
#             graph = build_graph_embeddings(hidden_states)
#         elif lat_rep == "a":
#             attention_matrix = get_attention(text, model, tokenizer)
#             graph = build_graph_attention(attention_matrix)
#         else:
#             print("could not make a graph")
#             return
#         tda_features = compute_tda_features(graph)
#         data.append(tda_features)

#     columns = ["Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
#                "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1"]
#     return pd.DataFrame(data, columns=columns)
