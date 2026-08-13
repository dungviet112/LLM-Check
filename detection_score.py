import numpy as np
from matplotlib import pyplot as plt
import pickle
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, precision_recall_curve

from tqdm import tqdm
from model import *


gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 16
dropout_mlp = 0.5
dropout_gru = 0.25
learning_rate = 1e-4
weight_decay = 1e-2

# load train and test scores from /data
with open('data/only_answer/scores_fava_annot_llama3_layer21_500samp.pkl', 'rb') as f:
    scores, sample_indiv_scores, generated_embeddings, sample_labels = pickle.load(f)
    # scores, sample_indiv_scores, sample_labels = pickle.load(f)


def get_roc_auc_scores(scores: np.array, labels: np.array):
    """
    Calculate ROC AUC metrics, including the AUC score, accuracy, and other key values.

    Args:
        scores (np.array): Predicted scores.
        labels (np.array): Ground Truth binary labels.

    Returns:
        tuple: A tuple containing the following:
            - arc (float): Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve.
            - acc (float): Maximum accuracy derived from the ROC curve.
            - low (float): True Positive Rate (TPR) at the maximum False Positive Rate (FPR) < 0.05.
            - fpr (np.ndarray): False Positive Rates at various thresholds.
            - tpr (np.ndarray): True Positive Rates at various thresholds.
            - thresh_ind (int): Index of the threshold corresponding to maximum accuracy.
            - thresh (np.ndarray): Thresholds used for calculating the ROC curve.
    """
    fpr, tpr, thresh = roc_curve(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Calculate F1 score for each threshold, adding a small epsilon to avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    # Find the index of the maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[optimal_idx]

    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    thresh_ind = np.argmax(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<0.05)[0][-1]]
    return arc, acc, low, fpr, tpr, thresh_ind, thresh, best_f1


def get_thresh_val(thresh: np.array, acc: float, scores: np.array):
    """Find approx threshold that matches avg accuracy"""
    for t in thresh:
        pred_list = np.array([ 1 if x < t else 0 for x in scores])
        if np.mean(pred_list) <= acc:
            #print(f"Accuracy: {acc:.2f} , {np.mean(pred_list):.2f}")
            return t, pred_list


def get_balanced_scores(scores: np.array, sample_labels: np.array):
    """Get balanced scores"""
    num_samp = min(sum(sample_labels), len(sample_labels)-sum(sample_labels))
    bal_sc = np.concatenate([scores[:num_samp] , scores[-num_samp:]])
    bal_labels = np.concatenate([sample_labels[:num_samp], sample_labels[-num_samp:]])
    return bal_sc, bal_labels


ly_scores = -np.array(sample_indiv_scores['logit']["perplexity"])
arc, acc, low, fpr, tpr, thresh_ind, thresh, f1 = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
# print(f"AUROC:{arc*100:.2f}, Acc:{acc*100:.2f}, TPR@5%FPR:{low*100:.2f}")
print(f"PPL & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} & {f1*100:.2f} \\\\")

ly_scores = np.array(sample_indiv_scores['logit']["window_entropy"])
arc, acc, low, fpr, tpr, thresh_ind, thresh, f1 = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
print(f"Window Entropy & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} & {f1*100:.2f} \\\\")

ly_scores = np.array(sample_indiv_scores['logit']["logit_entropy"])
arc, acc, low, fpr, tpr, thresh_ind, thresh, f1 = get_roc_auc_scores(*get_balanced_scores(ly_scores,sample_labels))
print(f"Logit Entropy & {arc*100:.2f} & {acc*100:.2f} & {low*100:.2f} & {f1*100:.2f} \\\\")

num_layers = len(sample_indiv_scores['attns'].keys())  # The 7B and 8B models that are being evaluated have 32 layers, so num_layers=31
arc_list, acc_list, low_list = [], [], []

samp_preds = []
thresh_vals = []

print(f"Evaluating {num_layers} layers of attention scores:")
for layer_num in range(1, num_layers + 1):
    scores = -np.array(sample_indiv_scores['attns']["Attn"+str(layer_num)])
    bal_sc, bal_labels = get_balanced_scores(scores,sample_labels)
    arc, acc, low, fpr, tpr, thresh_ind, thresh, f1 = get_roc_auc_scores(bal_sc, bal_labels)
    thresh_val, pred_list = get_thresh_val(thresh, acc, bal_sc)
    samp_preds.append(pred_list)
    thresh_vals.append(thresh_val)
    print(f"Layer:{layer_num+1} - AUROC:{arc:.4f}, Acc:{acc:.4f}, TPR@5%FPR:{low:.4f}, F1={f1:.4f}")
    arc_list.append(arc)
    acc_list.append(acc)
    low_list.append(low)
    plt.plot(fpr, tpr, label = f'LY{layer_num}, AUC={arc*100:.2f}, Acc={acc*100:.2f}, TPR@5%FPR={low*100:.2f}') 

# clear the plot
plt.clf()
plt.xticks([1, 5, 10, 15, 20, 25, 30, 32])
plt.plot(range(1, len(arc_list) + 1), arc_list, label='AUROC')
plt.plot(range(1, len(arc_list) + 1), acc_list, label='ACC')
plt.plot(range(1, len(arc_list) + 1), low_list, label='TPR@5%FPR')
plt.xlabel('Layer Number')
plt.grid()
plt.legend()
plt.savefig("plots/llama_hidden_fava.png", dpi=200, bbox_inches='tight')


def gen_classifier_roc(inputs, labels, model):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=123)
    classifier_model = model(X_train.shape[1]).to(device)
    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for _ in tqdm(range(1001)):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()
    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:,1]>0.5).type(torch.long).cpu()
        y_score = pred[:,1].cpu().numpy()
        y_true = y_test.cpu().numpy()
        roc_auc = roc_auc_score(y_true, y_score)
        acc = (prediction_classes.numpy()==y_test.cpu().numpy()).mean()
        f1 = f1_score(y_true, prediction_classes)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        tpr_at_5_fpr = np.interp(0.05, fpr, tpr)
    return roc_auc, acc, tpr_at_5_fpr, f1

classifier = SingleMLP_Classifier
pooled_embeddings = []
for i in range(len(generated_embeddings)):
    embeddings = generated_embeddings[i]
    emb_sequence = torch.tensor(embeddings, dtype=torch.float32)
    pooled_emb = torch.mean(emb_sequence, dim=0)
    pooled_embeddings.append(pooled_emb)

pooled_embeddings_tensor = torch.stack(pooled_embeddings)
context_emb_roc, context_emb_acc, context_emb_tpr_at_5_fpr, context_emb_f1 = gen_classifier_roc(pooled_embeddings_tensor, sample_labels, classifier)

print(f"Contextual Embeddings - AUROC: {context_emb_roc:.4f}, Accuracy: {context_emb_acc:.4f}")
print(f"f1_score: {context_emb_f1:.4f}")
print(f"TPR at 5% FPR: {context_emb_tpr_at_5_fpr:.4f}")