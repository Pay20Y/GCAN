import string
import editdistance
import numpy as np
from data_provider.data_utils import get_vocabulary
import editdistance

def _normalize_text(text):
    text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
    return text.lower()

def idx2label(inputs, id2char=None, char2id=None):

    if id2char is None:
        voc, char2id, id2char = get_vocabulary(voc_type="ALLCASES_SYMBOLS")

    def end_cut(ins):
        cut_ins = []
        for id in ins:
            if id != char2id['EOS']:
                if id != char2id['UNK']:
                    cut_ins.append(id2char[id])
            else:
                break
        return cut_ins

    if isinstance(inputs, np.ndarray):
        assert len(inputs.shape) == 2, "input's rank should be 2"
        results = [''.join([ch for ch in end_cut(ins)]) for ins in inputs]
        return results
    else:
        print("input to idx2label should be numpy array")
        return inputs

def calc_metrics(predicts, labels, metrics_type='accuracy'):
    assert metrics_type in ['accuracy', 'editdistance'], "Unsupported metrics type {}".format(metrics_type)
    predicts = [_normalize_text(pred) for pred in predicts]
    labels = [_normalize_text(targ) for targ in labels]
    if metrics_type == 'accuracy':
        acc_list = [(pred == tar) for pred, tar in zip(predicts, labels)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        return accuracy
    elif metrics_type == 'editdistance':
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(predicts, labels)]
        eds = sum(ed_list)
        return eds

    return -1

def calc_metrics_length(predicts, labels, metrics_type='accuracy'):
    assert metrics_type in ['accuracy', 'editdistance'], "Unsupported metrics type {}".format(metrics_type)
    predicts = [_normalize_text(pred) for pred in predicts]
    labels = [_normalize_text(targ) for targ in labels]
    lenghts = [len(la) for la in labels]

    predicts_ = np.array(predicts)
    labels_ = np.array(labels)
    lenghts_ = np.array(lenghts)

    len_acc = {}
    for l in range(lenghts_.min(), lenghts_.max()):
        group_predicts = predicts_[lenghts_==l].tolist()
        group_labels = labels_[lenghts_==l].tolist()
        assert len(group_predicts) == len(group_labels)
        if len(group_predicts) == 0:
            continue
        acc_list = [(pred == tar) for pred, tar in zip(group_predicts, group_labels)]
        # accuracy = 1.0 * sum(acc_list) / len(acc_list)
        len_acc[l] = (sum(acc_list), len(acc_list))

    return len_acc

def _lexicon_search(lexicon, word):
    edit_distances = []
    lexicon = [_normalize_text(lex) for lex in lexicon]
    for lex_word in lexicon:
        edit_distances.append(editdistance.eval(_normalize_text(lex_word), _normalize_text(word)))
    edit_distances = np.asarray(edit_distances, dtype=np.int)
    argmin = np.argmin(edit_distances)
    return lexicon[argmin]

def calc_metrics_lexicon(predicts, labels, lexicons, metrics_type='accuracy'):
    assert metrics_type in ['accuracy', 'editdistance'], "Unsupported metrics type {}".format(metrics_type)
    predicts = [_normalize_text(pred) for pred in predicts]
    labels = [_normalize_text(targ) for targ in labels]
    refine_predicts = [_lexicon_search(l, p) for l, p in zip(lexicons, predicts)]

    if metrics_type == 'accuracy':
        acc_list = [(pred == tar) for pred, tar in zip(refine_predicts, labels)]
        accuracy = 1.0 * sum(acc_list) / len(acc_list)
        return accuracy
    elif metrics_type == 'editdistance':
        ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refine_predicts, labels)]
        eds = sum(ed_list)
        return eds