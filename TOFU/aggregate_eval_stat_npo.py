
from omegaconf import OmegaConf
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(unlearn_forget_result['avg_paraphrased_loss'])
    unlearn_perturbed_np_values = np.array(unlearn_forget_result['average_perturb_loss'])
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)
    # print(unlearn_perturbed_np_values)
    # print(unlearn_paraphrase_np_values)
    retain_paraphrase_np_values = np.array(retain_forget_result['avg_paraphrased_loss'])
    retain_perturbed_np_values = np.array(retain_forget_result['average_perturb_loss'])
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)
    # print(retain_perturbed_np_values)
    # print(retain_paraphrase_np_values)
    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return ({'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic},
            {'Unlearn Truth Ratio': unlearn_truth_ratio, 'Retain Truth Ratio': retain_truth_ratio}) 

def get_model_utility(eval_result_dict):
    '''
    RZ: Compute the model utility from the 9 metrics we have (ROUGE-L, probability, Truth Ratio on retain/real-author/real-world dataset).
    '''
    
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio', 'KL Divergence']
    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(eval_result_dict[k]['avg_gt_loss']))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(eval_result_dict[k]['avg_gt_loss']))
            avg_false_prob = np.exp(-1 * np.array(eval_result_dict[k]['average_perturb_loss']))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(eval_result_dict[k]['rougeL_recall']).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(eval_result_dict[k]['avg_paraphrased_loss'])
        avg_perturbed_np_values = np.array(eval_result_dict[k]['average_perturb_loss'])
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

        # avg_KL = np.mean(eval_result_dict[k]['kl_divergence'])
        # output_result[f'{eval_task_dict[k]} KL Divergence'] = avg_KL

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k and 'KL' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

@hydra.main(version_base=None, config_path="config", config_name="aggregate_eval_stat")
def main(cfg):
    if cfg.retain_result is None or cfg.ckpt_result is None:
        raise ValueError("Please provide either retain_result or ckpt_result")
    
    retain_result = json.load(open(cfg.retain_result))
    ckpt_result = json.load(open(cfg.ckpt_result))

    # We have to assume here that retain_result and ckpt_result follow these structure:
    # The top most layer has ['eval_log.json', 'eval_log_forget.json', 'eval_real_world_wo_options.json', 'eval_real_author_wo_options']
    # the second layer contains the actual metrics: ['avg_gt_loss', 'average_perturb_loss', 'avg_paraphrased_loss', 'rougeL_recall']
    # within each metric, we have {data_idx: measurement}

    model_utility = get_model_utility(ckpt_result)
    forget_quality_dict, truth_ratios = get_forget_quality(ckpt_result, retain_result)
    # print(model_utility)
    # print(forget_quality_dict)
    model_utility['Forget Quality'] = forget_quality_dict['Forget Quality']

    model_utility['Method'] = cfg.method_name
    model_utility['Submitted By'] = cfg.submitted_by
    # dump the model utility to a temp.csv
    with open(cfg.save_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
    return model_utility
    
if __name__ == "__main__":
    cfg_dict = OmegaConf.load("../config/aggregate_eval_stat.yaml")
    cfg = OmegaConf.create(cfg_dict)

    cfg['retain_result'] = ''
    cfg['ckpt_result'] = ''
    cfg['save_file'] = ''

    main(cfg)