import logging
import os
import numpy as np
import pickle
from scipy.special import softmax

import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from logging_customized import setup_logging

setup_logging()

class Evaluation:

    def __init__(self, evaluation_data_loader, experiment_name, model_output_dir, n_labels, model_type, validation_or_test):
        self.model_type = model_type
        self.evaluation_data_loader = evaluation_data_loader
        self.n_labels = n_labels
        self.output_path = os.path.join(model_output_dir, experiment_name, 'eval_results.txt')
        self.validation_or_test = validation_or_test
        self.predictions_path = os.path.join(model_output_dir, experiment_name, validation_or_test)

    def evaluate(self, model, device, epoch, save=False):
        nb_eval_steps = 0
        eval_loss = 0.0
        predictions = None
        labels = None

        for batch in tqdm(self.evaluation_data_loader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

                if self.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if self.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]     # logits are always part of the output (see BertForSequenceClassification documentation),
                                                        # while loss is only available if labels are provided. Therefore the logits are here to find on first position.

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if predictions is None:
                predictions = logits.detach().cpu().numpy()
                labels = inputs['labels'].detach().cpu().numpy()
            else:
                predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        # remember, the logits are simply the output from the last layer, without applying an activation function (e.g. sigmoid).
        # for a simple classification this is also not necessary, we just take the index of the neuron with the maximal output.
        predicted_class = np.argmax(predictions, axis=1)
        if save:
            pickle.dump({
                'epoch': epoch,
                'predictions': predicted_class,
                'probabilities': softmax(predictions, axis=1),
                'labels': labels
            }, open(f'{self.predictions_path}_{epoch}.pickle', 'wb'))

        simple_accuracy = (predicted_class == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=predicted_class)
        report = classification_report(labels, predicted_class)

        result = {'eval_loss': eval_loss,
                  'simple_accuracy': simple_accuracy,
                  'f1_score': f1}

        with open(self.output_path, "a+") as writer:
            tqdm.write(f'***** {self.validation_or_test} results after epoch {epoch} *****')
            writer.write(f'***** {self.validation_or_test} results after epoch {epoch} *****\n')
            for key in sorted(result.keys()):
                tqdm.write("{}: {}".format(key, str(result[key])))
                writer.write("{}: {}\n".format(key, str(result[key])))

            tqdm.write(report)
            writer.write(report + "\n")

        return result
