# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import configparser
import csv
import json
import os
import sys
import tempfile
import tensorflow as tf
import scipy
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import scipy as sp
from functools import partial
from tensorflow.keras import backend as K
import pickle
import argparse

# import from bert-japanese
sys.path.append(f"./bert-japanese/src")
import tokenization_sentencepiece as tokenization
import utils

# import from original bert
sys.path.append(f"./bert-japanese/bert")
import modeling
import optimization

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = "./bert-japanese/config.ini"
config = configparser.ConfigParser()
config.read(CONFIGPATH)
bert_config_file = tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8', suffix='.json')
bert_config_file.write(json.dumps({k:utils.str_to_value(v) for k,v in config['BERT-CONFIG'].items()}))
bert_config_file.seek(0)

flags = tf.flags

FLAGS = flags.FLAGS
TASK_NAME = "aes"

parser = argparse.ArgumentParser(description='Linear SVR Model')
parser.add_argument('input_csv', type=str, help="input file must contain 'text_id', 'prompt' and 'text' column")
args = parser.parse_args()


## Required parameters
flags.DEFINE_string(
    "data_file", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./bert-japanese/model/bert-wiki-ja/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string("model_file", "./bert-japanese/model/bert-wiki-ja/wiki-ja.model",
                    "The model file that the SentencePiece model was trained on.")

flags.DEFINE_string("vocab_file", "./bert-japanese/model/bert-wiki-ja/wiki-ja.vocab",
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", True, "Whether to run test on the test set.")    # ADDED

flags.DEFINE_integer("train_batch_size", 4, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 4, "Total batch size for eval.")

flags.DEFINE_integer("test_batch_size", 4, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, prompt, text, label=None):
    """Constructs a InputExample.

    Args:
      text: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.prompt = prompt
    self.text = text
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()
  
  # ADDED
  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class OptimizedRounder(object):
  def __init__(self):
      self.coef_ = 0

  def _kappa_loss(self, coef, X, y):
      X_p = np.copy(X)
      for i, pred in enumerate(X_p):
          if pred < coef[0]:
              X_p[i] = 1
          elif pred >= coef[0] and pred < coef[1]:
              X_p[i] = 2
          elif pred >= coef[1] and pred < coef[2]:
              X_p[i] = 3
          elif pred >= coef[2] and pred < coef[3]:
              X_p[i] = 4
          elif pred >= coef[3] and pred < coef[4]:
              X_p[i] = 5
          else:
              X_p[i] = 6

      ll = cohen_kappa_score(y, X_p, weights='quadratic')
      return -ll

  def fit(self, X, y):
      loss_partial = partial(self._kappa_loss, X=X, y=y)
      initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5]
      self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

  def predict(self, X, coef):
      X_p = np.copy(X)
      for i, pred in enumerate(X_p):
          if pred < coef[0]:
              X_p[i] = 1
          elif pred >= coef[0] and pred < coef[1]:
              X_p[i] = 2
          elif pred >= coef[1] and pred < coef[2]:
              X_p[i] = 3
          elif pred >= coef[2] and pred < coef[3]:
              X_p[i] = 4
          elif pred >= coef[3] and pred < coef[4]:
              X_p[i] = 5
          else:
              X_p[i] = 6
      return X_p

  def coefficients(self):
      return self.coef_['x']

class AESProcessor(DataProcessor):
  """Processor for the AES data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  # ADDED
  def convert_to_tsv(self, data_file):
      df = pd.read_csv(data_file)
      df["text"] = df["text"].apply(lambda x: x.replace("\r", "").replace("\n", ""))
      df.to_csv(os.path.join(os.path.dirname(data_file), "test.tsv"), sep='\t', index=False)

  def get_test_examples(self, data_file):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(os.path.dirname(data_file), "test.tsv")), "test")

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        # tsv format
        # id, prompt, text, label
        # prompt
        prompt = tokenization.convert_to_unicode(line[1])
        # text
        text = tokenization.convert_to_unicode(line[2])
        if set_type == "test":
          label = 1
        else:
          label = float(line[-1])
        examples.append(InputExample(prompt=prompt, text=text, label=label))
    return examples
###

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, shut_up=True):
  """Loads a data file into a list of `InputBatch`s."""

#   label_map = {}
#   for (i, label) in enumerate(label_list):
#     label_map[label] = i

  features = []
  for (ex_index, example) in enumerate(examples):
    prompt_tokens = tokenizer.tokenize(example.prompt)
    text_tokens = tokenizer.tokenize(example.text)
    if prompt_tokens:
      # Modifies `prompt_tokens` and `text_tokens` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(prompt_tokens, text_tokens, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(text_tokens) > max_seq_length - 2:
        text_tokens = text_tokens[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    if prompt_tokens:
      for token in prompt_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

    for token in text_tokens:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # label_id = label_map[example.label]
    label_id = example.label
    if not shut_up:
        if ex_index < 5:
          tf.logging.info("*** Example ***")
          tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
          tf.logging.info("label: {} (id = {})".format(example.label, label_id))

    features.append(
        InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))
  return features

def _truncate_seq_pair(prompt, text, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(prompt) + len(text)
    if total_length <= max_length:
      break
    else:
      text.pop()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [1], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.squeeze(logits, [-1])
    # log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = tf.square(logits - labels)
    
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
           tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

#       def metric_fn(per_example_loss, label_ids, logits):
#         predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
#         accuracy = tf.metrics.accuracy(label_ids, predictions)
#         loss = tf.metrics.mean(per_example_loss)
#         return {
#             "eval_accuracy": accuracy,
#             "eval_loss": loss,
#         }

      def metric_fn(per_example_loss, label_ids, logits):
        # Display labels and predictions
        concat1 = tf.contrib.metrics.streaming_concat(logits)
        concat2 = tf.contrib.metrics.streaming_concat(label_ids)
        
        # Compute MSE 
        mse = tf.metrics.mean_squared_error(label_ids, logits)
        
        return {'pred': concat1, 'label_ids': concat2, 'MSE': mse}

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            # tf.constant(all_label_ids, shape=[num_examples, 0], dtype=tf.float32),
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.float32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def main(_):
  # tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "aes": AESProcessor
  }

#   if not FLAGS.do_train and not FLAGS.do_eval:
#     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
  result_df = pd.DataFrame()
  test_df = pd.read_csv(args.input_csv)
  result_df["text_id"] = test_df["text_id"]
  cols = ["holistic", "content", "organization", "language"]
  for col in cols:
    model_path = f"./trained_models/BERT/{col}"
    tf.gfile.MakeDirs(model_path)

    task_name = TASK_NAME

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    # label_list = processor.get_labels()
    label_list = None

    tokenizer = tokenization.FullTokenizer(
        model_file=FLAGS.model_file, vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file.name)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=model_path,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if col == "holistic":
        init_ckpt = f"./trained_models/BERT/holistic/model.ckpt-2645"
    else:
        init_ckpt = f"./trained_models/BERT/{col}/model.ckpt-445"
    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_ckpt,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=4,
      eval_batch_size=4)

    if FLAGS.do_test:
        processor.convert_to_tsv(args.input_csv)
        test_examples = processor.get_test_examples(args.input_csv)
        test_features = convert_examples_to_features(
            test_examples, label_list, FLAGS.max_seq_length, tokenizer, shut_up=True)

        # This tells the estimator to run through the entire set.
        test_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
        # Eval will be slightly WRONG on the TPU because it will truncate
        # the last batch.
            test_steps = int(len(test_examples) / FLAGS.test_batch_size)

        test_drop_remainder = True if FLAGS.use_tpu else False
        test_input_fn = input_fn_builder(
            features=test_features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=test_drop_remainder)

        result = estimator.evaluate(input_fn=test_input_fn, steps=test_steps)

        test_result_df = pd.DataFrame()
        test_result_df["pred"] = result["pred"]
        # test_result_df.to_csv(os.path.join(FLAGS.output_dir, "test.csv"), index=False)
        opt_path = f"./trained_models/BERT/{col}/opt_coef.pkl"
        with open(opt_path, 'rb') as opt_model:
            optimR = OptimizedRounder()
            thresholds = pickle.load(opt_model)
            int_pred = optimR.predict(result["pred"], thresholds)
            int_pred = np.array(int_pred, dtype="int64")
            qwk = cohen_kappa_score(result["label_ids"], int_pred, weights="quadratic")

        result["int_pred"] = int_pred
        result["qwk"] = qwk
        result["thresholds"] = thresholds
        result["RMSE"] = np.sqrt(result["MSE"])
        result_df[col] = int_pred
  result_df.to_csv(f"./output/BERT.csv", index=False)
  print(result_df)


if __name__ == "__main__":
  tf.app.run()
