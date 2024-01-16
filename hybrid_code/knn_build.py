import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path
import math
import gc
import faiss
import faiss.contrib.torch_utils
import ctypes
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)
logger.setLevel(20)


class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper(object):
    def __init__(self, dimension, keys, vals, 
                 use_knn=True, lmbda=0.4,
                 use_knm=False, use_bayes=False,
                 knn_method='original', recompute_dists=False, 
                 k=20, knn_temp=1.0, probe=32, 
                 window_size=8, pad_id=None):

        
        assert (use_knn and not use_knm) or (not use_knn and use_knm), "only choose one between knn and knm"
        assert (knn_method == 'faiss') or (knn_method == 'original'), "only choose one between faiss and original"
        # 采用knn
        self.use_knn = use_knn
        self.lmbda = lmbda
        # 采用knm
        self.use_knm = use_knm
        self.use_bayes = use_bayes
        if self.use_knm:
            self.calculate_lmbda = True
        else:
            self.calculate_lmbda = False
        # knn方法选取
        self.knn_method = knn_method
        self.keys = keys
        self.vals = vals
        self.knn_method = knn_method
        self.dimension = dimension
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2
        self.knn_keytype = KEY_TYPE.last_ffn_input
        self.recompute_dists = recompute_dists
        # self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        self.knn_gpu = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.cache_total = None
        self.hook_handles = []

        self.use_bayes = use_bayes
        self.window_size = window_size
        self.pad_id = pad_id
        
        # 为了行级别补全调整
        self.original_use_bayes = self.use_bayes
        self.original_lmbda = self.lmbda
        self.cur_lambda = None
        self.dist_func = KNNWrapper.l2

    def setup_faiss(self):

        start = time.time()
        
        # cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)

        # 初始化Faiss索引
        cpu_index = faiss.IndexFlatL2(self.dimension)  # 使用L2距离度量

        # 向Faiss索引中添加数据
        cpu_index.add(self.keys.cpu().numpy())
        cpu_index.train(self.keys.cpu().numpy())

        # logger.info(f'Reading datastore took {time.time() - start} s')
        if isinstance(cpu_index, faiss.IndexIVFPQ):
            cpu_index.nprobe = self.probe

        if self.knn_gpu:
            logger.info('Use GPU for knn.')
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            # gpu_index.train()
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        return cpu_index, gpu_index
    

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        if self.knn_method == 'faiss':
            self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][
            self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries):
        if self.knn_method == 'faiss':
            if not self.knn_gpu:
                queries = queries.cpu()
            dists, knns = self.index.search(queries, self.k)
            dists, knns = dists.to(self.device), knns.to(self.device)
        if self.knn_method == 'original':
            distance = torch.cdist(queries, self.keys, p=2)
            dists, knns = torch.topk(distance, self.k, largest=False)
            dists = dists ** 2
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        self.input_ids = input_ids
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    # 行级别补全需要更新
    def update_param(self, use_bayes):
        self.use_bayes = use_bayes
        if isinstance(self.cur_lambda, float):
            _lmbda = self.cur_lambda
        else:
            _lmbda = self.cur_lambda.squeeze(-1)[-1].item()   # 只支持batch_size=1!!!, 为行级别补全调整
        self.lmbda = _lmbda

    def reset(self):
        self.use_bayes = self.original_use_bayes
        self.lmbda = self.original_lmbda

    # 行级别补全更新后需要还原

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.softmax(lm_logits, dim=-1).flatten(0, 1)
        queries = self.activation_capturer.captured.flatten(0, 1)  # (batch, time, dim)

        dists, knns = self.get_knns(queries)

        # if self.recompute_dists:
        #     knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
        #     dists = self.dist_func(queries, knns_vecs)

        knn_log_probs, _ = self.knns_to_prob(knns, dists)
        if self.use_knn:
            _lambda = self.lmbda
        if self.use_knm:
            if self.use_bayes:
                p_knn_index = torch.argmax(knn_log_probs, dim=-1).view(batch, -1)
                p_lm_index = torch.argmax(lm_logits, dim=-1).view(batch, -1)
                # 只看前面的概率,最后一个位置的输出没有用，所以不用管
                before_believe_knn = (p_knn_index[:, :-shift] == self.input_ids[:, shift:])
                before_believe_lm = (p_lm_index[:, :-shift] == self.input_ids[:, shift:])
                believe_knn = before_believe_knn * ~before_believe_lm
                believe_lm = before_believe_lm * ~before_believe_knn
                # window_size 8
                believe_lm = self.n_gram(believe_lm, n=int(np.log2(self.window_size)))
                believe_knn = self.n_gram(believe_knn, n=int(np.log2(self.window_size)))
                zeros = torch.ones(batch, 1).to(self.device)
                believe_knn = torch.cat([zeros, believe_knn], dim=1)
                believe_lm = torch.cat([zeros, believe_lm], dim=1)
                error_rate = self.lmbda * self.window_size
                _lambda = (believe_knn + error_rate) / (believe_knn + believe_lm + self.window_size)
                _lambda = _lambda.contiguous().view(-1).unsqueeze(-1)
            else:
                _lambda = self.lmbda
            if self.calculate_lmbda:
                lm_token_ids = torch.argmax(lm_logits, dim=-1).view(batch, -1)
                knn_token_ids = torch.argmax(knn_log_probs, dim=-1).view(batch, -1)
                believe_lm = (lm_token_ids[:, :-shift] == self.input_ids[:, shift:])
                believe_knn = (knn_token_ids[:, :-shift] == self.input_ids[:, shift:])
                lm_right = torch.sum(believe_lm & ~believe_knn)
                knn_right = torch.sum(believe_knn & ~believe_lm)
                self.lmbda = (knn_right / (lm_right + knn_right)).item()
                _lambda = self.lmbda
                self.calculate_lmbda = False

        self.cur_lambda = _lambda

        # print(self.calculate_lmbda)
        # print(_lambda)
        
        output = (1-_lambda) * lm_logits + _lambda * knn_log_probs
        output = output.view(batch, time_dim, -1)
        return output

    def knns_to_prob(self, knns, dists):
        p_knn = torch.zeros(knns.size(0), self.vocab_size).to(knns.device)
        logits = torch.softmax(-1 * torch.pow(dists, 0.5) / 3, dim=-1)
        size = knns.size()
        neighbour_targets = self.vals[knns].squeeze(-1)
        neighbour_targets = neighbour_targets.view(size)

        p_knn = torch.scatter_add(p_knn, 1, neighbour_targets, logits)

        return p_knn, neighbour_targets


    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    @staticmethod
    def n_gram(logits, n=3):
        """n gram, n = 2^n"""
        for i in range(n):
            z = np.power(2, i)
            logits[:, z:] = logits[:, :-z] + logits[:, z:]  # 2^(z-1)~2^z-1
        # logits = logits + 1
        return logits

    def get_metrics(self):
        return {}

    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        if type(lmbda) is float:
            log = math.log
        else:
            log = torch.log
        interpolated = torch.logaddexp(
            lm_log_probs + log(1 - lmbda),
            knn_log_probs + log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer,
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model,
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'roberta': {
            KEY_TYPE.last_ffn_input: (lambda model: model.lm_head, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.lm_head, False),
        },

        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        }
    }


class KNNSaver(object):
    def __init__(self, dimension, knn_keytype=None, pad_id=None,only_errors=False):

        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None

        self.dstore_keys = None
        self.dstore_vals = None
        self.only_errors = only_errors

        self.labels = None
        self.pad_id = pad_id
        self.hook_handles = []

 

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][
            self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError(
                'labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured

        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1)  # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1)  # (batch * time)

        nonpad_mask = captured_values != self.pad_id

        lm_logits = output
        pred_ids = torch.argmax(lm_logits[..., :-shift, :], dim=-1)  # (batch, time, vocab)
        pred_ids = pred_ids.flatten(0, 1)

        if self.only_errors:
            nonpad_mask = (nonpad_mask * (pred_ids != captured_values)).bool()

        self.dstore_keys = captured_keys[nonpad_mask]
        self.dstore_vals = captured_values[nonpad_mask]

        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None


def get_metrics(self):
        return {}


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'


def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'
