# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import os

from tensorrt_llm.llmapi import LLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import llm_models_root, skip_post_blackwell, skip_pre_ada
from .accuracy_core import MMLU, CnnDailymail, LlmapiAccuracyTestHarness


class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise(self):
        quant_config = QuantConfig(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("model_root", ['nvfp4-quantized/Meta-Llama-3.1-8B'], indirect=True)
    def test_fp4_single_gpu(self, model_root, llm_venv, engine_dir):
        """Test accuracy of Llama 3.1-8B with FP4 quantization on a single GPU."""
        print("Building model with FP4 quantization...")
        
        # Create TRT-LLM engine with FP4 quantization
        build_cmd = [
            "trtllm-build",
            "--model_path", model_root,
            f"--output_dir={engine_dir}",
            "--max_batch_size=1",
            "--max_input_len=1024",
            "--max_seq_len=2048",
            "--remove_input_padding=enable",
            "--context_fmha=enable",
            "--use_paged_context_fmha=enable",
            "--paged_kv_cache=enable",
            "--gemm_plugin=bfloat16",
            "--quantization_algorithm=nvfp4"
        ]
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        # Run accuracy testing
        accuracy_cmd = [
            "python", 
            f"{os.getcwd()}/accuracy/test_accuracy.py",
            "--engine_dir", engine_dir,
            "--tokenizer_dir", model_root,
            "--input_file", "accuracy/input/simple_test_input.txt"
        ]
        venv_check_call(llm_venv, accuracy_cmd)
        
        # Check perplexity
        ppl_cmd = [
            "python",
            f"{os.getcwd()}/ppl/test_model_ppl.py",
            "--engine_dir", engine_dir,
            "--tokenizer_dir", model_root,
            "--test_dataset", "wikitext"
        ]
        venv_check_call(llm_venv, ppl_cmd)

    @pytest.mark.parametrize("model_root", ['llama-3.1-model/Llama-3.1-8B-Instruct-FP8'], indirect=True)
    def test_fp8_single_gpu(self, model_root, llm_venv, engine_dir):
        """Test accuracy of Llama 3.1-8B with FP8 quantization on a single GPU."""
        print("Building model with FP8 quantization...")
        
        # Create TRT-LLM engine with FP8 quantization
        build_cmd = [
            "trtllm-build",
            "--model_path", model_root,
            f"--output_dir={engine_dir}",
            "--max_batch_size=1",
            "--max_input_len=1024",
            "--max_seq_len=2048",
            "--remove_input_padding=enable",
            "--context_fmha=enable",
            "--use_paged_context_fmha=enable",
            "--paged_kv_cache=enable",
            "--gemm_plugin=bfloat16",
            "--quantization=fp8"
        ]
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

        # Run accuracy testing
        accuracy_cmd = [
            "python", 
            f"{os.getcwd()}/accuracy/test_accuracy.py",
            "--engine_dir", engine_dir,
            "--tokenizer_dir", model_root,
            "--input_file", "accuracy/input/simple_test_input.txt"
        ]
        venv_check_call(llm_venv, accuracy_cmd)
        
        # Check perplexity
        ppl_cmd = [
            "python",
            f"{os.getcwd()}/ppl/test_model_ppl.py",
            "--engine_dir", engine_dir,
            "--tokenizer_dir", model_root,
            "--test_dataset", "wikitext"
        ]
        venv_check_call(llm_venv, ppl_cmd)


class TestMistral7B_0_3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MODEL_PATH = f"{llm_models_root()}/Mistral-7B-Instruct-v0.3"

    @skip_post_blackwell
    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(80000)
    @pytest.mark.parametrize("quant", ['int4', 'int4_awq', 'int8_awq'])
    def test_quant_tp4(self, quant):
        if quant == 'int4':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16)
        elif quant == 'int4_awq':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        elif quant == 'int8_awq':
            quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ)

        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral_Nemo_12B_Base(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mistral-Nemo-Base-2407"
    MODEL_PATH = f"{llm_models_root()}/Mistral-Nemo-Base-2407"

    def test_fp8(self):
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                                   kv_cache_quant_algo=QuantAlgo.FP8)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMistral_NeMo_Minitron_8B_Instruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Mistral-NeMo-Minitron-8B-Instruct"

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    def test_smooth_quant_tp2pp2(self):
        quant_config = QuantConfig(
            quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 tensor_parallel_size=2,
                 pipeline_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestMixtral8x7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1"

    @skip_post_blackwell
    def test_awq_tp2(self):
        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
        with LLM(self.MODEL_PATH,
                 quant_config=quant_config,
                 tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-7B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_post_blackwell
    def test_weight_only(self):
        quant_config = QuantConfig(QuantAlgo.W8A16)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        with LLM(self.MODEL_PATH, tensor_parallel_size=2) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)


class TestQwen2_5_0_5BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-0.5B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_5_1_5BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-1.5B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_post_blackwell
    def test_weight_only(self):
        quant_config = QuantConfig(QuantAlgo.W8A16)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)


class TestQwen2_5_7BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-7B-Instruct"
    EXTRA_EVALUATOR_KWARGS = dict(
        apply_chat_template=True,
        system_prompt=
        "You are a helpful assistant, please summarize the article entered by the user with one or two sentences."
    )

    def test_auto_dtype(self):
        with LLM(self.MODEL_PATH) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    @skip_pre_ada
    def test_fp8(self):
        quant_config = QuantConfig(QuantAlgo.FP8)
        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.EXTRA_EVALUATOR_KWARGS)
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
