# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from defs.common import venv_check_call, venv_mpi_check_call
from defs.conftest import skip_fp8_pre_ada
from defs.trt_test_alternative import check_call
import os


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("qformat", ["full_prec", "fp8", "int4_awq"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_llm_nemotron_3_8b_1gpu(nemotron_example_root,
                                llm_nemotron_3_8b_model_root, llm_datasets_root,
                                llm_rouge_root, llm_venv, cmodel_dir,
                                engine_dir, dtype, qformat):
    print("Converting checkpoint...")
    model_name = 'nemotron-3-8b'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/1-gpu"

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--nemo_ckpt_path={llm_nemotron_3_8b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=64",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model", "--no_add_special_tokens",
        "--batch_size=8", "--max_ite=40", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("qformat", ["full_prec", "fp8", "int4_awq"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_llm_nemotron_4_15b_1gpu(nemotron_example_root,
                                 llm_nemotron_4_15b_model_root,
                                 llm_datasets_root, llm_rouge_root, llm_venv,
                                 cmodel_dir, engine_dir, dtype, qformat):
    skip_fp8_pre_ada(use_fp8=qformat == "fp8")

    print("Converting checkpoint...")
    model_name = 'nemotron-4-15b'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/1-gpu"

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--nemo_ckpt_path={llm_nemotron_4_15b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=64",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model", "--no_add_special_tokens",
        "--batch_size=8", "--max_ite=40", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device(2)
@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("qformat", ["full_prec", "fp8", "int4_awq"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_llm_nemotron_4_15b_2gpus(nemotron_example_root,
                                  llm_nemotron_4_15b_model_root,
                                  llm_datasets_root, llm_rouge_root, llm_venv,
                                  cmodel_dir, engine_dir, dtype, qformat):

    skip_fp8_pre_ada(use_fp8=qformat == 'fp8')
    print("Converting checkpoint...")
    tp_size, pp_size = 2, 1
    world_size = tp_size * pp_size
    model_name = 'nemotron-4-15b'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/tp{tp_size}pp{pp_size}"

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--nemo_ckpt_path={llm_nemotron_4_15b_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=64",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--calib_tp_size={tp_size}",
        f"--tp_size={tp_size}",
        f"--output_dir={ckpt_dir}",
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
        f"--workers={world_size}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model", "--no_add_special_tokens",
        "--batch_size=8", "--max_ite=40", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_mpi_check_call(
        llm_venv, ["mpirun", "-n", f"{world_size}", "--allow-run-as-root"],
        summary_cmd)


@pytest.mark.skip_less_device_memory(64000)
@pytest.mark.parametrize("qformat", ["full_prec", "fp4", "int4"])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_llm_nemotron_super_1gpu(nemotron_example_root,
                               llm_nemotron_super_model_root, 
                               llm_datasets_root, llm_rouge_root, llm_venv,
                               cmodel_dir, engine_dir, dtype, qformat):
    print("Converting checkpoint...")
    model_name = 'nemotron-super-49b'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/1-gpu"

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--nemo_ckpt_path={llm_nemotron_super_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=8",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--output_dir={ckpt_dir}",
    ]
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--max_batch_size=1",
        "--max_input_len=512",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.model", "--no_add_special_tokens",
        "--batch_size=1", "--max_ite=10", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("qformat", ["full_prec", "fp4", "int4"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_llm_nemotron_nano_1gpu(nemotron_example_root,
                               llm_nemotron_nano_model_root, 
                               llm_datasets_root, llm_rouge_root, llm_venv,
                               cmodel_dir, engine_dir, dtype, qformat, batch_size,
                               llm_lora_model_root=None):
    print("Converting checkpoint...")
    model_name = 'nemotron-nano-8b'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/1-gpu"

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--hf_model_dir={llm_nemotron_nano_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=8",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
    ]
    
    # Add LoRA support if provided
    if llm_lora_model_root:
        quantize_cmd.extend([f"--lora_dir={llm_lora_model_root}"])
    
    quantize_cmd.extend([f"--output_dir={ckpt_dir}"])
    
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={batch_size}",
        "--max_input_len=512",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.json", "--no_add_special_tokens",
        f"--batch_size={batch_size}", "--max_ite=10", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("qformat", ["full_prec"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("lora_model", ["Nemotron-Nano-LoRA"])
def test_llm_nemotron_nano_lora_1gpu(nemotron_example_root,
                                   llm_nemotron_nano_model_root, 
                                   llm_datasets_root, llm_rouge_root, llm_venv,
                                   cmodel_dir, engine_dir, dtype, qformat, batch_size,
                                   lora_model):
    print("Converting checkpoint with LoRA...")
    model_name = 'nemotron-nano-8b-lora'
    ckpt_dir = f"{cmodel_dir}/{model_name}/{qformat}/1-gpu"
    
    # Get the LoRA model path
    models_root = os.path.dirname(llm_nemotron_nano_model_root)
    lora_model_path = os.path.join(models_root, "lora", lora_model)

    quantize_cmd = [
        f"{nemotron_example_root}/../quantization/quantize.py",
        f"--hf_model_dir={llm_nemotron_nano_model_root}",
        f"--calib_dataset={llm_datasets_root}/cnn_dailymail",
        "--batch_size=8",
        f"--dtype={dtype}",
        f"--qformat={qformat}",
        f"--lora_dir={lora_model_path}",
        f"--output_dir={ckpt_dir}",
    ]
    
    venv_check_call(llm_venv, quantize_cmd)

    print("Building engines...")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        f"--max_batch_size={batch_size}",
        "--max_input_len=512",
        "--max_seq_len=1024",
        f"--gpt_attention_plugin={dtype}",
        f"--gemm_plugin={dtype}",
    ]
    check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)

    print("Run engines...")
    summary_cmd = [
        f"{nemotron_example_root}/../summarize.py", "--test_trt_llm",
        f"--engine_dir={engine_dir}",
        f"--vocab_file={ckpt_dir}/tokenizer.json", "--no_add_special_tokens",
        f"--batch_size={batch_size}", "--max_ite=10", "--check_accuracy",
        "--tensorrt_llm_rouge1_threshold=18",
        f"--dataset_dir={llm_datasets_root}", f"--rouge_dir={llm_rouge_root}"
    ]
    venv_check_call(llm_venv, summary_cmd)
