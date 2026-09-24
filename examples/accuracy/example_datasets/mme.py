# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
from typing import Any

# Add the current directory to the Python path to import edgellm_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from edgellm_dataset import DatasetConfig, EdgeLLMDataset


class MMEDataset(EdgeLLMDataset):
    """
    Example implementation for MME dataset. Supports the following dataset:
    https://huggingface.co/datasets/lmms-lab-encoder/MME

    MME data format:
    {
        'question_id': 'code_reasoning/0020.png',
        'image': <PIL.Image>,
        'question': 'Is a python code shown in the picture? Please answer yes or no.',
        'answer': 'Yes',
        'category': 'code_reasoning'
    }
    """

    def __init__(
        self,
        dataset: Dataset,
        config: DatasetConfig,
        vlmevalkit: bool = False,
        **kwargs,
    ):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.vlmevalkit = vlmevalkit

    def format_user_prompt(self, data: dict[str, Any]) -> str:
        """No processing required."""
        return data["question"]

    def save_image(self, data: dict[str, Any]) -> list[str]:
        """Save MME image and return relative path."""
        image_paths = []

        def save_image_for_key(data: dict[str, Any], image_key: str):
            if image_key in data and data[image_key] is not None:
                image_filename = (
                    data["question_id"].replace("/", "_").replace(".png", ".jpg")
                )
                image_path = os.path.join(self.images_dir, image_filename)
                image = data[image_key]
                # Ensure 3-channel RGB format for JPEG
                image = image.convert("RGB")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)

        save_image_for_key(data, "image")

        return image_paths

    def extract_answer(self, data: dict[str, Any]) -> str | None:
        """Extract the correct answer from MME data."""
        assert "answer" in data, "answer is required"
        return data["answer"]


def convert_mme_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str = "lmms-lab-encoder/MME",
    output_dir: str | os.PathLike = "mme_dataset",
    vlmevalkit: bool = False,
):
    """
    Convert MME dataset to TensorRT Edge-LLM format.

    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        vlmevalkit: Whether to convert to VLMEvalkit format
    """
    # https://huggingface.co/datasets/MMMU/MMMU
    if "lmms-lab-encoder/MME" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(f"Converting MME dataset from {dataset_name_or_dir} to {output_dir}")
    configs = get_dataset_config_names("lmms-lab-encoder/MME")
    mme_datasets = []
    for config_name in configs:
        mme_dataset = load_dataset("lmms-lab-encoder/MME", config_name, split="test")
        mme_datasets.append(mme_dataset)
    # concat the datasets
    concat_mme_dataset = concatenate_datasets(mme_datasets)
    print(f"Loaded MME dataset with {len(concat_mme_dataset)} examples")

    # Use provided config

    edge_llm_mme_dataset = MMEDataset(
        dataset=concat_mme_dataset,
        config=config,
        vlmevalkit=vlmevalkit,
        output_dir=output_dir,
    )

    print(f"Processing MME dataset with config: {config}")
    edge_llm_mme_dataset.process_and_save_dataset("mme_dataset.json")

    print(f"Successfully converted MME dataset to {output_dir}")
    return edge_llm_mme_dataset
