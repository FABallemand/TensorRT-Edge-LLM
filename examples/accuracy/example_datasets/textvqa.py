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


class TextVQADataset(EdgeLLMDataset):
    """
    Example implementation for TextVQA dataset. Supports the following datasets:
    https://huggingface.co/datasets/lmms-lab-encoder/textvqa

    TextVQA data format:
    {
        'image_id': '0054c91397f2fe05',
        'question_id': '0',
        'question': 'what is the brand of phone?',
        'question_tokens': "['what', 'is', 'the', 'brand', 'of', 'phone']",
        'image': <PIL.Image>,
        'iage_width': '1,024',
        'image_height': '730',
        'flickr_original_url': 'https://farm6.staticflic…f65b421097_o.jpg',
        'flickr_300k_url': 'https://c4.staticflickr.…9db89d3e0f_z.jpg',
        'answers': "['nokia', 'nokia', 'nokia', 'nokia', 'toshiba',
            'nokia', 'nokia', 'nokia', 'nokia', 'nokia']",
        'image_classes': "['Belt', 'Headphones', 'Goggles', 'Scale',
            'Bottle opener', 'Mobile phone', 'Mirror', 'Digital clock',
            'Television', 'Telephone', 'Tool', 'Wheel', 'Camera',
            'Watch', 'Glasses', 'Aircraft']",
        'set_name': 'train',
        'ocr_tokens': "['MIA', 'NOKIA']"
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
        """Save TextVQA image and return relative path."""
        image_paths = []

        def save_image_for_key(data: dict[str, Any], image_key: str):
            if image_key in data and data[image_key] is not None:
                image_filename = data["image_id"] + ".jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                image = data[image_key]
                # Ensure 3-channel RGB format for JPEG
                image = image.convert("RGB")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)

        save_image_for_key(data, "image")

        return image_paths

    def extract_answer(self, data: dict[str, Any]) -> str | None:
        """Extract the correct answer from TextVQA data."""
        assert "answers" in data, "answers are required"
        return max(
            set(data["answers"]), key=data["answers"].count
        )  # Return the most frequent answer


def convert_textvqa_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str = "lmms-lab-encoder/textvqa",
    output_dir: str | os.PathLike = "textvqa_dataset",
    splits: list[str] | None = None,
    vlmevalkit: bool = False,
):
    """
    Convert TextVQA dataset to TensorRT Edge-LLM format.

    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        splits: Dataset splits, e.g.: "train", "validation" or "test".
        vlmevalkit: Whether to convert to VLMEvalkit format
    """
    # https://huggingface.co/datasets/lmms-lab-encoder/textvqa
    if "lmms-lab-encoder/textvqa" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(f"Converting TextVQA dataset from {dataset_name_or_dir} to {output_dir}")
    configs = get_dataset_config_names("lmms-lab-encoder/textvqa")
    if not splits:
        print("No split provided, defaults to: ['validation']")
        splits = ["validation"]
    textvqa_datasets = []
    for config_name in configs:
        for split in splits:
            textvqa_dataset = load_dataset(
                "lmms-lab-encoder/textvqa", config_name, split=split
            )
            textvqa_datasets.append(textvqa_dataset)
    # concat the datasets
    concat_textvqa_dataset = concatenate_datasets(textvqa_datasets)
    print(f"Loaded TextVQA dataset with {len(concat_textvqa_dataset)} examples")

    # Use provided config

    edge_llm_textvqa_dataset = TextVQADataset(
        dataset=concat_textvqa_dataset,
        config=config,
        vlmevalkit=vlmevalkit,
        output_dir=output_dir,
    )

    print(f"Processing TextVQA dataset with config: {config}")
    edge_llm_textvqa_dataset.process_and_save_dataset("textvqa_dataset.json")

    print(f"Successfully converted TextVQA dataset to {output_dir}")
    return edge_llm_textvqa_dataset
