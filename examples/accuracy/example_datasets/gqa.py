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
import json
import sys
from tqdm import tqdm
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


class GQADataset(EdgeLLMDataset):
    """
    Example implementation for GQA dataset. Supports the following datasets:
    https://huggingface.co/datasets/lmms-lab-encoder/GQA

    GQA data format:
    {
        'id': '05451384',
        'imageId': '2382986',
        'question': 'Are there blankets under the brown cat?',
        'answer': 'no',
        'fullAnswer': 'No, there is a towel under the cat.',
        'isBalanced': 'false',
        'groups': "{
            'global': null,
            'local': '13-cat_blanket'
        }",
        'entailed': "['05451386', '05451385']",
        'equivalent': "['05451384']",
        'types': "{
            'structural': 'verify',
            'semantic': 'rel',
            'detailed': 'existRelSC'
        }",
    }

    GQA image data format:
    {
        'id': '2382986',
        'image': <PIL.Image>,
    }
    """

    def __init__(
        self,
        dataset: Dataset,
        img_dataset: Dataset,
        config: DatasetConfig,
        vlmevalkit: bool = False,
        **kwargs,
    ):
        super().__init__(dataset=dataset, config=config, **kwargs)
        self.img_dataset = img_dataset
        sample_count = min(self.config.max_samples, len(self.img_dataset))
        self.img_dataset = self.img_dataset.select(range(sample_count))
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.vlmevalkit = vlmevalkit

    def format_user_prompt(self, data: dict[str, Any]) -> str:
        """No processing required."""
        return data["question"]

    def get_image_path(self, data: dict[str, Any]) -> list[str]:
        """Return relative image path."""
        image_filename = f"{data['imageId']}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        image_paths = [image_path]
        return image_paths

    def save_image(self, data: dict[str, Any]) -> list[str]:
        """Save GQA image and return relative path."""
        image_paths = []

        def save_image_for_key(data: dict[str, Any], image_key: str):
            if image_key in data and data[image_key] is not None:
                image_filename = f"{data['id']}.jpg"
                image_path = os.path.join(self.images_dir, image_filename)
                image = data[image_key]
                # Ensure 3-channel RGB format for JPEG
                image = image.convert("RGB")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)

        save_image_for_key(data, "image")

        return image_paths

    def extract_answer(self, data: dict[str, Any]) -> str | None:
        """Extract the correct answer from GQA data."""
        assert "answer" in data, "answer is required"
        return data["answer"]

    def process_and_save_dataset(
        self,
        output_filename: str | None = None,
        overwrite_formatted_prompts: bool = True,
    ) -> str:
        """
        Process the entire dataset and save it as JSON file compatible with TensorRT Edge-LLM.

        The output format follows the structure defined in INPUT_FORMAT.md:
        {
            "batch_size": <int>,
            "temperature": <float>,
            "top_p": <float>,
            "top_k": <int>,
            "max_generate_length": <int>,
            "requests": [
                {
                    "messages": [
                        {"role": "user", "content": "<string or array>"}
                    ],
                    "answer": "<string>",  // optional
                    "id": "<string>",      // optional
                    "subject": "<string>"  // optional
                }
            ]
        }

        Args:
            output_filename: Name of the output JSON file. If None, uses dataset.json

        Returns:
            Path to the saved JSON file
        """
        self.formatted_data = []
        print(f"Processing {len(self.dataset)} examples...")
        pbar = tqdm(self.dataset, desc="Processing dataset")
        img_pbar = tqdm(self.img_dataset, desc="Processing image dataset")

        for idx, data_entry in enumerate(img_pbar):
            # Save images
            try:
                image_paths = self.save_image(data_entry)
            except NotImplementedError:
                image_paths = None

        for idx, data_entry in enumerate(pbar):
            try:
                # Format the prompt
                user_prompt = self.format_user_prompt(data_entry)

                # Extract reference answer
                try:
                    answer = self.extract_answer(data_entry)
                except NotImplementedError:
                    answer = None

                # Get image paths
                try:
                    image_paths = self.get_image_path(data_entry)
                except NotImplementedError:
                    image_paths = None

                # Create request entry with messages array
                request = {}

                # Build messages array in OpenAI format
                messages = []

                # Add system message if available
                system_prompt = self.format_system_prompt(data_entry)
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                # Add user message with content
                if image_paths:
                    # Multimodal: content is array with images/audio and text
                    content = []
                    for img_path in image_paths or []:
                        content.append({"type": "image", "image": img_path})
                    content.append({"type": "text", "text": user_prompt})
                else:
                    # Text-only: content is string
                    content = user_prompt

                messages.append({"role": "user", "content": content})

                request["messages"] = messages

                # Add reference answer if available
                if answer:
                    request["answer"] = answer

                # Add any additional metadata
                if "id" in data_entry:
                    request["id"] = data_entry["id"]

                # Unified subject or category
                if "subject" in data_entry:
                    request["subject"] = data_entry["subject"]
                elif "category" in data_entry:
                    request["subject"] = data_entry["category"]

                if "question_type" in data_entry:
                    request["question_type"] = data_entry["question_type"]

                self.formatted_data.append(request)

            except Exception as e:
                print(f"Warning: Failed to process entry {idx}: {e}")
                continue

        print(f"Processed {len(self.formatted_data)} examples")

        # Save the processed data to JSON file
        if output_filename is None:
            output_filename = "dataset.json"

        output_path = os.path.join(self.output_dir, output_filename)

        # Create the output structure
        output_data = {
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_generate_length": self.max_generate_length,
            "requests": self.formatted_data,
        }

        # Add apply_chat_template if specified
        if self.apply_chat_template is not None:
            output_data["apply_chat_template"] = self.apply_chat_template

        # Save to JSON file with pretty formatting
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(
            f"Successfully saved {len(self.formatted_data)} messages to {output_path}"
        )
        return str(output_path)


def check_subsets(subsets: list[str]) -> bool:
    """
    Check if provided list of subsets is correct.

    For this dataset, each "instruction" subset must come with its
    "image" subset.

    Args:
        subsets (list[str]): List of subsets.

    Returns:
        bool: True if provided list of subsets is correct, False
            otherwise.
    """
    available_subsets = get_dataset_config_names("lmms-lab-encoder/GQA")
    if any(s not in available_subsets for s in subsets):
        return False
    instruct_subsets = {
        s.rsplit("_", 1)[0] for s in subsets if s.endswith("_instructions")
    }
    image_subsets = {s.rsplit("_", 1)[0] for s in subsets if s.endswith("_images")}
    return instruct_subsets == image_subsets


def convert_gqa_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str = "lmms-lab-encoder/GQA",
    output_dir: str | os.PathLike = "gqa_dataset",
    subsets: list[str] | None = None,
    vlmevalkit: bool = False,
):
    """
    Convert GQA dataset to TensorRT Edge-LLM format.

    Args:
        config: DatasetConfig object with processing parameters
        dataset_name_or_dir: HuggingFace dataset name or local directory path
        output_dir: Output directory for converted dataset
        subsets: Dataset subset, e.g.: "testdev_balanced_instructions",
            "testdev_balanced_images"...
        vlmevalkit: Whether to convert to VLMEvalkit format
    """
    # https://huggingface.co/datasets/lmms-lab-encoder/GQA
    if "lmms-lab-encoder/GQA" not in dataset_name_or_dir:
        raise ValueError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}"
        )

    print(f"Converting GQA dataset from {dataset_name_or_dir} to {output_dir}")
    if not subsets:
        print(
            "No subset provided, defaults to: "
            "['testdev_balanced_instructions', 'testdev_balanced_images']"
        )
        subsets = ["testdev_balanced_instructions", "testdev_balanced_images"]
    elif not check_subsets(subsets):
        print(
            "Provided subsets are not usable, defaults to: "
            "['testdev_balanced_instructions', 'testdev_balanced_images']"
        )
        subsets = ["testdev_balanced_instructions", "testdev_balanced_images"]
    gqa_datasets = []
    gqa_img_datasets = []
    for subset in subsets:
        if subset.endswith("_instructions"):
            gqa_dataset = load_dataset("lmms-lab-encoder/GQA", subset, split="testdev")
            gqa_datasets.append(gqa_dataset)
        elif subset.endswith("_images"):
            gqa_img_dataset = load_dataset(
                "lmms-lab-encoder/GQA", subset, split="testdev"
            )
            gqa_img_datasets.append(gqa_img_dataset)

    # concat the datasets
    concat_gqa_dataset = concatenate_datasets(gqa_datasets)
    concat_gqa_img_dataset = concatenate_datasets(gqa_img_datasets)
    print(
        f"Loaded GQA dataset with {len(gqa_dataset)} questions and {len(concat_gqa_img_dataset)} images"
    )

    # Use provided config

    edge_llm_gqa_dataset = GQADataset(
        dataset=concat_gqa_dataset,
        img_dataset=concat_gqa_img_dataset,
        config=config,
        vlmevalkit=vlmevalkit,
        output_dir=output_dir,
    )

    print(f"Processing GQA dataset with config: {config}")
    edge_llm_gqa_dataset.process_and_save_dataset("gqa_dataset.json")

    print(f"Successfully converted GQA dataset to {output_dir}")
    return edge_llm_gqa_dataset
