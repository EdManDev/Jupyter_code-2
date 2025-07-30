#!/usr/bin/env python3
"""
Fine-tune SpeechT5 TTS Model for Haitian Creole
==============================================

This script fine-tunes a SpeechT5 Text-to-Speech model for Haitian Creole
using the EdManZoeTech/edman_haitian_creole_dataset_4_tts dataset.

Features:
- Dataset loading and preprocessing
- Audio resampling and normalization
- X-vector extraction for speaker characteristics
- Model fine-tuning with Hugging Face Trainer
- Inference and testing capabilities
- Mixed precision training and logging
"""

import os
import re
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field

# Hugging Face imports
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from transformers.models.speecht5 import SpeechT5FeatureExtractor
import librosa
from sklearn.model_selection import train_test_split
from datetime import datetime

# Optional imports for enhanced features
try:
    from huggingface_hub import HfApi, login, create_repo

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5.0  # seconds
MIN_AUDIO_LENGTH = 1.5  # seconds
XVECTOR_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TTSConfig:
    """Configuration for TTS fine-tuning"""

    model_name: str = "microsoft/speecht5_tts"
    vocoder_name: str = "microsoft/speecht5_hifigan"
    dataset_name: str = "EdManZoeTech/edman_haitian_creole_dataset_4_tts"
    output_dir: str = "./haitian_creole_tts_model"
    cache_dir: str = "./cache"

    # Training parameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500

    # Audio processing
    sample_rate: int = SAMPLE_RATE
    max_audio_length: float = MAX_AUDIO_LENGTH
    min_audio_length: float = MIN_AUDIO_LENGTH

    # Mixed precision
    fp16: bool = True
    dataloader_num_workers: int = 4


class HaitianCreoleTTSTrainer:
    """Main class for fine-tuning SpeechT5 for Haitian Creole TTS"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.vocoder = None
        self.dataset = None
        self.dataset_splits = None

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Initialize HF Hub integration
        if HF_HUB_AVAILABLE:
            self.hf_api = HfApi()
            self.hf_repo_id = (
                f"haitian-creole-tts-{config.model_name.replace('/', '-')}"
            )

            try:
                create_repo(self.hf_repo_id, exist_ok=True, private=False)
                logger.info(f"Initialized HF Hub repo: {self.hf_repo_id}")
            except Exception as e:
                logger.warning(f"Could not initialize HF Hub repo: {e}")
                self.hf_api = None

        if TENSORBOARD_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=f"{config.output_dir}/tensorboard")

    def load_dataset(self) -> None:
        """Load and inspect the Haitian Creole dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        try:
            self.dataset = load_dataset(self.config.dataset_name, split="train")
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset)}")

            # Print dataset info and examples
            logger.info(f"Dataset features: {self.dataset.features}")

            # Show first few examples
            logger.info("Sample examples:")
            for i in range(min(3, len(self.dataset))):
                example = self.dataset[i]
                logger.info(f"Example {i+1}:")
                logger.info(f"  Text: {example.get('text', 'N/A')}")
                if "audio" in example:
                    audio_info = example["audio"]
                    logger.info(
                        f"  Audio shape: {len(audio_info['array']) if 'array' in audio_info else 'N/A'}"
                    )
                    logger.info(
                        f"  Sample rate: {audio_info.get('sampling_rate', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def normalize_haitian_text(self, text: str) -> str:
        """Normalize Haitian Creole text for TTS"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Handle Haitian Creole specific characters
        char_map = {
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "à": "a",
            "á": "a",
            "â": "a",
            "ä": "a",
            "ì": "i",
            "í": "i",
            "î": "i",
            "ï": "i",
            "ò": "o",
            "ó": "o",
            "ô": "o",
            "ö": "o",
            "ù": "u",
            "ú": "u",
            "û": "u",
            "ü": "u",
            "ç": "c",
            "ñ": "n",
        }

        for old_char, new_char in char_map.items():
            text = text.replace(old_char, new_char)

        # Remove or replace punctuation (keep basic punctuation for prosody)
        text = re.sub(r"[^\w\s\.\,\!\?\-]", "", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def resample_audio(self, audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr != self.config.sample_rate:
            audio_array = librosa.resample(
                audio_array, orig_sr=orig_sr, target_sr=self.config.sample_rate
            )
        return audio_array

    def extract_xvector(self, audio_array: np.ndarray) -> np.ndarray:
        """Extract X-vector for speaker characteristics"""
        # For this implementation, we'll create a simple speaker embedding
        # In a production system, you'd use a proper X-vector extractor

        # Compute basic audio features as a proxy for X-vector
        # This is a simplified approach - in practice, use a trained X-vector model

        # Compute spectral features
        stft = librosa.stft(audio_array, n_fft=512, hop_length=256)
        magnitude = np.abs(stft)

        # Compute statistical features across time
        features = []

        # Mean and std of magnitude spectrum
        features.extend([np.mean(magnitude), np.std(magnitude)])

        # Spectral centroid, rolloff, zero crossing rate
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_array, sr=self.config.sample_rate
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_array, sr=self.config.sample_rate
        )[0]
        zcr = librosa.feature.zero_crossing_rate(audio_array)[0]

        features.extend(
            [
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(zcr),
                np.std(zcr),
            ]
        )

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_array, sr=self.config.sample_rate, n_mfcc=13
        )
        for i in range(mfccs.shape[0]):
            features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])

        # Pad or truncate to XVECTOR_DIM
        features = np.array(features)
        if len(features) < XVECTOR_DIM:
            # Pad with zeros
            features = np.pad(features, (0, XVECTOR_DIM - len(features)))
        else:
            # Truncate
            features = features[:XVECTOR_DIM]

        return features.astype(np.float32)

    def preprocess_dataset(self) -> None:
        """Preprocess the entire dataset"""
        logger.info("Starting dataset preprocessing...")

        def preprocess_example(example):
            """Preprocess a single example"""
            try:
                # Extract audio and text
                audio_data = example["audio"]
                text = example.get("text", "")

                # Get audio array and sample rate
                audio_array = np.array(audio_data["array"], dtype=np.float32)
                orig_sr = audio_data["sampling_rate"]

                # Resample audio
                audio_array = self.resample_audio(audio_array, orig_sr)

                # Filter by audio length
                audio_duration = len(audio_array) / self.config.sample_rate

                if (
                    audio_duration < self.config.min_audio_length
                    or audio_duration > self.config.max_audio_length
                ):
                    return None  # Will be filtered out

                # Normalize text
                normalized_text = self.normalize_haitian_text(text)
                if not normalized_text:
                    return None

                # Extract X-vector
                xvector = self.extract_xvector(audio_array)

                return {
                    "audio": audio_array,
                    "text": normalized_text,
                    "xvector": xvector,
                    "duration": audio_duration,
                    "sample_rate": self.config.sample_rate,
                }

            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                return None

        # Apply preprocessing
        logger.info("Applying preprocessing to all examples...")
        processed_examples = []

        for i, example in enumerate(self.dataset):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(self.dataset)} examples")

            processed = preprocess_example(example)
            if processed is not None:
                processed_examples.append(processed)

        logger.info(
            f"Preprocessing completed. Kept {len(processed_examples)}/{len(self.dataset)} examples"
        )

        # Convert back to Dataset
        if processed_examples:
            self.dataset = Dataset.from_list(processed_examples)
        else:
            raise ValueError("No valid examples after preprocessing")

    def split_dataset(self) -> None:
        """Split dataset into train/validation/test sets"""
        logger.info("Splitting dataset...")

        # First split: 80% train, 20% temp
        train_data, temp_data = train_test_split(
            list(range(len(self.dataset))), test_size=0.2, random_state=42
        )

        # Second split: 10% val, 10% test from the 20% temp
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=42
        )

        # Create dataset splits
        self.dataset_splits = DatasetDict(
            {
                "train": self.dataset.select(train_data),
                "validation": self.dataset.select(val_data),
                "test": self.dataset.select(test_data),
            }
        )

        logger.info(f"Dataset split sizes:")
        logger.info(f"  Train: {len(self.dataset_splits['train'])}")
        logger.info(f"  Validation: {len(self.dataset_splits['validation'])}")
        logger.info(f"  Test: {len(self.dataset_splits['test'])}")

    def pad_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        target_length = int(self.config.max_audio_length * self.config.sample_rate)

        if len(audio_array) > target_length:
            # Truncate
            return audio_array[:target_length]
        else:
            # Pad with zeros
            return np.pad(audio_array, (0, target_length - len(audio_array)))

    def load_models(self) -> None:
        """Load SpeechT5 processor, model, and vocoder"""
        logger.info("Loading SpeechT5 models...")

        # Load processor
        self.processor = SpeechT5Processor.from_pretrained(self.config.model_name)

        # Load model
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.config.model_name)
        self.model.to(DEVICE)

        # Load vocoder
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.config.vocoder_name)
        self.vocoder.to(DEVICE)

        logger.info("Models loaded successfully")


class TTSDataCollator:
    """Data collator for SpeechT5 TTS training"""

    def __init__(self, processor: SpeechT5Processor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch for training"""
        # Extract components
        texts = [item["text"] for item in batch]
        audio_arrays = [item["audio"] for item in batch]
        xvectors = [item["xvector"] for item in batch]

        # Process texts to input_ids
        text_inputs = self.processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Process audio to speech features
        speech_inputs = self.processor.feature_extractor(
            audio_arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )

        # Stack X-vectors
        xvectors_tensor = torch.stack([torch.tensor(xv) for xv in xvectors])

        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": speech_inputs.input_values,
            "speaker_embeddings": xvectors_tensor,
        }


class TTSTrainer(Trainer):
    """Custom trainer for SpeechT5 TTS"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute training loss"""
        labels = inputs.pop("labels")
        speaker_embeddings = inputs.pop("speaker_embeddings")

        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            speaker_embeddings=speaker_embeddings,
            return_dict=True,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def log_metrics(self, logs: Dict[str, float], step: int) -> None:
        """Log training metrics to HF Hub and tensorboard"""
        super().log(logs)

        # Log to HF Hub (if available)
        if hasattr(self.model, "hf_api") and self.model.hf_api is not None:
            try:
                # Create metrics file for this step
                metrics_data = {
                    "step": step,
                    "timestamp": datetime.now().isoformat(),
                    **logs,
                }

                import json

                metrics_file = f"metrics_step_{step}.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics_data, f, indent=2)

                # Upload to HF Hub
                self.model.hf_api.upload_file(
                    path_or_fileobj=metrics_file,
                    path_in_repo=f"training_logs/{metrics_file}",
                    repo_id=self.model.hf_repo_id,
                    commit_message=f"Add training metrics for step {step}",
                )

                # Clean up local file
                import os

                os.remove(metrics_file)

            except Exception as e:
                logger.warning(f"Failed to log metrics to HF Hub: {e}")


def main():
    """Main execution function"""
    logger.info("Starting Haitian Creole TTS fine-tuning")

    # Initialize configuration
    config = TTSConfig()
    tts_trainer = HaitianCreoleTTSTrainer(config)

    # Step 1: Load dataset
    tts_trainer.load_dataset()

    # Step 2: Preprocess dataset
    tts_trainer.preprocess_dataset()

    # Step 3: Split dataset
    tts_trainer.split_dataset()

    # Step 4: Load models
    tts_trainer.load_models()

    # Step 5: Setup training
    data_collator = TTSDataCollator(tts_trainer.processor)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=["tensorboard"] if TENSORBOARD_AVAILABLE else None,
    )

    # Step 6: Initialize trainer
    trainer = TTSTrainer(
        model=tts_trainer.model,
        args=training_args,
        train_dataset=tts_trainer.dataset_splits["train"],
        eval_dataset=tts_trainer.dataset_splits["validation"],
        data_collator=data_collator,
        tokenizer=tts_trainer.processor.tokenizer,
    )

    # Step 7: Train model
    logger.info("Starting training...")
    trainer.train()

    # Step 8: Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tts_trainer.processor.save_pretrained(config.output_dir)

    # Step 9: Test inference
    logger.info("Testing inference...")
    test_haitian_tts(tts_trainer, config)

    logger.info("Training completed successfully!")


def test_haitian_tts(tts_trainer: HaitianCreoleTTSTrainer, config: TTSConfig):
    """Test the fine-tuned model with sample Haitian Creole text"""

    # Sample Haitian Creole texts
    test_texts = [
        "Bonjou, kijan ou ye?",  # Hello, how are you?
        "Mwen renmen pale kreyol ayisyen.",  # I like to speak Haitian Creole
        "Nou ap aprann teknoloji nouvo yo.",  # We are learning new technologies
    ]

    logger.info("Testing with sample Haitian Creole texts...")

    for i, text in enumerate(test_texts):
        logger.info(f"Generating speech for: '{text}'")

        # Normalize text
        normalized_text = tts_trainer.normalize_haitian_text(text)

        # Tokenize
        inputs = tts_trainer.processor.tokenizer(
            normalized_text, return_tensors="pt"
        ).to(DEVICE)

        # Use a sample X-vector from test set
        if len(tts_trainer.dataset_splits["test"]) > 0:
            sample_xvector = (
                torch.tensor(tts_trainer.dataset_splits["test"][0]["xvector"])
                .unsqueeze(0)
                .to(DEVICE)
            )
        else:
            # Create a dummy X-vector if no test data
            sample_xvector = torch.randn(1, XVECTOR_DIM).to(DEVICE)

        # Generate speech
        with torch.no_grad():
            tts_trainer.model.eval()
            speech = tts_trainer.model.generate_speech(
                inputs.input_ids,
                speaker_embeddings=sample_xvector,
                vocoder=tts_trainer.vocoder,
            )

        # Save generated audio
        output_path = f"{config.output_dir}/generated_sample_{i+1}.wav"
        torchaudio.save(output_path, speech.cpu().unsqueeze(0), config.sample_rate)

        logger.info(f"Generated audio saved to: {output_path}")

    logger.info("Inference testing completed!")


if __name__ == "__main__":
    main()
