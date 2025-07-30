# Haitian Creole TTS Fine-tuning with SpeechT5

This project fine-tunes a SpeechT5 Text-to-Speech model for Haitian Creole using the publicly available dataset from Hugging Face.

## Features

- **Dataset Processing**: Automatic loading and preprocessing of Haitian Creole audio-text pairs
- **Audio Normalization**: 16kHz resampling and duration filtering
- **Text Preprocessing**: Haitian Creole text normalization with special character handling
- **X-vector Extraction**: Speaker characteristic embeddings for voice consistency
- **Mixed Precision Training**: Efficient GPU utilization with FP16
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Inference Testing**: Built-in testing with sample Haitian Creole phrases

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python finetune_haitian_creole_tts.py
```

The script will:
- Load the `EdManZoeTech/edman_haitian_creole_dataset_4_tts` dataset
- Preprocess audio (resample to 16kHz, filter by duration)
- Normalize Haitian Creole text
- Extract speaker X-vectors
- Split data into train/validation/test sets (80/10/10)
- Fine-tune SpeechT5 model
- Test with sample Haitian Creole phrases
- Save the fine-tuned model to `./haitian_creole_tts_model/`

### 3. Generated Outputs

After training, you'll find:
- **Model files**: `./haitian_creole_tts_model/`
- **Generated audio samples**: `generated_sample_1.wav`, `generated_sample_2.wav`, etc.
- **Training logs**: TensorBoard logs in `./haitian_creole_tts_model/tensorboard/`

## Configuration

Modify the `TTSConfig` class in the script to adjust:

```python
@dataclass
class TTSConfig:
    # Model settings
    model_name: str = "microsoft/speecht5_tts"
    vocoder_name: str = "microsoft/speecht5_hifigan"
    
    # Training parameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-5
    
    # Audio processing
    max_audio_length: float = 5.0  # seconds
    min_audio_length: float = 1.5  # seconds
```

## Sample Test Phrases

The script automatically tests with these Haitian Creole phrases:
- "Bonjou, kijan ou ye?" (Hello, how are you?)
- "Mwen renmen pale kreyol ayisyen." (I like to speak Haitian Creole)
- "Nou ap aprann teknoloji nouvo yo." (We are learning new technologies)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training
- Internet connection for dataset download

## Performance Notes

- Training typically takes 20-30 minutes on modern GPUs
- The model supports mixed precision training for efficiency
- Checkpoint saving allows resuming interrupted training
- Memory usage is optimized for batch processing

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size`
2. **Dataset Download Issues**: Check internet connection and Hugging Face access
3. **Audio Processing Errors**: Ensure `librosa` and `torchaudio` are properly installed

## Architecture Details

The implementation uses:
- **SpeechT5ForTextToSpeech**: Pre-trained transformer-based TTS model
- **SpeechT5HifiGan**: High-quality neural vocoder
- **Custom X-vector Extraction**: Speaker embedding computation
- **Hugging Face Trainer**: Optimized training loop with automatic mixed precision

## License

This project follows the licensing terms of the underlying models and datasets used.