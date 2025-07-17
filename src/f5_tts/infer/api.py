from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, Dict
import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import io
import logging
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

# Assuming f5_tts is installed and available in the environment
# If these imports fail, ensure f5_tts is correctly installed and accessible.
try:
    from utils_infer import (
        cfg_strength,
        cross_fade_duration,
        device,
        fix_duration,
        infer_process,
        load_model,
        load_vocoder,
        mel_spec_type,
        nfe_step,
        preprocess_ref_audio_text,
        remove_silence_for_generated_wav,
        speed,
        sway_sampling_coef,
        target_rms,
    )
except ImportError as e:
    logging.error(f"Failed to import from f5_tts or utils_infer. Please ensure 'f5_tts' is installed and its modules are accessible. Error: {e}")
    # Exit or raise a more specific error if this is critical for startup
    raise RuntimeError("Required f5_tts modules could not be imported. Please check your installation.") from e


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print("startingup")
# Configuration defaults (can be overridden by environment variables or request)
# Set a default reference audio file path via an environment variable
DEFAULT_REF_AUDIO_PATH = os.environ.get("DEFAULT_REF_AUDIO_PATH", "") # Initialize empty
# Attempt to set default from f5_tts package if not set by env var
if not DEFAULT_REF_AUDIO_PATH:
    try:
        DEFAULT_REF_AUDIO_PATH = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
    except Exception as e:
        logger.warning(f"Could not set default reference audio path from f5_tts package: {e}. "
                       "Please set DEFAULT_REF_AUDIO_PATH environment variable or provide a ref_audio_file.")

if DEFAULT_REF_AUDIO_PATH and not os.path.exists(DEFAULT_REF_AUDIO_PATH):
    logger.warning(f"Default reference audio path does not exist: {DEFAULT_REF_AUDIO_PATH}. "
                   "Please ensure the path is correct or set the DEFAULT_REF_AUDIO_PATH environment variable.")


# Global variables for models and base configuration
vocoder_instance = None
ema_model_instance = None
base_config = {} # Will be loaded in lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Models are loaded here to ensure they are ready before the server starts accepting requests.
    """
    global vocoder_instance, ema_model_instance, base_config

    logger.info("Starting up F5-TTS Inference API...")

    # Load initial configuration from the TOML file
    try:
        initial_config_path = str(files("f5_tts").joinpath("infer/examples/basic/basic.toml"))
        with open(initial_config_path, "rb") as f:
            base_config = tomli.load(f)
        logger.info(f"Initial configuration loaded from: {initial_config_path}")
    except Exception as e:
        logger.error(f"FATAL ERROR: Could not load initial config from {initial_config_path}. "
                     f"Please ensure the f5_tts package is correctly installed and the file exists. Error: {e}")
        raise RuntimeError("Failed to load base configuration during startup.") from e

    try:
        # Vocoder
        vocoder_name_startup = base_config.get("vocoder_name", mel_spec_type)
        load_vocoder_from_local_startup = base_config.get("load_vocoder_from_local", False)
        vocoder_local_path_startup = ""
        if vocoder_name_startup == "vocos":
            vocoder_local_path_startup = "../checkpoints/vocos-mel-24khz"
        elif vocoder_name_startup == "bigvgan":
            vocoder_local_path_startup = "../checkpoints/bigvgan_v2_24khz_100band_256x"

        logger.info(f"Loading vocoder: {vocoder_name_startup} (local: {load_vocoder_from_local_startup})...")
        vocoder_instance = load_vocoder(
            vocoder_name=vocoder_name_startup,
            is_local=load_vocoder_from_local_startup,
            local_path=vocoder_local_path_startup,
            device=device # Using global device from utils_infer
        )
        logger.info("Vocoder loaded successfully.")

        # TTS Model
        model_startup = base_config.get("model", "F5TTS_v1_Base")
        model_cfg_path_startup = base_config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model_startup}.yaml")))
        ckpt_file_startup = base_config.get("ckpt_file", "")
        vocab_file_startup = base_config.get("vocab_file", "")

        try:
            model_cfg_startup = OmegaConf.load(model_cfg_path_startup)
        except Exception as e:
            raise RuntimeError(f"Failed to load model config from {model_cfg_path_startup}: {e}")

        model_cls_startup = get_class(f"f5_tts.model.{model_cfg_startup.model.backbone}")
        model_arc_startup = model_cfg_startup.model.arch

        repo_name_startup, ckpt_step_startup, ckpt_type_startup = "F5-TTS", 1250000, "safetensors"

        if model_startup != "F5TTS_Base":
            if vocoder_name_startup != model_cfg_startup.model.mel_spec.mel_spec_type:
                logger.warning(f"Vocoder name '{vocoder_name_startup}' does not match model config's mel_spec_type '{model_cfg_startup.model.mel_spec.mel_spec_type}'.")

        # Override for previous models based on the original script's logic
        if model_startup == "F5TTS_Base":
            if vocoder_name_startup == "vocos":
                ckpt_step_startup = 1200000
            elif vocoder_name_startup == "bigvgan":
                model_startup = "F5TTS_Base_bigvgan"
                ckpt_type_startup = "pt"
        elif model_startup == "E2TTS_Base":
            repo_name_startup = "E2-TTS"
            ckpt_step_startup = 1200000

        if not ckpt_file_startup:
            # This is where model checkpoints are downloaded.
            # Ensure good network connectivity and sufficient disk space.
            try:
                ckpt_file_startup = str(cached_path(f"hf://SWivid/{repo_name_startup}/{model_startup}/model_{ckpt_step_startup}.{ckpt_type_startup}"))
            except Exception as e:
                raise RuntimeError(f"Failed to download model checkpoint from Hugging Face. Check network or path: {e}")

        logger.info(f"Loading TTS model: {model_startup} from {ckpt_file_startup}...")
        ema_model_instance = load_model(
            model_cls_startup, model_arc_startup, ckpt_file_startup,
            mel_spec_type=vocoder_name_startup, vocab_file=vocab_file_startup, device=device
        )
        logger.info("TTS model loaded successfully.")

        yield # Application is ready to handle requests

    except Exception as e:
        logger.critical(f"FATAL ERROR during model loading or startup: {e}", exc_info=True)
        # Re-raise the exception to prevent the application from starting if models fail to load
        raise RuntimeError(f"Application failed to start due to model loading error: {e}") from e
    finally:
        logger.info("F5-TTS Inference API shutting down.")


app = FastAPI(
    title="F5-TTS Inference API",
    description="API for E2/F5 TTS with Advanced Batch Processing.",
    version="1.0.0",
    lifespan=lifespan # Use the new lifespan context manager
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to F5-TTS Inference API. Use /infer to generate audio."}

@app.post("/infer")
async def infer_tts(
    gen_text: str = Form(..., description="The text to synthesize speech from."),
    ref_audio_file: Optional[UploadFile] = File(None, description="Optional reference audio file."),
    ref_text: Optional[str] = Form("", description="The transcript/subtitle for the reference audio."),
    gen_file: Optional[UploadFile] = File(None, description="A file with text to generate. If provided, `gen_text` will be ignored."),
    output_file_name: Optional[str] = Form("infer_cli_basic.wav", description="The name of the output audio file (e.g., 'output.wav')."),
    save_chunk: bool = Form(False, description="To save each audio chunks during inference."), # Default to False, will use base_config later
    no_legacy_text: bool = Form(False, description="Not to use lossy ASCII transliterations of unicode text in saved file names."), # Default to False
    remove_silence: bool = Form(False, description="To remove long silence found in output."), # Default to False
    vocoder_name: Optional[str] = Form(mel_spec_type, description=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}"), # Default to None, will use base_config later
    target_rms: Optional[float] = Form(target_rms, description=f"Target output speech loudness normalization value, default {target_rms}"), # Default to None
    cross_fade_duration: Optional[float] = Form(cross_fade_duration, description=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}"), # Default to None
    nfe_step: Optional[int] = Form(nfe_step, description=f"The number of function evaluation (denoising steps), default {nfe_step}"), # Default to None
    cfg_strength: Optional[float] = Form(cfg_strength, description=f"Classifier-free guidance strength, default {cfg_strength}"), # Default to None
    sway_sampling_coef: Optional[float] = Form(sway_sampling_coef, description=f"Sway Sampling coefficient, default {sway_sampling_coef}"), # Default to None
    speed: Optional[float] = Form(speed, description=f"The speed of the generated audio, default {speed}"), # Default to None
    fix_duration: Optional[float] = Form(fix_duration, description=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}"), # Default to None
    device_override: Optional[str] = Form(0, description="Specify the device to run on (e.g., 'cpu', 'cuda:0'). Overrides default device."),
):
    """
    Synthesize speech from text using F5-TTS model.
    """
    # Apply defaults from base_config if not provided in the request
    _save_chunk = save_chunk if save_chunk is not None else base_config.get("save_chunk", False)
    _no_legacy_text = no_legacy_text if no_legacy_text is not None else base_config.get("no_legacy_text", False)
    _remove_silence = remove_silence if remove_silence is not None else base_config.get("remove_silence", False)
    _vocoder_name = vocoder_name if vocoder_name is not None else base_config.get("vocoder_name", mel_spec_type)
    _target_rms = target_rms if target_rms is not None else base_config.get("target_rms", target_rms)
    _cross_fade_duration = cross_fade_duration if cross_fade_duration is not None else base_config.get("cross_fade_duration", cross_fade_duration)
    _nfe_step = nfe_step if nfe_step is not None else base_config.get("nfe_step", nfe_step)
    _cfg_strength = cfg_strength if cfg_strength is not None else base_config.get("cfg_strength", cfg_strength)
    _sway_sampling_coef = sway_sampling_coef if sway_sampling_coef is not None else base_config.get("sway_sampling_coef", sway_sampling_coef)
    _speed = speed if speed is not None else base_config.get("speed", speed)
    _fix_duration = fix_duration if fix_duration is not None else base_config.get("fix_duration", fix_duration)


    if vocoder_instance is None or ema_model_instance is None:
        logger.error("Inference requested but models are not loaded. Server might not have started correctly.")
        raise HTTPException(status_code=503, detail="Models not loaded. Please wait or check server logs for startup errors.")

    current_device = device_override if device_override else device
    logger.info(f"Using device: {current_device}")

    # Handle gen_file
    if gen_file:
        try:
            gen_text_to_process = (await gen_file.read()).decode("utf-8")
            logger.info(f"Generating from provided text file: {gen_file.filename}")
        except Exception as e:
            logger.error(f"Could not read generation text file: {e}")
            raise HTTPException(status_code=400, detail=f"Could not read generation text file: {e}")
    else:
        gen_text_to_process = gen_text
        logger.info(f"Generating from provided text: '{gen_text_to_process[:50]}...'")

    # Handle ref_audio
    local_ref_audio_path = None
    if ref_audio_file:
        # Save the uploaded file temporarily
        try:
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            local_ref_audio_path = temp_dir / ref_audio_file.filename
            with open(local_ref_audio_path, "wb") as buffer:
                buffer.write(await ref_audio_file.read())
            logger.info(f"Reference audio file saved temporarily to: {local_ref_audio_path}")
        except Exception as e:
            logger.error(f"Could not save reference audio file: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save reference audio file: {e}")
    else:
        if not DEFAULT_REF_AUDIO_PATH or not os.path.exists(DEFAULT_REF_AUDIO_PATH):
             logger.error("No reference audio provided and default reference audio path is not configured or found.")
             raise HTTPException(status_code=400, detail="No reference audio provided and default reference audio path is not configured or found. Set DEFAULT_REF_AUDIO_PATH environment variable or upload a file.")
        local_ref_audio_path = DEFAULT_REF_AUDIO_PATH
        logger.info(f"Using default reference audio from: {local_ref_audio_path}")

    # Use the ref_text from the form, or default from the initial config if available and no override
    effective_ref_text = ref_text if ref_text is not None else base_config.get("ref_text", "Some call me nature, others call me mother nature.")
    logger.info(f"Effective reference text: '{effective_ref_text[:50]}...'")

    # Prepare voice configuration similar to the CLI script
    main_voice = {"ref_audio": local_ref_audio_path, "ref_text": effective_ref_text}
    voices: Dict[str, Dict] = {"main": main_voice}
    if "voices" in base_config:
        # Deep copy to avoid modifying the global base_config
        voices_from_config = base_config["voices"].copy()
        for voice_key, voice_data in voices_from_config.items():
            # Apply patches for pip pkg user as in original script
            voice_ref_audio = voice_data.get("ref_audio", "")
            if "infer/examples/" in voice_ref_audio:
                try:
                    voice_data["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))
                except Exception as e:
                    logger.warning(f"Could not resolve path for voice '{voice_key}' ref_audio '{voice_ref_audio}': {e}. Using original path.")
            voices[voice_key] = voice_data
        voices["main"] = main_voice # Ensure 'main' voice is always present and updated by input

    generated_audio_segments = []
    final_sample_rate = 24000  # Default or determined by vocoder

    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text_to_process)
    reg2 = r"\[(\w+)\]"

    try:
        for text_chunk in chunks:
            if not text_chunk.strip():
                continue
            match = re.match(reg2, text_chunk)
            voice = "main" # Default to main voice
            if match:
                voice = match[1]
            else:
                logger.info("No voice tag found for chunk, using main.")

            if voice not in voices:
                logger.warning(f"Voice {voice} not found in configured voices, using main.")
                voice = "main"

            current_text_to_gen = re.sub(reg2, "", text_chunk).strip()

            if not current_text_to_gen:
                continue

            ref_audio_path_for_chunk = voices[voice]["ref_audio"]
            ref_text_for_chunk = voices[voice]["ref_text"]

            # Preprocess reference audio and text
            preprocessed_ref_audio, preprocessed_ref_text = preprocess_ref_audio_text(
                ref_audio_path_for_chunk, ref_text_for_chunk
            )

            local_speed_for_chunk = voices[voice].get("speed", _speed)

            logger.info(f"Generating for voice: {voice}, text: '{current_text_to_gen[:50]}...' (Ref audio: {ref_audio_path_for_chunk})")
            audio_segment, final_sample_rate, _ = infer_process(
                preprocessed_ref_audio,
                preprocessed_ref_text,
                current_text_to_gen,
                ema_model_instance,
                vocoder_instance,
                mel_spec_type=_vocoder_name,
                target_rms=_target_rms,
                cross_fade_duration=_cross_fade_duration,
                nfe_step=_nfe_step,
                cfg_strength=_cfg_strength,
                sway_sampling_coef=_sway_sampling_coef,
                speed=local_speed_for_chunk,
                fix_duration=_fix_duration,
                device=current_device,
            )
            generated_audio_segments.append(audio_segment)

            if _save_chunk:
                # This part needs a proper way to save chunks, potentially to a temporary directory
                # and then either return them or clean them up. For simplicity, we'll skip
                # direct chunk saving in the API response for now, focusing on the final audio.
                # If needed, this would require zipping multiple files or returning a list of audio streams.
                logger.info("Chunk saving is enabled but not fully implemented for API response. Only final audio will be returned.")
                pass

        if not generated_audio_segments:
            logger.error("No audio segments were generated. Check input text and voice configurations.")
            raise HTTPException(status_code=400, detail="No audio segments were generated. Check your input text.")

        final_wave = np.concatenate(generated_audio_segments)
        logger.info(f"Concatenated {len(generated_audio_segments)} audio segments.")

        # Remove silence if requested
        if _remove_silence:
            temp_output_dir = Path("temp_outputs")
            temp_output_dir.mkdir(exist_ok=True)
            temp_wav_path = temp_output_dir / f"temp_{datetime.now().strftime(r'%Y%m%d_%H%M%S_%f')}.wav"
            sf.write(temp_wav_path, final_wave, final_sample_rate)
            logger.info(f"Temporary WAV saved for silence removal: {temp_wav_path}")
            remove_silence_for_generated_wav(str(temp_wav_path))
            final_wave, final_sample_rate = sf.read(temp_wav_path)
            os.remove(temp_wav_path) # Clean up temp file
            logger.info("Silence removed from generated audio.")

        # Prepare audio for streaming response
        output_buffer = io.BytesIO()
        sf.write(output_buffer, final_wave, final_sample_rate, format='WAV')
        output_buffer.seek(0)

        media_type = "audio/wav"
        filename = output_file_name if output_file_name else f"generated_audio_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
        if _no_legacy_text:
            filename = unidecode(filename) # Apply unidecode if requested

        logger.info(f"Returning generated audio file: {filename}")
        return StreamingResponse(output_buffer, media_type=media_type,
                                 headers={"Content-Disposition": f"attachment; filename={filename}"})

    except Exception as e:
        logger.exception(f"Error during inference: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error during audio generation: {e}")
    finally:
        # Clean up temporary reference audio file if it was uploaded
        if local_ref_audio_path and local_ref_audio_path != Path(DEFAULT_REF_AUDIO_PATH) and local_ref_audio_path.exists():
            try:
                os.remove(local_ref_audio_path)
                logger.info(f"Cleaned up temporary reference audio file: {local_ref_audio_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary reference audio file {local_ref_audio_path}: {e}")

