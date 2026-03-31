from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_nas3r import EncoderNAS3RCfg, EncoderNAS3R

ENCODERS = {
   
    "nas3r": (EncoderNAS3R, None),
}

EncoderCfg = EncoderNAS3RCfg 

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
