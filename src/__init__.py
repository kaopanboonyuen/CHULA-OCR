"""
CHULA-OCR: Uncertainty-Aware Robust Text Recognition
for National-Scale Land Title Deed Digitization

Author: Teerapong Panboonyuen (Kao), Ph.D.
        C2F Postdoctoral Fellow, Chulalongkorn University
Contact: teerapong.panboonyuen@gmail.com
Project: https://kaopanboonyuen.github.io/CHULA-OCR/

"""

from .model import CHULA_OCR, CHULAOCRLoss, UncertaintyGating
from .dataset import ThaiOCRTokenizer, ThaiLandTitleDeedDataset

__version__ = "1.0.0"
__author__ = "Teerapong Panboonyuen (Kao)"
__email__ = "teerapong.panboonyuen@gmail.com"

__all__ = [
    "CHULA_OCR",
    "CHULAOCRLoss",
    "UncertaintyGating",
    "ThaiOCRTokenizer",
    "ThaiLandTitleDeedDataset",
]
