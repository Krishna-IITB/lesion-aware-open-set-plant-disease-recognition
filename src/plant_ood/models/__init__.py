"""Model components for global, textual, lesion, fusion, and open-set evidence."""

from plant_ood.models.fusion import ThreeViewGate
from plant_ood.models.lesion import LesionDecoder, lesion_aware_pool
from plant_ood.models.prototypes import PrototypeBank

__all__ = ["LesionDecoder", "PrototypeBank", "ThreeViewGate", "lesion_aware_pool"]
