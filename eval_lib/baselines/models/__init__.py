"""
统一模型接口 - 导入所有模型

约定：
- 每个模型提供统一接口（BaseModel 子类）
- 每个模型提供 create_* 工厂函数
"""
from .base_model import BaseModel

from .cellblast_model import CellBLASTModel, create_cellblast_model
from .gmvae_model import GMVAEModel, create_gmvae_model
from .scalex_model import SCALEXModel, create_scalex_model
from .scdiffusion_model import scDiffusionModel, create_scdiffusion_model
from .sivae_model import siVAEModel, create_sivae_model

from .clear_model import CLEARModel, create_clear_model
from .scdac_model import scDACModel, create_scdac_model
from .scdeepcluster_model import scDeepClusterModel, create_scdeepcluster_model
from .scdhmap_model import scDHMapModel, create_scdhmap_model
from .scgnn_model import scGNNModel, create_scgnn_model
from .scgcc_model import scGCCModel, create_scgcc_model
from .scsmd_model import scSMDModel, create_scsmd_model
from .disentanglement_vae_model import DisentanglementVAEModel, create_disentanglement_vae_model

# scVI-family (requires scvi-tools)
try:
    from .scvi_family_model import (
        SCVIModel, create_scvi_model,
        PeakVIModel, create_peakvi_model,
        PoissonVIModel, create_poissonvi_model)
    _SCVI_AVAILABLE = True
except ImportError:
    _SCVI_AVAILABLE = False

__all__ = [
    # base
    "BaseModel",
    # models
    "CellBLASTModel",
    "GMVAEModel",
    "SCALEXModel",
    "scDiffusionModel",
    "siVAEModel",
    "CLEARModel",
    "scDACModel",
    "scDeepClusterModel",
    "scDHMapModel",
    "scGNNModel",
    "scGCCModel",
    "scSMDModel",
    "DisentanglementVAEModel",
    # factories
    "create_cellblast_model",
    "create_gmvae_model",
    "create_scalex_model",
    "create_scdiffusion_model",
    "create_sivae_model",
    "create_clear_model",
    "create_scdac_model",
    "create_scdeepcluster_model",
    "create_scdhmap_model",
    "create_scgnn_model",
    "create_scgcc_model",
    "create_scsmd_model",
    "create_disentanglement_vae_model",
    # scVI-family (optional)
    "SCVIModel",
    "PeakVIModel",
    "PoissonVIModel",
    "create_scvi_model",
    "create_peakvi_model",
    "create_poissonvi_model",
]