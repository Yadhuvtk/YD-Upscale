from .loss_factory import build_loss
from .charbonnier import CharbonnierLoss
from .pixel_loss import L1Loss, MSELoss
from .edge_loss import EdgeLoss
from .perceptual_loss import PerceptualLoss
from .gan_loss import GeneratorLoss, DiscriminatorLoss
from .illustration_loss import IllustrationLoss
from .text_consistency_loss import TextConsistencyLoss
