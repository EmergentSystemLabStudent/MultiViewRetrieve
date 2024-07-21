from __future__ import annotations

import sys
from pathlib import Path 

import torch 
from torch import Tensor
from torch import nn
from torchvision.models.resnet import ResNet

sys.path.append(str(Path(__file__).parent.absolute()))
from backbone.resnet import ResNet18, ResNet50

class SimSiam(nn.Module) :
    
    _backbone: ResNet
    
    def __init__(
        self, 
        backbone: str="res50", 
        num_classes: int=2048, 
        prediction_dim: int=512, 
        imagenet_pretrain_path: str=None,
        freeze_partial_param: bool=False
    ) -> None:
        
        super(SimSiam, self).__init__()
        
        if backbone == "res50" :
            self._backbone = ResNet50(num_classes)
            
        elif backbone == "res18" :
            self._backbone = ResNet18(num_classes)
            
        else :
            raise f"{backbone} is not supported"
        
        hidden_dim = self._backbone.fc.weight.shape[1]
        
        self._projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, hidden_dim, bias=False),
                                   nn.BatchNorm1d(hidden_dim), 
                                   nn.ReLU(inplace=True),
                                   self._backbone.fc, 
                                   nn.BatchNorm1d(num_classes, affine=False))
        
        self._backbone.fc = nn.Identity()
        self._projector[6].bias.requires_grad = False
        
        self._predictor = nn.Sequential(
                           nn.Linear(num_classes, prediction_dim, bias=False),
                           nn.BatchNorm1d(prediction_dim),
                           nn.ReLU(inplace=True), # hidden layer
                           nn.Linear(prediction_dim, num_classes)) # output layer
        
        if imagenet_pretrain_path is not None:
            self.load_imagenet(imagenet_pretrain_path)
            
        if freeze_partial_param:
            for name, param in self.named_parameters():
                if name.startswith("_backbone") and "layer4" not in name:
                    param.requires_grad = False
                # if name.startswith("_projector") or name.startswith("_predictor"):
                #     param.requires_grad = False
                    
                print(f"{name} : grad {param.requires_grad}")
        
    def forward(self, x1:Tensor, x2:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        z1 = self._backbone(x1) # NxC
        z1 = self._projector(z1) # NxC
        
        z2 = self._backbone(x2) # NxC
        z2 = self._projector(z2) # NxC

        p1 = self._predictor(z1) # NxC
        p2 = self._predictor(z2) # NxC
        
        return p1, p2, z1.detach(), z2.detach()
    
    def get_embedding(self, image:Tensor, auto_cast: bool = True) -> Tensor:
        
        if auto_cast: 
            with torch.cuda.amp.autocast():
                embedding = self._backbone(image)
                
        else:
            embedding = self._backbone(image)
            
        return embedding.detach().cpu().numpy()
    
    #==============================================================================================
    #
    # Methods for loading pre-trained weights
    #
    #==============================================================================================
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        backbone_state_dict = {}
        projector_state_dict = {}
        predictor_state_dict = {}
                
        for key, value in state_dict.items():
            if key.startswith("_backbone"):
                backbone_state_dict[key.replace("_backbone.", "")] = value
            elif key.startswith("_projector"):
                projector_state_dict[key.replace("_projector.", "")] = value
            elif key.startswith("_predictor"):
                predictor_state_dict[key.replace("_predictor.", "")] = value
            else:
                raise ValueError(f"unexpected key: {key}")
            
        backbone_msg = self._backbone.load_state_dict(backbone_state_dict, strict=True)
        projector_msg = self._projector.load_state_dict(projector_state_dict, strict=True)
        predictor_msg = self._predictor.load_state_dict(predictor_state_dict, strict=True)
        print(f"Backbone Msg: {backbone_msg}")
        print(f"Projector Msg: {projector_msg}")
        print(f"Predictor Msg: {predictor_msg}")
        
        
    def load_imagenet(self, imagenet_path: str) -> None:
        
        checkpoint = torch.load(imagenet_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}
        projector_state_dict = {}
        predictor_state_dict = {}
        
        
        for key, value in state_dict.items():
            if key.startswith("module.encoder") and not key.startswith("module.encoder.fc"):
                encoder_state_dict[key.replace("module.encoder.", "")] = value
            elif key.startswith("module.encoder.fc"):
                projector_state_dict[key.replace("module.encoder.fc.", "")] = value
            elif key.startswith("module.predictor"):
                predictor_state_dict[key.replace("module.predictor.", "")] = value
            else:
                raise ValueError(f"unexpected key: {key}")
            
        encoder_msg = self._backbone.load_state_dict(encoder_state_dict, strict=True)
        projector_msg = self._projector.load_state_dict(projector_state_dict, strict=True)
        predictor_msg = self._predictor.load_state_dict(predictor_state_dict, strict=True)
        print(f"Encoder Msg: {encoder_msg}")
        print(f"Projector Msg: {projector_msg}")
        print(f"Predictor Msg: {predictor_msg}")

        
