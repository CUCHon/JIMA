import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.entity_predictor import EntityPredictor, compute_entity_loss


class R2GenMultiTaskModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenMultiTaskModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.idx2token)
        
        # Visual feature extractor
        self.visual_extractor = VisualExtractor(args)
        
        # Report generation decoder
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        # Entity prediction head
        self.entity_predictor = EntityPredictor(
            visual_feat_size=args.d_vf,  # Should match the output dimension of VisualExtractor
            vocab_size=self.vocab_size,
            hidden_size=args.d_model,
            dropout=args.dropout,
            dataset_name=args.dataset_name
        )
        
        # Set forward method based on dataset
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, entity_targets=None, mode='train', task=None):
        # Extract visual features from the two images
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        
        # Concatenate features from both images
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        # Determine the output based on mode and task
        if mode == 'train':
            # Perform specific task or both tasks
            if task == 'entity':
                # Only perform entity prediction
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            elif task == 'report':
                # Only generate report
                output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
                return output
            else:
                raise ValueError(f"Did not specify task")
                
        elif mode == 'sample':
            # Sampling mode (for evaluation or inference)
            if task == 'entity':
                # Use concatenated features for both images
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            else:
                # Default to report generation for sampling
                output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
                return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_mimic_cxr(self, images, targets=None, entity_targets=None, mode='train', task=None):
        # Extract features from a single image
        att_feats, fc_feats = self.visual_extractor(images)
        
        # Determine the output based on mode and task
        if mode == 'train':
            # Perform specific task or both tasks
            if task == 'entity':
                # Only perform entity prediction
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            elif task == 'report':
                # Only generate report
                output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
                return output
            else:
                # Perform both tasks (task is None)
                entity_logits = self.entity_predictor(fc_feats)
                output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
                return output, entity_logits
                
        elif mode == 'sample':
            # Sampling mode (for evaluation or inference)
            if task == 'entity':
                entity_logits = self.entity_predictor(fc_feats)
                return entity_logits
            else:
                # Default to report generation for sampling
                output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
                return output
        else:
            raise ValueError(f"Unsupported mode: {mode}")