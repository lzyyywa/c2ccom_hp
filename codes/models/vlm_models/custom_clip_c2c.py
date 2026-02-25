import torch
import torch.nn as nn
import math

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models.vlm_models.text_learner import get_text_learner
from models.vlm_models import lorentz as L  

import torch.nn.functional as F
from einops import rearrange

_tokenizer = _Tokenizer()

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Linear(incoming, outgoing, bias=bias))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Linear(incoming, out_dim, bias=bias))

        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))

            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))

        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))

        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):  
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj
        self.num_frames = cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)
        return out


class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        self.curv = nn.Parameter(torch.tensor(1.0).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}
        self.visual_alpha = nn.Parameter(torch.tensor(cfg.emb_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(cfg.emb_dim**-0.5).log())

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_OE2 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)
        self.c2c_VE2 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers, dropout=False, norm=True, layers=layers)

        self.c2c_f_v_e_o_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)
        self.c2c_f_o_e_v_com = nn.Linear(2 * cfg.emb_dim, cfg.emb_dim, bias=True)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.hyp_proj_v_vis = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.hyp_proj_o_vis = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.hyp_proj_v_text = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.hyp_proj_o_text = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        
        nn.init.eye_(self.hyp_proj_v_vis.weight)
        nn.init.zeros_(self.hyp_proj_v_vis.bias)
        nn.init.eye_(self.hyp_proj_o_vis.weight)
        nn.init.zeros_(self.hyp_proj_o_vis.bias)
        nn.init.eye_(self.hyp_proj_v_text.weight)
        nn.init.zeros_(self.hyp_proj_v_text.bias)
        nn.init.eye_(self.hyp_proj_o_text.weight)
        nn.init.zeros_(self.hyp_proj_o_text.bias)

    def forward(self, video, pairs=None):
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        video_features = self.video_encoder(video)

        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        o_feat_normed = torch.nan_to_num(F.normalize(o_feat, dim=1, eps=1e-5))
        v_feat_normed = torch.nan_to_num(F.normalize(v_feat, dim=1, eps=1e-5))

        verb_text_features_norm = verb_text_features / (verb_text_features.norm(dim=-1, keepdim=True) + 1e-5)
        verb_text_features_norm = torch.nan_to_num(verb_text_features_norm)
        
        obj_text_features_norm = obj_text_features / (obj_text_features.norm(dim=-1, keepdim=True) + 1e-5)
        obj_text_features_norm = torch.nan_to_num(obj_text_features_norm)

        verb_logits_eucl = torch.nan_to_num(v_feat_normed @ verb_text_features_norm.t())
        obj_logits_eucl = torch.nan_to_num(o_feat_normed @ obj_text_features_norm.t())

        verb_logits_prob = verb_logits_eucl * 0.5 + 0.5
        obj_logits_prob = obj_logits_eucl * 0.5 + 0.5

        b = video_features.shape[0]
        c = verb_text_features.shape[-1]
        n_v = verb_logits_prob.shape[-1]
        n_o = obj_logits_prob.shape[-1]

        # [最后一块防弹衣]: 给推断区的原始特征加上保护
        o_feat_c = torch.nan_to_num(self.c2c_OE2(video_features.mean(dim=-1)))
        v_feat_c = torch.nan_to_num(self.c2c_VE2(video_features).mean(dim=-1))

        p_v_con_o, p_o_con_v = self.condition_module(v_feat_c, o_feat_c, verb_text_features, obj_text_features, n_o, b, c, n_v)
        p_pair_o = p_v_con_o * obj_logits_prob.unsqueeze(1)  
        p_pair_v = p_o_con_v * verb_logits_prob.unsqueeze(-1)  

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        with torch.autocast(video_features.device.type, dtype=torch.float32):
            v_feat_tangent = self.hyp_proj_v_vis(v_feat_normed.float())
            o_feat_tangent = self.hyp_proj_o_vis(o_feat_normed.float())
            verb_text_tangent = self.hyp_proj_v_text(verb_text_features_norm.float())
            obj_text_tangent = self.hyp_proj_o_text(obj_text_features_norm.float())

            v_feat_hyp = L.exp_map0(v_feat_tangent * self.visual_alpha.exp().float(), _curv)
            o_feat_hyp = L.exp_map0(o_feat_tangent * self.visual_alpha.exp().float(), _curv)
            verb_text_hyp = L.exp_map0(verb_text_tangent * self.textual_alpha.exp().float(), _curv)
            obj_text_hyp = L.exp_map0(obj_text_tangent * self.textual_alpha.exp().float(), _curv)

            verb_logits_hyp = -L.pairwise_dist(v_feat_hyp, verb_text_hyp, _curv)
            obj_logits_hyp = -L.pairwise_dist(o_feat_hyp, obj_text_hyp, _curv)

        if self.training:
            return verb_logits_eucl, obj_logits_eucl, verb_logits_hyp, obj_logits_hyp, p_pair_v, p_pair_o, video_features, o_feat, v_feat, p_v_con_o, p_o_con_v, v_feat_hyp, o_feat_hyp, verb_text_hyp, obj_text_hyp, _curv
        else:
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            com_logits = p_pair_o[:, verb_idx, obj_idx] + p_pair_v[:, verb_idx, obj_idx]
            return com_logits

    def condition_module(self, v_feat_c, o_feat_c, v_emb, o_emb, n_o, b, c, n_v):
        # [彻底封死欧式推理区溢出]
        v_emb_normed = torch.nan_to_num(F.normalize(v_emb, dim=1, eps=1e-5))
        o_emb_normed = torch.nan_to_num(F.normalize(o_emb, dim=1, eps=1e-5))

        f_v_e_o = self.c2c_f_v_e_o_com(
            torch.cat([v_feat_c.unsqueeze(1).repeat(1, n_o, 1), o_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2)) 
        f_v_e_o_norm = torch.nan_to_num(F.normalize(f_v_e_o, dim=-1, eps=1e-5)).view(b, n_o, c)

        f_o_e_v = self.c2c_f_o_e_v_com(
            torch.cat([o_feat_c.unsqueeze(1).repeat(1, n_v, 1), v_emb.unsqueeze(0).repeat(b, 1, 1)], dim=-1).view(-1, c * 2)) 
        f_o_e_v_norm = torch.nan_to_num(F.normalize(f_o_e_v, dim=-1, eps=1e-5)).view(b, n_v, c)

        p_v_con_o = torch.einsum('bnc,mc->bnm', f_v_e_o_norm, v_emb_normed) * 0.5 + 0.5  
        p_v_con_o = p_v_con_o.permute(0, 2, 1)  
        p_o_con_v = torch.einsum('bnc,mc->bnm', f_o_e_v_norm, o_emb_normed) * 0.5 + 0.5  
        
        return torch.nan_to_num(p_v_con_o), torch.nan_to_num(p_o_con_v)


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def build_model(train_dataset,cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    print("Building custom CLIP")
    model = CustomCLIP(cfg, train_dataset, clip_model)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop' and 'prompt_vectors' in name:
                    param.requires_grad_(True)
                elif cfg.learn_input_method == 'csp' and ('obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name):
                    param.requires_grad_(True)
                elif cfg.learn_input_method == 'spm' and ('prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name):
                    param.requires_grad_(True)
        elif 'video_encoder' in name and ('temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name):
            param.requires_grad = True
        elif 'c2c' in name or 'curv' in name or 'alpha' in name or 'hyp_proj' in name:
            param.requires_grad = True
    return model