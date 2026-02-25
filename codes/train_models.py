import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
# [HYPERBOLIC] 引入纯净损失类
from loss import EntailmentConeLoss, HyperbolicHardNegativeAlignmentLoss
from loss import KLLoss
from torch.nn.modules.loss import CrossEntropyLoss
import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca
from clip import clip

def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    print("Pre-extracting coarse text Euclidean features using frozen CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_clip, _ = clip.load(config.backbone, device=device)
    frozen_clip.eval()
    
    coarse_v_feats_dict = {}
    coarse_o_feats_dict = {}
    
    with torch.no_grad():
        for cv in set(train_dataset.verb_hierarchy.values()):
            tokens = clip.tokenize(f"an action of {cv}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            coarse_v_feats_dict[cv] = feat
            
        for co in set(train_dataset.obj_hierarchy.values()):
            tokens = clip.tokenize(f"a photo of a {co}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            coarse_o_feats_dict[co] = feat
            
    del frozen_clip
    print("Pre-extraction complete.")

    model.train()
    best_loss = 1e5
    best_metric = 0
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []
        
        epoch_ent_losses = []
        epoch_ali_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        
        actual_model = model.module if hasattr(model, 'module') else model

        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            
            batch_coarse_verb = batch[4]
            batch_coarse_obj = batch[5]

            # -------------------------------------------------------------------------
            # 1. 允许欧式空间的计算在 FP16 (AMP) 下进行
            # -------------------------------------------------------------------------
            with torch.cuda.amp.autocast(enabled=True):
                # [关键修复] 接收前向传播返回的双轨参数：_eucl 保基线，_hyp 做正则
                verb_logits_eucl, obj_logits_eucl, verb_logits_hyp, obj_logits_hyp, p_pair_v, p_pair_o, vid_feat, o_feat, v_feat, p_v_con_o, p_o_con_v, \
                v_feat_hyp, o_feat_hyp, verb_text_hyp, obj_text_hyp, _curv = model(batch_img)
                
                # ====== 欧式基线托底区 (保证 C2C 能力不跌) ======
                loss_verb_eucl = Loss_fn(verb_logits_eucl * config.cosine_scale, batch_verb)
                loss_obj_eucl = Loss_fn(obj_logits_eucl * config.cosine_scale, batch_obj)
                
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)
                
                w_att_obj = config.att_obj_w
                loss_eucl_base = loss_com + w_att_obj * (loss_verb_eucl + loss_obj_eucl)
                
                # ====== 提取粗粒度特征 (欧式切空间) ======
                coarse_v_raw = torch.cat([coarse_v_feats_dict[cv] for cv in batch_coarse_verb], dim=0).cuda()
                coarse_o_raw = torch.cat([coarse_o_feats_dict[co] for co in batch_coarse_obj], dim=0).cuda()
                
                coarse_v_eucl = actual_model.c2c_text_v(coarse_v_raw)
                coarse_o_eucl = actual_model.c2c_text_o(coarse_o_raw)
                
            # -------------------------------------------------------------------------
            # 2. 强制退出半精度，执行双曲增强与结构正则化 (FP32)
            # -------------------------------------------------------------------------
            with torch.cuda.amp.autocast(enabled=False):
                _curv_fp32 = _curv.float()
                v_alpha_fp32 = actual_model.textual_alpha.exp().float()
                
                # [关键修复] 粗粒度特征必须经过相同的切空间投影头，防止几何冲突
                coarse_v_tangent = actual_model.hyp_proj_v_text(coarse_v_eucl.float())
                coarse_o_tangent = actual_model.hyp_proj_o_text(coarse_o_eucl.float())
                
                import models.vlm_models.lorentz as L
                coarse_v_hyp = L.exp_map0(coarse_v_tangent * v_alpha_fp32, _curv_fp32)
                coarse_o_hyp = L.exp_map0(coarse_o_tangent * v_alpha_fp32, _curv_fp32)
                
                batch_fine_v_hyp = verb_text_hyp[batch_verb].float()
                batch_fine_o_hyp = obj_text_hyp[batch_obj].float()
                v_feat_hyp_fp32 = v_feat_hyp.float()
                o_feat_hyp_fp32 = o_feat_hyp.float()
                
                # ====== 实例化双曲公式 ======
                entail_loss_fn = EntailmentConeLoss(margin=0.01)
                align_loss_fn = HyperbolicHardNegativeAlignmentLoss(margin=0.2)
                
                # 层次蕴含损失 (Entailment: 子节点 -> 父节点)
                loss_ent_v = entail_loss_fn(batch_fine_v_hyp, coarse_v_hyp, _curv_fp32)
                loss_ent_o = entail_loss_fn(batch_fine_o_hyp, coarse_o_hyp, _curv_fp32)
                loss_ent_vid_v = entail_loss_fn(v_feat_hyp_fp32, batch_fine_v_hyp, _curv_fp32)
                loss_ent_vid_o = entail_loss_fn(o_feat_hyp_fp32, batch_fine_o_hyp, _curv_fp32)
                
                loss_entailment = loss_ent_v + loss_ent_o + loss_ent_vid_v + loss_ent_vid_o
                
                # 判别对齐损失 (Alignment: 拉开难负样本距离)
                loss_align_v = align_loss_fn(verb_logits_hyp.float(), batch_verb)
                loss_align_o = align_loss_fn(obj_logits_hyp.float(), batch_obj)
                
                loss_alignment = loss_align_v + loss_align_o
                
                # 双曲空间的交叉熵 (作为辅助分类正则化，使用安全的温度系数20.0)
                loss_verb_hyp = Loss_fn(verb_logits_hyp.float() * 20.0, batch_verb)
                loss_obj_hyp = Loss_fn(obj_logits_hyp.float() * 20.0, batch_obj)
                
                # ====== 终极融合：欧式基线 + 双曲正则 ======
                w_align = config.lambda_align
                w_entail = config.lambda_entail

                # 双曲辅助损失 = 双曲分类 + 层次蕴含 + 难负样本排斥
                loss_hyp_aux = w_att_obj * (loss_verb_hyp + loss_obj_hyp) + w_entail * loss_entailment + w_align * loss_alignment
                
                # 混合流形总损失
                loss = loss_eucl_base + loss_hyp_aux
                
                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                
                # 强制梯度截断，防止双曲空间特有的梯度突刺撕裂参数
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            # 记录欧式 loss 以监控基线表现
            epoch_vv_losses.append(loss_verb_eucl.item())
            epoch_oo_losses.append(loss_obj_eucl.item())
            
            epoch_ent_losses.append(loss_entailment.item())
            epoch_ali_losses.append(loss_alignment.item())

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")
        
        log_training.write(f"epoch {i + 1} entailment loss {np.mean(epoch_ent_losses)}\n")
        log_training.write(f"epoch {i + 1} alignment loss {np.mean(epoch_ali_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))