import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca

# --- Added: import clip to process coarse text features ---
from clip import clip
# ----------------------------------------------------------

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
    # key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    # key_set = [ "attr_acc", "obj_acc",'attr_acc_open','obj_acc_open',"ub_seen","ub_unseen","ub_all","ub_open_seen","ub_open_unseen","ub_open_all","best_seen", "best_unseen", "best_hm","AUC"]
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        # if key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


# ========conditional train=
def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
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

    # --- Added: Pre-extract coarse text Euclidean features using frozen CLIP to save VRAM and time ---
    print("Pre-extracting coarse text Euclidean features using frozen CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_clip, _ = clip.load(config.backbone, device=device)
    frozen_clip.eval()
    
    coarse_v_feats_dict = {}
    coarse_o_feats_dict = {}
    
    with torch.no_grad():
        # Process unique coarse verbs
        for cv in set(train_dataset.verb_hierarchy.values()):
            tokens = clip.tokenize(f"an action of {cv}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            coarse_v_feats_dict[cv] = feat
            
        # Process unique coarse objects
        for co in set(train_dataset.obj_hierarchy.values()):
            tokens = clip.tokenize(f"a photo of a {co}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            coarse_o_feats_dict[co] = feat
            
    # Free memory
    del frozen_clip
    print("Pre-extraction complete.")
    # -----------------------------------------------------------------------------------------------

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
        
        # --- Added: track entailment and alignment losses ---
        epoch_ent_losses = []
        epoch_ali_losses = []
        # --------------------------------------------------

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        
        actual_model = model.module if hasattr(model, 'module') else model

        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            
            # Extract coarse texts for current batch
            batch_coarse_verb = batch[4]
            batch_coarse_obj = batch[5]

            with torch.cuda.amp.autocast(enabled=True):
                p_v_hyp, p_o_hyp, p_pair_v, p_pair_o, vid_feat, o_feat, v_feat, p_v_con_o, p_o_con_v, \
                v_feat_hyp, o_feat_hyp, verb_text_hyp, obj_text_hyp, _curv = model(batch_img)
                
                # component loss (using hyperbolic distance logits)
                loss_verb = Loss_fn(p_v_hyp * config.cosine_scale, batch_verb)
                loss_obj = Loss_fn(p_o_hyp * config.cosine_scale, batch_obj)
                
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                pred_com_train = (p_pair_v + p_pair_o)[:, train_v_inds, train_o_inds]
                loss_com = Loss_fn(pred_com_train * config.cosine_scale, batch_target)
                
                # --- Added: Coarse concept hyperbolic mapping and Hierarchical Losses ---
                
                # 1. Lookup and process coarse features through C2C linear projection to match dimensions
                coarse_v_raw = torch.cat([coarse_v_feats_dict[cv] for cv in batch_coarse_verb], dim=0).cuda()
                coarse_o_raw = torch.cat([coarse_o_feats_dict[co] for co in batch_coarse_obj], dim=0).cuda()
                
                coarse_v_eucl = actual_model.c2c_text_v(coarse_v_raw)
                coarse_o_eucl = actual_model.c2c_text_o(coarse_o_raw)
                
                # 2. Map coarse Euclidean features to Hyperbolic space
                v_alpha = actual_model.textual_alpha.exp()
                import models.vlm_models.lorentz as L
                coarse_v_hyp = L.exp_map0(coarse_v_eucl * v_alpha, _curv)
                coarse_o_hyp = L.exp_map0(coarse_o_eucl * v_alpha, _curv)
                
                # 3. Calculate 4 sets of Hierarchical Entailment Losses
                # Select the fine hyperbolic features corresponding to current batch ground truths
                batch_fine_v_hyp = verb_text_hyp[batch_verb]
                batch_fine_o_hyp = obj_text_hyp[batch_obj]
                
                # 3a. Semantic Entailment (Fine Concept -> Coarse Concept)
                loss_ent_v = HierarchicalEntailmentLoss(batch_fine_v_hyp, coarse_v_hyp, _curv)
                loss_ent_o = HierarchicalEntailmentLoss(batch_fine_o_hyp, coarse_o_hyp, _curv)
                
                # 3b. Compositional Entailment (Video Features -> Fine Concept)
                loss_ent_vid_v = HierarchicalEntailmentLoss(v_feat_hyp, batch_fine_v_hyp, _curv)
                loss_ent_vid_o = HierarchicalEntailmentLoss(o_feat_hyp, batch_fine_o_hyp, _curv)
                
                loss_entailment = loss_ent_v + loss_ent_o + loss_ent_vid_v + loss_ent_vid_o
                
                # 4. Calculate Discriminative Alignment Loss (Visual -> Coarse Text)
                loss_align_v = HyperbolicAlignmentLoss(v_feat_hyp, coarse_v_hyp, _curv)
                loss_align_o = HyperbolicAlignmentLoss(o_feat_hyp, coarse_o_hyp, _curv)
                
                loss_alignment = loss_align_v + loss_align_o
                
                # 5. Total Loss Fusion (Using safe preset hyper-parameter weight 0.1 for new terms)
                loss = loss_com + 0.2 * (loss_verb + loss_obj) + 0.1 * loss_entailment + 0.1 * loss_alignment
                # ------------------------------------------------------------------------
                
                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())
            
            # --- Added: Track new losses ---
            epoch_ent_losses.append(loss_entailment.item())
            epoch_ali_losses.append(loss_alignment.item())
            # -------------------------------

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")
        
        # --- Added: Write new losses to log ---
        log_training.write(f"epoch {i + 1} entailment loss {np.mean(epoch_ent_losses)}\n")
        log_training.write(f"epoch {i + 1} alignment loss {np.mean(epoch_ali_losses)}\n")
        # --------------------------------------

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
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
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
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
