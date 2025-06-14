import time
import torch
import numpy as np
from modules import utils
import modules.evaluation as evaluate
from modules.box_utils import decode
from modules.utils import get_individual_labels
import torch.utils.data as data_utils
from data import custum_collate
from sklearn.metrics import f1_score, classification_report
from modules.utils import get_individual_labels

logger = utils.get_logger(__name__)

def val(args, net, val_dataset):
    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custum_collate)
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(args.EVAL_EPOCHS[0])
    logger.info('Loaded model from :: '+args.MODEL_PATH)
    net.load_state_dict(torch.load(args.MODEL_PATH))
    mAP, ap_all, ap_strs = validate(args, net,  val_data_loader, val_dataset, args.EVAL_EPOCHS[0])
    label_types = args.label_types + ['ego_action']
    all_classes = args.all_classes + [args.ego_classes]
    for nlt in range(args.num_label_types+1):
        for ap_str in ap_strs[nlt]:
            logger.info(ap_str)
        ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
        logger.info(ptr_str)

def validate(args, net, val_data_loader, val_dataset, iteration_num):
    iou_thresh = args.IOU_THRESH
    num_samples = len(val_dataset)
    logger.info('Validating at ' + str(iteration_num) + ' number of samples:: ' + str(num_samples))

    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()

    ego_pds = []
    ego_gts = []

    det_boxes = []
    gt_boxes_all = []

    concept_preds_all = []
    concept_labels_all = []

    for nlt in range(args.num_label_types):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        gt_boxes_all.append([])

    net.eval()
    with torch.no_grad():
        # Added concept_labels to the validation loop
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh, concept_labels) in enumerate(val_data_loader):
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            images = images.cuda(0, non_blocking=True)

            # Added concept_labels to the validation loop
            concept_labels = concept_labels.cuda(0, non_blocking=True)

            outputs = net(images)

            if isinstance(outputs, tuple):
                # Added handling for concept predictions
                if len(outputs) == 5:
                    decoded_boxes, confidence, ego_preds, concept_probs, _ = outputs
                    concept_preds_sigmoid = activation(concept_probs)
                    concept_preds_all.append(concept_preds_sigmoid.detach().cpu().numpy())
                    concept_labels_all.append(concept_labels.detach().cpu().numpy())
                elif len(outputs) == 4:
                    decoded_boxes, confidence, ego_preds, concept_probs = outputs
                    concept_preds_sigmoid = activation(concept_probs)
                    concept_preds_all.append(concept_preds_sigmoid.detach().cpu().numpy())
                    concept_labels_all.append(concept_labels.detach().cpu().numpy())
                elif len(outputs) == 3:
                    decoded_boxes, confidence, ego_preds = outputs
                else:
                    raise ValueError(f"[validate] Numero inatteso di output dal modello: {len(outputs)}")
            else:
                raise TypeError("[validate] L'output del modello non è una tupla!")

            ego_preds = activation(ego_preds).cpu().numpy()
            ego_labels = ego_labels.numpy()
            confidence = activation(confidence)

            if print_time and val_itr % val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                logger.info('Forward Time {:0.3f}'.format(tf - t1))

            seq_len = gt_targets.size(1)
            for b in range(batch_size):
                for s in range(seq_len):
                    if args.DATASET == 'ava' and batch_counts[b, s] < 1:
                        continue

                    if ego_labels[b, s] > -1:
                        ego_pds.append(ego_preds[b, s, :])
                        ego_gts.append(ego_labels[b, s])

                    width, height = wh[b][0], wh[b][1]
                    gt_boxes_batch = gt_boxes[b, s, :batch_counts[b, s], :].numpy()
                    gt_labels_batch = gt_targets[b, s, :batch_counts[b, s]].numpy()
                    decoded_boxes_frame = decoded_boxes[b, s].clone()

                    cc = 0
                    for nlt in range(args.num_label_types):
                        num_c = args.num_classes_list[nlt]
                        tgt_labels = gt_labels_batch[:, cc:cc+num_c]
                        frame_gt = get_individual_labels(gt_boxes_batch, tgt_labels)
                        gt_boxes_all[nlt].append(frame_gt)

                        for cl_ind in range(num_c):
                            scores = confidence[b, s, :, cc].clone().squeeze()
                            cc += 1
                            cls_dets = utils.filter_detections(args, scores, decoded_boxes_frame)
                            det_boxes[nlt][cl_ind].append(cls_dets)
                count += 1

            if print_time and val_itr % val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('detections done: {:d}/{:d} time taken {:0.3f}'.format(count, num_samples, te - ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
                logger.info('NMS stuff Time {:0.3f}'.format(te - tf))

    logger.info('Evaluating detections for epoch number ' + str(iteration_num))
    mAP, ap_all, ap_strs = evaluate.evaluate(gt_boxes_all, det_boxes, args.all_classes, iou_thresh=iou_thresh)
    mAP_ego, ap_all_ego, ap_strs_ego = evaluate.evaluate_ego(np.asarray(ego_gts), np.asarray(ego_pds), args.ego_classes)

    # === CEM Final Evaluation ===
    if concept_preds_all and concept_labels_all:
        concept_preds_all = np.concatenate(concept_preds_all, axis=0).reshape(-1, concept_preds_all[0].shape[-1])
        concept_labels_all = np.concatenate(concept_labels_all, axis=0).reshape(-1, concept_preds_all.shape[-1])

        concept_preds_bin = (concept_preds_all > 0.5).astype(int)
        concept_labels_bin = concept_labels_all.astype(int)

        cem_accuracy = (concept_preds_bin == concept_labels_bin).mean()
        cem_f1_micro = f1_score(concept_labels_bin, concept_preds_bin, average='micro')
        cem_f1_macro = f1_score(concept_labels_bin, concept_preds_bin, average='macro')

        logger.info(f'[CEM] Concept Prediction Accuracy: {cem_accuracy:.4f}')
        logger.info(f'[CEM] Concept F1 Micro: {cem_f1_micro:.4f}')
        logger.info(f'[CEM] Concept F1 Macro: {cem_f1_macro:.4f}')

        # === Class-wise performance
        try:
            report = classification_report(
                concept_labels_bin, concept_preds_bin,
                target_names=args.triplet_labels if hasattr(args, 'triplet_labels') else None,
                zero_division=0
            )
            logger.info(f'[CEM] Classification Report:\n{report}')
        except Exception as e:
            logger.warning(f'[CEM] Impossibile stampare classification_report: {e}')

        # === Error Analysis: concetti peggiori
        error_counts = (concept_preds_bin != concept_labels_bin).sum(axis=0)
        concept_counts = concept_labels_bin.sum(axis=0)
        top_errors = np.argsort(error_counts)[-10:]

        logger.info("[CEM] Top 10 concetti più difficili (errori/frequenza):")
        for i in top_errors:
            cname = args.triplet_labels[i] if hasattr(args, 'triplet_labels') else f"Concept {i}"
            logger.info(f"{cname}: errors={error_counts[i]}, freq={concept_counts[i]}")

    return mAP + [mAP_ego], ap_all + [ap_all_ego], ap_strs + [ap_strs_ego]

