import torch
import torch.nn.functional as F
import numpy as np


def extract_features_per_sample(model, dataloader, device="cpu"):
    model.eval()

    features_list = []
    labels_list = []
    memberships_list = []

    for _, x, y, m in dataloader:  # Assuming dataloader yields (id, image, label, membership)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        y_tensor = torch.tensor([y]).to(device)

        # x.requires_grad = True
        # if hasattr(model, 'forward_features'):
        #     features = model.forward_features(x)  # shape: [1, feature_dim]
        #     logits = model.fc(features)           # shape: [1, num_classes]
        #     features = features.squeeze(0)        # [feature_dim]
        # else:
        #     # If model does not have this, you need to modify it.
        #     # For example, if using torchvision resnet:
        #     features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(x))))))))
        #     features = torch.flatten(features, 1).squeeze(0)
        #     logits = model.fc(features.unsqueeze(0)).squeeze(0)

        # Enable gradient computation for gradient-based features        
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        logit_vec = logits[0]
        prob_vec = probs[0]

        # # Loss-based features
        # nll_loss = F.cross_entropy(logits, y_tensor, reduction='none').item()

        # Gradient norm w.r.t. input (or logits)
        # model.zero_grad()
        # loss = F.cross_entropy(logits, y_tensor)
        # loss.backward()
        # grad_norm = x.grad.norm().item()

        # Logit features
        logit_max = logit_vec.max().item()
        logit_std = logit_vec.std().item()
        logit_mean = logit_vec.mean().item()
        logit_max_minus_mean = logit_max - logit_mean

        # Probabilities
        prediction_confidence = prob_vec.max().item()
        eps = 1e-12
        prediction_entropy = (-prob_vec * (prob_vec + eps).log()).sum().item()
        entropy_normalized = prediction_entropy / np.log(prob_vec.shape[0])
        entropy_over_logit_std = prediction_entropy / (logit_std + eps)

        predicted_class = prob_vec.argmax().item()
        pred_prob = prob_vec[predicted_class]
        negative_log_pred_prob = -pred_prob.log().item()
        correct_prediction = int(predicted_class == y)

        # Top-k probs and logits
        top_probs = prob_vec.topk(2).values
        top1_prob = top_probs[0].item()
        top2_prob = top_probs[1].item()
        top1_minus_top2 = top1_prob - top2_prob

        top_logits = logit_vec.topk(2).values
        margin_confidence = (top_logits[0] - top_logits[1]).item()

        # True class rank
        true_class_rank = (prob_vec > prob_vec[y]).sum().item()

        # Cosine similarity (requires access to final FC weights if available)
        # cosine_similarity = np.nan
        # if hasattr(model, 'fc') and hasattr(model.fc, 'weight'):
        #     weight = model.fc.weight.data  # [num_classes, feature_dim]
        #     cosine_similarity = F.cosine_similarity(features.unsqueeze(0), weight[y].unsqueeze(0)).item()

        # Final feature vector
        features = [
            logit_std,
            prediction_confidence,
            prediction_entropy,
            correct_prediction,
            top1_minus_top2,
            margin_confidence,
            true_class_rank,
            # nll_loss,
            # grad_norm,
            # cosine_similarity
        ]
        features_list.append(features)
        labels_list.append(y)
        memberships_list.append(m)

    return np.array(features_list), np.array(labels_list), np.array(memberships_list)

