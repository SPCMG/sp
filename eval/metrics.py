import torch

def compute_car(model, motion_embs, text_embs, shuffled_text_embs, device="cuda"):
    num_motions = motion_embs.shape[0]
    num_correct = 0

    for i in range(num_motions):
        original_similarity = model.compute_similarity(motion_embs[i], text_embs[i])
        shuffled_similarity = model.compute_similarity(motion_embs[i], shuffled_text_embs[i])

        if original_similarity > shuffled_similarity:
            num_correct += 1

    car = num_correct / num_motions
    return car

def compute_r_precision(model, motion_embs, text_embs, device="cuda", k_values=[1, 3, 5]):
    num_motions = motion_embs.shape[0]
    r_precision_values = {}

    for k in k_values:
        num_correct = 0
        for i in range(num_motions):
            similarities = []
            for j in range(num_motions):
                similarity = model.compute_similarity(motion_embs[i], text_embs[j])
                similarities.append((similarity, j))

            similarities.sort(reverse=True)
            retrieved_indices = [idx for _, idx in similarities[:k]]

            if i in retrieved_indices:
                num_correct += 1

        r_precision = num_correct / num_motions
        r_precision_values[f"r_precision_top_{k}"] = r_precision

    return r_precision_values