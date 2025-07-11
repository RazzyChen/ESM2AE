import torch
import torch.nn.functional as F


def Simclr_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # cosine similarity matrix: (2N, 2N)
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # positive samples: z1 vs z2
    pos_sim = torch.diag(sim_matrix, diagonal=N)  # shape (N,)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # (2N,)

    # mask self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    logits = sim_matrix / temperature
    labels = torch.arange(2 * N, device=z.device)
    labels = (labels + N) % (2 * N)  # positive pair index

    return F.cross_entropy(logits, labels)
