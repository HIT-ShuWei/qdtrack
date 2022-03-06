import torch
import torch.nn.functional as F


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    assert method in ['dot_product', 'cosine']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())

def cal_weighted_similarity(key_embeds,
                            ref_embeds,
                            key_scores,
                            ref_scores,
                            method='dot_product',
                            temperature=-1):
    assert method in ['dot_product', 'cosine']
    num_regions = key_embeds.size(1)

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=2)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=2)
        dists = []
        weights = []
        
        for region_ind in range(num_regions):
            dist = torch.mm(key_embeds[:,region_ind,:], ref_embeds[:,region_ind,:].t())
            weight = torch.mul(key_scores[:,region_ind,:], ref_scores[:, region_ind,:].t())
            weights.append(weight)
            dists.append(torch.mul(dist, weight))

        # concat from list
        dists = torch.cat(dists).view(num_regions, key_embeds.size(0), ref_embeds.size(0))
        weights = torch.cat(weights).view(num_regions, key_scores.size(0), ref_scores.size(0))
        
        # sum all regions
        dists = torch.sum(dists, dim=0)
        weights = torch.sum(dist, dim=0)
        
        # cal result
        res = torch.mul(dist, weights.pow(-1))
        return res
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            dists = []
            weights = []
            for region_ind in range(num_regions):
                dist = torch.mm(key_embeds[:,region_ind,:], ref_embeds[:,region_ind,:].t())
                weight = torch.mul(key_scores[:,region_ind,:], ref_scores[:, region_ind,:].t())
                weights.append(weight)
                dists.append(torch.mul(dist, weight))
            
            # concat from list
            dists = torch.cat(dists).view(num_regions, key_embeds.size(0), ref_embeds.size(0))
            weights = torch.cat(weights).view(num_regions, key_scores.size(0), ref_scores.size(0))
            
            # sum all regions
            dists = torch.sum(dists, dim=0)
            weights = torch.sum(dist, dim=0)
            
            # cal result
            res = torch.mul(dist, weights.pow(-1))
            return res