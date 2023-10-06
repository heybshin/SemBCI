import torch
import networkx as nx
from hierarchy.relation import ClassRelations
import numpy as np


class RelationLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(RelationLoss, self).__init__()
        relations = ClassRelations(cfg, is_a=True)
        self.graph = relations.get_relation_graph().rgraph()
        self.sorted_nodes = list(nx.topological_sort(self.graph))
        cfg.MODEL.n_nodes = len(self.sorted_nodes)
        self.name_to_idx = {
            name: idx for idx, name in enumerate(self.sorted_nodes)
        }
        self.class_names = cfg.DATA.class_long
        self.class_set = set(self.class_names)

    def embed(self, labels):
        # Embed the labels into the hierarchy (mark all ancestors as present)
        embedding = np.zeros((len(labels), len(self.name_to_idx)))
        for i, label in enumerate(labels):
            embedding[i, self.name_to_idx[label]] = 1.0
            for ancestor in nx.ancestors(self.graph, label):
                embedding[i, self.name_to_idx[ancestor]] = 1.0

        return embedding

    def _get_conditional_probabilities(self, pred):
        cond_probs = {
            node_name: pred[i].clone() for node_name, i in self.name_to_idx.items()
        }
        return cond_probs

    def _calculate_unconditional_probabilities(self, cond_probs):
        uncond_probs = {}
        for node_name in self.sorted_nodes:
            uncond_prob = cond_probs[node_name]

            no_parent = 1.0
            has_parents = False
            for parent in self.graph.predecessors(node_name):
                has_parents = True
                no_parent *= 1.0 - uncond_probs[parent]

            if has_parents:
                uncond_prob *= 1.0 - no_parent

            uncond_probs[node_name] = uncond_prob

        return uncond_probs

    def _extrapolate(self, targets, preds):
        ground_truth = []
        for i, target in enumerate(targets):
            cond_probs = self._get_conditional_probabilities(preds[i])
            target = self.class_names[target]
            uncond_probs = self._calculate_unconditional_probabilities(cond_probs)
            ground_truth += [(target, uncond_probs)]

        extrapolated = []
        for target, uncond_probs in ground_truth:
            allowed_candidates = set(nx.descendants(self.graph, target))

            candidates = [
                uid
                for (uid, probability) in uncond_probs.items()
                if uid in self.class_set.intersection(allowed_candidates)
            ]

            if len(candidates) > 0:
                # add a very small amount of noise to break ties when sorting probabilities
                candidates = list(
                    sorted(
                        candidates,
                        key=lambda x: uncond_probs[x] + np.random.normal(0, 0.0001),
                        reverse=True,
                    )
                )
                extrapolated += [candidates[0]]
            else:
                extrapolated += [target]

        return extrapolated

    def deembed(self, preds):
        return [
            self._deembed(pred) for pred in preds
        ]

    def _deembed(self, pred):
        cond_probs = self._get_conditional_probabilities(pred)
        uncond_probs = self._calculate_unconditional_probabilities(cond_probs)
        sorted_probs = list(sorted(uncond_probs.items(), key=lambda x: x[1], reverse=True))
        for i, (uid, _) in enumerate(sorted_probs):
            if uid not in self.class_names:
                sorted_probs[i] = (uid, 0.0)

        total = sum([prob for _, prob in sorted_probs])
        if total > 0:
            sorted_probs = [(uid, prob / total) for uid, prob in sorted_probs]

        return list(sorted_probs)

    def forward(self, preds, targets):

        # Extrapolate ground truth
        ground_truth = self._extrapolate(targets, preds)

        # Compute loss mask
        loss_mask = np.zeros((len(ground_truth), len(self.name_to_idx)))
        for i, target in enumerate(ground_truth):
            loss_mask[i, self.name_to_idx[target]] = 1.0

            for ancestor in nx.ancestors(self.graph, target):
                loss_mask[i, self.name_to_idx[ancestor]] = 1.0
                for successor in self.graph.successors(ancestor):
                    loss_mask[i, self.name_to_idx[successor]] = 1.0
                    # This should also cover the node itself, but we do it anyway

        # Embed ground truth
        ground_truth_emb = self.embed(ground_truth)

        # Send to device
        loss_mask = torch.from_numpy(loss_mask).to(preds.device)
        ground_truth_emb = torch.from_numpy(ground_truth_emb).to(preds.device)

        clipped_probs = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        loss = -(ground_truth_emb * torch.log(clipped_probs)
                 + (1.0 - ground_truth_emb) * torch.log(1.0 - clipped_probs))

        sum_per_batch_element = torch.sum(loss * loss_mask , dim=1)

        loss = torch.mean(sum_per_batch_element)

        preds_dist = self.deembed(preds)
        y_hat = [sorted(pred_dist, key=lambda x: x[1], reverse=True)[0][0] for pred_dist in preds_dist]
        y_hat = torch.tensor([self.class_names.index(y) for y in y_hat]).to(preds.device)

        return loss, y_hat



