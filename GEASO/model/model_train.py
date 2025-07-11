import torch
from tqdm import trange
from collections import OrderedDict
from GEASO.model.graph_aug import random_aug
from timeit import default_timer as timer


def train(
    model,
    inputs,
    args,
):
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_log = []
    time_now = timer()
    t = trange(args.epochs, desc='', leave=True)

    for epoch in t:
        model.train()
        optimizer.zero_grad()

        for name, graph, feat in inputs:
            with torch.no_grad():
                N = graph.number_of_nodes()
                graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
                graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)
                graph1 = graph1.add_self_loop()
                graph2 = graph2.add_self_loop()

            z1, z2 = model(graph1, feat1, graph2, feat2)
            c = torch.mm(z1.T, z2) / N
            c1 = torch.mm(z1.T, z1) / N
            c2 = torch.mm(z2.T, z2) / N
            loss_inv = -torch.diagonal(c).sum()
            iden = torch.eye(c.size(0), device=args.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()
            loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)
            loss.backward()
            optimizer.step()

        loss_log.append(loss.item())
        time_step = timer() - time_now
        time_now = time_step + time_now
        t.set_description(f'Epoch: {epoch}, Loss: {loss.item():.3f} step time={time_step:.3f}s')
        t.refresh()

    with torch.no_grad():
        model.eval()
        embedding = OrderedDict()
        for name, graph, feat in inputs:
            embedding[name] = model.get_embedding(graph, feat)
        return embedding, loss_log, model
