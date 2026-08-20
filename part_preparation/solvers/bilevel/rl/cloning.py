"""
Behaviour cloning: supervised imitation of a hand-written rule, used as an RL warm start.

Why this exists. On the storage-dominated objective, policy gradient from scratch reaches
~1,134,000 over three seeds (barely 10% better than random) while a one-line rule scores
~491,000. Cloning that rule takes 70 seconds and lands at ~480,000 -- better than the teacher.
So the network can represent a good policy and supervised learning finds it; what fails is the
gradient-policy search. Starting RL from these weights means the agent refines a good policy
instead of rediscovering one.

The expert is entirely FEATURE-EXPRESSIBLE, which the myopic rollout used elsewhere is not:
that one picks `candidates[0]` at the engine level, a positional choice no network without a
position input could ever imitate. Cloning it would fail for a reason unrelated to capacity.
"""
import numpy as np
import torch
import torch.nn.functional as F


# Expert decision per level, each a pure function of that level's features.
#   engine  -> the one most expensive to leave unplaced   (feature 5, skip cost)
#   msn     -> cheapest reachable travel                  (feature 2)
#   factory -> best immediate reward                      (feature 5, what step will pay)
EXPERT = (
    lambda feats: int(np.argmax(feats[:, 5])),
    lambda feats: int(np.argmin(feats[:, 2])),
    lambda feats: int(np.argmax(feats[:, 5])),
)


def expert_episode(env, extractor):
    """
    Run the expert once, recording (features, chosen position) at each level.

    Returns (samples, cost). The cost is the rule's own score, i.e. the bar the clone has to
    match -- worth keeping so a clone that silently underperforms its teacher is visible.
    """
    state, cost, samples = env.reset(), 0.0, []
    while True:
        engine_feats, engine_idx = extractor.engine_candidates(state)
        if len(engine_idx) == 0:
            break
        engine_k = EXPERT[0](engine_feats)

        msn_feats, msn_idx = extractor.msn_candidates(state, int(engine_idx[engine_k]))
        if len(msn_idx) == 0:
            break
        msn_k = EXPERT[1](msn_feats)

        factory_feats, factory_idx = extractor.factory_candidates(
            state, int(engine_idx[engine_k]), int(msn_idx[msn_k]))
        if len(factory_idx) == 0:
            break
        factory_k = EXPERT[2](factory_feats)

        samples.append((engine_feats, engine_k, msn_feats, msn_k, factory_feats, factory_k))
        state, reward, done, _ = env.step((
            env.engines[int(engine_idx[engine_k])].id,
            env.factories[int(factory_idx[factory_k])].id,
            env.msns[int(msn_idx[msn_k])].id))
        cost -= reward
        if done:
            break
    return samples, cost


def collect(envs, extractors):
    """Expert demonstrations over every sub-problem. ~6,200 samples for 80 sub-problems."""
    samples, costs = [], []
    for env, extractor in zip(envs, extractors):
        episode, cost = expert_episode(env, extractor)
        samples += episode
        costs.append(cost)
    return samples, float(np.mean(costs))


def clone(network, samples, epochs=12, lr=1e-3, seed=0, log_every=3):
    """
    Cross-entropy on the expert's choice, at all three levels at once.

    Only the ACTOR heads are trained -- the critic never appears in this loss and stays random.
    That matters for what comes next: starting RL here without warming the critic first would
    feed the actor advantages computed from a random V, and the cloned policy would be destroyed
    in the first few updates.

    Accuracy is reported per level because they differ enormously in difficulty: 4 candidates at
    the factory level against ~80 at the MSN level.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    heads = (network.score_engines, network.score_msns, network.score_factories)

    for epoch in range(epochs):
        np.random.shuffle(samples)
        correct, total = np.zeros(3), 0

        for engine_feats, engine_k, msn_feats, msn_k, factory_feats, factory_k in samples:
            losses = []
            for head, feats, target in zip(heads,
                                           (engine_feats, msn_feats, factory_feats),
                                           (engine_k, msn_k, factory_k)):
                logits = head(torch.as_tensor(feats)).unsqueeze(0)
                losses.append(F.cross_entropy(logits, torch.tensor([target])))
                correct[len(losses) - 1] += int(logits.argmax()) == target

            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
            total += 1

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:2d} | accuracy  engine {100 * correct[0] / total:5.1f}%  "
                  f"msn {100 * correct[1] / total:5.1f}%  "
                  f"factory {100 * correct[2] / total:5.1f}%")

    return network


def warm_critic(envs, extractors, network, rng, iterations=30, episodes=4, lr=1e-3,
                gamma=1.0, lam=0.95, log_every=10):
    """
    Fit V on the cloned policy's own returns, with the ACTOR FROZEN.

    Cloning never touches the critic, so V comes out of it random. Handing that to GAE would
    produce advantages made of noise, and the first few policy updates would undo the clone --
    losing in seconds what supervised learning just built.

    Freezing the actor is what makes this safe: the policy cannot move while V is wrong, and by
    the time it is released the advantages mean something. The trajectories are collected under
    the cloned policy, which is exactly the distribution V will have to evaluate.
    """
    from part_preparation.solvers.bilevel.rl.training import gae, run_episode

    actor = [p for name, p in network.named_parameters() if not name.startswith("value")]
    critic = [p for name, p in network.named_parameters() if name.startswith("value")]
    for parameter in actor:
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(critic, lr=lr)
    scale = None

    for iteration in range(iterations):
        optimizer.zero_grad()
        losses = []
        for _ in range(episodes):
            index = int(rng.integers(len(envs)))
            _, values, rewards, _, _ = run_episode(envs[index], network, extractors[index])
            if not rewards:
                continue
            if scale is None:
                scale = max(abs(np.mean(rewards)) * len(rewards) / 100.0, 1e-8)
            _, returns = gae([r / scale for r in rewards], values, gamma, lam)
            loss = F.mse_loss(torch.stack(values), torch.as_tensor(returns))
            (loss / episodes).backward()
            losses.append(float(loss.detach()))

        torch.nn.utils.clip_grad_norm_(critic, 1.0)
        optimizer.step()
        if iteration % log_every == 0 or iteration == iterations - 1:
            print(f"  critic iter {iteration:3d} | value loss {np.mean(losses):10,.2f}")

    for parameter in actor:
        parameter.requires_grad_(True)
    return scale