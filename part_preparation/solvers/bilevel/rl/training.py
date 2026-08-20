"""
Actor-critic training loop with GAE.

The action is autoregressive -- (engine, MSN, factory) drawn in that order inside one
environment step -- so the three log-probabilities are SUMMED into one scalar per step. That
sum is log pi(e, m, f | s) by the chain rule, and the algorithm never needs to know the action
was composite.

Expects an extractor exposing four pure functions of (state, choices made so far):

    engine_candidates(state)                    -> feats (n_e, 5),  indices (n_e,)
    msn_candidates(state, engine_i)             -> feats (n_m, 5),  indices (n_m,)
    factory_candidates(state, engine_i, msn_i)  -> feats (4, 11),   indices (4,)
    state_features(state)                       -> (state_dim,)

Row k of `feats` describes entity `indices[k]`: only legal candidates are returned, so the
network samples a POSITION and `indices` maps it back to the real entity.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def pick(score_fn, feats, greedy=False):
    """
    Score the candidates, draw one, return its position, log-prob and entropy.

    greedy=False (sampling) is the only valid mode for training: the policy-gradient estimator
    requires the action to come from pi_theta. greedy=True is for evaluation, where a
    deterministic policy scores better.

    Entropy is divided by log(n_candidates). The candidate count swings from 4 factories to
    ~500 MSNs, then down to 1 near the end of the episode, so raw nats would let the MSN level
    dominate the entropy bonus early and vanish late. Normalized, 1.0 reads as uniform and 0.0
    as fully committed, at every level and at every point of the episode.

    The caller must check that `feats` is non-empty. Categorical raises on empty logits, and
    raises again on all-(-inf) logits because it validates its arguments -- so the failure is
    loud by default. It becomes silent only if validate_args is turned off, which is a common
    speed tweak in RL, so guard rather than rely on it.
    """
    logits = score_fn(torch.as_tensor(feats))
    dist = Categorical(logits=logits)
    k = logits.argmax() if greedy else dist.sample()
    max_entropy = np.log(max(len(feats), 2))
    return int(k), dist.log_prob(k), dist.entropy() / max_entropy


def run_episode(env, network, extractor, greedy=False):
    """
    One full episode on one engine type.

    The three levels are drawn in sequence inside a single environment step: the state does not
    change between them, the conditioning travels through the arguments handed to the extractor
    (engine_i, then msn_i). env.step is called once, with the complete triple.

    Updating EPISODICALLY rather than at every step is what GAE requires: A_t is a backward
    recursion over the whole trajectory, so it cannot be evaluated before the episode ends. An
    online update would force lam = 0, i.e. plain TD(0). Episodes here run 80 to 500 steps, so
    the retained graph stays small; on much longer ones a truncated buffer would replace this.

    Returns:
        logps, values, rewards, entropies (one entry per step) and the last info dict.
    """
    state = env.reset()
    logps, values, rewards, entropies = [], [], [], []
    info = {}

    while True:
        # Every level is guarded before a distribution is built: Categorical raises on an
        # empty candidate set, and a crash mid-training is not a useful failure mode.
        engine_feats, engine_idx = extractor.engine_candidates(state)
        if len(engine_idx) == 0:
            break
        k, engine_logp, engine_entropy = pick(network.score_engines, engine_feats, greedy)
        engine_i = int(engine_idx[k])

        msn_feats, msn_idx = extractor.msn_candidates(state, engine_i)
        if len(msn_idx) == 0:
            break
        k, msn_logp, msn_entropy = pick(network.score_msns, msn_feats, greedy)
        msn_i = int(msn_idx[k])

        factory_feats, factory_idx = extractor.factory_candidates(state, engine_i, msn_i)
        if len(factory_idx) == 0:
            break
        k, factory_logp, factory_entropy = pick(network.score_factories, factory_feats, greedy)
        factory_i = int(factory_idx[k])

        # V(s) on the state BEFORE the action, and never from a representation that already
        # depends on the chosen engine or MSN: a baseline must be a function of the state alone,
        # otherwise the policy gradient becomes biased.
        value = network.state_value(torch.as_tensor(extractor.state_features(state)))

        state, reward, done, info = env.step((
            env.engines[engine_i].id, env.factories[factory_i].id, env.msns[msn_i].id))

        # Chain rule: log pi(e, m, f | s) is the SUM of the three. Keeping only one would leave
        # the other two levels without any gradient.
        logps.append(engine_logp + msn_logp + factory_logp)
        entropies.append(engine_entropy + msn_entropy + factory_entropy)
        values.append(value)
        rewards.append(reward)

        if done:
            break

    return logps, values, rewards, entropies, info


def gae(rewards, values, gamma=1.0, lam=0.95):
    """
    Generalized Advantage Estimation -- the TD(lambda) forward view, used as an advantage.

        A_t = delta_t + (gamma*lam) A_{t+1}      with  delta = r + gamma*V(s') - V(s)

    The recursion is the whole point: A_t contains the sum for A_{t+1}, so walking backwards
    turns an O(T^2) sum into O(T). `running` holds A_{t+1} when the loop reaches step t.

    Two knobs, living in different places:
      gamma belongs to the OBJECTIVE -- it says what a good policy is. Left at 1 because the
        real cost here is an undiscounted sum of euros over a finite episode, and discounting
        would optimise something else (0.99^80 = 0.45, the last engine worth half the first).
      lam belongs to the ESTIMATOR only -- it never appears in the objective, it trades bias
        against variance. Its bias vanishes as V improves; gamma's never does.

    Effective horizon is 1/(1 - gamma*lam), so lam=0.95 means an action's advantage sees about
    20 steps of consequences -- enough to feel the queue it creates, not so much that it drowns
    in the noise of 60 later decisions.

    Values are detached to plain floats: the backward recursion must not build a graph through
    itself. The last step bootstraps with 0, not V(s_terminal): the episode ends because no
    engine or no MSN is left, so there is genuinely no future reward.
    """
    values = [float(v.detach()) for v in values]
    advantages = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0

    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running

    # A_t + V(s_t) is exactly the lambda-return, so regressing V on this trains the critic
    # by TD(lambda) with the same lam.
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def reference_scores(envs, network, extractors):
    """
    One reference cost per sub-problem, from an untrained pass.

    Raw episode rewards cannot be plotted against each other: on medium, type 1 costs ~187k and
    type 2 ~151k, so a curve of raw batch means would mostly show WHICH sub-problems happened to
    be drawn, not whether the policy improved. Dividing by this reference puts every sub-problem
    on the same scale, where 1.0 means "as good as an untrained network".
    """
    with torch.no_grad():
        return np.array([
            max(abs(sum(run_episode(env, network, extractor)[2])), 1e-8)
            for env, extractor in zip(envs, extractors)])


def train(envs, network, extractors, iterations=200, episodes_per_batch=4, lr=1e-3,
          gamma=1.0, lam=0.95, value_coef=0.5, entropy_coef=0.01, seed=0, log_every=10,
          eval_fn=None):
    """
    Args:
        envs: one Env per (instance, engine type). One is drawn at random per episode.
        extractors: aligned with envs.
        value_coef: actor and critic share no parameter here, so this is redundant with the
            critic's learning rate -- it only rescales the value gradient. Tune one, not both.
        eval_fn: called every log_every iterations with no argument, returns a scalar cost on a
            FIXED held-out set played greedily. This is the curve to trust: no sampling noise,
            no mixing of sub-problems, and it measures what the policy knows rather than what it
            is exploring. The two can diverge sharply -- on an earlier attempt the sampled score
            sat at 810k while the greedy one fell to 754k, which is how we saw the policy had
            learned a ranking but stayed near-uniform when sampling.

    Returns:
        history, one record per iteration: raw reward, reward normalised by the sub-problem
        reference, entropy, and the eval score when one was computed.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Actor and critic share no parameter, so they must be clipped SEPARATELY. Clipping the
    # global norm recouples them: the critic's gradient starts around 36 (it regresses returns
    # of magnitude ~100) against 0.11 for the actor, so a global clip at 1.0 divides everything
    # by ~37 and the policy ends up training at lr/37. Measured: actor gradient 0.0031 instead
    # of 0.223, a factor 73. That is what froze the entropy at 0.978 per level.
    actor_params = [p for name, p in network.named_parameters() if not name.startswith("value")]
    critic_params = [p for name, p in network.named_parameters() if name.startswith("value")]

    reference = reference_scores(envs, network, extractors)
    # Rewards are euros: ~-3000 a step, so a return is ~-1e5 to -1e6. Left raw, MSE(V, G) would
    # be ~1e12, drowning the policy term and saturating the gradient clipping.
    scale = max(reference.mean() / 100.0, 1e-8)

    history = []
    # Keep the best-scoring weights, not the last. Training can degrade late -- observed here,
    # 68,724 at iteration 80 against 74,170 at 199 -- and saving only the final state throws
    # away the run's actual result.
    best = {"eval": float("inf"), "state": None, "iteration": None}

    for iteration in range(iterations):
        optimizer.zero_grad()
        raw, normalised, batch_entropy = [], [], []

        for _ in range(episodes_per_batch):
            i = int(rng.integers(len(envs)))
            logps, values, rewards, entropies, _ = run_episode(envs[i], network, extractors[i])
            if not rewards:
                continue

            advantages, returns = gae([r / scale for r in rewards], values, gamma, lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # gae() returns plain numpy, so the advantage is already detached. That matters: if
            # the policy term could reach the critic, V would learn to shrink the advantage
            # instead of predicting the return.
            policy_loss = -(torch.stack(logps) * torch.as_tensor(advantages)).mean()
            value_loss = F.mse_loss(torch.stack(values), torch.as_tensor(returns))
            # Each level is normalised to [0, 1] and run_episode SUMS the three, so the
            # maximum is 3. Divide to keep the logged number readable as "per level".
            entropy = torch.stack(entropies).mean() / 3.0

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            # Gradients accumulate across the batch, one optimizer step happens after the loop.
            (loss / episodes_per_batch).backward()

            raw.append(-sum(rewards))
            normalised.append(-sum(rewards) / reference[i])
            batch_entropy.append(float(entropy.detach()))

        torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
        torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
        optimizer.step()

        record = {
            "iteration": iteration,
            "cost": float(np.mean(raw)) if raw else float("nan"),
            "cost_normalised": float(np.mean(normalised)) if normalised else float("nan"),
            "entropy": float(np.mean(batch_entropy)) if batch_entropy else float("nan"),
            "eval": None,
        }

        if iteration % log_every == 0 or iteration == iterations - 1:
            if eval_fn is not None:
                record["eval"] = float(eval_fn())
            shown = f"{record['eval']:11,.0f}" if record["eval"] is not None else " " * 11
            # Entropy PER LEVEL: 1.0 is uniform, 0.0 fully committed. Pinned near 1.0
            # means the sampled policy is still uniform even if argmax has learned a ranking --
            # the logits carry a consistent tilt too small for sampling to follow.
            print(f"  iter {iteration:4d} | sampled {record['cost_normalised']:.3f} x ref "
                  f"| greedy {shown} | entropy {record['entropy']:.3f}")

        if record["eval"] is not None and record["eval"] < best["eval"]:
            best.update(eval=record["eval"], iteration=iteration,
                        state=copy.deepcopy(network.state_dict()))

        history.append(record)

    return history, best