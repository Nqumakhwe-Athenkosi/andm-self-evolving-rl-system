"""Temporary independent numerical audit of the supplied ANDM model archives."""

from pathlib import Path
from zipfile import ZipFile
from collections import OrderedDict
import io
import json
import pickle

import numpy as np
import pandas as pd


import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package', type=Path, required=True, help='Extracted ANDM_Final_Measured_RL_Package folder')
parser.add_argument('--output', type=Path, default=Path('verification'))
args = parser.parse_args()
ROOT = args.package.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
PHASES = ['Morning', 'Afternoon', 'Night']
SEEDS = list(range(501, 516))


class FloatStorage:
    pass


def rebuild_tensor(storage, offset, shape, strides, requires_grad, hooks, metadata=None):
    shape = tuple(int(x) for x in shape)
    strides = tuple(int(x) for x in strides)
    offset = int(offset)
    if len(shape) != len(strides) or any(x < 0 for x in shape + strides) or offset < 0:
        raise ValueError('Invalid tensor view')
    last = offset + sum((size - 1) * stride for size, stride in zip(shape, strides) if size)
    if last >= storage.size:
        raise ValueError('Tensor view exceeds storage')
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=shape,
        strides=tuple(x * storage.dtype.itemsize for x in strides),
        writeable=False,
    ).copy()


class RestrictedTensorReader(pickle.Unpickler):
    """Read only standard float tensors; reject every other pickle global."""

    def __init__(self, buffer, archive, prefix):
        super().__init__(buffer)
        self.archive = archive
        self.prefix = prefix
        self.storage_cache = {}

    def find_class(self, module, name):
        permitted = {
            ('collections', 'OrderedDict'): OrderedDict,
            ('torch', 'FloatStorage'): FloatStorage,
            ('torch._utils', '_rebuild_tensor_v2'): rebuild_tensor,
        }
        if (module, name) not in permitted:
            raise pickle.UnpicklingError(f'Unsupported model object: {module}.{name}')
        return permitted[(module, name)]

    def persistent_load(self, identifier):
        kind, storage_type, key, device, size = identifier
        if kind != 'storage' or storage_type is not FloatStorage or device != 'cpu':
            raise pickle.UnpicklingError('Unsupported tensor storage')
        if key not in self.storage_cache:
            if not str(key).isdigit():
                raise pickle.UnpicklingError('Unexpected storage key')
            payload = self.archive.read(f'{self.prefix}/data/{key}')
            if len(payload) != int(size) * 4:
                raise ValueError('Tensor storage byte length mismatch')
            self.storage_cache[key] = np.frombuffer(payload, dtype='<f4').copy()
        return self.storage_cache[key]


def read_weights(path):
    with ZipFile(path) as outer:
        metadata = json.loads(outer.read('data'))
        with ZipFile(io.BytesIO(outer.read('policy.pth'))) as inner:
            pickle_path = next(x for x in inner.namelist() if x.endswith('/data.pkl'))
            prefix = pickle_path.rsplit('/', 1)[0]
            if inner.read(f'{prefix}/byteorder').decode().strip() != 'little':
                raise ValueError('Unsupported model byte order')
            weights = RestrictedTensorReader(
                io.BytesIO(inner.read(pickle_path)), inner, prefix
            ).load()
    for name, value in weights.items():
        if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
            raise ValueError(f'Invalid tensor: {name}')
    return metadata, weights


def make_predictor(weights, algorithm):
    if algorithm == 'DQN':
        layers = ['q_net.q_net.0', 'q_net.q_net.2', 'q_net.q_net.4']
        activation = lambda value: np.maximum(value, np.float32(0))
    elif algorithm == 'PPO':
        layers = ['mlp_extractor.policy_net.0', 'mlp_extractor.policy_net.2', 'action_net']
        activation = np.tanh
    else:
        raise ValueError(algorithm)
    if [weights[f'{layer}.weight'].shape for layer in layers] != [(64, 9), (64, 64), (3, 64)]:
        raise ValueError('Unexpected network architecture')

    def predict(observation):
        value = np.asarray(observation, dtype=np.float32)
        for index, layer in enumerate(layers):
            value = weights[f'{layer}.weight'] @ value + weights[f'{layer}.bias']
            if index < len(layers) - 1:
                value = activation(value)
        return int(np.argmax(value))

    return predict


def split(frame, first, last):
    return pd.concat([
        frame[frame['simulated_time'] == phase].sort_values('sample').iloc[first:last]
        for phase in PHASES
    ], ignore_index=True)


def load_frames():
    frames = {}
    for scenario in ['baseline', 'rural']:
        cpu = pd.read_csv(ROOT / f'data/mininet_{scenario}_cpu.csv')
        network = pd.read_csv(ROOT / f'data/mininet_{scenario}_network_raw.csv')
        selected = network[(network['switch'] == 's1') & (network['port'] == '1:')].sort_values('sample')
        if not np.array_equal(cpu['timestamp'].to_numpy(), selected['timestamp'].to_numpy()):
            raise ValueError('CPU and network timestamps differ')
        frame = cpu.merge(
            selected.drop(columns=['timestamp', 'scenario', 'simulated_time']),
            on='sample', validate='one_to_one',
        )
        frame['delivery_ratio'] = np.clip(frame['delivered_rate_mbps'] / frame['offered_load_mbps'], 0, 1)
        frame['latency_for_model_ms'] = frame['latency_ms'].fillna(1000.0)
        frames[scenario] = frame
    calibration = 70.0 / split(frames['baseline'], 0, 36)['cpu_used'].quantile(0.95)
    for frame in frames.values():
        frame['calibrated_compute_load'] = np.clip(frame['cpu_used'] * calibration, 0, 160)
    return frames, calibration


def state(row, vcpu):
    return np.array([
        np.clip(row['calibrated_compute_load'] / 160, 0, 1),
        np.clip(row['offered_load_mbps'] / 30, 0, 1),
        np.clip(row['delivery_ratio'], 0, 1),
        np.clip(row['latency_for_model_ms'] / 1000, 0, 1),
        np.clip(row['packet_loss_pct'] / 100, 0, 1),
        vcpu / 4,
        *[float(row['simulated_time'] == phase) for phase in PHASES],
    ], dtype=np.float32)


def episode(frame, policy, seed, predictor=None, start_offset=None):
    offset = int(np.random.default_rng(seed).integers(0, len(frame))) if start_offset is None else start_offset
    order = np.roll(np.arange(len(frame)), -offset)
    vcpu = 2
    total_reward = 0.0
    trace = []
    for index in order:
        row = frame.iloc[index]
        previous = vcpu
        if predictor is not None:
            action = predictor(state(row, vcpu))
        elif policy == 'Fixed-vCPU-2':
            action = 1 if vcpu < 2 else 2 if vcpu > 2 else 0
        elif policy == 'CPU-rule':
            projected = row['calibrated_compute_load'] * 2 / vcpu
            action = 1 if projected > 85 and vcpu < 4 else 2 if projected < 40 and vcpu > 1 else 0
        else:
            raise ValueError(policy)
        if action == 1:
            vcpu = min(4, vcpu + 1)
        elif action == 2:
            vcpu = max(1, vcpu - 1)
        projected = row['calibrated_compute_load'] * 2 / vcpu
        satisfaction = min(1.0, 100 / max(projected, 1e-6)) * row['delivery_ratio']
        reward = 2 * satisfaction - 1.5 * max(projected - 85, 0) / 85 - 0.1 * vcpu - (0.05 if vcpu != previous else 0)
        total_reward += reward
        trace.append({
            'policy': policy, 'seed': seed, 'action': action, 'reward': reward,
            'sample': int(row['sample']), 'simulated_time': row['simulated_time'],
            'measured_cpu_pct': float(row['cpu_used']), 'projected_cpu_pct': float(projected),
            'delivery_ratio': float(row['delivery_ratio']),
            'latency_ms': float(row['latency_for_model_ms']),
            'packet_loss_pct': float(row['packet_loss_pct']),
            'service_satisfaction': float(satisfaction), 'vcpu': vcpu,
            'overload': bool(projected > 100),
        })
    trace = pd.DataFrame(trace)
    return {
        'policy': policy, 'seed': seed, 'episode_reward': total_reward,
        'mean_satisfaction': trace['service_satisfaction'].mean(),
        'mean_vcpu': trace['vcpu'].mean(), 'overload_rate': trace['overload'].mean(),
    }, trace


def evaluate(frame, policy, predictor=None):
    episodes = []
    traces = []
    for seed in SEEDS:
        metrics, trace = episode(frame, policy, seed, predictor)
        episodes.append(metrics)
        traces.append(trace)
    return pd.DataFrame(episodes), pd.concat(traces, ignore_index=True)


def summary(episodes):
    return {
        'mean_reward': episodes['episode_reward'].mean(),
        'std_reward': episodes['episode_reward'].std(),
        'mean_satisfaction': episodes['mean_satisfaction'].mean(),
        'mean_vcpu': episodes['mean_vcpu'].mean(),
        'mean_overload_rate': episodes['overload_rate'].mean(),
    }


def bootstrap(first, second, seed):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    generator = np.random.default_rng(seed)
    means = np.array([generator.choice(differences, len(differences), replace=True).mean() for _ in range(10000)])
    return {'mean_difference': float(differences.mean()), 'ci_low': float(np.percentile(means, 2.5)), 'ci_high': float(np.percentile(means, 97.5))}


def main():
    frames, calibration = load_frames()
    validation = split(frames['baseline'], 36, 48)
    saved = pd.read_csv(ROOT / 'rl_outputs/algorithm_seed_results.csv')
    model_results = []
    predictors = {}
    for path in sorted((ROOT / 'rl_outputs/models').glob('*.zip')):
        algorithm = 'PPO' if path.name.startswith('ppo') else 'DQN'
        metadata, weights = read_weights(path)
        predictor = make_predictor(weights, algorithm)
        predictors[path.name] = predictor
        if path.name.startswith('candidate'):
            continue
        seed = int(metadata['seed'])
        results, _ = evaluate(validation, f'{algorithm}-seed-{seed}', predictor)
        computed = summary(results)
        expected = saved[(saved['algorithm'] == algorithm) & (saved['seed'] == seed)].iloc[0]
        maximum_error = max(abs(computed[key] - expected[f'validation_{key}']) for key in computed)
        model_results.append({'algorithm': algorithm, 'seed': seed, 'max_metric_error': maximum_error, 'num_timesteps': metadata['num_timesteps'], **computed})
        print(f'{path.name}: max saved-metric difference {maximum_error:.3g}', flush=True)
        if maximum_error > 1e-8:
            raise ValueError(f'Saved model does not reproduce recorded results: {path.name}')

    model_results = pd.DataFrame(model_results)
    model_results.to_csv(OUT / 'independent_model_validation.csv', index=False)
    champion = predictors['dqn_seed_42.zip']
    candidate = predictors['candidate_dqn_v2.zip']
    rural_validation = split(frames['rural'], 36, 48)
    champion_validation, _ = evaluate(rural_validation, 'Champion-v1', champion)
    candidate_validation, _ = evaluate(rural_validation, 'Candidate-v2', candidate)
    promotion = bootstrap(candidate_validation['episode_reward'], champion_validation['episode_reward'], 2027)
    print('REPRODUCED PROMOTION:', json.dumps(promotion), flush=True)

    final_rows = []
    final_traces = []
    test = split(frames['rural'], 48, 60)
    for policy, predictor in [('Champion-v1', champion), ('Deployed-v2', candidate), ('Fixed-vCPU-2', None), ('CPU-rule', None)]:
        results, trace = evaluate(test, policy, predictor)
        final_rows.append(results)
        final_traces.append(trace)
    all_episodes = pd.concat(final_rows, ignore_index=True)
    all_traces = pd.concat(final_traces, ignore_index=True)
    all_episodes.to_csv(OUT / 'independent_rural_test_episodes.csv', index=False)
    all_traces.to_csv(OUT / 'independent_rural_test_traces.csv', index=False)
    comparison = pd.DataFrame([{'policy': policy, **summary(group)} for policy, group in all_episodes.groupby('policy', sort=False)])
    comparison.to_csv(OUT / 'independent_rural_test_summary.csv', index=False)
    print('INDEPENDENT FINAL TEST\n' + comparison.to_string(index=False), flush=True)

    old_episodes = pd.read_csv(ROOT / 'rl_outputs/final_rural_test_episodes.csv')
    repeated = all_episodes.merge(old_episodes, on=['policy', 'seed'], suffixes=('_verified', '_saved'))
    errors = {column: float(np.max(np.abs(repeated[f'{column}_verified'] - repeated[f'{column}_saved']))) for column in ['episode_reward', 'mean_satisfaction', 'mean_vcpu', 'overload_rate']}
    if len(repeated) != len(old_episodes) or max(errors.values()) > 1e-8:
        raise ValueError('Final saved episode results do not match')
    _, natural_trace = episode(test, 'Deployed-v2', 2026, candidate, start_offset=0)
    original_trace = pd.read_csv(ROOT / 'rl_outputs/deployed_rural_trace.csv')
    if not np.array_equal(natural_trace['vcpu'].to_numpy(), original_trace['vcpu'].to_numpy()):
        raise ValueError('Deployed natural-order decisions do not match')
    natural_trace.to_csv(OUT / 'independent_deployed_natural_trace.csv', index=False)
    v1 = all_episodes[all_episodes['policy'] == 'Champion-v1'].sort_values('seed')
    v2 = all_episodes[all_episodes['policy'] == 'Deployed-v2'].sort_values('seed')
    rule = all_episodes[all_episodes['policy'] == 'CPU-rule'].sort_values('seed')
    fixed = all_episodes[all_episodes['policy'] == 'Fixed-vCPU-2'].sort_values('seed')
    intervals = {
        'v2_minus_v1': bootstrap(v2['episode_reward'], v1['episode_reward'], 2028),
        'v2_minus_fixed': bootstrap(v2['episode_reward'], fixed['episode_reward'], 2029),
        'rule_minus_v2': bootstrap(rule['episode_reward'], v2['episode_reward'], 2030),
    }
    audit = {
        'method': 'Independent NumPy inference from restricted float-tensor archive parsing; no fresh training.',
        'models_reproduced': len(model_results), 'saved_final_episodes_reproduced': len(repeated),
        'saved_deployed_trace_decisions_reproduced': len(natural_trace),
        'final_episode_metric_errors': errors,
        'cpu_calibration_factor': calibration,
        'promotion_interval_reproduced': promotion,
        'additional_final_test_paired_intervals': intervals,
        'interval_boundary': 'Conditional on 15 cyclic offsets of one 36-row rural test trace; not independent network captures.',
    }
    (OUT / 'verification_record.json').write_text(json.dumps(audit, indent=2))
    print('VERIFICATION RECORD\n' + json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
