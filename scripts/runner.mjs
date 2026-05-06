// Usage:
//   node scripts/runner.mjs [--episodes 100] [--seed 0] [--policy rule|stub]
//                           [--ndrones 5] [--nanti 5] [--timeout 300] [--detail]
import { DEFAULT_PARAMS } from '../src/constants.js';
import { RuleBasedDronePolicy } from '../src/policies/dronePolicy.js';
import { createTeamMlStubPolicy } from '../src/policies/teamMlStub.js';
import { Simulation } from '../src/simulation.js';

const args = parseArgs(process.argv.slice(2));
const episodes   = args.episodes  ?? 100;
const seedStart  = args.seed      ?? 0;
const policyName = args.policy    ?? 'rule';
const timeoutSec = args.timeout   ?? 300;
const detail     = 'detail' in args;

const params = {
  ...DEFAULT_PARAMS,
  ...(args.ndrones != null && { ndrones: args.ndrones }),
  ...(args.nanti   != null && { nanti:   args.nanti   }),
};

const records = [];

for (let i = 0; i < episodes; i++) {
  const seed = seedStart + i;
  const sim  = new Simulation({ params, seed, dronePolicy: makePolicy(policyName) });

  while (sim.simTime < timeoutSec && !allDone(sim)) {
    sim.step();
  }

  const total       = sim.drones.length;
  const hits        = sim.drones.filter((d) => d.mode === 'hit').length;
  const intercepted = sim.drones.filter((d) => d.mode === 'intercepted').length;
  const timedOut    = sim.simTime >= timeoutSec;
  const reward      = hits * 1.0 - intercepted * 0.5 - (sim.simTime / timeoutSec) * 0.1;

  records.push({ seed, reward, hits, intercepted, total, time: Math.round(sim.simTime), timedOut });
}

const n       = records.length;
const avg     = (key) => records.reduce((s, r) => s + r[key], 0) / n;
const ratioAvg = (num, den) => records.reduce((s, r) => s + r[num] / r[den], 0) / n;

const summary = {
  mean_reward:    r3(avg('reward')),
  hit_rate:       r3(ratioAvg('hits', 'total')),
  intercept_rate: r3(ratioAvg('intercepted', 'total')),
  mean_time:      r3(avg('time')),
  timeout_rate:   r3(records.filter((r) => r.timedOut).length / n),
};

const output = {
  policy:   policyName,
  params:   { ndrones: params.ndrones, nanti: params.nanti },
  episodes: n,
  summary,
  ...(detail && { records }),
};

console.log(JSON.stringify(output, null, 2));

function makePolicy(name) {
  return name === 'stub' ? createTeamMlStubPolicy() : new RuleBasedDronePolicy();
}

function allDone(sim) {
  return sim.drones.every((d) => !d.alive);
}

function r3(v) {
  return Math.round(v * 1000) / 1000;
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i].replace(/^--/, '');
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      result[key] = true;
    } else {
      result[key] = Number.isFinite(Number(next)) ? Number(next) : next;
      i++;
    }
  }
  return result;
}
