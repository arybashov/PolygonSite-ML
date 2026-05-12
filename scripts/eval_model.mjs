// Usage:
//   node scripts/eval_model.mjs [--model data/model_web.onnx] [--episodes 200]
//                               [--seed 0] [--ndrones 5] [--nanti 5]
//                               [--out results.json] [--label "BC model"]
import { readFileSync, writeFileSync } from 'fs';
import * as ort from 'onnxruntime-node';
import { DEFAULT_PARAMS, DRONE_DETECT_R } from '../src/constants.js';
import { encodeTeamObservation } from '../src/ml/observations.js';
import { RuleBasedDronePolicy, TeamWaypointMlDronePolicy } from '../src/policies/dronePolicy.js';
import { Simulation } from '../src/simulation.js';

const args       = parseArgs(process.argv.slice(2));
const modelPath  = args.model    ?? 'data/model_web.onnx';
const episodes   = args.episodes ?? 200;
const seedStart  = args.seed     ?? 0;
const timeoutSec = args.timeout  ?? 1000;
const outFile    = args.out      ?? null;
const label      = args.label    ?? 'BC model';

const params = {
  ...DEFAULT_PARAMS,
  ...(args.ndrones != null && { ndrones: args.ndrones }),
  ...(args.nanti   != null && { nanti:   args.nanti   }),
};

process.stderr.write(`Loading ${modelPath} ...\n`);
const session = await ort.InferenceSession.create(modelPath);
process.stderr.write(`Model loaded. Running ${episodes} episodes ...\n`);

const records = [];

for (let ep = 0; ep < episodes; ep++) {
  const seed = seedStart + ep;

  // Shared cache: actionProvider reads from here, prefetch writes here
  let cachedOutput = null;

  const policy = new TeamWaypointMlDronePolicy({
    actionProvider:    () => cachedOutput,
    deploymentProvider: null,
    fallback:          new RuleBasedDronePolicy(),
    actionScale:       DRONE_DETECT_R,
  });

  const sim = new Simulation({ params, seed, dronePolicy: policy });

  while (sim.simTime < timeoutSec && !allDone(sim)) {
    // Async inference before synchronous sim.step()
    const obs    = encodeTeamObservation(sim);
    const tensor = new ort.Tensor('float32', obs, [1, obs.length]);
    const out    = await session.run({ obs: tensor });
    cachedOutput = out.action.data;

    sim.step();
  }

  const total       = sim.drones.length;
  const hits        = sim.drones.filter((d) => d.mode === 'hit').length;
  const intercepted = sim.drones.filter((d) => d.mode === 'intercepted').length;
  const timedOut    = sim.simTime >= timeoutSec;
  const reward      = hits * 1.0 - intercepted * 0.5 - (sim.simTime / timeoutSec) * 0.1;

  records.push({ seed, reward, hits, intercepted, total, time: Math.round(sim.simTime), timedOut });

  if ((ep + 1) % 50 === 0 || ep + 1 === episodes) {
    process.stderr.write(`ep ${ep + 1}/${episodes}\n`);
  }
}

const n        = records.length;
const avg      = (key) => records.reduce((s, r) => s + r[key], 0) / n;
const ratioAvg = (num, den) => records.reduce((s, r) => s + r[num] / r[den], 0) / n;

const summary = {
  mean_reward:    r3(avg('reward')),
  hit_rate:       r3(ratioAvg('hits', 'total')),
  intercept_rate: r3(ratioAvg('intercepted', 'total')),
  mean_time:      r3(avg('time')),
  timeout_rate:   r3(records.filter((r) => r.timedOut).length / n),
};

console.log(JSON.stringify({ policy: label, params: { ndrones: params.ndrones, nanti: params.nanti }, episodes: n, summary }, null, 2));

if (outFile) {
  const run = {
    id:        Date.now(),
    timestamp: new Date().toISOString(),
    label,
    policy:    'onnx',
    params:    { ndrones: params.ndrones, nanti: params.nanti },
    summary,
    records,
  };

  let existing = [];
  try { existing = JSON.parse(readFileSync(outFile, 'utf8')); } catch {}
  existing.push(run);
  writeFileSync(outFile, JSON.stringify(existing, null, 2));
  process.stderr.write(`Saved to ${outFile} (${existing.length} runs total)\n`);
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
    const key  = argv[i].replace(/^--/, '');
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
