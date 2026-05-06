// Usage:
//   node scripts/recorder.mjs [--episodes 200] [--seed 0] [--policy rule|stub]
//                             [--ndrones 5] [--nanti 5] [--timeout 300]
//                             [--out data/episodes.ndjson]
//
// Output: NDJSON — one episode per line
//   {"ep":0,"seed":0,"policy":"rule","nsteps":143,"steps":[{"obs":[...],"action":[...],"reward":0,"done":false},...]}
import { appendFileSync, rmSync } from 'fs';
import { DEFAULT_PARAMS, DRONE_DETECT_R, MAX_DRONES } from '../src/constants.js';
import { clamp } from '../src/math.js';
import { encodeTeamObservation } from '../src/ml/observations.js';
import { RuleBasedDronePolicy } from '../src/policies/dronePolicy.js';
import { createTeamMlStubPolicy } from '../src/policies/teamMlStub.js';
import { Simulation } from '../src/simulation.js';

const SCALE = DRONE_DETECT_R;

const args       = parseArgs(process.argv.slice(2));
const episodes   = args.episodes ?? 200;
const seedStart  = args.seed     ?? 0;
const policyName = args.policy   ?? 'rule';
const timeoutSec = args.timeout  ?? 300;
const outFile    = args.out      ?? 'data/episodes.ndjson';

const params = {
  ...DEFAULT_PARAMS,
  ...(args.ndrones != null && { ndrones: args.ndrones }),
  ...(args.nanti   != null && { nanti:   args.nanti   }),
};

try { rmSync(outFile); } catch {}

let totalSteps = 0;

for (let ep = 0; ep < episodes; ep++) {
  const seed    = seedStart + ep;
  const { policy, innerPolicy } = makeRecordingPolicy(policyName);
  const sim     = new Simulation({ params, seed, dronePolicy: policy });
  const steps   = [];

  while (sim.simTime < timeoutSec && !allDone(sim)) {
    const obs = Array.from(encodeTeamObservation(sim));

    innerPolicy.resetActionBuffer();
    const prevModes = sim.drones.map((d) => d.mode);
    sim.step();

    const reward = computeStepReward(sim, prevModes);
    const done   = allDone(sim) || sim.simTime >= timeoutSec;

    steps.push({ obs, action: Array.from(innerPolicy.actionBuffer), reward, done });
    if (done) break;
  }

  const line = JSON.stringify({ ep, seed, policy: policyName, nsteps: steps.length, steps });
  appendFileSync(outFile, line + '\n');
  totalSteps += steps.length;

  if ((ep + 1) % 50 === 0 || ep + 1 === episodes) {
    process.stderr.write(`ep ${ep + 1}/${episodes}  steps ${totalSteps}\n`);
  }
}

process.stderr.write(`Done. Wrote ${episodes} episodes, ${totalSteps} steps → ${outFile}\n`);

// — helpers —

function makeRecordingPolicy(name) {
  const base = name === 'stub' ? createTeamMlStubPolicy() : new RuleBasedDronePolicy();

  const actionBuffer = new Float32Array(MAX_DRONES * 3);

  const wrapper = {
    actionBuffer,
    resetActionBuffer() { actionBuffer.fill(0); },

    getAction(sim, drone) {
      const action = base.getAction(sim, drone);
      captureAction(drone, action, actionBuffer);
      return action;
    },

    getTeamActions(sim) {
      const map = base.getTeamActions?.(sim);
      if (!map) return null;
      sim.drones.forEach((drone) => {
        const action = map.get(drone.id);
        if (action) captureAction(drone, action, actionBuffer);
      });
      return map;
    },

    getDeployment(sim, droneCount) {
      return base.getDeployment?.(sim, droneCount) ?? null;
    },
  };

  return { policy: wrapper, innerPolicy: wrapper };
}

function captureAction(drone, action, buf) {
  if (action?.kind !== 'guidance') return;
  const offset = drone.id * 3;
  buf[offset]     = clamp((action.tx - drone.x) / SCALE, -1, 1);
  buf[offset + 1] = clamp((action.ty - drone.y) / SCALE, -1, 1);
  buf[offset + 2] = action.intent === 'evade' ? -1 : 1;
}

function computeStepReward(sim, prevModes) {
  let r = 0;
  sim.drones.forEach((drone, i) => {
    if (prevModes[i] === drone.mode) return;
    if (drone.mode === 'hit')         r += 1.0;
    if (drone.mode === 'intercepted') r -= 0.5;
  });
  return r;
}

function allDone(sim) {
  return sim.drones.every((d) => !d.alive);
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
