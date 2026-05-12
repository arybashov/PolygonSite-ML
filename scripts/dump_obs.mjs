// Dumps initial encodeTeamObservation for N seeds as JSON to stdout.
// Usage: node scripts/dump_obs.mjs [--seeds 20] [--ndrones 5] [--nanti 5]
import { DEFAULT_PARAMS } from '../src/constants.js';
import { encodeTeamObservation } from '../src/ml/observations.js';
import { Simulation } from '../src/simulation.js';

const args = parseArgs(process.argv.slice(2));
const nSeeds  = args.seeds   ?? 20;
const ndrones = args.ndrones ?? DEFAULT_PARAMS.ndrones;
const nanti   = args.nanti   ?? DEFAULT_PARAMS.nanti;

const params = { ...DEFAULT_PARAMS, ndrones, nanti };
const result = [];

for (let seed = 0; seed < nSeeds; seed++) {
  const sim = new Simulation({ params, seed, dronePolicy: null });
  result.push(Array.from(encodeTeamObservation(sim)));
}

process.stdout.write(JSON.stringify(result));

function parseArgs(argv) {
  const r = {};
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i].replace(/^--/, '');
    const v = argv[i + 1];
    if (v === undefined || v.startsWith('--')) { r[k] = true; }
    else { r[k] = Number.isFinite(Number(v)) ? Number(v) : v; i++; }
  }
  return r;
}
