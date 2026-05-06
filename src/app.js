import { DEFAULT_PARAMS, MAX_DRONES, PARAM_DEFS } from './constants.js';
import { initResultsPanel } from './charts.js';
import {
  DEPLOYMENT_OBSERVATION_SCHEMA,
  TEAM_OBSERVATION_SCHEMA,
  encodeDeploymentObservation,
  encodeTeamObservation,
} from './ml/observations.js';
import { RuleBasedDronePolicy } from './policies/dronePolicy.js';
import { createTeamMlStubPolicy } from './policies/teamMlStub.js';
import { CanvasRenderer } from './render.js';
import { Simulation } from './simulation.js';

const canvas = document.getElementById('c');
const resetButton = document.getElementById('btnReset');
const policyMode = document.getElementById('policyMode');
const policyStatus = document.getElementById('policyStatus');
const controls = new Map(PARAM_DEFS.map((def) => [def.id, document.getElementById(def.id)]));
const valueLabels = new Map(PARAM_DEFS.map((def) => [def.id, document.getElementById(`${def.id}V`)]));

const sim = new Simulation({ params: readParams(), seed: Date.now(), dronePolicy: createPolicy() });
const renderer = new CanvasRenderer(canvas, sim);

window.sim = sim;

syncControlLabels();
bindControls();
syncMlPanel();
updateStats();
requestAnimationFrame(loop);

initResultsPanel(
  document.getElementById('btnResults'),
  document.getElementById('resultsOverlay'),
  document.getElementById('btnCloseResults'),
);

function bindControls() {
  resetButton.addEventListener('click', () => {
    resetScenario();
  });

  policyMode.addEventListener('change', () => {
    resetScenario();
  });

  for (const def of PARAM_DEFS) {
    const input = controls.get(def.id);
    input.addEventListener('input', () => {
      valueLabels.get(def.id).textContent = input.value;
    });
  }
}

function loop() {
  sim.setParams(readParams());
  sim.step();
  renderer.draw();
  updateStats();
  requestAnimationFrame(loop);
}

function resetScenario() {
  sim.setParams(readParams());
  sim.setDronePolicy(createPolicy());
  sim.reset({ seed: Date.now() });
  renderer.resetView();
  syncMlPanel();
  updateStats();
}

function readParams() {
  const params = { ...DEFAULT_PARAMS };
  for (const def of PARAM_DEFS) {
    const input = controls.get(def.id);
    params[def.id] = Number(input?.value ?? DEFAULT_PARAMS[def.id]);
  }
  return params;
}

function syncControlLabels() {
  for (const def of PARAM_DEFS) {
    const input = controls.get(def.id);
    const label = valueLabels.get(def.id);
    if (input) input.value = DEFAULT_PARAMS[def.id];
    if (label) label.textContent = DEFAULT_PARAMS[def.id];
  }
}

function updateStats() {
  const stats = sim.getStats();
  document.getElementById('sDrones').textContent = stats.drones;
  document.getElementById('sHits').textContent = stats.hits;
  document.getElementById('sInt').textContent = stats.intercepted;
  document.getElementById('sAD').textContent = stats.activeAnti;
  document.getElementById('sTime').textContent = stats.time;
  updateMlMode();
  updateIntentStats();
}

function createPolicy() {
  return policyMode?.value === 'team-stub'
    ? createTeamMlStubPolicy()
    : new RuleBasedDronePolicy();
}

function getPolicyLabel() {
  return policyMode?.value === 'team-stub' ? 'Team ML Stub' : 'Rule-based';
}

function syncMlPanel() {
  setText('mlDroneSlots', MAX_DRONES);
  setText('mlDeploymentObs', encodeDeploymentObservation(sim).length);
  setText('mlDeploymentAction', DEPLOYMENT_OBSERVATION_SCHEMA.actionSize);
  setText('mlTeamObs', encodeTeamObservation(sim).length);
  setText('mlTeamAction', TEAM_OBSERVATION_SCHEMA.actionSize);
  setText('mlSeed', sim.seed);
  updateMlMode();
}

function updateMlMode() {
  const label = getPolicyLabel();
  policyStatus.textContent = label;
  setText('mlMode', label);
}

function updateIntentStats() {
  const live = sim.drones.filter((drone) => {
    return drone.alive && drone.mode !== 'hit' && drone.mode !== 'intercepted';
  });
  const evade = live.filter((drone) => drone.intent === 'evade').length;
  setText('mlAttackIntent', live.length - evade);
  setText('mlEvadeIntent', evade);
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}
