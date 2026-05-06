import { Simulation } from '../src/simulation.js';
import { MAX_DRONES } from '../src/constants.js';
import {
  encodeDeploymentObservation,
  encodeDroneObservation,
  encodeTeamObservation,
} from '../src/ml/observations.js';
import { TeamWaypointMlDronePolicy } from '../src/policies/dronePolicy.js';
import { createTeamMlStubPolicy } from '../src/policies/teamMlStub.js';

const sim = new Simulation({ seed: 1 });

for (let i = 0; i < 60 * 20; i++) {
  sim.step();
}

const stats = sim.getStats();
const drone = sim.drones.find((item) => item.alive) ?? sim.drones[0];
const observation = encodeDroneObservation(sim, drone);
const teamObservation = encodeTeamObservation(sim);
const deploymentObservation = encodeDeploymentObservation(sim);
const teamPolicySim = new Simulation({
  seed: 2,
  dronePolicy: new TeamWaypointMlDronePolicy({
    deploymentProvider: (_observation, _sim, droneCount) => {
      const output = new Float32Array(MAX_DRONES);
      for (let i = 0; i < droneCount; i++) {
        output[i] = droneCount === 1 ? 0 : -1 + (2 * i) / (droneCount - 1);
      }
      return output;
    },
    actionProvider: () => new Float32Array(MAX_DRONES * 3),
  }),
});
const firstTeamPolicySpawn = {
  x: Math.round(teamPolicySim.drones[0].x),
  y: Math.round(teamPolicySim.drones[0].y),
};

for (let i = 0; i < 60; i++) {
  teamPolicySim.step();
}

const stubSim = new Simulation({
  seed: 3,
  dronePolicy: createTeamMlStubPolicy(),
});
const firstStubSpawn = {
  x: Math.round(stubSim.drones[0].x),
  y: Math.round(stubSim.drones[0].y),
};

for (let i = 0; i < 60 * 5; i++) {
  stubSim.step();
}

if (!Number.isFinite(stats.time) || stats.time <= 0) {
  throw new Error('Simulation time did not advance.');
}

if (!(observation instanceof Float32Array) || observation.length === 0) {
  throw new Error('Drone observation encoder returned an invalid vector.');
}

if (!(teamObservation instanceof Float32Array) || teamObservation.length === 0) {
  throw new Error('Team observation encoder returned an invalid vector.');
}

if (!(deploymentObservation instanceof Float32Array) || deploymentObservation.length === 0) {
  throw new Error('Deployment observation encoder returned an invalid vector.');
}

console.log(JSON.stringify({
  ok: true,
  stats,
  observationSize: observation.length,
  teamObservationSize: teamObservation.length,
  deploymentObservationSize: deploymentObservation.length,
  firstTeamPolicySpawn,
  teamPolicyStats: teamPolicySim.getStats(),
  firstStubSpawn,
  stubStats: stubSim.getStats(),
}, null, 2));
