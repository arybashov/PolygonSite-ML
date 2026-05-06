import {
  ATTACK_COMMIT_DIST,
  DRONE_DETECT_R,
  MAX_DRONES,
  POLY,
} from '../constants.js';
import { clamp, distance } from '../math.js';
import { TeamWaypointMlDronePolicy } from './dronePolicy.js';

export function createTeamMlStubPolicy() {
  return new TeamWaypointMlDronePolicy({
    deploymentProvider: createSpreadDeployment,
    actionProvider: createCooperativeWaypoints,
    actionScale: DRONE_DETECT_R,
  });
}

function createSpreadDeployment(_observation, _sim, droneCount) {
  const output = new Float32Array(MAX_DRONES);

  for (let i = 0; i < droneCount; i++) {
    output[i] = droneCount === 1 ? 0 : -0.95 + (1.9 * i) / (droneCount - 1);
  }

  return output;
}

function createCooperativeWaypoints(_observation, sim) {
  const output = new Float32Array(MAX_DRONES * 3);
  const liveDrones = sim.drones.filter((drone) => {
    return drone.alive && drone.mode !== 'hit' && drone.mode !== 'intercepted';
  });
  const openTargets = sim.targets
    .map((target, index) => ({ target, index }))
    .filter((item) => !item.target.hit);

  if (openTargets.length === 0) {
    return output;
  }

  liveDrones.forEach((drone, slot) => {
    const target = selectTargetForDrone(sim, drone, openTargets, slot);
    const threat = findClosestThreat(sim, drone);
    const distToTarget = distance(drone.x, drone.y, target.x, target.y);
    const shouldEvade = threat && distToTarget > ATTACK_COMMIT_DIST;
    const goal = shouldEvade
      ? buildEvadePoint(sim, drone, target, threat)
      : buildGroupApproachPoint(sim, drone, target, slot, liveDrones.length);
    const dx = goal.x - drone.x;
    const dy = goal.y - drone.y;
    const offset = drone.id * 3;

    output[offset] = clamp(dx / DRONE_DETECT_R, -1, 1);
    output[offset + 1] = clamp(dy / DRONE_DETECT_R, -1, 1);
    output[offset + 2] = shouldEvade ? -1 : 1;
  });

  return output;
}

function selectTargetForDrone(sim, drone, openTargets, slot) {
  const assignedTarget = sim.targets[drone.targetIdx];
  if (assignedTarget && !assignedTarget.hit) {
    return assignedTarget;
  }

  const baseTarget = openTargets.find((item) => item.target.x === POLY / 2 && item.target.y === POLY / 2);
  if (baseTarget && slot % 3 === 0) {
    return baseTarget.target;
  }

  return openTargets[(drone.id + slot) % openTargets.length].target;
}

function buildGroupApproachPoint(sim, drone, target, slot, liveCount) {
  const dist = distance(drone.x, drone.y, target.x, target.y);
  if (dist < 320) {
    return { x: target.x, y: target.y };
  }

  const angle = liveCount <= 1 ? 0 : (slot / liveCount) * Math.PI * 2;
  const radius = Math.min(260, Math.max(120, dist * 0.14));
  const timePhase = sim.simTime * 0.15;

  return {
    x: clamp(target.x + Math.cos(angle + timePhase) * radius, 0, POLY),
    y: clamp(target.y + Math.sin(angle + timePhase) * radius, 0, POLY),
  };
}

function findClosestThreat(sim, drone) {
  let best = null;
  let bestDist = Infinity;

  sim.antidrones.forEach((anti) => {
    if (!anti.alive || anti.mode === 'base' || anti.mode === 'waiting') return;

    const dist = distance(drone.x, drone.y, anti.x, anti.y);
    if (dist > DRONE_DETECT_R * 1.2 || dist >= bestDist) return;

    best = anti;
    bestDist = dist;
  });

  return best;
}

function buildEvadePoint(sim, drone, target, threat) {
  const avoid = buildAntidroneAvoidance(sim, drone);
  const targetBias = {
    x: (target.x - drone.x) * 0.25,
    y: (target.y - drone.y) * 0.25,
  };
  const tangent = buildThreatTangent(drone, threat);

  return {
    x: clamp(drone.x + avoid.x + targetBias.x + tangent.x, 0, POLY),
    y: clamp(drone.y + avoid.y + targetBias.y + tangent.y, 0, POLY),
  };
}

function buildThreatTangent(drone, threat) {
  const dx = drone.x - threat.x;
  const dy = drone.y - threat.y;
  const len = Math.max(1, Math.sqrt(dx * dx + dy * dy));
  return {
    x: (-dy / len) * 220,
    y: (dx / len) * 220,
  };
}

function buildAntidroneAvoidance(sim, drone) {
  let ax = 0;
  let ay = 0;

  sim.antidrones.forEach((anti) => {
    if (!anti.alive || anti.mode === 'base' || anti.mode === 'waiting') return;

    const dist = distance(drone.x, drone.y, anti.x, anti.y);
    if (dist <= 1 || dist > DRONE_DETECT_R * 1.2) return;

    const force = (DRONE_DETECT_R * 1.2 - dist) / (DRONE_DETECT_R * 1.2);
    ax += ((drone.x - anti.x) / dist) * force * 320;
    ay += ((drone.y - anti.y) / dist) * force * 320;
  });

  return { x: ax, y: ay };
}
