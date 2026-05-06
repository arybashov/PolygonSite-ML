import {
  CAMERA_COUNT,
  DEPLOYMENT_MARGIN,
  DRONE_DETECT_R,
  MAX_ANTIDRONES,
  MAX_DRONES,
  POLY,
} from '../constants.js';
import { clamp, distance } from '../math.js';

export const DRONE_OBSERVATION_SCHEMA = Object.freeze({
  description: 'Drone-centric fixed vector for debugging and decentralized experiments.',
  outputAction: 'waypoint_delta',
  actionSize: 2,
  actionRange: [-1, 1],
  antiSlots: MAX_ANTIDRONES,
  targetSlots: MAX_DRONES,
});

export const TEAM_OBSERVATION_SCHEMA = Object.freeze({
  description: 'Centralized cooperative observation for one policy controlling all drones.',
  outputAction: 'waypoint_delta_and_intent_per_drone',
  actionSize: MAX_DRONES * 3,
  actionRange: [-1, 1],
  droneSlots: MAX_DRONES,
  antiSlots: MAX_ANTIDRONES,
  targetSlots: MAX_DRONES,
  cameraSlots: CAMERA_COUNT,
});

export const DEPLOYMENT_OBSERVATION_SCHEMA = Object.freeze({
  description: 'Episode-level observation for cooperative drone placement on the perimeter.',
  outputAction: 'perimeter_position_per_drone',
  actionSize: MAX_DRONES,
  actionRange: [-1, 1],
  droneSlots: MAX_DRONES,
  antiSlots: MAX_ANTIDRONES,
  targetSlots: MAX_DRONES,
  cameraSlots: CAMERA_COUNT,
});

export function encodeDeploymentObservation(sim, droneCount = sim.getParam('ndrones')) {
  const values = [];

  values.push(droneCount / MAX_DRONES);
  pushTeamTargets(values, sim);
  pushTeamCameras(values, sim);
  pushTeamAntidrones(values, sim);
  pushGlobalParams(values, sim);

  return new Float32Array(values);
}

export function decodePerimeterDeployment(output, droneCount, options = {}) {
  const spawns = [];
  const margin = options.margin ?? DEPLOYMENT_MARGIN;

  for (let i = 0; i < droneCount; i++) {
    const value = clamp(Number(output?.[i] ?? 0), -1, 1);
    spawns.push(perimeterValueToPoint(value, margin));
  }

  return spawns;
}

export function perimeterValueToPoint(value, margin = DEPLOYMENT_MARGIN) {
  const side = POLY - margin * 2;
  const perimeter = side * 4;
  const t = (clamp(value, -1, 1) + 1) / 2;
  let d = t * perimeter;

  if (d < side) {
    return { x: margin + d, y: margin };
  }

  d -= side;
  if (d < side) {
    return { x: POLY - margin, y: margin + d };
  }

  d -= side;
  if (d < side) {
    return { x: POLY - margin - d, y: POLY - margin };
  }

  d -= side;
  return { x: margin, y: POLY - margin - d };
}

export function encodeTeamObservation(sim) {
  const values = [];

  pushTeamDrones(values, sim);
  pushTeamAntidrones(values, sim);
  pushTeamTargets(values, sim);
  pushTeamCameras(values, sim);
  pushGlobalParams(values, sim);

  return new Float32Array(values);
}

export function decodeTeamWaypointActions(output, sim, options = {}) {
  const actions = new Map();
  const scale = options.scale ?? DRONE_DETECT_R;

  for (let i = 0; i < MAX_DRONES; i++) {
    const drone = sim.drones[i];
    if (!drone) continue;

    const offset = i * 3;
    const dx = clamp(Number(output?.[offset] ?? 0), -1, 1) * scale;
    const dy = clamp(Number(output?.[offset + 1] ?? 0), -1, 1) * scale;
    const intent = Number(output?.[offset + 2] ?? 0) < 0 ? 'evade' : 'attack';
    actions.set(drone.id, {
      kind: 'guidance',
      tx: clamp(drone.x + dx, 0, POLY),
      ty: clamp(drone.y + dy, 0, POLY),
      intent,
      mode: intent === 'evade' ? 'team-evade' : 'team-attack',
      cruiseKmh: sim.getParam('dspeed'),
      minKmh: Math.min(sim.getParam('dspeed') * 0.6, sim.getParam('dspeed') - 10),
    });
  }

  return actions;
}

export function encodeDroneObservation(sim, drone) {
  const values = [];

  pushSelf(values, sim, drone);
  pushAntidrones(values, sim, drone);
  pushTargets(values, sim, drone);
  pushGlobalParams(values, sim);

  return new Float32Array(values);
}

function pushTeamDrones(values, sim) {
  for (let i = 0; i < MAX_DRONES; i++) {
    const drone = sim.drones[i];
    if (!drone) {
      values.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      continue;
    }

    const target = sim.targets[drone.targetIdx];
    const tx = target ? target.x : drone.rtx;
    const ty = target ? target.y : drone.rty;

    values.push(
      1,
      drone.alive ? 1 : 0,
      drone.x / POLY,
      drone.y / POLY,
      Math.sin(drone.angle),
      Math.cos(drone.angle),
      drone.speed / (sim.getParam('dspeed') / 3.6),
      tx / POLY,
      ty / POLY,
      distance(drone.x, drone.y, tx, ty) / POLY,
      target?.hit ? 1 : 0,
      drone.evading ? 1 : 0,
      drone.mode === 'hit' || drone.mode === 'intercepted' ? 1 : 0,
    );
  }
}

function pushTeamAntidrones(values, sim) {
  for (let i = 0; i < MAX_ANTIDRONES; i++) {
    const anti = sim.antidrones[i];
    if (!anti) {
      values.push(0, 0, 0, 0, 0, 0, 0, 0);
      continue;
    }

    values.push(
      1,
      anti.alive ? 1 : 0,
      anti.mode === 'base' ? 0 : 1,
      anti.x / POLY,
      anti.y / POLY,
      Math.sin(anti.angle),
      Math.cos(anti.angle),
      anti.speed / (sim.getParam('aspeed') / 3.6),
    );
  }
}

function pushTeamTargets(values, sim) {
  for (let i = 0; i < MAX_DRONES; i++) {
    const target = sim.targets[i];
    if (!target) {
      values.push(0, 0, 0, 0, 0);
      continue;
    }

    values.push(
      1,
      target.x / POLY,
      target.y / POLY,
      target.hit ? 1 : 0,
      target.x === POLY / 2 && target.y === POLY / 2 ? 1 : 0,
    );
  }
}

function pushTeamCameras(values, sim) {
  for (let i = 0; i < CAMERA_COUNT; i++) {
    const camera = sim.cameras[i];
    if (!camera) {
      values.push(0, 0, 0, 0, 0, 0);
      continue;
    }

    values.push(
      1,
      camera.x / POLY,
      camera.y / POLY,
      Math.sin(camera.angle),
      Math.cos(camera.angle),
      camera.detected ? 1 : 0,
    );
  }
}

export function decodeWaypointAction(output, sim, drone, options = {}) {
  const scale = options.scale ?? DRONE_DETECT_R;
  const dx = clamp(Number(output?.[0] ?? 0), -1, 1) * scale;
  const dy = clamp(Number(output?.[1] ?? 0), -1, 1) * scale;

  return {
    kind: 'guidance',
    tx: clamp(drone.x + dx, 0, POLY),
    ty: clamp(drone.y + dy, 0, POLY),
    intent: 'attack',
    mode: 'ml',
    cruiseKmh: sim.getParam('dspeed'),
    minKmh: Math.min(sim.getParam('dspeed') * 0.6, sim.getParam('dspeed') - 10),
  };
}

function pushSelf(values, sim, drone) {
  const target = sim.targets[drone.targetIdx];
  const tx = target ? target.x : drone.rtx;
  const ty = target ? target.y : drone.rty;
  const targetDist = distance(drone.x, drone.y, tx, ty);

  values.push(
    drone.x / POLY,
    drone.y / POLY,
    Math.sin(drone.angle),
    Math.cos(drone.angle),
    drone.speed / (sim.getParam('dspeed') / 3.6),
    (tx - drone.x) / POLY,
    (ty - drone.y) / POLY,
    targetDist / POLY,
    drone.evading ? 1 : 0,
  );
}

function pushAntidrones(values, sim, drone) {
  const sorted = [...sim.antidrones].sort((a, b) => {
    return distance(drone.x, drone.y, a.x, a.y) - distance(drone.x, drone.y, b.x, b.y);
  });

  for (let i = 0; i < MAX_ANTIDRONES; i++) {
    const anti = sorted[i];
    if (!anti) {
      values.push(0, 0, 0, 0, 0, 0, 0);
      continue;
    }

    values.push(
      (anti.x - drone.x) / POLY,
      (anti.y - drone.y) / POLY,
      Math.sin(anti.angle),
      Math.cos(anti.angle),
      anti.speed / (sim.getParam('aspeed') / 3.6),
      anti.alive ? 1 : 0,
      anti.mode === 'base' ? 0 : 1,
    );
  }
}

function pushTargets(values, sim, drone) {
  const sorted = [...sim.targets].sort((a, b) => {
    return distance(drone.x, drone.y, a.x, a.y) - distance(drone.x, drone.y, b.x, b.y);
  });

  for (let i = 0; i < MAX_DRONES; i++) {
    const target = sorted[i];
    if (!target) {
      values.push(0, 0, 0, 0);
      continue;
    }

    values.push(
      (target.x - drone.x) / POLY,
      (target.y - drone.y) / POLY,
      distance(drone.x, drone.y, target.x, target.y) / POLY,
      target.hit ? 1 : 0,
    );
  }
}

function pushGlobalParams(values, sim) {
  values.push(
    sim.getParam('dspeed') / 300,
    sim.getParam('aspeed') / 500,
    sim.getParam('camrange') / POLY,
    sim.getParam('arange') / POLY,
    sim.simTime / 300,
  );
}
