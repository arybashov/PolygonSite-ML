import {
  ARRIVAL_R,
  ATTACK_COMMIT_DIST,
  DEFAULT_DRONE_TURN_RATE,
  DRONE_DETECT_R,
  DRONE_FOV,
  MS,
} from '../constants.js';
import { angleDiff, distance, inCone } from '../math.js';
import {
  decodeTeamWaypointActions,
  decodeWaypointAction,
  decodePerimeterDeployment,
  encodeDeploymentObservation,
  encodeDroneObservation,
  encodeTeamObservation,
} from '../ml/observations.js';

export class RuleBasedDronePolicy {
  getAction(sim, drone) {
    if (!drone.alive || drone.mode === 'hit' || drone.mode === 'intercepted') {
      return { kind: 'idle' };
    }

    if (sim.targets[drone.targetIdx]?.hit) {
      const closest = sim.findClosestUnattackedTarget(drone);
      if (!closest) return { kind: 'deactivate' };
      retarget(drone, closest);
    }

    const cruiseKmh = sim.getParam('dspeed');
    const minKmh = Math.min(cruiseKmh * 0.6, cruiseKmh - 10);
    const turnRateRad = DEFAULT_DRONE_TURN_RATE;
    const turnRadius = (drone.speed * MS) / turnRateRad;
    const deadZoneRadius = turnRadius * 1.5;

    if (drone.evadeCooldown > 0) {
      drone.evadeCooldown -= sim.dt;
    }

    const distToTarget = distance(drone.x, drone.y, drone.rtx, drone.rty);
    if (distToTarget < ATTACK_COMMIT_DIST) {
      clearEvasion(drone);
      drone.tx = drone.rtx;
      drone.ty = drone.rty;
      drone.mode = 'approach';
      if (distToTarget < ARRIVAL_R) return { kind: 'hitTarget' };
    } else {
      const threat = findThreat(sim, drone);
      if (threat) {
        if (!drone.evading) {
          const closest = sim.findClosestUnattackedTarget(drone);
          if (closest) retarget(drone, closest);
        }

        if (drone.threatId !== threat.id || !drone.evadeWpt || drone.evadeCooldown <= 0) {
          const result = computeEvadeWaypoint(sim, drone, threat);
          drone.evadeWpt = result.wpt;
          drone.predictPt = result.pred;
          drone.threatId = threat.id;
          drone.evadeCooldown = 0.5;
        }

        drone.evading = true;
        drone.tx = drone.evadeWpt.x;
        drone.ty = drone.evadeWpt.y;
        drone.mode = 'evade';
      } else {
        updateNonThreatFlight(drone, deadZoneRadius, turnRadius);
        if (drone.mode !== 'evade' && drone.mode !== 'reapproach') {
          if (distance(drone.x, drone.y, drone.rtx, drone.rty) < ARRIVAL_R) {
            return { kind: 'hitTarget' };
          }
        }
      }
    }

    return {
      kind: 'guidance',
      tx: drone.tx,
      ty: drone.ty,
      intent: drone.mode === 'evade' ? 'evade' : 'attack',
      mode: drone.mode,
      cruiseKmh,
      minKmh,
      turnRateRad,
    };
  }
}

export class WaypointMlDronePolicy {
  constructor({ actionProvider, fallback = new RuleBasedDronePolicy(), actionScale = DRONE_DETECT_R } = {}) {
    this.actionProvider = actionProvider;
    this.fallback = fallback;
    this.actionScale = actionScale;
  }

  getAction(sim, drone) {
    if (!this.actionProvider) {
      return this.fallback.getAction(sim, drone);
    }

    const observation = encodeDroneObservation(sim, drone);
    const output = this.actionProvider(observation, sim, drone);
    if (!output) {
      return this.fallback.getAction(sim, drone);
    }

    const action = decodeWaypointAction(output, sim, drone, { scale: this.actionScale });
    drone.tx = action.tx;
    drone.ty = action.ty;
    drone.mode = action.mode;
    return action;
  }
}

export class TeamWaypointMlDronePolicy {
  constructor({
    actionProvider,
    deploymentProvider,
    fallback = new RuleBasedDronePolicy(),
    actionScale = DRONE_DETECT_R,
  } = {}) {
    this.actionProvider = actionProvider;
    this.deploymentProvider = deploymentProvider;
    this.fallback = fallback;
    this.actionScale = actionScale;
  }

  getDeployment(sim, droneCount) {
    if (!this.deploymentProvider) return null;

    const observation = encodeDeploymentObservation(sim, droneCount);
    const output = this.deploymentProvider(observation, sim, droneCount);
    if (!output) return null;

    return decodePerimeterDeployment(output, droneCount);
  }

  getTeamActions(sim) {
    if (!this.actionProvider) {
      return buildFallbackActions(sim, this.fallback);
    }

    const observation = encodeTeamObservation(sim);
    const output = this.actionProvider(observation, sim);
    if (!output) {
      return buildFallbackActions(sim, this.fallback);
    }

    const actions = decodeTeamWaypointActions(output, sim, { scale: this.actionScale });

    sim.drones.forEach((drone) => {
      const maintenanceAction = getMaintenanceAction(sim, drone);
      if (maintenanceAction) {
        actions.set(drone.id, maintenanceAction);
        return;
      }

      const action = actions.get(drone.id);
      if (!action) {
        actions.set(drone.id, this.fallback.getAction(sim, drone));
        return;
      }

      drone.tx = action.tx;
      drone.ty = action.ty;
      drone.mode = action.mode;
    });

    return actions;
  }
}

function buildFallbackActions(sim, fallback) {
  const actions = new Map();
  sim.drones.forEach((drone) => {
    actions.set(drone.id, fallback.getAction(sim, drone));
  });
  return actions;
}

function getMaintenanceAction(sim, drone) {
  if (!drone.alive || drone.mode === 'hit' || drone.mode === 'intercepted') {
    return { kind: 'idle' };
  }

  if (sim.targets[drone.targetIdx]?.hit) {
    const closest = sim.findClosestUnattackedTarget(drone);
    if (!closest) return { kind: 'deactivate' };
    retarget(drone, closest);
  }

  const distToTarget = distance(drone.x, drone.y, drone.rtx, drone.rty);
  if (distToTarget < ARRIVAL_R) {
    return { kind: 'hitTarget' };
  }

  return null;
}

function retarget(drone, closest) {
  drone.targetIdx = closest.i;
  drone.rtx = closest.t.x;
  drone.rty = closest.t.y;
  drone.tx = closest.t.x;
  drone.ty = closest.t.y;
  drone.mode = 'approach';
}

function clearEvasion(drone) {
  drone.evading = false;
  drone.evadeWpt = null;
  drone.predictPt = null;
  drone.threatId = null;
}

function findThreat(sim, drone) {
  return sim.antidrones.find((anti) => {
    return anti.alive
      && anti.mode !== 'base'
      && inCone(drone.x, drone.y, drone.angle, DRONE_FOV, anti.x, anti.y, DRONE_DETECT_R);
  }) ?? null;
}

function computeEvadeWaypoint(sim, drone, threat) {
  const droneSpeed = drone.speed * MS;
  const antiSpeed = sim.getParam('aspeed') / 3.6 * MS;
  const avx = Math.cos(threat.angle) * antiSpeed;
  const avy = Math.sin(threat.angle) * antiSpeed;
  const distDroneAnti = distance(drone.x, drone.y, threat.x, threat.y);
  const timeToCross = Math.max(1, distDroneAnti / (droneSpeed + antiSpeed));
  const predX = threat.x + avx * timeToCross;
  const predY = threat.y + avy * timeToCross;
  const lineAngle = Math.atan2(predY - threat.y, predX - threat.x);
  const perpAngle = lineAngle + Math.PI / 2;
  const safeR = Math.max(150, distDroneAnti * 0.4);
  const midX = (drone.x + predX) / 2;
  const midY = (drone.y + predY) / 2;
  const c1x = midX + Math.cos(perpAngle) * safeR;
  const c1y = midY + Math.sin(perpAngle) * safeR;
  const c2x = midX - Math.cos(perpAngle) * safeR;
  const c2y = midY - Math.sin(perpAngle) * safeR;
  const d1 = distance(c1x, c1y, drone.rtx, drone.rty);
  const d2 = distance(c2x, c2y, drone.rtx, drone.rty);

  return {
    wpt: d1 < d2 ? { x: c1x, y: c1y } : { x: c2x, y: c2y },
    pred: { x: predX, y: predY },
  };
}

function updateNonThreatFlight(drone, deadZoneRadius, turnRadius) {
  if (drone.evading) {
    clearEvasion(drone);
    drone.tx = drone.x + Math.cos(drone.angle) * turnRadius * 2;
    drone.ty = drone.y + Math.sin(drone.angle) * turnRadius * 2;
    drone.mode = 'reapproach';
    return;
  }

  if (drone.mode === 'reapproach') {
    if (distance(drone.x, drone.y, drone.tx, drone.ty) < turnRadius) {
      drone.tx = drone.rtx;
      drone.ty = drone.rty;
      drone.mode = 'approach';
    }
    return;
  }

  if (drone.mode !== 'bypass') drone.mode = 'approach';
  const rdx = drone.rtx - drone.x;
  const rdy = drone.rty - drone.y;
  const rdist = distance(drone.x, drone.y, drone.rtx, drone.rty);
  const targetAngle = Math.atan2(rdy, rdx);
  const headingError = Math.abs(angleDiff(drone.angle, targetAngle));
  const inDeadZone = rdist < deadZoneRadius && headingError > Math.PI / 3;

  if (inDeadZone && drone.mode !== 'bypass') {
    drone.deadZone = true;
    drone.mode = 'bypass';
    const cross = Math.cos(drone.angle) * rdy - Math.sin(drone.angle) * rdx;
    const side = cross > 0 ? 1 : -1;
    drone.tx = drone.rtx + Math.sin(drone.angle) * turnRadius * 1.5 * side;
    drone.ty = drone.rty - Math.cos(drone.angle) * turnRadius * 1.5 * side;
    return;
  }

  if (!inDeadZone) {
    drone.deadZone = false;
    if (drone.mode === 'bypass') {
      if (distance(drone.x, drone.y, drone.tx, drone.ty) < ARRIVAL_R * 4) {
        drone.mode = 'approach';
        drone.tx = drone.rtx;
        drone.ty = drone.rty;
      }
    } else {
      drone.tx = drone.rtx;
      drone.ty = drone.rty;
    }
  }
}
