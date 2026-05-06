import {
  ANTI_FOV,
  ARRIVAL_R,
  BASE_X,
  BASE_Y,
  CAMERA_COUNT,
  CAM_FOV,
  CENTER_ZONE,
  DEFAULT_ANTI_TURN_RATE,
  DEFAULT_DRONE_TURN_RATE,
  DEFAULT_PARAMS,
  DEPLOYMENT_MARGIN,
  DRONE_COLS,
  DT,
  INTERCEPT_R,
  MS,
  POLY,
  TRAIL_LEN,
} from './constants.js';
import { angleDiff, clamp, createRng, distance, inCone, lerp } from './math.js';
import { RuleBasedDronePolicy } from './policies/dronePolicy.js';

export class Simulation {
  constructor({ params = {}, seed = Date.now(), dronePolicy = new RuleBasedDronePolicy() } = {}) {
    this.dt = DT;
    this.params = { ...DEFAULT_PARAMS, ...params };
    this.seed = seed;
    this.rng = createRng(seed);
    this.dronePolicy = dronePolicy;

    this.simTime = 0;
    this.drones = [];
    this.targets = [];
    this.cameras = [];
    this.antidrones = [];
    this.explosions = [];

    this.reset({ seed });
  }

  setParams(params) {
    for (const [key, value] of Object.entries(params)) {
      if (Number.isFinite(Number(value))) {
        this.params[key] = Number(value);
      }
    }
  }

  getParam(id) {
    return Number(this.params[id]);
  }

  setDronePolicy(policy) {
    this.dronePolicy = policy;
  }

  reset({ seed = Date.now() } = {}) {
    this.seed = seed;
    this.rng = createRng(seed);
    this.simTime = 0;
    this.drones = [];
    this.targets = [];
    this.cameras = [];
    this.antidrones = [];
    this.explosions = [];

    const droneCount = Math.floor(this.getParam('ndrones'));
    const antiCount = Math.floor(this.getParam('nanti'));
    const targets = Array.from({ length: Math.max(0, droneCount - 1) }, () => this.spawnTarget());
    targets.push({ x: BASE_X, y: BASE_Y, hit: false });
    this.targets.push(...targets);
    this.cameras = this.placeCamerasAroundTargets();
    for (let i = 0; i < antiCount; i++) {
      this.antidrones.push(this.makeAnti(i, antiCount));
    }

    const spawns = this.getDeploymentSpawns(droneCount);
    const assigned = new Set();
    spawns.forEach((spawn, i) => {
      let bestIdx = -1;
      let bestDist = Infinity;
      targets.forEach((target, ti) => {
        if (assigned.has(ti)) return;
        const dist = distance(spawn.x, spawn.y, target.x, target.y);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = ti;
        }
      });

      assigned.add(bestIdx);
      this.drones.push(this.makeDrone(i, spawn, targets[bestIdx], bestIdx));
    });
  }

  step() {
    this.simTime += this.dt;
    const teamActions = this.dronePolicy?.getTeamActions?.(this) ?? null;
    this.drones.forEach((drone) => this.updateDrone(drone, teamActions?.get(drone.id)));
    this.updateCameras();
    this.updateAnti();
    this.updateExplosions();
  }

  getStats() {
    return {
      drones: this.drones.filter((d) => d.alive && d.mode !== 'hit' && d.mode !== 'intercepted').length,
      hits: this.drones.filter((d) => d.mode === 'hit').length,
      intercepted: this.drones.filter((d) => d.mode === 'intercepted').length,
      activeAnti: this.antidrones.filter((a) => a.alive && a.mode !== 'base').length,
      time: Math.round(this.simTime),
    };
  }

  getState() {
    return {
      seed: this.seed,
      time: this.simTime,
      params: { ...this.params },
      drones: this.drones.map(copyAgent),
      antidrones: this.antidrones.map(copyAgent),
      targets: this.targets.map((target, id) => ({ id, ...target })),
      cameras: this.cameras.map((camera) => ({
        id: camera.id,
        x: camera.x,
        y: camera.y,
        angle: camera.angle,
        detectedDroneId: camera.detected?.id ?? null,
      })),
    };
  }

  findClosestUnattackedTarget(drone) {
    let best = null;
    let bestDist = Infinity;
    this.targets.forEach((target, i) => {
      if (target.hit) return;
      const dist = distance(drone.x, drone.y, target.x, target.y);
      if (dist < bestDist) {
        bestDist = dist;
        best = { t: target, i };
      }
    });
    return best;
  }

  markDroneHitTarget(drone) {
    if (this.targets[drone.targetIdx]) {
      this.targets[drone.targetIdx].hit = true;
    }
    this.explosions.push({ x: drone.x, y: drone.y, t: 1.0 });
    drone.alive = false;
    drone.trail = [];
    drone.mode = 'hit';
  }

  markDroneIntercepted(drone) {
    this.explosions.push({ x: drone.x, y: drone.y, t: 1.0 });
    drone.alive = false;
    drone.mode = 'intercepted';
    drone.trail = [];
  }

  getDeploymentSpawns(droneCount) {
    const policySpawns = this.dronePolicy?.getDeployment?.(this, droneCount);
    if (!Array.isArray(policySpawns) || policySpawns.length < droneCount) {
      return Array.from({ length: droneCount }, () => this.spawnEdge());
    }

    return policySpawns.slice(0, droneCount).map((spawn) => {
      if (!Number.isFinite(spawn?.x) || !Number.isFinite(spawn?.y)) {
        return this.spawnEdge();
      }

      return {
        x: clamp(spawn.x, DEPLOYMENT_MARGIN, POLY - DEPLOYMENT_MARGIN),
        y: clamp(spawn.y, DEPLOYMENT_MARGIN, POLY - DEPLOYMENT_MARGIN),
      };
    });
  }

  spawnEdge() {
    const side = this.rng.int(4);
    const margin = DEPLOYMENT_MARGIN;
    if (side === 0) return { x: this.rng.range(margin, POLY - margin), y: margin };
    if (side === 1) return { x: this.rng.range(margin, POLY - margin), y: POLY - margin };
    if (side === 2) return { x: margin, y: this.rng.range(margin, POLY - margin) };
    return { x: POLY - margin, y: this.rng.range(margin, POLY - margin) };
  }

  spawnTarget() {
    const angle = this.rng.range(0, Math.PI * 2);
    const radius = this.rng.range(100, CENTER_ZONE);
    return {
      x: POLY / 2 + Math.cos(angle) * radius,
      y: POLY / 2 + Math.sin(angle) * radius,
      hit: false,
    };
  }

  makeDrone(id, spawn, target, targetIdx) {
    return {
      id,
      x: spawn.x,
      y: spawn.y,
      angle: Math.atan2(POLY / 2 - spawn.y, POLY / 2 - spawn.x),
      speed: this.getParam('dspeed') / 3.6,
      tx: target.x,
      ty: target.y,
      rtx: target.x,
      rty: target.y,
      targetIdx,
      trail: [],
      mode: 'approach',
      intent: 'attack',
      deadZone: false,
      alive: true,
      col: DRONE_COLS[id % DRONE_COLS.length],
      evading: false,
      evadeWpt: null,
      predictPt: null,
      threatId: null,
      evadeCooldown: 0,
    };
  }

  makeCamera(id, x, y, angle) {
    return { id, x, y, angle, detected: null, cooldown: 0 };
  }

  placeCamerasAroundTargets() {
    const radius = POLY * 0.15;
    return Array.from({ length: CAMERA_COUNT }, (_, i) => {
      const angle = (i / CAMERA_COUNT) * Math.PI * 2;
      const x = POLY / 2 + Math.cos(angle) * radius;
      const y = POLY / 2 + Math.sin(angle) * radius;
      return this.makeCamera(i, x, y, angle);
    });
  }

  makeAnti(id, total) {
    const angle = total > 1 ? (id / total) * Math.PI * 2 : 0;
    const radius = total > 1 ? 40 : 0;
    return {
      id,
      x: BASE_X + Math.cos(angle) * radius,
      y: BASE_Y + Math.sin(angle) * radius,
      angle: 0,
      speed: 0,
      target: null,
      lastKnownX: null,
      lastKnownY: null,
      mode: 'base',
      trail: [],
      alive: true,
      launchDelay: 0,
      pendingTarget: null,
    };
  }

  updateDrone(drone, action = null) {
    const resolvedAction = action ?? this.dronePolicy.getAction(this, drone);
    this.applyDroneAction(drone, resolvedAction);
  }

  applyDroneAction(drone, action) {
    if (!action || action.kind === 'idle') return;
    if (action.kind === 'hitTarget') {
      this.markDroneHitTarget(drone);
      return;
    }
    if (action.kind === 'deactivate') {
      drone.alive = false;
      drone.trail = [];
      return;
    }
    if (action.kind !== 'guidance') return;

    drone.intent = action.intent ?? drone.intent ?? 'attack';
    if (action.mode) {
      drone.mode = action.mode;
      drone.evading = action.intent === 'evade';
    }

    const turnRate = action.turnRateRad ?? DEFAULT_DRONE_TURN_RATE;
    const maxTurn = turnRate * this.dt;
    const headingTarget = Math.atan2(action.ty - drone.y, action.tx - drone.x);
    const diff = angleDiff(drone.angle, headingTarget);
    drone.angle += clamp(diff, -maxTurn, maxTurn);

    const dot = Math.max(0, Math.cos(diff));
    const cruiseSpeed = (action.cruiseKmh ?? this.getParam('dspeed')) / 3.6;
    const minSpeed = (action.minKmh ?? Math.min(this.getParam('dspeed') * 0.6, this.getParam('dspeed') - 10)) / 3.6;
    drone.speed = lerp(drone.speed, lerp(minSpeed, cruiseSpeed, dot), this.dt / 0.5);
    drone.x += Math.cos(drone.angle) * drone.speed * this.dt * MS;
    drone.y += Math.sin(drone.angle) * drone.speed * this.dt * MS;
    drone.x = clamp(drone.x, 0, POLY);
    drone.y = clamp(drone.y, 0, POLY);

    drone.trail.push({ x: drone.x, y: drone.y });
    if (drone.trail.length > TRAIL_LEN) drone.trail.shift();
  }

  updateCameras() {
    const camRange = this.getParam('camrange');
    const delay = this.getParam('adelay');

    this.cameras.forEach((camera) => {
      if (camera.cooldown > 0) {
        camera.cooldown -= this.dt;
        camera.detected = null;
        return;
      }

      camera.detected = null;
      let best = null;
      let bestDist = camRange;
      this.drones.forEach((drone) => {
        if (!drone.alive || drone.mode === 'hit' || drone.mode === 'intercepted') return;
        if (!inCone(camera.x, camera.y, camera.angle, CAM_FOV, drone.x, drone.y, camRange)) return;

        const dist = distance(camera.x, camera.y, drone.x, drone.y);
        if (dist < bestDist) {
          bestDist = dist;
          best = drone;
        }
      });

      if (!best) return;

      camera.detected = best;
      const freeAnti = this.antidrones.find((anti) => anti.alive && anti.mode === 'base' && anti.launchDelay === 0);
      if (freeAnti) {
        freeAnti.launchDelay = delay;
        freeAnti.pendingTarget = best;
        freeAnti.mode = 'waiting';
        camera.cooldown = 3;
      }
    });
  }

  findNewAntiTarget(anti) {
    const antiRange = this.getParam('arange');
    const inView = this.drones.filter((drone) => {
      return drone.alive
        && drone.mode !== 'hit'
        && drone.mode !== 'intercepted'
        && inCone(anti.x, anti.y, anti.angle, ANTI_FOV, drone.x, drone.y, antiRange);
    });

    if (inView.length > 0) {
      inView.sort((a, b) => distance(anti.x, anti.y, a.x, a.y) - distance(anti.x, anti.y, b.x, b.y));
      return inView[0];
    }

    const signal = this.cameras.find((camera) => {
      return camera.detected && camera.detected.alive && camera.detected.mode !== 'hit' && camera.detected.mode !== 'intercepted';
    });

    return signal ? signal.detected : null;
  }

  updateAnti() {
    const antiSpeedMS = this.getParam('aspeed') / 3.6;
    const antiRange = this.getParam('arange');
    const maxTurn = DEFAULT_ANTI_TURN_RATE * this.dt;

    this.antidrones.forEach((anti) => {
      if (!anti.alive) return;

      if (anti.mode === 'waiting') {
        anti.launchDelay -= this.dt;
        if (anti.launchDelay <= 0) {
          anti.launchDelay = 0;
          if (anti.pendingTarget && anti.pendingTarget.alive && anti.pendingTarget.mode !== 'hit' && anti.pendingTarget.mode !== 'intercepted') {
            anti.target = anti.pendingTarget;
            anti.lastKnownX = anti.target.x;
            anti.lastKnownY = anti.target.y;
            anti.angle = Math.atan2(anti.target.y - anti.y, anti.target.x - anti.x);
            anti.mode = 'intercept';
          } else {
            const newTarget = this.findNewAntiTarget(anti);
            if (newTarget) {
              anti.target = newTarget;
              anti.lastKnownX = newTarget.x;
              anti.lastKnownY = newTarget.y;
              anti.angle = Math.atan2(newTarget.y - anti.y, newTarget.x - anti.x);
              anti.mode = 'intercept';
            } else {
              anti.mode = 'base';
            }
          }
          anti.pendingTarget = null;
        }
        return;
      }

      if (anti.mode === 'base') return;

      if (anti.mode !== 'base' && anti.mode !== 'waiting') {
        const inView = this.drones.filter((drone) => {
          return drone.alive
            && drone.mode !== 'hit'
            && drone.mode !== 'intercepted'
            && inCone(anti.x, anti.y, anti.angle, ANTI_FOV, drone.x, drone.y, antiRange);
        });
        if (inView.length > 0) {
          inView.sort((a, b) => distance(anti.x, anti.y, a.x, a.y) - distance(anti.x, anti.y, b.x, b.y));
          const closest = inView[0];
          if (anti.target !== closest) {
            anti.target = closest;
            anti.lastKnownX = closest.x;
            anti.lastKnownY = closest.y;
          }
          anti.mode = 'chase';
        }
      }

      if (anti.mode === 'intercept' || anti.mode === 'chase' || anti.mode === 'lastknown') {
        this.updateActiveAnti(anti, antiSpeedMS, maxTurn, antiRange);
      } else if (anti.mode === 'return') {
        this.updateReturningAnti(anti, antiSpeedMS, maxTurn);
      }

      anti.trail.push({ x: anti.x, y: anti.y });
      if (anti.trail.length > TRAIL_LEN) anti.trail.shift();
    });
  }

  updateActiveAnti(anti, antiSpeedMS, maxTurn, antiRange) {
    if (anti.mode !== 'lastknown') {
      if (!anti.target || !anti.target.alive || anti.target.mode === 'hit' || anti.target.mode === 'intercepted') {
        const newTarget = this.findNewAntiTarget(anti);
        if (newTarget) {
          anti.target = newTarget;
          anti.lastKnownX = newTarget.x;
          anti.lastKnownY = newTarget.y;
          anti.mode = 'intercept';
        } else {
          anti.mode = 'return';
          anti.target = null;
          return;
        }
      } else if (inCone(anti.x, anti.y, anti.angle, ANTI_FOV, anti.target.x, anti.target.y, antiRange)) {
        anti.lastKnownX = anti.target.x;
        anti.lastKnownY = anti.target.y;
        anti.mode = 'chase';
      } else {
        anti.mode = 'lastknown';
      }
    }

    let gx;
    let gy;
    if (anti.mode === 'chase' && anti.target && anti.target.alive) {
      gx = anti.target.x;
      gy = anti.target.y;
    } else if (anti.lastKnownX !== null) {
      gx = anti.lastKnownX;
      gy = anti.lastKnownY;
    } else {
      anti.mode = 'return';
      return;
    }

    if (anti.target && anti.target.alive && anti.target.mode !== 'hit' && anti.target.mode !== 'intercepted') {
      if (distance(anti.x, anti.y, anti.target.x, anti.target.y) < INTERCEPT_R) {
        this.markDroneIntercepted(anti.target);
        anti.alive = false;
        return;
      }
    }

    if (anti.mode === 'lastknown' && distance(anti.x, anti.y, gx, gy) < 30) {
      const newTarget = this.findNewAntiTarget(anti);
      if (newTarget) {
        anti.target = newTarget;
        anti.lastKnownX = newTarget.x;
        anti.lastKnownY = newTarget.y;
        anti.mode = 'intercept';
      } else {
        anti.mode = 'return';
        anti.target = null;
        return;
      }
    }

    const diff = angleDiff(anti.angle, Math.atan2(gy - anti.y, gx - anti.x));
    anti.angle += clamp(diff, -maxTurn, maxTurn);
    anti.speed = lerp(anti.speed, antiSpeedMS, this.dt / 0.3);
    anti.x += Math.cos(anti.angle) * anti.speed * this.dt * MS;
    anti.y += Math.sin(anti.angle) * anti.speed * this.dt * MS;
    anti.x = clamp(anti.x, 0, POLY);
    anti.y = clamp(anti.y, 0, POLY);
  }

  updateReturningAnti(anti, antiSpeedMS, maxTurn) {
    const homeX = BASE_X + Math.cos(anti.id / this.antidrones.length * Math.PI * 2) * 40;
    const homeY = BASE_Y + Math.sin(anti.id / this.antidrones.length * Math.PI * 2) * 40;
    const distHome = distance(anti.x, anti.y, homeX, homeY);
    if (distHome < 15) {
      anti.x = homeX;
      anti.y = homeY;
      anti.mode = 'base';
      anti.speed = 0;
      anti.trail = [];
      return;
    }

    const diff = angleDiff(anti.angle, Math.atan2(homeY - anti.y, homeX - anti.x));
    anti.angle += clamp(diff, -maxTurn, maxTurn);
    anti.speed = lerp(anti.speed, antiSpeedMS * 0.7, this.dt / 0.3);
    anti.x += Math.cos(anti.angle) * anti.speed * this.dt * MS;
    anti.y += Math.sin(anti.angle) * anti.speed * this.dt * MS;
    anti.x = clamp(anti.x, 0, POLY);
    anti.y = clamp(anti.y, 0, POLY);
  }

  updateExplosions() {
    for (let i = this.explosions.length - 1; i >= 0; i--) {
      this.explosions[i].t -= this.dt * 1.5;
      if (this.explosions[i].t <= 0) {
        this.explosions.splice(i, 1);
      }
    }
  }
}

function copyAgent(agent) {
  return {
    id: agent.id,
    x: agent.x,
    y: agent.y,
    angle: agent.angle,
    speed: agent.speed,
    mode: agent.mode,
    intent: agent.intent ?? null,
    alive: agent.alive,
    targetId: agent.target?.id ?? agent.targetIdx ?? null,
  };
}
