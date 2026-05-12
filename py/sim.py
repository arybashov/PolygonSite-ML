"""Python port of simulation.js + observations.js for RL training."""
import math
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
DT                     = 1 / 60
POLY                   = 2500.0
MS                     = 1
ARRIVAL_R              = 20.0
INTERCEPT_R            = 25.0
ATTACK_COMMIT_DIST     = 250.0
CAM_FOV                = 50 * math.pi / 180
ANTI_FOV               = 50 * math.pi / 180
BASE_X                 = POLY / 2
BASE_Y                 = POLY / 2
CAMERA_COUNT           = 10
CENTER_ZONE            = 800.0
DEPLOYMENT_MARGIN      = 50.0
DEFAULT_DRONE_TURN_RATE = 30 * (1 - (50 - 10) / 190 * 0.5) * math.pi / 180
DEFAULT_ANTI_TURN_RATE  = 60 * math.pi / 180
DRONE_DETECT_R         = 600.0
MAX_DRONES             = 15
MAX_ANTIDRONES         = 15

OBS_DIM    = 455   # 15*13 + 15*8 + 15*5 + 10*6 + 5
ACTION_DIM = 45    # 15 * 3
STEP_SKIP  = 1     # no frame skip — lam=1.0 handles credit assignment

DEFAULT_PARAMS = dict(
    dspeed=200, aspeed=300, camrange=900,
    arange=600, ndrones=5, nanti=5, adelay=5,
)

# ── Math ─────────────────────────────────────────────────────────────────────
def _clamp(v, lo, hi):  return max(lo, min(hi, v))
def _lerp(a, b, t):     return a + (b - a) * t
def _dist(ax, ay, bx, by): return math.hypot(ax - bx, ay - by)

def _angle_diff(a, b):
    d = b - a
    while d >  math.pi: d -= math.pi * 2
    while d < -math.pi: d += math.pi * 2
    return d

def _in_cone(ox, oy, ang, fov, tx, ty, rng):
    dx, dy = tx - ox, ty - oy
    d = math.hypot(dx, dy)
    return d <= rng and abs(_angle_diff(ang, math.atan2(dy, dx))) <= fov / 2

# ── RNG (xorshift32 — matches JS createRng exactly) ──────────────────────────
class _Rng:
    def __init__(self, seed):
        s = int(seed) & 0xFFFFFFFF
        self.state = s if s else 0x9e3779b9

    def next(self):
        s = self.state
        s = (s ^ ((s << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
        s = (s ^ (s >> 17)) & 0xFFFFFFFF
        s = (s ^ ((s << 5)  & 0xFFFFFFFF)) & 0xFFFFFFFF
        self.state = s
        return s / 0x100000000

    def rrange(self, lo, hi): return lo + self.next() * (hi - lo)
    def rint(self, n):        return int(self.next() * n)

# ── Simulation ────────────────────────────────────────────────────────────────
class Simulation:
    def __init__(self, params=None, seed=0):
        self.params   = {**DEFAULT_PARAMS, **(params or {})}
        self.seed     = seed
        self.rng      = _Rng(seed)
        self.sim_time = 0.0
        self.drones      = []
        self.antidrones  = []
        self.targets     = []
        self.cameras     = []

    def _p(self, k): return float(self.params[k])

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.rng      = _Rng(self.seed)
        self.sim_time = 0.0
        self.drones      = []
        self.antidrones  = []
        self.targets     = []
        self.cameras     = []

        nd = int(self._p('ndrones'))
        na = int(self._p('nanti'))

        targets = [self._spawn_target() for _ in range(max(0, nd - 1))]
        targets.append({'x': BASE_X, 'y': BASE_Y, 'hit': False})
        self.targets = targets
        self.cameras = self._place_cameras()
        for i in range(na):
            self.antidrones.append(self._make_anti(i, na))

        spawns   = [self._spawn_edge() for _ in range(nd)]
        assigned = set()
        for i, sp in enumerate(spawns):
            bi, bd = -1, math.inf
            for ti, t in enumerate(targets):
                if ti in assigned: continue
                d = _dist(sp['x'], sp['y'], t['x'], t['y'])
                if d < bd: bd, bi = d, ti
            assigned.add(bi)
            self.drones.append(self._make_drone(i, sp, targets[bi], bi))

        return self._encode_obs()

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        """Run STEP_SKIP physics steps for one RL step. Action applied once."""
        total_reward = 0.0
        obs = None
        done = False
        for skip_i in range(STEP_SKIP):
            obs, reward, done, _ = self._physics_step(action if skip_i == 0 else None)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, {}

    def _physics_step(self, action):
        """One physics tick. action=None reuses current waypoints."""
        self.sim_time += DT
        prev_modes = [d['mode'] for d in self.drones]

        scale = DRONE_DETECT_R
        for drone in self.drones:
            if not drone['alive'] or drone['mode'] in ('hit', 'intercepted'):
                continue
            # retarget if assigned target was hit
            if self.targets[drone['targetIdx']]['hit']:
                best = self._closest_unattacked(drone)
                if not best:
                    drone['alive'] = False
                    drone['mode']  = 'intercepted'
                    continue
                drone.update(targetIdx=best['i'],
                             rtx=best['t']['x'], rty=best['t']['y'],
                             tx=best['t']['x'],  ty=best['t']['y'],
                             mode='approach')
            # hit check
            if _dist(drone['x'], drone['y'], drone['rtx'], drone['rty']) < ARRIVAL_R:
                self._hit(drone)
                continue
            # apply RL action (only on first physics step of RL step)
            if action is not None:
                i  = drone['id']
                dx = _clamp(float(action[i * 3]),     -1, 1) * scale
                dy = _clamp(float(action[i * 3 + 1]), -1, 1) * scale
                intent = float(action[i * 3 + 2])
                target = self.targets[drone['targetIdx']]
                targetX = target['x'] if target else drone['rtx']
                targetY = target['y'] if target else drone['rty']
                rdx = targetX - drone['x']
                rdy = targetY - drone['y']
                dist = math.hypot(rdx, rdy) or 1.0
                baseTx = drone['x'] + (rdx / dist) * scale
                baseTy = drone['y'] + (rdy / dist) * scale
                self._guidance(drone,
                               _clamp(baseTx + dx, 0, POLY),
                               _clamp(baseTy + dy, 0, POLY),
                               'team-evade' if intent < 0 else 'team-attack',
                               intent < 0)

        self._update_cameras()
        self._update_anti()

        reward = 0.0
        for d, pm in zip(self.drones, prev_modes):
            if d['mode'] != pm:
                if d['mode'] == 'hit': reward += 1.0

        done = all(not d['alive'] for d in self.drones) or self.sim_time > 500.0
        if done and self.sim_time > 500.0:
            for d in self.drones:
                if d['alive'] and d['mode'] not in ('hit', 'intercepted'):
                    reward -= 0.5
        return self._encode_obs(), reward, done, {}

    # ── drone helpers ─────────────────────────────────────────────────────────
    def _guidance(self, d, tx, ty, mode, evading):
        d['mode'] = mode; d['evading'] = evading
        d['tx'] = tx;     d['ty'] = ty
        turn = DEFAULT_DRONE_TURN_RATE * DT
        heading = math.atan2(ty - d['y'], tx - d['x'])
        diff    = _angle_diff(d['angle'], heading)
        d['angle'] += _clamp(diff, -turn, turn)
        dot    = max(0.0, math.cos(diff))
        cruise = self._p('dspeed') / 3.6
        mins   = min(cruise * 0.6, cruise - 10 / 3.6)
        d['speed'] = _lerp(d['speed'], _lerp(mins, cruise, dot), DT / 0.5)
        d['x'] = _clamp(d['x'] + math.cos(d['angle']) * d['speed'] * DT, 0, POLY)
        d['y'] = _clamp(d['y'] + math.sin(d['angle']) * d['speed'] * DT, 0, POLY)

    def _hit(self, d):
        self.targets[d['targetIdx']]['hit'] = True
        d['alive'] = False; d['mode'] = 'hit'

    def _intercept(self, d):
        d['alive'] = False; d['mode'] = 'intercepted'

    def _closest_unattacked(self, drone):
        best, bd = None, math.inf
        for i, t in enumerate(self.targets):
            if t['hit']: continue
            d = _dist(drone['x'], drone['y'], t['x'], t['y'])
            if d < bd: bd = d; best = {'t': t, 'i': i}
        return best

    # ── cameras ───────────────────────────────────────────────────────────────
    def _update_cameras(self):
        cr    = self._p('camrange')
        delay = self._p('adelay')
        for cam in self.cameras:
            if cam['cooldown'] > 0:
                cam['cooldown'] -= DT; cam['detected'] = None; continue
            cam['detected'] = None
            best, bd = None, cr
            for d in self.drones:
                if not d['alive'] or d['mode'] in ('hit', 'intercepted'): continue
                if not _in_cone(cam['x'], cam['y'], cam['angle'], CAM_FOV,
                                d['x'], d['y'], cr): continue
                dist_ = _dist(cam['x'], cam['y'], d['x'], d['y'])
                if dist_ < bd: bd = dist_; best = d
            if best is None: continue
            cam['detected'] = best
            free = next((a for a in self.antidrones
                         if a['alive'] and a['mode'] == 'base'
                         and a['launchDelay'] == 0), None)
            if free:
                free['launchDelay'] = delay
                free['pendingTarget'] = best
                free['mode'] = 'waiting'
                cam['cooldown'] = 3.0

    # ── antidrones ────────────────────────────────────────────────────────────
    def _new_anti_target(self, anti):
        ar = self._p('arange')
        iv = [d for d in self.drones
              if d['alive'] and d['mode'] not in ('hit', 'intercepted')
              and _in_cone(anti['x'], anti['y'], anti['angle'],
                           ANTI_FOV, d['x'], d['y'], ar)]
        if not iv: return None
        iv.sort(key=lambda d: _dist(anti['x'], anti['y'], d['x'], d['y']))
        return iv[0]

    def _update_anti(self):
        aspd  = self._p('aspeed') / 3.6
        ar    = self._p('arange')
        mturn = DEFAULT_ANTI_TURN_RATE * DT
        for a in self.antidrones:
            if not a['alive']: continue
            if a['mode'] == 'base': continue

            a['battery'] -= DT
            if a['battery'] <= 0:
                a['alive'] = False
                continue

            if a['mode'] == 'waiting':
                a['launchDelay'] -= DT
                if a['launchDelay'] <= 0:
                    a['launchDelay'] = 0
                    pt = a['pendingTarget']
                    if pt and pt['alive'] and pt['mode'] not in ('hit', 'intercepted'):
                        a['target'] = pt
                        a['lastKnownX'] = pt['x']; a['lastKnownY'] = pt['y']
                        a['angle'] = math.atan2(pt['y'] - a['y'], pt['x'] - a['x'])
                        a['mode'] = 'intercept'
                    else:
                        nt = self._new_anti_target(a)
                        if nt:
                            a['target'] = nt
                            a['lastKnownX'] = nt['x']; a['lastKnownY'] = nt['y']
                            a['angle'] = math.atan2(nt['y'] - a['y'], nt['x'] - a['x'])
                            a['mode'] = 'intercept'
                        else:
                            a['mode'] = 'base'
                    a['pendingTarget'] = None
                continue
            # camera always active — can acquire any drone in FOV at any time
            iv = [d for d in self.drones
                  if d['alive'] and d['mode'] not in ('hit', 'intercepted')
                  and _in_cone(a['x'], a['y'], a['angle'], ANTI_FOV, d['x'], d['y'], ar)]
            if iv:
                iv.sort(key=lambda d: _dist(a['x'], a['y'], d['x'], d['y']))
                cl = iv[0]
                if a.get('target') is not cl:
                    a['target'] = cl
                    a['lastKnownX'] = cl['x']; a['lastKnownY'] = cl['y']
                a['mode'] = 'chase'
            if a['mode'] in ('intercept', 'chase', 'lastknown'):
                self._active_anti(a, aspd, mturn, ar)
            elif a['mode'] == 'circle':
                self._circle_anti(a, aspd)

    def _active_anti(self, a, aspd, mturn, ar):
        if a['mode'] != 'lastknown':
            t = a.get('target')
            if not t or not t['alive'] or t['mode'] in ('hit', 'intercepted'):
                nt = self._new_anti_target(a)
                if nt:
                    a['target'] = nt
                    a['lastKnownX'] = nt['x']; a['lastKnownY'] = nt['y']
                    a['mode'] = 'intercept'
                else:
                    a['mode'] = 'circle'; a['target'] = None; return
            elif _in_cone(a['x'], a['y'], a['angle'], ANTI_FOV, t['x'], t['y'], ar):
                a['lastKnownX'] = t['x']; a['lastKnownY'] = t['y']
                a['mode'] = 'chase'
            elif a['mode'] == 'chase':
                a['mode'] = 'circle'; return
            else:
                a['mode'] = 'lastknown'

        t = a.get('target')
        if a['mode'] == 'chase' and t and t['alive']:
            gx, gy = t['x'], t['y']
        elif a.get('lastKnownX') is not None:
            gx, gy = a['lastKnownX'], a['lastKnownY']
        else:
            a['mode'] = 'circle'; return

        t = a.get('target')
        if t and t['alive'] and t['mode'] not in ('hit', 'intercepted'):
            if _dist(a['x'], a['y'], t['x'], t['y']) < INTERCEPT_R:
                self._intercept(t); a['alive'] = False; return

        if a['mode'] == 'lastknown' and _dist(a['x'], a['y'], gx, gy) < 30:
            nt = self._new_anti_target(a)
            if nt:
                a['target'] = nt
                a['lastKnownX'] = nt['x']; a['lastKnownY'] = nt['y']
                a['mode'] = 'intercept'
            else:
                a['mode'] = 'circle'; a['target'] = None; return

        diff = _angle_diff(a['angle'], math.atan2(gy - a['y'], gx - a['x']))
        a['angle'] += _clamp(diff, -mturn, mturn)
        a['speed']  = _lerp(a['speed'], aspd, DT / 0.3)
        a['x'] = _clamp(a['x'] + math.cos(a['angle']) * a['speed'] * DT, 0, POLY)
        a['y'] = _clamp(a['y'] + math.sin(a['angle']) * a['speed'] * DT, 0, POLY)

    def _circle_anti(self, a, aspd):
        circle_rate = 15 * math.pi / 180
        a['angle'] += circle_rate * DT
        a['speed']  = _lerp(a['speed'], aspd, DT / 0.3)
        a['x'] = _clamp(a['x'] + math.cos(a['angle']) * a['speed'] * DT, 0, POLY)
        a['y'] = _clamp(a['y'] + math.sin(a['angle']) * a['speed'] * DT, 0, POLY)

    # ── factories ─────────────────────────────────────────────────────────────
    def _make_drone(self, id_, sp, t, ti):
        return dict(id=id_, x=sp['x'], y=sp['y'],
                    angle=math.atan2(POLY/2 - sp['y'], POLY/2 - sp['x']),
                    speed=self._p('dspeed') / 3.6,
                    tx=t['x'], ty=t['y'], rtx=t['x'], rty=t['y'],
                    targetIdx=ti, mode='approach', intent='attack',
                    alive=True, evading=False,
                    evadeWpt=None, predictPt=None, threatId=None,
                    evadeCooldown=0.0, deadZone=False)

    def _make_camera(self, id_, x, y, ang):
        return dict(id=id_, x=x, y=y, angle=ang, detected=None, cooldown=0.0)

    def _place_cameras(self):
        r = POLY * 0.15
        return [self._make_camera(i,
                    POLY/2 + math.cos(i / CAMERA_COUNT * math.pi * 2) * r,
                    POLY/2 + math.sin(i / CAMERA_COUNT * math.pi * 2) * r,
                    i / CAMERA_COUNT * math.pi * 2)
                for i in range(CAMERA_COUNT)]

    def _make_anti(self, id_, total):
        ang = (id_ / total * math.pi * 2) if total > 1 else 0.0
        r   = 40.0 if total > 1 else 0.0
        return dict(id=id_, x=BASE_X + math.cos(ang)*r, y=BASE_Y + math.sin(ang)*r,
                    angle=0.0, speed=0.0, target=None,
                    lastKnownX=None, lastKnownY=None,
                    mode='base', alive=True,
                    launchDelay=0.0, pendingTarget=None, battery=600.0)

    def _spawn_edge(self):
        m = DEPLOYMENT_MARGIN
        s = self.rng.rint(4)
        if s == 0: return dict(x=self.rng.rrange(m, POLY-m), y=m)
        if s == 1: return dict(x=self.rng.rrange(m, POLY-m), y=POLY-m)
        if s == 2: return dict(x=m, y=self.rng.rrange(m, POLY-m))
        return dict(x=POLY-m, y=self.rng.rrange(m, POLY-m))

    def _spawn_target(self):
        ang = self.rng.rrange(0, math.pi * 2)
        r   = self.rng.rrange(100, CENTER_ZONE)
        return dict(x=POLY/2 + math.cos(ang)*r, y=POLY/2 + math.sin(ang)*r, hit=False)

    # ── observation encoding (matches observations.js encodeTeamObservation) ──
    def _encode_obs(self):
        v    = []
        dspd = self._p('dspeed') / 3.6
        aspd = self._p('aspeed') / 3.6

        for i in range(MAX_DRONES):
            d = self.drones[i] if i < len(self.drones) else None
            if d is None:
                v += [0.0] * 13; continue
            t  = self.targets[d['targetIdx']] if d['targetIdx'] < len(self.targets) else None
            tx = t['x'] if t else d['rtx']
            ty = t['y'] if t else d['rty']
            v += [1,
                  1 if d['alive'] else 0,
                  d['x']/POLY, d['y']/POLY,
                  math.sin(d['angle']), math.cos(d['angle']),
                  d['speed']/dspd,
                  tx/POLY, ty/POLY,
                  _dist(d['x'], d['y'], tx, ty)/POLY,
                  1 if (t and t['hit']) else 0,
                  1 if d['evading'] else 0,
                  1 if d['mode'] in ('hit', 'intercepted') else 0]

        for i in range(MAX_ANTIDRONES):
            a = self.antidrones[i] if i < len(self.antidrones) else None
            if a is None:
                v += [0.0] * 8; continue
            v += [1,
                  1 if a['alive'] else 0,
                  0 if a['mode'] == 'base' else 1,
                  a['x']/POLY, a['y']/POLY,
                  math.sin(a['angle']), math.cos(a['angle']),
                  a['speed']/aspd]

        for i in range(MAX_DRONES):
            t = self.targets[i] if i < len(self.targets) else None
            if t is None:
                v += [0.0] * 5; continue
            is_base = abs(t['x'] - POLY/2) < 1 and abs(t['y'] - POLY/2) < 1
            v += [1, t['x']/POLY, t['y']/POLY,
                  1 if t['hit'] else 0,
                  1 if is_base else 0]

        for i in range(CAMERA_COUNT):
            c = self.cameras[i] if i < len(self.cameras) else None
            if c is None:
                v += [0.0] * 6; continue
            v += [1, c['x']/POLY, c['y']/POLY,
                  math.sin(c['angle']), math.cos(c['angle']),
                  1 if c['detected'] else 0]

        v += [self._p('dspeed')/300, self._p('aspeed')/500,
              self._p('camrange')/POLY, self._p('arange')/POLY,
              self.sim_time/300]

        return np.array(v, dtype=np.float32)

    def get_stats(self):
        return dict(
            hits=sum(1 for d in self.drones if d['mode'] == 'hit'),
            intercepted=sum(1 for d in self.drones if d['mode'] == 'intercepted'),
            total=len(self.drones), time=self.sim_time,
        )
