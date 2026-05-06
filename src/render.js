import {
  ANTI_FOV,
  CAM_FOV,
  CENTER_ZONE,
  DRONE_DETECT_R,
  DRONE_FOV,
  POLY,
} from './constants.js';

export class CanvasRenderer {
  constructor(canvas, sim) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.sim = sim;
    this.width = 0;
    this.height = 0;
    this.camOffX = 0;
    this.camOffY = 0;
    this.camUserZoom = 1.0;
    this.userPan = false;
    this.panStart = { x: 0, y: 0 };
    this.camStart = { x: 0, y: 0 };

    this.resize();
    this.bindInput();
    window.addEventListener('resize', () => this.resize());
  }

  resetView() {
    this.camOffX = 0;
    this.camOffY = 0;
    this.camUserZoom = 1.0;
  }

  resize() {
    const area = this.canvas.parentElement;
    this.width = area.clientWidth || 600;
    this.height = area.clientHeight || 600;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
  }

  baseZoom() {
    return this.width / POLY * 0.95;
  }

  getZoom() {
    return this.baseZoom() * this.camUserZoom;
  }

  w2s(wx, wy) {
    const zoom = this.getZoom();
    const cx = POLY / 2 + this.camOffX;
    const cy = POLY / 2 + this.camOffY;
    return {
      x: (wx - cx) * zoom + this.width / 2,
      y: (wy - cy) * zoom + this.height / 2,
    };
  }

  s2w(sx, sy) {
    const zoom = this.getZoom();
    const cx = POLY / 2 + this.camOffX;
    const cy = POLY / 2 + this.camOffY;
    return {
      x: (sx - this.width / 2) / zoom + cx,
      y: (sy - this.height / 2) / zoom + cy,
    };
  }

  bindInput() {
    this.canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 0.9;
      const rect = this.canvas.getBoundingClientRect();
      const mx = (event.clientX - rect.left) * (this.width / rect.width);
      const my = (event.clientY - rect.top) * (this.height / rect.height);
      const before = this.s2w(mx, my);
      this.camUserZoom = Math.max(0.3, Math.min(8, this.camUserZoom * factor));
      const after = this.s2w(mx, my);
      this.camOffX -= after.x - before.x;
      this.camOffY -= after.y - before.y;
    }, { passive: false });

    this.canvas.addEventListener('mousedown', (event) => {
      if (event.button !== 2) return;
      this.userPan = true;
      this.panStart = { x: event.clientX, y: event.clientY };
      this.camStart = { x: this.camOffX, y: this.camOffY };
    });

    this.canvas.addEventListener('mousemove', (event) => {
      if (!this.userPan) return;
      const zoom = this.getZoom();
      this.camOffX = this.camStart.x - (event.clientX - this.panStart.x) / zoom;
      this.camOffY = this.camStart.y - (event.clientY - this.panStart.y) / zoom;
    });

    this.canvas.addEventListener('mouseup', () => {
      this.userPan = false;
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.userPan = false;
    });

    this.canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  draw() {
    const ctx = this.ctx;
    const zoom = this.getZoom();

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = '#0d0f0e';
    ctx.fillRect(0, 0, this.width, this.height);

    this.drawMap(zoom);
    this.drawCameras(zoom);
    this.drawTrails();
    this.drawTargets(zoom);
    this.drawDrones(zoom);
    this.drawWaitingAntidrones();
    this.drawActiveAntidrones(zoom);
    this.drawExplosions(zoom);

    const p0 = this.w2s(0, 0);
    const p1 = this.w2s(POLY, POLY);
    ctx.fillStyle = 'rgba(78,203,113,0.2)';
    ctx.font = '10px "Share Tech Mono"';
    ctx.fillText('2500m x 2500m', p0.x + 6, p1.y - 6);
  }

  drawMap(zoom) {
    const ctx = this.ctx;
    const p0 = this.w2s(0, 0);
    const p1 = this.w2s(POLY, POLY);

    ctx.strokeStyle = 'rgba(78,203,113,0.15)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);
    ctx.fillStyle = 'rgba(78,203,113,0.015)';
    ctx.fillRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y);

    ctx.strokeStyle = 'rgba(78,203,113,0.06)';
    ctx.lineWidth = 1;
    for (let g = 0; g <= POLY; g += 500) {
      const sx = this.w2s(g, 0);
      const sy = this.w2s(0, g);
      ctx.beginPath();
      ctx.moveTo(sx.x, p0.y);
      ctx.lineTo(sx.x, p1.y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p0.x, sy.y);
      ctx.lineTo(p1.x, sy.y);
      ctx.stroke();
    }

    const center = this.w2s(POLY / 2, POLY / 2);
    ctx.beginPath();
    ctx.arc(center.x, center.y, CENTER_ZONE * zoom, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(224,75,74,0.15)';
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(center.x, center.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(224,75,74,0.6)';
    ctx.fill();
    ctx.fillStyle = 'rgba(78,203,113,0.4)';
    ctx.font = '10px "Share Tech Mono"';
    ctx.fillText('BASE', center.x + 10, center.y + 4);
  }

  drawCameras(zoom) {
    const ctx = this.ctx;
    const camRange = this.sim.getParam('camrange');

    this.sim.cameras.forEach((camera) => {
      const cs = this.w2s(camera.x, camera.y);
      const active = camera.detected !== null;
      ctx.beginPath();
      ctx.moveTo(cs.x, cs.y);
      ctx.arc(cs.x, cs.y, camRange * zoom, camera.angle - CAM_FOV / 2, camera.angle + CAM_FOV / 2);
      ctx.closePath();
      ctx.fillStyle = active ? 'rgba(232,168,48,0.08)' : 'rgba(91,163,232,0.04)';
      ctx.fill();
      ctx.strokeStyle = active ? 'rgba(232,168,48,0.55)' : 'rgba(91,163,232,0.18)';
      ctx.lineWidth = active ? 1.5 : 0.8;
      ctx.stroke();
      ctx.fillStyle = active ? '#e8a830' : 'rgba(91,163,232,0.5)';
      ctx.beginPath();
      ctx.arc(cs.x, cs.y, Math.max(2, 3 * zoom), 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawTrails() {
    const ctx = this.ctx;

    this.sim.drones.forEach((drone) => {
      if (!drone.alive || drone.trail.length < 2) return;
      for (let i = 1; i < drone.trail.length; i++) {
        const s0 = this.w2s(drone.trail[i - 1].x, drone.trail[i - 1].y);
        const s1 = this.w2s(drone.trail[i].x, drone.trail[i].y);
        const alpha = i / drone.trail.length;
        const col = drone.col;
        const r = parseInt(col.slice(1, 3), 16);
        const g = parseInt(col.slice(3, 5), 16);
        const b = parseInt(col.slice(5, 7), 16);
        ctx.beginPath();
        ctx.moveTo(s0.x, s0.y);
        ctx.lineTo(s1.x, s1.y);
        ctx.strokeStyle = `rgba(${r},${g},${b},${alpha * 0.55})`;
        ctx.lineWidth = alpha * 2.5;
        ctx.stroke();
      }
    });

    this.sim.antidrones.forEach((anti) => {
      if (!anti.alive || anti.trail.length < 2) return;
      for (let i = 1; i < anti.trail.length; i++) {
        const s0 = this.w2s(anti.trail[i - 1].x, anti.trail[i - 1].y);
        const s1 = this.w2s(anti.trail[i].x, anti.trail[i].y);
        const alpha = i / anti.trail.length;
        ctx.beginPath();
        ctx.moveTo(s0.x, s0.y);
        ctx.lineTo(s1.x, s1.y);
        ctx.strokeStyle = `rgba(224,75,74,${alpha * 0.55})`;
        ctx.lineWidth = alpha * 2.5;
        ctx.stroke();
      }
    });
  }

  drawTargets(zoom) {
    const ctx = this.ctx;
    this.sim.targets.forEach((target, i) => {
      if (target.hit) return;
      const ts = this.w2s(target.x, target.y);
      const pulse = 0.5 + 0.5 * Math.sin(this.sim.simTime * 3 + i);
      ctx.strokeStyle = `rgba(232,168,48,${0.35 + pulse * 0.3})`;
      ctx.lineWidth = 1.5;
      const size = Math.max(4, 7 * zoom);
      ctx.beginPath();
      ctx.moveTo(ts.x - size, ts.y);
      ctx.lineTo(ts.x + size, ts.y);
      ctx.moveTo(ts.x, ts.y - size);
      ctx.lineTo(ts.x, ts.y + size);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(ts.x, ts.y, size * 0.55, 0, Math.PI * 2);
      ctx.stroke();
    });
  }

  drawDrones(zoom) {
    const ctx = this.ctx;

    this.sim.drones.forEach((drone) => {
      if (!drone.alive || drone.mode === 'intercepted') return;
      const ds = this.w2s(drone.x, drone.y);
      const scale = Math.max(0.4, Math.min(1.8, zoom * 3.2));

      if (drone.evading && drone.evadeWpt && drone.predictPt) {
        this.drawDroneEvasion(drone, ds, scale, zoom);
      }

      this.drawShape(ds.x, ds.y, drone.angle, scale, drone.col, drone.col);
      ctx.fillStyle = 'rgba(200,216,192,0.35)';
      ctx.font = `${Math.max(9, 10 * zoom)}px "Share Tech Mono"`;
      ctx.fillText(`D${drone.id + 1}`, ds.x + 13 * scale, ds.y - 9 * scale);
    });
  }

  drawDroneEvasion(drone, ds, scale, zoom) {
    const ctx = this.ctx;

    ctx.beginPath();
    ctx.moveTo(ds.x, ds.y);
    ctx.arc(ds.x, ds.y, DRONE_DETECT_R * zoom, drone.angle - DRONE_FOV / 2, drone.angle + DRONE_FOV / 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(232,168,48,0.07)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(232,168,48,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();

    const pp = this.w2s(drone.predictPt.x, drone.predictPt.y);
    ctx.strokeStyle = 'rgba(224,75,74,0.55)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pp.x - 5, pp.y - 5);
    ctx.lineTo(pp.x + 5, pp.y + 5);
    ctx.moveTo(pp.x + 5, pp.y - 5);
    ctx.lineTo(pp.x - 5, pp.y + 5);
    ctx.stroke();

    if (drone.threatId !== null) {
      const threat = this.sim.antidrones.find((anti) => anti.id === drone.threatId);
      if (threat && threat.alive) {
        const as = this.w2s(threat.x, threat.y);
        ctx.beginPath();
        ctx.moveTo(as.x, as.y);
        ctx.lineTo(pp.x, pp.y);
        ctx.strokeStyle = 'rgba(224,75,74,0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    const wp = this.w2s(drone.evadeWpt.x, drone.evadeWpt.y);
    ctx.beginPath();
    ctx.arc(wp.x, wp.y, 6, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(91,163,232,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(wp.x, wp.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#5ba3e8';
    ctx.fill();

    const target = this.w2s(drone.rtx, drone.rty);
    ctx.beginPath();
    ctx.moveTo(ds.x, ds.y);
    ctx.lineTo(wp.x, wp.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = 'rgba(91,163,232,0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e8a830';
    ctx.font = `bold ${Math.max(10, 11 * zoom)}px "Share Tech Mono"`;
    ctx.fillText('!', ds.x - 3, ds.y - 15 * scale);
  }

  drawWaitingAntidrones() {
    const ctx = this.ctx;
    const delay = Math.max(1, this.sim.getParam('adelay'));

    this.sim.antidrones.forEach((anti) => {
      if (!anti.alive || anti.mode !== 'waiting') return;
      const ds = this.w2s(anti.x, anti.y);
      const frac = anti.launchDelay / delay;
      ctx.beginPath();
      ctx.arc(ds.x, ds.y, 12, -Math.PI / 2, Math.PI * 2 * frac - Math.PI / 2);
      ctx.strokeStyle = 'rgba(232,168,48,0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = 'rgba(232,168,48,0.8)';
      ctx.font = '9px "Share Tech Mono"';
      ctx.fillText(`A${anti.id + 1}:${Math.ceil(anti.launchDelay)}s`, ds.x + 16, ds.y);
    });
  }

  drawActiveAntidrones(zoom) {
    const ctx = this.ctx;
    const antiRange = this.sim.getParam('arange');

    this.sim.antidrones.forEach((anti) => {
      if (!anti.alive || anti.mode === 'base' || anti.mode === 'waiting') return;
      const ds = this.w2s(anti.x, anti.y);
      const scale = Math.max(0.4, Math.min(1.8, zoom * 3.2));

      if (anti.mode === 'chase' || anti.mode === 'intercept' || anti.mode === 'lastknown') {
        ctx.beginPath();
        ctx.moveTo(ds.x, ds.y);
        ctx.arc(ds.x, ds.y, antiRange * zoom, anti.angle - ANTI_FOV / 2, anti.angle + ANTI_FOV / 2);
        ctx.closePath();
        ctx.fillStyle = anti.mode === 'chase' ? 'rgba(224,75,74,0.1)' : 'rgba(224,75,74,0.04)';
        ctx.fill();
        ctx.strokeStyle = anti.mode === 'chase' ? 'rgba(224,75,74,0.4)' : 'rgba(224,75,74,0.18)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      if (anti.target && anti.target.alive) {
        const ts = this.w2s(anti.target.x, anti.target.y);
        ctx.beginPath();
        ctx.moveTo(ds.x, ds.y);
        ctx.lineTo(ts.x, ts.y);
        ctx.strokeStyle = 'rgba(224,75,74,0.28)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (anti.mode === 'lastknown' && anti.lastKnownX !== null) {
        const ls = this.w2s(anti.lastKnownX, anti.lastKnownY);
        ctx.beginPath();
        ctx.arc(ls.x, ls.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(224,75,74,0.4)';
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(ds.x, ds.y);
        ctx.lineTo(ls.x, ls.y);
        ctx.strokeStyle = 'rgba(224,75,74,0.18)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      this.drawShape(ds.x, ds.y, anti.angle, scale, '#e04b4b', '#c03030');
      ctx.fillStyle = 'rgba(224,75,74,0.75)';
      ctx.font = `${Math.max(9, 10 * zoom)}px "Share Tech Mono"`;
      ctx.fillText(`A${anti.id + 1}`, ds.x + 13 * scale, ds.y - 9 * scale);
    });
  }

  drawExplosions(zoom) {
    const ctx = this.ctx;

    this.sim.explosions.forEach((explosion) => {
      const es = this.w2s(explosion.x, explosion.y);
      const radius = explosion.t;
      ctx.beginPath();
      ctx.arc(es.x, es.y, 50 * radius * zoom, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(224,75,74,${radius * 0.45})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(es.x, es.y, 25 * radius * zoom, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232,168,48,${radius * 0.65})`;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(es.x, es.y, 10 * radius * zoom, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${radius * 0.85})`;
      ctx.fill();
    });
  }

  drawShape(x, y, angle, scale, bodyColor, wingColor) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(16 * scale, 0);
    ctx.lineTo(-9 * scale, -7 * scale);
    ctx.lineTo(-5 * scale, 0);
    ctx.lineTo(-9 * scale, 7 * scale);
    ctx.closePath();
    ctx.fillStyle = bodyColor;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(3 * scale, 0);
    ctx.lineTo(-4 * scale, -12 * scale);
    ctx.lineTo(-8 * scale, -12 * scale);
    ctx.lineTo(-5 * scale, 0);
    ctx.lineTo(-8 * scale, 12 * scale);
    ctx.lineTo(-4 * scale, 12 * scale);
    ctx.closePath();
    ctx.fillStyle = wingColor;
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-5 * scale, 0);
    ctx.lineTo(-9 * scale, -4 * scale);
    ctx.lineTo(-11 * scale, 0);
    ctx.lineTo(-9 * scale, 4 * scale);
    ctx.closePath();
    ctx.fillStyle = bodyColor;
    ctx.fill();
    ctx.restore();
  }
}
