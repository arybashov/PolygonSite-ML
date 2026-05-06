export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function angleDiff(a, b) {
  let d = b - a;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

export function distance(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return Math.sqrt(dx * dx + dy * dy);
}

export function inCone(ox, oy, angle, fov, tx, ty, range) {
  const dx = tx - ox;
  const dy = ty - oy;
  const d = Math.sqrt(dx * dx + dy * dy);
  return d <= range && Math.abs(angleDiff(angle, Math.atan2(dy, dx))) <= fov / 2;
}

export function createRng(seed = Date.now()) {
  let state = seed >>> 0;
  if (state === 0) state = 0x9e3779b9;

  return {
    next() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return ((state >>> 0) / 0x100000000);
    },
    range(min, max) {
      return min + this.next() * (max - min);
    },
    int(maxExclusive) {
      return Math.floor(this.next() * maxExclusive);
    },
  };
}
