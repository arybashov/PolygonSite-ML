export const DT = 1 / 60;
export const POLY = 2500;
export const MS = 1;

export const ARRIVAL_R = 20;
export const INTERCEPT_R = 25;
export const ATTACK_COMMIT_DIST = 250;

export const CAM_FOV = 50 * Math.PI / 180;
export const ANTI_FOV = 50 * Math.PI / 180;
export const DRONE_FOV = Math.PI * 2;
export const DRONE_DETECT_R = 600;

export const BASE_X = POLY / 2;
export const BASE_Y = POLY / 2;
export const CAMERA_COUNT = 10;
export const CENTER_ZONE = 800;
export const TRAIL_LEN = 600;
export const DEPLOYMENT_MARGIN = 50;

export const DEFAULT_DRONE_TURN_RATE = 30 * (1 - (50 - 10) / 190 * 0.5) * Math.PI / 180;
export const DEFAULT_ANTI_TURN_RATE = 60 * Math.PI / 180;

export const MAX_DRONES = 15;
export const MAX_ANTIDRONES = 15;

export const DEFAULT_PARAMS = Object.freeze({
  dspeed: 200,
  aspeed: 300,
  camrange: 900,
  arange: 600,
  ndrones: 5,
  nanti: 5,
  adelay: 5,
});

export const PARAM_DEFS = Object.freeze([
  { id: 'dspeed', label: 'Скорость дрона (км/ч)', min: 80, max: 300, step: 10 },
  { id: 'aspeed', label: 'Скорость антидрона (км/ч)', min: 150, max: 500, step: 10 },
  { id: 'camrange', label: 'Дальность камеры (м)', min: 300, max: 2500, step: 50 },
  { id: 'arange', label: 'Дальность антидрона (м)', min: 200, max: 800, step: 50 },
  { id: 'ndrones', label: 'Кол-во дронов', min: 1, max: MAX_DRONES, step: 1 },
  { id: 'nanti', label: 'Кол-во антидронов', min: 1, max: MAX_ANTIDRONES, step: 1 },
  { id: 'adelay', label: 'Задержка вылета (с)', min: 0, max: 30, step: 1 },
]);

export const DRONE_COLS = Object.freeze([
  '#5ba3e8',
  '#3ecaa5',
  '#7ec87a',
  '#e8a830',
  '#c87aad',
  '#9a8fe0',
  '#e87a5b',
  '#5be8c8',
  '#e8d85b',
  '#a0e85b',
  '#5b8be8',
  '#e85ba0',
  '#aaa0ff',
  '#ffaa60',
  '#60ffcc',
]);
