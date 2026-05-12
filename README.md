# PolygonSite-ML

ML-песочница для симулятора перехвата дронов на полигоне 2500×2500 м.
Атакующие дроны управляются ML-политикой, антидроны и камеры остаются rule-based.

## Архитектура

```
.
├── index.html                  # браузерный интерфейс
├── styles.css
├── src/
│   ├── app.js                  # связывает UI, симуляцию и рендер
│   ├── constants.js            # параметры симуляции
│   ├── math.js                 # геометрия, angleDiff, createRng
│   ├── simulation.js           # JS-симулятор (браузер + eval)
│   ├── render.js               # Canvas-отрисовка
│   ├── charts.js               # Chart.js графики результатов и обучения
│   ├── policies/
│   │   ├── dronePolicy.js      # RuleBasedDronePolicy, TeamWaypointMlDronePolicy
│   │   └── teamMlStub.js       # заглушка-ML для проверки контракта
│   └── ml/
│       └── observations.js     # encodeTeamObservation, decodeTeamWaypointActions
├── scripts/
│   ├── smoke-test.mjs          # headless-проверка симуляции
│   ├── runner.mjs              # запуск эпизодов, запись результатов
│   ├── recorder.mjs            # запись эпизодов для behavioral cloning
│   ├── eval_model.mjs          # оценка ONNX-модели в Node.js
│   └── dump_obs.mjs            # дамп наблюдений JS-сима (для теста паритета)
├── py/
│   └── sim.py                  # Python-порт симулятора (для RL-обучения)
├── train.py                    # behavioral cloning (BC, устарел)
├── train_rl.py                 # PPO RL обучение
├── export_onnx.py              # экспорт model.pt → model_web.onnx
├── validate_onnx.py            # проверка ONNX-файла
└── test_sim_parity.py          # проверка совпадения JS и Python симуляторов
```

## ML-контракт

### Наблюдение — 455 float32

| Блок          | Слоты | Фичи | Итого |
|---------------|-------|------|-------|
| Дроны         | 15    | 13   | 195   |
| Антидроны     | 15    | 8    | 120   |
| Цели          | 15    | 5    | 75    |
| Камеры        | 10    | 6    | 60    |
| Глобальные    | —     | 5    | 5     |

### Действие — 45 float32, диапазон [-1, 1]

На каждый слот дрона: `[dx, dy, intent]`
- `dx`, `dy` — смещение waypoint, масштабируется на `DRONE_DETECT_R = 600 м`
- `intent < 0` → режим уклонения, `intent ≥ 0` → атака

### Reward

```
+1.0  дрон поразил цель
 0.0  дрон перехвачен (жертвовать дроном легитимно — decoy тактика)
−0.5  дрон вышел за лимит времени не достигнув цели
```

## Запуск браузера

```bash
npm run serve        # http://localhost:8000
npm run smoke        # headless-проверка
```

## Обучение (PPO RL)

```bash
# запуск обучения
python train_rl.py --updates 2000 --envs 8

# возобновление с чекпоинта
python train_rl.py --updates 2000 --resume data/model_rl.pt

# ключевые параметры (дефолты)
--updates 2000      # число PPO-апдейтов
--envs 8            # параллельных сред
--rollout 1024      # шагов на среду на апдейт (~5.5 сек/апдейт)
--eval-every 20     # оценивать каждые N апдейтов
--eval-eps 20       # эпизодов на оценку
--lr 1e-4           # learning rate
--clip 0.1          # PPO clip ratio
--out results.json  # куда писать результаты
--log data/train_log.json
```

После каждого нового лучшего результата модель автоматически экспортируется
в `data/model_web.onnx` и сразу доступна в браузере через кнопку ML Policy → ONNX.

**Архитектура политики:** ActorCritic 455→256→128, Gaussian с фиксированным
`log_std = −0.5` (std ≈ 0.61). Std не обучается — предотвращает дрейф энтропии
при sparse rewards.

## Оценка модели

```bash
# оценить ONNX в Node.js (тот же контракт что и браузер)
npm run eval
# или явно
node scripts/eval_model.mjs --model data/model_web.onnx --episodes 200
```

## Паритет симуляторов

JS-симулятор (`src/simulation.js`) и Python-порт (`py/sim.py`) должны давать
идентичные наблюдения для одинаковых сидов. Проверяется тестом:

```bash
npm run parity       # или: python test_sim_parity.py
```

Запускать после изменений в любом из двух симуляторов.

## Behavioral Cloning (устарел)

```bash
# запись эпизодов rule-based политики
npm run record       # → data/episodes.ndjson

# обучение BC-модели
python train.py

# экспорт в ONNX
python export_onnx.py
```

BC-модель давала hit_rate ≈ 0.01 из-за covariate shift.
Поэтому перешли на PPO RL.

## Результаты

Кнопка **ML Results** в браузере показывает (автообновление каждые 5 сек):
- **Bar chart** — Hit rate / Intercept rate / Mean reward по политикам
- **Таблица** — все прогоны с параметрами и временем обучения
- **Entropy** — кривая энтропии политики из `data/train_log.json`
