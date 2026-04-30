# Lie Detection from Text

Small local web app for classifying a pasted review as more likely authentic or computer-generated.

## Run the Front End

```bash
python3 app.py --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

## Saved Model

The current saved model is:

```text
models/deception_model.pkl
```

The included artifact has been trained from:

```text
fake reviews dataset.csv
```

Current training size:

```text
40,432 rows
```

## Retrain With Your Dataset

Place `fake reviews dataset.csv` in this folder, then run:

```bash
python3 train_model.py --data "fake reviews dataset.csv"
```

The training script supports common column names:

- Text column: `text`, `text_`, `review`, `content`, or `message`
- Label column: `label`, `target`, `class`, or `category`
- Computer-generated labels: `CG`, `computer`, `AI`, `1`, `deceptive`, or `fake`
- Authentic labels: `OR`, `original`, `human`, `0`, `genuine`, `authentic`, or `real`

Restart the server after retraining so the app loads the new model.

## Deploy Online With Vercel

This repo includes Vercel-ready files:

```text
public/
api/
vercel.json
models/deception_model.pkl
```

From the project folder:

```bash
npx vercel deploy
```

For a production URL:

```bash
npx vercel deploy --prod
```

During setup, use these choices if Vercel asks:

```text
Framework Preset: Other
Build Command: none
Output Directory: public
Install Command: none
```

The frontend is served from `public/`, and prediction requests go to `/api/predict`.
