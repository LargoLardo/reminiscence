# Vercel Frontend + Modal GPU Deployment

The public React frontend runs on Vercel. Modal hosts a small CPU API and starts
one A10 GPU worker only while a reconstruction is running. Users do not need an
API key.

## Billing Guarantee

Set the Modal workspace budget to **$29/month** under:

**Modal Settings > Usage & Billing > Workspace budget**

This Modal-controlled workspace budget is the hard billing stop. It is below
the Starter plan's `$30/month` free compute credit, so Modal usage cannot exceed
the free credit while that budget remains enabled.

The app also fails closed after 30 accepted jobs per UTC month. Its explicit
resource ceilings produce a conservative maximum estimated compute cost of
`$23.26/month` at the Modal prices recorded in `modal_app.py`. The estimate
assumes:

- All 30 jobs run for the full 20-minute timeout.
- Every GPU container remains idle for its full scale-down window.
- The public API is kept busy continuously for a 31-day month.
- The daily cleanup job runs for its full timeout every day.

The `$29` workspace budget remains necessary because only Modal can enforce a
true dollar limit across price changes, startup overhead, and every app in the
workspace.

## App Limits

The limits are defined near the top of `modal_app.py`:

```text
GPU workers:          1 maximum
GPU job timeout:      20 minutes
Monthly jobs:         30 maximum
Upload size:          200 MB maximum
Result retention:     7 days
Training iterations:  7000
GPU allocation:       4 CPU cores and 32 GiB RAM
Web allocation:       0.125 CPU cores and 512 MiB RAM
```

There is no daily limit. All 30 monthly jobs can be used in one day. Accepted
jobs count toward the monthly limit even if they later fail, which prevents
failed-job spam from creating extra GPU spend.

Check the currently deployed limits and estimate:

```powershell
Invoke-RestMethod https://andy-jy-zhang--reminiscence-web.modal.run/api/health
```

Check the current monthly job counter:

```powershell
python -m modal dict items reminiscence-usage
```

Check Modal's recorded workspace usage for the current month:

```powershell
python -m modal billing report --for "this month" --json
```

## First Deployment

Create free accounts at [Modal](https://modal.com) and
[Vercel](https://vercel.com), then install and authenticate their CLIs:

```powershell
python -m pip install -r deploy\requirements.txt
python -m modal token new
npx vercel login
```

Deploy both services:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\deploy_cloud.ps1
```

The script deploys the Modal API and GPU worker, saves the endpoint to ignored
file `deploy/.modal-state.json`, configures `VITE_API_BASE_URL`, and deploys the
React frontend to Vercel.

## Redeploy Services

Redeploy Modal after changing `modal_app.py` or GPU behavior:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\deploy_modal.ps1
```

Redeploy Vercel after changing the React frontend:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\deploy_vercel.ps1
```

## Local Frontend Against Modal

After Modal has deployed, copy its URL from `deploy/.modal-state.json` into
`frontend/.env.local`:

```text
VITE_API_BASE_URL=https://your-modal-web-endpoint.modal.run
```

Then run:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` and upload a video.

## Architecture

- Vercel serves the static React/Vite app.
- Modal's CPU web function receives uploads and polls jobs.
- A Modal Volume stores uploaded videos and generated PLY files.
- A queued Modal A10 function runs COLMAP and FastGS. COLMAP's SIFT
  extraction and matching use CPU mode because its GPU mode requires an
  OpenGL display context; FastGS training and rendering still use the A10 GPU.
- Completed jobs return a downloadable PLY file.
- A daily cleanup job removes files older than seven days.

Modal function results also expire after seven days, so old job URLs stop
working even if a stale browser tab remains open.
