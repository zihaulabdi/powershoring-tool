# Morocco Powershoring: Industry Identification Tool

An interactive simulation tool that ranks energy-intensive products Morocco could attract by offering cheap renewable electricity. Users set the assumptions (energy thresholds, relocation theory, scoring weights) and the tool shows the resulting ranking.

Developed by the Harvard Growth Lab with the UM6P Economic Complexity Unit, as part of the OCP powershoring study.

## Pages

| Page | What it does |
|---|---|
| Introduction | What the tool is, what powershoring is, and the identification process |
| 1 · Candidate pool | Energy, electricity and trade thresholds that define the candidate products |
| 2 · Score and rank | Likelihood of relocation, then feasibility and attractiveness for Morocco |
| 3 · Scenarios | Runs four relocation theories and shows which industries rank highly under several |
| Compare | Puts two saved results side by side |

## Requirements

- Python 3.11 (3.9 to 3.12 also work)
- About 1 GB of memory. The app keeps the dataset (under 1 MB) and cached results in memory.
- No database, no external API calls and no secrets. All data ships in `data/`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run How_To.py
```

The app opens at http://localhost:8501.

## Run with Docker

```bash
docker build -t powershoring-tool .
docker run -d --restart unless-stopped -p 8501:8501 --name powershoring powershoring-tool
```

Health check: `GET /_stcore/health` returns `ok`.

## Hosting on a website

Streamlit is a Python web server, not a set of static files. It cannot be uploaded to a static web host; it needs a server or container that stays running.

1. **Run the app** on a server or container platform, using Docker (above) or `streamlit run How_To.py --server.port 8501 --server.address 0.0.0.0`.
2. **Put it behind your reverse proxy** (nginx, Apache, a cloud load balancer) on your domain. The proxy must forward **WebSocket** connections: Streamlit uses a WebSocket at `/_stcore/stream`, and the app will not load without it.
3. **To serve it under a sub-path** such as `https://example.org/powershoring/`, start Streamlit with `--server.baseUrlPath=powershoring` and route that path to the app.
4. **To embed it in an existing page**, point an `<iframe>` at the app's URL with `?embed=true` appended.

Example nginx block:

```nginx
location /powershoring/ {
    proxy_pass http://127.0.0.1:8501/powershoring/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
}
```

Theme and server settings are in `.streamlit/config.toml`.

## Versions and updates

Releases are tagged (for example `v1.0`). To host a fixed version:

```bash
git clone https://github.com/zihaulabdi/powershoring-tool.git
cd powershoring-tool
git checkout v1.0
```

To update later, `git fetch --tags`, check out the new tag, and rebuild or restart.

## Data and methodology

| Source | Used for |
|---|---|
| Growth Lab country-product exports, HS 2012 six-digit (2021 base year; growth 2012 to 2024) | Trade volumes, market size, growth, market concentration |
| US EPA USEEIO (2017) | Energy and electricity intensity per dollar of output |
| Atlas of Economic Complexity | Morocco RCA, density, COG; product complexity (PCI) |
| EU CBAM annex | CBAM coverage |
| Product distance data (HS 1992, mapped with WITS concordance) | Average trade distance |

Limitations: energy intensities assume US production technology for every country, and biomass raises the figures for paper and wood. Scores are percentile ranks within the pool the user defines, so they change when the pool changes.

The scoring logic is in `utils.py`. The methodology version is shown at the bottom of the Introduction page.

## Citation

> Abdi, Z. (2026). *Morocco Powershoring: Industry Identification Tool*. Harvard Growth Lab and UM6P Economic Complexity Unit. https://github.com/zihaulabdi/powershoring-tool

## License

Code: MIT License (see `LICENSE`). The data in `data/` is derived from the sources listed above and remains subject to their terms.

## Contact

Zihaul Abdi, Harvard Growth Lab: zihaulabdi@hks.harvard.edu
